"""
Unified training script for AmbiDrop and baseline FT-JNF models.

Examples:
    # Baseline FT-JNF (microphone input)
    python scripts/train.py --mode baseline \
        --data-dir datasets/experiment_full_anm \
        --train-split mic_train_ds_preprocessed_merged \
        --val-split mic_val_ds_preprocessed_merged \
        --input-dim 14 --hidden1 64 --hidden2 64

    # AmbiDrop FT-JNF (Ambisonics + SHChannelDropout)
    python scripts/train.py --mode ambidrop \
        --data-dir . \
        --train-split si_tr_s_preprocessed_full \
        --val-split si_dt_05_preprocessed_full \
        --input-dim 18 --hidden1 64 --hidden2 64 \
        --dropout-type SHChannelDropout --drop-prob 0.4 --max-drop 3

    # AmbiDrop with PerChDropout
    python scripts/train.py --mode ambidrop \
        --input-dim 18 --hidden1 64 --hidden2 64 \
        --dropout-type PerChDropout \
        --drop-probs 0,0.1,0.45,0.1,0.45,1,0.75,1,0.45
"""

import os
import sys
import argparse
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ambidrop.models import FT_JNF
from ambidrop.losses import si_snr
from ambidrop.datasets import SimDS_preprocessed
from ambidrop.checkpoint import load_checkpoint, save_checkpoint
from ambidrop.signal_utils import get_lr
from ambidrop.constants import N_FFT, HOP_LENGTH, WIN_LENGTH, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Train FT-JNF (baseline or AmbiDrop)")
    p.add_argument('--mode', choices=['baseline', 'ambidrop'], required=True)

    p.add_argument('--data-dir', default='datasets/experiment_full_anm')
    p.add_argument('--train-split', default='mic_train_ds_preprocessed_merged')
    p.add_argument('--val-split', default='mic_val_ds_preprocessed_merged')

    p.add_argument('--input-dim', type=int, default=14)
    p.add_argument('--hidden1', type=int, default=64)
    p.add_argument('--hidden2', type=int, default=64)
    p.add_argument('--output-dim', type=int, default=2)

    p.add_argument('--dropout-type', default=None, choices=[None, 'SHChannelDropout', 'PerChDropout'])
    p.add_argument('--drop-prob', type=float, default=0.4)
    p.add_argument('--max-drop', type=int, default=3)
    p.add_argument('--drop-probs', type=str, default=None,
                   help='Comma-separated per-channel drop probabilities')

    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=1e-6)
    p.add_argument('--max-batches', type=int, default=None,
                   help='Limit training batches per epoch (for testing)')

    p.add_argument('--checkpoint', default=None, help='Resume from checkpoint')
    p.add_argument('--save-dir', default='checkpoints')

    p.add_argument('--wandb-project', default='speech-enhancement')
    p.add_argument('--wandb-entity', default='tatarjit-ben-gurion-university-of-the-negev')
    p.add_argument('--no-wandb', action='store_true')

    return p.parse_args()


def training_step_baseline(net, data, device):
    """Baseline training step: per-sample reference channel from dataset."""
    x = data['noisy'].to(device)
    s = data['clean'].float().to(device)
    ref_id = data['ref_id']

    B = x.shape[0]
    num_ch = x.shape[-1] // 2
    batch_idx = torch.arange(B, device=device)

    s = s[batch_idx, ref_id, :]

    M = net(x)
    Ms = M[..., 0] + 1j * M[..., 1]

    Y = x[batch_idx, :, :, ref_id] + 1j * x[batch_idx, :, :, num_ch + ref_id]

    S_hat = Ms * Y
    win = torch.hamming_window(WIN_LENGTH, device=device)
    s_hat = torch.istft(S_hat.permute(0, 2, 1), n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window=win, center=True,
                        normalized=False, onesided=True, return_complex=False,
                        length=s.shape[1])

    if torch.isnan(s_hat).any() or torch.isnan(s).any():
        return torch.tensor(0.0, requires_grad=True, device=device)

    return -si_snr(s_hat, s).mean()


def training_step_ambidrop(net, data, device):
    """AmbiDrop training step: always uses channel 0 (a00) as reference."""
    x = data['noisy'].to(device)
    s = data['clean'].float().to(device)

    num_sh = x.shape[-1] // 2

    M = net(x)
    Ms = M[..., 0] + 1j * M[..., 1]

    Y = x[:, :, :, 0] + 1j * x[:, :, :, num_sh]

    S_hat = Ms * Y
    win = torch.hamming_window(WIN_LENGTH, device=device)
    s_hat = torch.istft(S_hat.permute(0, 2, 1), n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window=win, center=True,
                        normalized=False, onesided=True, return_complex=False,
                        length=s.shape[1])

    if torch.isnan(s_hat).any() or torch.isnan(s).any():
        return torch.tensor(0.0, requires_grad=True, device=device)

    return -si_snr(s_hat, s).mean()


def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(0)
    current_time = datetime.now()

    train_step_fn = training_step_baseline if args.mode == 'baseline' else training_step_ambidrop

    drop_probs = None
    if args.drop_probs:
        drop_probs = [float(x) for x in args.drop_probs.split(',')]

    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_dir}/{args.train_split}")
    print(f"Network: input_dim={args.input_dim}, hidden=({args.hidden1},{args.hidden2})")
    if args.mode == 'ambidrop':
        print(f"Dropout: {args.dropout_type}, drop_prob={args.drop_prob}, max_drop={args.max_drop}")
        if drop_probs:
            print(f"  Per-channel probs: {drop_probs}")

    train_ds = SimDS_preprocessed(args.data_dir, args.train_split)
    val_ds = SimDS_preprocessed(args.data_dir, args.val_split)

    generator = torch.Generator(device=device)
    train_sampler = RandomSampler(train_ds, generator=generator)

    trainloader = DataLoader(train_ds, sampler=train_sampler,
                             batch_size=args.batch_size, drop_last=True)
    valloader = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, drop_last=True)

    net = FT_JNF(
        input_dim=args.input_dim,
        hidden1_dim=args.hidden1,
        hidden2_dim=args.hidden2,
        output_dim=args.output_dim,
        dropout_type=args.dropout_type,
        drop_prob=args.drop_prob,
        max_drop=args.max_drop,
        drop_probs=drop_probs,
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    prev_loss = float('inf')
    if args.checkpoint:
        info = load_checkpoint(args.checkpoint, net=net, optimizer=optimizer)
        start_epoch = info['epoch'] + 1
        prev_loss = info['loss'] if info['loss'] is not None else float('inf')

    if not args.no_wandb:
        wandb.login()
        prefix = "FT_JNF" if args.mode == 'baseline' else "SH_FT_JNF"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{prefix}_train_{current_time:%Y%m%d_%H%M%S}",
            config=vars(args),
        )
        wandb.watch(net, log="all", log_freq=100)

    writer = SummaryWriter()
    os.makedirs(args.save_dir, exist_ok=True)
    prefix = "FT_JNF" if args.mode == 'baseline' else "SH_FT_JNF"
    ckpt_filename = f'{prefix},{current_time:%Y-%m-%d_%H-%M-%S}.pt'
    ckpt_path = os.path.join(args.save_dir, ckpt_filename)

    print(f"Training for {args.epochs} epochs, saving to {ckpt_path}")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        train_loss = 0.0

        for i, data in enumerate(trainloader):
            if args.max_batches and i >= args.max_batches:
                break
            optimizer.zero_grad()
            loss = train_step_fn(net, data, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 99:
                print(f'  epoch {epoch+1}, iter {i+1}, loss: {train_loss/i:.3f}')

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        n_batches = min(i + 1, args.max_batches) if args.max_batches else i + 1
        total_train_loss = train_loss / max(n_batches, 1)
        writer.add_scalar("Loss", total_train_loss, epoch)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                loss = train_step_fn(net, data, device)
                val_loss += loss.item()
            total_val_loss = val_loss / max(i + 1, 1)

        writer.add_scalar("val_loss", total_val_loss, epoch)
        print(f'Epoch {epoch+1}: train_loss={total_train_loss:.4f}, val_loss={total_val_loss:.4f}')

        if total_val_loss < prev_loss:
            prev_loss = total_val_loss
            save_checkpoint(ckpt_path, epoch, net, optimizer, total_val_loss)
            print(f'  Model saved as {ckpt_filename}')

        if not args.no_wandb:
            wandb.log({"lr": get_lr(optimizer), "train_loss": total_train_loss,
                       "val_loss": total_val_loss}, step=epoch + 1)

    print('Finished Training')
    if not args.no_wandb:
        wandb.finish()
    writer.close()


if __name__ == '__main__':
    main()
