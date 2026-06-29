"""
Quick smoke test: verifies that the ambidrop package, training, and inference
pipelines all work end-to-end on the locally available data.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader

from FT_JNF.model import FT_JNF
from ambidrop.losses import si_snr
from FT_JNF.datasets import SimDS_preprocessed
from ambidrop.checkpoint import load_checkpoint
from ambidrop.constants import N_FFT, HOP_LENGTH, WIN_LENGTH
from FT_JNF.constants import CHECKPOINT_REGISTRY

BASE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)


def train_one_step(mode, net, data):
    """Run one training step, return loss."""
    x = data['noisy']
    s = data['clean'].float()
    num_ch = x.shape[-1] // 2

    M = net(x)
    Ms = M[..., 0] + 1j * M[..., 1]

    if mode == 'baseline':
        ref_id = data['ref_id']
        B = x.shape[0]
        idx = torch.arange(B)
        s = s[idx, ref_id, :]
        Y = x[idx, :, :, ref_id] + 1j * x[idx, :, :, num_ch + ref_id]
    else:
        Y = x[:, :, :, 0] + 1j * x[:, :, :, num_ch]

    S_hat = Ms * Y
    win = torch.hamming_window(WIN_LENGTH)
    s_hat = torch.istft(S_hat.permute(0, 2, 1), n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window=win, center=True,
                        normalized=False, onesided=True, return_complex=False,
                        length=s.shape[-1])
    return -si_snr(s_hat, s).mean()


print("=" * 60)
print("TEST 1: Baseline FT-JNF training step")
print("=" * 60)
ds1 = SimDS_preprocessed(os.path.join(BASE, 'datasets/experiment_full_anm'), 'mic_train_ds_preprocessed_merged')
loader1 = DataLoader(ds1, batch_size=2, shuffle=False)
net1 = FT_JNF(input_dim=14, hidden1_dim=64, hidden2_dim=64, output_dim=2)
opt1 = torch.optim.Adam(net1.parameters(), lr=0.001)
net1.train()
for data in loader1:
    opt1.zero_grad()
    loss = train_one_step('baseline', net1, data)
    loss.backward()
    opt1.step()
    print(f"  loss: {loss.item():.4f}")
    break
print("  PASSED\n")

print("=" * 60)
print("TEST 2: AmbiDrop FT-JNF training step")
print("=" * 60)
ds2 = SimDS_preprocessed(BASE, 'si_tr_s_preprocessed_full')
loader2 = DataLoader(ds2, batch_size=2, shuffle=False)
net2 = FT_JNF(input_dim=18, hidden1_dim=64, hidden2_dim=64, output_dim=2,
              dropout_type='SHChannelDropout', drop_prob=0.4, max_drop=3)
opt2 = torch.optim.Adam(net2.parameters(), lr=0.001)
net2.train()
for data in loader2:
    opt2.zero_grad()
    loss = train_one_step('ambidrop', net2, data)
    loss.backward()
    opt2.step()
    print(f"  loss: {loss.item():.4f}")
    break
print("  PASSED\n")

print("=" * 60)
print("TEST 3: Checkpoint loading + inference")
print("=" * 60)
ckpt_file = 'SH_FT_JNF,2025-12-01_10-08-18.pt'
ckpt_path = os.path.join(BASE, 'checkpoints', 'FT_JNF', ckpt_file)
if os.path.exists(ckpt_path):
    reg = CHECKPOINT_REGISTRY[ckpt_file]
    net3 = FT_JNF(input_dim=reg['input_dim'], hidden1_dim=reg['hidden1'],
                  hidden2_dim=reg['hidden2'], output_dim=2,
                  dropout_type=reg.get('dropout'), drop_prob=reg.get('drop_prob', 0),
                  max_drop=reg.get('max_drop', 0))
    info = load_checkpoint(ckpt_path, target_epoch=200, net=net3)
    net3.eval()
    x_test = torch.randn(1, 10, 257, 18)
    out = net3(x_test)
    print(f"  Loaded epoch {info['epoch']}, output shape: {out.shape}")
    print("  PASSED\n")
else:
    print(f"  SKIPPED (checkpoint not found)\n")

print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
