"""
Training loop manager for IC Conv-TasNet with learning-rate halving and early stopping.

Public interface:
    Solver — orchestrates train/validation epochs, checkpoint saving, and wandb logging
"""
import os
import time

import torch
from ConvTasNet.loss import cal_loss
import wandb


class Solver(object):

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        self.use_cuda = args.use_cuda
        self.mode = getattr(args, 'mode', 'ambidrop')
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm

        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        self.print_freq = args.print_freq

        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _unwrap(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _reset(self):
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from, map_location="cpu")
            self._unwrap().load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Training...")
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            if self.checkpoint:
                file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self._unwrap().serialize(self._unwrap(), self.optimizer,
                           epoch + 1, tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print('Saving checkpoint model to %s' % file_path)

            print('Cross validation...')
            self.model.eval()
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] /= 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self._unwrap().serialize(self._unwrap(), self.optimizer,
                           epoch + 1, tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print("Find better validated model, saving to %s" % file_path)

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": tr_avg_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "best_val_loss": min(self.best_val_loss, val_loss)
                })

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, data in enumerate(data_loader):
            if isinstance(data, (tuple, list)) and len(data) >= 5:
                noisy_batch, clean_batch, ref_ids, _, _ = data
                batch_idx = torch.arange(clean_batch.shape[0]).to(clean_batch.device)
                clean_ref_mic = clean_batch[batch_idx, ref_ids, :]
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                noisy_batch, clean_batch = data
                ref_ids = None
                if clean_batch.dim() == 3 and clean_batch.shape[1] > 1:
                    clean_ref_mic = clean_batch[:, 0, :]
                else:
                    clean_ref_mic = clean_batch.squeeze(1) if clean_batch.dim() == 3 else clean_batch
            else:
                noisy_batch, clean_batch = data[0], data[1]
                ref_ids = None
                clean_ref_mic = clean_batch.squeeze(1) if clean_batch.dim() == 3 else clean_batch

            clean_energy = torch.sqrt(torch.mean(clean_ref_mic**2, dim=-1))
            if (clean_energy < 1e-4).any():
                if i % self.print_freq == 0:
                    print(f"Skipping Batch {i}: Silent or quiet clean reference.")
                continue

            batch_size = noisy_batch.shape[0]
            num_samples = noisy_batch.shape[2]
            mixture_lengths = torch.full((batch_size,), num_samples, dtype=torch.int64).to(noisy_batch.device)

            padded_mixture = noisy_batch
            padded_source = clean_ref_mic if clean_ref_mic.dim() == 3 else clean_ref_mic.unsqueeze(1)
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
                if ref_ids is not None:
                    ref_ids = ref_ids.cuda()

            estimate_source = self.model(padded_mixture, ref_ids=ref_ids)

            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

        return total_loss / (i + 1)
