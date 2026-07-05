"""
IC Conv-TasNet model for multichannel time-domain speech enhancement.

Public interface:
    TasNet — Conv1d encoder → dilated TCN → ConvTranspose1d decoder; supports ambidrop mode (9 SH channels, a00 reference) and baseline mode (M mic channels, per-sample reference)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import ConvTasNet.modules as modules
from ambidrop.dropouts import SHChannelDropout1D, PerChDropout1D


class TasNet(nn.Module):
    """
    IC Conv-TasNet for multichannel speech enhancement.

    Supports two modes:
      - ambidrop: Ambisonics input with channel dropout, reference = channel 0 (a00)
      - baseline: Microphone input, no dropout, reference = per-sample ref_ids
    """

    def __init__(self, mic_num, num_spk, enc_dim, feature_dim, ch_dim,
                 sample_rate, win, layer, stack, kernel, causal,
                 mode='ambidrop', drop_prob=0.4, max_drop=3,
                 drop_probs=None, dropout_type="SHChannelDropout"):
        super(TasNet, self).__init__()
        self.mic_num = mic_num
        self.num_spk = num_spk
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.ch_dim = ch_dim
        self.win = int(sample_rate * win / 1000)
        self.stride = self.win // 2
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.causal = causal
        self.mode = mode

        self.encoder = torch.nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        self.TCN = modules.TCN(
            self.mic_num, self.ch_dim, self.enc_dim, self.enc_dim * self.num_spk,
            self.feature_dim, self.feature_dim * 4,
            self.layer, self.stack, self.kernel, causal=self.causal
        )

        self.receptive_field = self.TCN.receptive_field

        self.decoder = torch.nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

        self.has_dropout = mode == 'ambidrop'
        if self.has_dropout:
            if dropout_type == "SHChannelDropout":
                self.channel_dropout = SHChannelDropout1D(drop_prob=drop_prob, max_drop=max_drop)
            elif dropout_type == "PerChDropout":
                self.channel_dropout = PerChDropout1D(drop_probs=drop_probs)

    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def forward(self, input, ref_ids=None):
        if self.has_dropout:
            input = self.channel_dropout(input)

        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        num_ch = output.size(1)
        enc_output = self.encoder(output.view(batch_size * num_ch, 1, -1)).view(
            batch_size, num_ch, self.enc_dim, -1)

        masks = torch.sigmoid(self.TCN(enc_output)).view(
            batch_size, self.num_spk, self.enc_dim, -1)

        if self.mode == 'ambidrop':
            masked_output = enc_output[:, 0:1] * masks
        else:
            if ref_ids is None:
                ref_ids = torch.zeros(batch_size, dtype=torch.long, device=input.device)
            batch_idx = torch.arange(batch_size).to(input.device)
            ref_enc_output = enc_output[batch_idx, ref_ids, :, :].unsqueeze(1)
            masked_output = ref_enc_output * masks

        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()
        output = output.view(batch_size, self.num_spk, -1)

        return output

    def serialize(self, model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'model_mic_num': self.mic_num,
            'model_enc_dim': self.enc_dim,
            'model_feature_dim': self.feature_dim,
            'model_win': self.win,
            'model_layer': self.layer,
            'model_stack': self.stack,
            'model_kernel': self.kernel,
            'model_num_spk': self.num_spk,
            'model_causal': self.causal,
            'model_mode': self.mode,
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
        if cv_loss is not None:
            package['cv_loss'] = cv_loss
        return package

    @classmethod
    def load_model_from_package(cls, package, mode='ambidrop',
                                 dropout_type="SHChannelDropout",
                                 drop_prob=0.4, max_drop=3, drop_probs=None):
        win = package['model_win'] * 1000 / 16000
        model = cls(
            mic_num=package['model_mic_num'],
            num_spk=package['model_num_spk'],
            enc_dim=package['model_enc_dim'],
            feature_dim=package['model_feature_dim'],
            ch_dim=package.get('model_ch_dim', 8),
            sample_rate=package.get('sample_rate', 16000),
            win=win,
            layer=package['model_layer'],
            stack=package['model_stack'],
            kernel=package['model_kernel'],
            causal=package['model_causal'],
            mode=mode,
            drop_prob=drop_prob,
            max_drop=max_drop,
            drop_probs=drop_probs,
            dropout_type=dropout_type,
        )
        model.load_state_dict(package['state_dict'], strict=False)
        return model

    @classmethod
    def load_model(cls, path, mode='ambidrop', dropout_type="SHChannelDropout",
                   drop_prob=0.4, max_drop=3, drop_probs=None):
        package = torch.load(path, map_location="cpu")
        model = cls.load_model_from_package(
            package, mode=mode, dropout_type=dropout_type,
            drop_prob=drop_prob, max_drop=max_drop, drop_probs=drop_probs
        )
        return model
