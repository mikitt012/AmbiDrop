import torch
from torch.autograd import Variable

import models_attention_set

import torch
import torch.nn as nn
import torch.nn.functional as F

class TasNet(torch.nn.Module):
    def __init__(self, mic_num, num_spk, enc_dim, feature_dim, ch_dim, sample_rate, win, layer, stack, kernel, causal):
        super(TasNet, self).__init__()
        # hyper parameters
        self.mic_num = mic_num
        self.num_spk = num_spk

        # increased enc dim
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim

        self.ch_dim = ch_dim

        self.win = int(sample_rate * win / 1000)
        self.stride = self.win // 2

        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal

        # input encoder
        self.encoder = torch.nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        # TCN separator
        self.TCN = models_attention_set.TCN(self.mic_num, self.ch_dim, self.enc_dim, self.enc_dim * self.num_spk,
                                            self.feature_dim, self.feature_dim * 4,  # single modified
                                            self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = torch.nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
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

    def forward(self, input, ref_ids):
        # Padding
        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        num_ch = output.size(1)
        enc_output = self.encoder(output.view(batch_size * num_ch, 1, -1)).view(batch_size, num_ch, self.enc_dim, -1)  # B, C, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L

        # reference ch
        batch_idx = torch.arange(input.shape[0]).to(input.device)
        ref_enc_output = enc_output[batch_idx, ref_ids, :, :].unsqueeze(1)
        masked_output = ref_enc_output * masks

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L

        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return output

    def serialize(self, model, optimizer, epoch, tr_loss=None, cv_loss=None):
        """Serialize model and optimizer state, and include hyperparameters."""
        package = {
            # Standard model state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            # Hyperparameters (The important part for the solver)
            'model_mic_num': self.mic_num,
            'model_enc_dim': self.enc_dim,
            'model_feature_dim': self.feature_dim,
            'model_win': self.win, # Use the ms value from config
            'model_layer': self.layer,
            'model_stack': self.stack,
            'model_kernel': self.kernel,
            'model_num_spk': self.num_spk,
            'model_causal': self.causal,
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
        if cv_loss is not None:
            package['cv_loss'] = cv_loss
        return package

    @classmethod
    def load_model_from_package(cls, package):
        # 1. Instantiate the model using the individual keys from the package
        # We match the keys saved in serialize() to the arguments expected by __init__
        win = package['model_win'] * 1000 / 16000

        model = cls(
            mic_num=package['model_mic_num'],
            num_spk=package['model_num_spk'],
            enc_dim=package['model_enc_dim'],
            feature_dim=package['model_feature_dim'],
            # If ch_dim wasn't in serialize, we provide a default (e.g., 8)
            ch_dim=package.get('model_ch_dim', 8), 
            sample_rate=package.get('sample_rate', 16000),
            win=win,
            layer=package['model_layer'],
            stack=package['model_stack'],
            kernel=package['model_kernel'],
            causal=package['model_causal'],
        )
        # 2. Load the weights
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model