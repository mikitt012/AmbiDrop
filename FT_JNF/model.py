import torch
import torch.nn as nn

from ambidrop.dropouts import SHChannelDropout, PerChDropout
from ambidrop.constants import get_device


class FT_JNF(nn.Module):
    """
    Frequency-Time Joint Non-linear Filter (FT-JNF).

    Two bidirectional LSTM layers (one across frequency, one across time)
    followed by a linear layer that outputs a complex Ideal Ratio Mask.

    Args:
        input_dim: Number of input features (2*num_channels: real + imag).
                   14 for baseline (7 mics), 18 for AmbiDrop (9 SH channels).
        hidden1_dim: Hidden size of the first BLSTM layer
        hidden2_dim: Hidden size of the second BLSTM layer
        output_dim: Output dimension (2 for real + imag mask)
        dropout_type: None for baseline, "SHChannelDropout" or "PerChDropout"
        drop_prob: Dropout probability (for SHChannelDropout)
        max_drop: Max channels to drop (for SHChannelDropout)
        drop_probs: Per-channel dropout probabilities (for PerChDropout)
    """

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim,
                 dropout_type=None, drop_prob=0.0, max_drop=0, drop_probs=None):
        super().__init__()

        self.has_dropout = dropout_type is not None and dropout_type != ""
        if self.has_dropout:
            if dropout_type == "SHChannelDropout":
                self.channel_dropout = SHChannelDropout(drop_prob=drop_prob, max_drop=max_drop)
            elif dropout_type == "PerChDropout":
                self.channel_dropout = PerChDropout(drop_probs=drop_probs)
            else:
                raise ValueError(f"Unknown dropout type: {dropout_type}")

        self.LSTM1 = nn.LSTM(
            input_size=input_dim, hidden_size=hidden1_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.LSTM2 = nn.LSTM(
            input_size=2 * hidden1_dim, hidden_size=hidden2_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(2 * hidden2_dim, output_dim)

        self.hamming_window = torch.hamming_window(512, device=get_device())

        self._init_weights()

    def _init_weights(self):
        for lstm in [self.LSTM1, self.LSTM2]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        B, T, F, C = x.shape

        if self.has_dropout:
            x = self.channel_dropout(x)

        x, _ = self.LSTM1(x.view(B * T, F, C))
        x = x.view(B, T, F, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * F, T, -1)
        x, _ = self.LSTM2(x)
        x = x.view(B, F, T, -1).permute(0, 2, 1, 3)
        x = self.linear(x)

        return x
