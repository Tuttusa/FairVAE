from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from evidentialdl.layers import DenseNormalGamma


@dataclass
class Enc:
    mu: Any = None
    logvar: Any = None

    v: Any = None
    alpha: torch.Tensor = None
    beta: torch.Tensor = None
    aleatoric: torch.Tensor = None
    epistemic: torch.Tensor = None


class Encoder(nn.Module):
    def __init__(self, input_size, compress_dims, embedding_dim, mode='AE'):
        super().__init__()

        self.mode = mode

        self.input_size = input_size

        seq = []
        dim = input_size
        for item in list(compress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item
        self.enc_seq = nn.Sequential(*seq)

        self.enc_mu = nn.Linear(dim, embedding_dim)

        if self.mode == 'VAE':
            self.enc_logvar = nn.Linear(dim, embedding_dim)

    @property
    def output_size(self):
        return self.enc_mu.out_features

    def forward(self, x):
        enc = Enc()

        h1 = self.enc_seq(x)

        enc.mu = self.enc_mu(h1)

        if self.mode == 'VAE':
            enc.logvar = self.enc_logvar(h1)

        return enc


class Decoder(nn.Module):
    def __init__(self, input_size, compress_dims, embedding_dim, mode='AE', uncertainty=False):
        super().__init__()

        self.mode = mode
        self.uncertainty = uncertainty

        seq = []
        dim = embedding_dim
        for item in list(compress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item
        self.dec_seq = nn.Sequential(*seq)

        if uncertainty:
            self.dec_mu = DenseNormalGamma(dim, input_size)
        else:
            self.dec_mu = nn.Linear(dim, input_size)


        if self.mode == 'VAE':
            self.dec_logvar = nn.Linear(dim, input_size)

    @property
    def output_size(self):
        return self.dec_mu.out_features

    def forward(self, x):

        enc = Enc()

        h1 = self.dec_seq(x)

        if self.uncertainty:
            enc.mu, enc.v, enc.alpha, enc.beta, enc.aleatoric, enc.epistemic = self.dec_mu(h1)
        else:
            enc.mu = self.dec_mu(h1)

        if self.mode == 'VAE':
            enc.std = self.dec_logvar(h1)

        return enc
