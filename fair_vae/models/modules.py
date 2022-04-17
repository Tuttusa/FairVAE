import torch.nn as nn


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
            self.enc_std = nn.Linear(dim, embedding_dim)

    @property
    def output_size(self):
        return self.enc_mu.out_features

    def forward(self, x):
        h1 = self.enc_seq(x)
        if self.mode == 'VAE':
            return self.enc_mu(h1), self.enc_std(h1)
        else:
            return self.enc_mu(h1), None


class Decoder(nn.Module):
    def __init__(self, input_size, compress_dims, embedding_dim, mode='AE'):
        super().__init__()

        self.mode = mode

        seq = []
        dim = embedding_dim
        for item in list(compress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item
        self.dec_seq = nn.Sequential(*seq)
        self.dec_mu = nn.Linear(dim, input_size)

        if self.mode == 'VAE':
            self.dec_std = nn.Linear(dim, input_size)

    @property
    def output_size(self):
        return self.dec_mu.out_features

    def forward(self, x):
        h1 = self.dec_seq(x)
        if self.mode == 'VAE':
            return self.dec_mu(h1), self.dec_std(h1)
        else:
            return self.dec_mu(h1), None
