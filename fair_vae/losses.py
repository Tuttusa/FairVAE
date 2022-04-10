import torch
from torch.nn.functional import cross_entropy

from fair_vae.datamodule import DataTransformer


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)


def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()


def kld_loss(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def reconstruction_loss(transformer: DataTransformer, recon_x, x, sigmas=None, loss_factor=1.0):
    st = 0
    loss = []
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn != "softmax":
                ed = st + span_info.dim
                loss_val = (x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2
                if sigmas is not None:
                    std = sigmas[:, st]
                    loss_val /= (std ** 2)
                loss_val = loss_val.sum()
                loss.append(loss_val)
                # loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    return sum(loss) * loss_factor / x.size()[0]
