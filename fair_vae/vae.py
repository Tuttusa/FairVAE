import numpy as np
import pytorch_lightning as pl
import torchmetrics
from torch.distributions import Normal
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.nn import L1Loss, Module

import warnings

from torchtest import assert_vars_change

from fair_vae.datamodule import DataTransformer, VAEDataModule
from fair_vae.losses import reconstruction_loss, kld_loss
from fair_vae.modules import Encoder, Decoder
from fair_vae.util import artifacts_path, pl_bar, set_wandb, MetricTracker
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from Fdatasets.real import Real
import os

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

mse_loss = L1Loss()


class VAE(LightningModule):
    def __init__(self, input_size: int, embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128),
                 loss_factor=2, l2scale=1e-5, cuda=True, transformer: DataTransformer = None, mode='AE', name='',
                 *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        assert mode in ['VAE', 'AE']

        self.name = name

        self.mode = mode
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.loss_factor = loss_factor
        self.l2scale = l2scale
        self.transformer = transformer

        if self.transformer is not None:
            input_size = self.transformer.output_dimensions

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self.encoder = Encoder(input_size, compress_dims, embedding_dim, mode=mode)
        self.decoder = Decoder(input_size, compress_dims, embedding_dim, mode=mode)

    def encode(self, x, use_transformer=False):
        if use_transformer:
            x = self.transformer.transform(x)
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), std

    def decode(self, z):
        return self.decoder(z)

    def predict(self, x):
        if self.mode == 'AE':
            mu = self.encode(x)
            rec = self.decode(mu)
            return dict(mu=mu, rec=rec)
        else:
            enc_mu, enc_logvar = self.encode(x)
            z, enc_std = self.reparameterize(enc_mu, enc_logvar)
            rec_mu, rec_logvar = self.decode(enc_mu)
            _, dec_std = self.reparameterize(rec_mu, rec_logvar)
            return dict(mu=enc_mu, logvar=enc_logvar, enc_st=enc_std, rec=rec_mu, rec_std=dec_std, z=z)

    def loss_fn(self, pred_result, x, return_loss_comps=False):

        ## Recon loss ##
        if self.transformer is not None:
            if self.mode == 'VAE':
                recon_loss = reconstruction_loss(self.transformer, pred_result['rec'], x, pred_result['rec_std'],
                                                 self.loss_factor)
            else:
                recon_loss = reconstruction_loss(self.transformer, pred_result['rec'], x)
        else:
            recon_loss = self.loss_factor * mse_loss(pred_result['rec'], x)

        ## recon metric ##
        recon_metric = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])

        ## KL Divergence ##
        kl = 0.0

        if self.mode == 'VAE':
            kl = kld_loss(pred_result['mu'], pred_result['logvar'])

        log_lik = 0.0

        loss = recon_loss + kl + log_lik

        if not return_loss_comps:
            return loss
        else:
            return loss, recon_loss, kl, log_lik, recon_metric

    def forward(self, x):
        pred_result = self.predict(x)

        loss, recon_loss, kl, log_lik, recon_metric = self.loss_fn(pred_result, x, return_loss_comps=True)

        return dict(x=x, loss=loss, kl=kl, recon_loss=recon_loss, log_lik=log_lik, recon_metric=recon_metric,
                    **pred_result)

    def reconstructed_probability(self, x):
        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['rec'], pred['rec_std'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def embedding_rec_prob(self, emb_x):
        with torch.no_grad():
            rec, rec_sigma = self.decode(emb_x)
        recon_dist = Normal(rec, rec_sigma)
        x = emb_x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def training_step(self, batch, batch_idx, *args, **kwargs):

        x = batch[0]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x)

        self.log(f"train_loss", pred_result['loss'], on_step=True, logger=True)
        # self.log('recon_loss', pred_result['recon_loss'], on_step=True, logger=True)
        # self.log('recon_metric', pred_result['recon_metric'], on_step=True, logger=True)
        # self.log('kl_loss', pred_result['kl'], on_step=True, logger=True)
        # self.log('log_lik_loss', pred_result['log_lik'], on_step=True, logger=True)

        logs = {"loss": pred_result['loss']}

        return logs

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x)

        self.log('val_loss', pred_result['loss'], on_step=True, logger=True)

        val_test_loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])

        self.log('val_test_loss', val_test_loss, on_step=True, logger=True)

        return {"loss": pred_result['loss']}

    def test_step(self, batch, batch_idx):
        x = batch[0]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x)

        loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])
        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), weight_decay=self.l2scale)


"""
Tests
"""
# %% Data set
db = Real.income()

# %%

from pytorch_lightning import Callback

basic_model_config = {
    'name': 'vae_leoleo',
    'input_size': db.x.shape[1],
    'latent_size': 128,
    'compress_dims': (128, 128),
    'decompress_dims': (128, 128),
    'num_resamples': 8,
    'device': 'cpu',
    'lr': 1e-4,
    'batch_size': 100,
    'epochs': 200,
    'no_progress_bar': False,
    'steps_log_loss': 10,
    'steps_log_norm_params': 10,
    'weight_decay': 1e-5,
    'loss_factor': 2,
    'mode': 'AE',
    'use_transformer': True,
    'data': db.x.to_numpy(),
    'patience': 40
}


def generate_test_case(model_config):
    model_path = artifacts_path.joinpath('vae_leoleo')

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.05, patience=100, verbose=False,
                                        mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_path,
        filename=model_config['name'] + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    vae_data = VAEDataModule(db.x.to_numpy(), batch_size=model_config['batch_size'])
    vae_data.setup()

    cb_log = MetricTracker()

    vae = VAE(input_size=model_config['input_size'], embedding_dim=model_config['latent_size'],
              compress_dims=model_config['compress_dims'], decompress_dims=model_config['decompress_dims'],
              loss_factor=model_config['loss_factor'], l2scale=model_config['weight_decay'],
              mode=model_config['mode'])

    callbacks = [checkpoint_callback, pl_bar, early_stop_callback, cb_log]

    return vae, vae_data, callbacks


# test
vae, vae_data, callbacks = generate_test_case(basic_model_config)

# %%

# AE converges
import matplotlib.pyplot as plt

basic_model_config['epochs'] = 100
basic_model_config['batch_size'] = 500

vae, vae_data, callbacks = generate_test_case(basic_model_config)
# logger = set_wandb(basic_model_config, project='leoleo', job_type='vae_leoleo')

trainer = pl.Trainer(max_epochs=basic_model_config['epochs'], log_every_n_steps=8,
                     callbacks=callbacks)

start_test_loss_log = trainer.test(vae, vae_data)
log_train = trainer.fit(vae, vae_data)
end_test_loss_log = trainer.test(vae, vae_data)

# diff_loss = start_test_loss_log[0]['test_loss'] - end_test_loss_log[0]['test_loss']
callbacks[3].plot('loss')
# plt.plot(loss_values)

# %%
# VAE converges

basic_model_config['mode'] = 'VAE'
basic_model_config['epochs'] = 100
basic_model_config['batch_size'] = 500
vae, vae_data, callbacks = generate_test_case(basic_model_config)

logger = set_wandb(basic_model_config, project='leoleo', job_type='vae_leoleo')

trainer = pl.Trainer(max_epochs=basic_model_config['epochs'], log_every_n_steps=8,
                     callbacks=callbacks, logger=logger)

start_test_loss = trainer.test(vae, vae_data)[0]
trainer.fit(vae, vae_data)

end_test_loss = trainer.test(vae, vae_data)[0]

# assert start_test_loss['test_loss'] - end_test_loss['test_loss'] > 0
