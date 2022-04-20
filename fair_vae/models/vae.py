import copy

import numpy as np
import torchmetrics
from torch.optim import Adam
import pytorch_lightning as pl
import torch
from torch.nn import L1Loss

import warnings

from fair_vae.configs import AEConfig
from fair_vae.losses import reconstruction_loss, kld_loss
from fair_vae.models.interface import VAEFrame
from fair_vae.models.modules import Encoder, Decoder
from fair_vae.util import artifacts_path, pl_bar

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

mse_loss = L1Loss()


class VAE(VAEFrame):
    model_impl = 'VAE'

    def __init__(self, ae_config: AEConfig, *args, **kwargs):
        super(VAE, self).__init__(ae_config, *args, **kwargs)

        self.encoder = Encoder(self.config.input_shape, self.config.compress_dims, self.config.embedding_dim,
                               mode=self.config.mode)
        self.decoder = Decoder(self.config.input_shape, self.config.compress_dims, self.config.embedding_dim,
                               mode=self.config.mode)

        if not self.config.device or not torch.cuda.is_available():
            device = 'cpu'
        else:
            device = 'cuda'

        self._device = torch.device(device)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), std

    def decode(self, z):
        return self.decoder(z)

    def predict(self, x):
        if self.config.mode == 'AE':
            mu, _ = self.encode(x)
            rec, _ = self.decode(mu)
            return dict(mu=mu, rec=rec)
        else:
            enc_mu, enc_logvar = self.encode(x)
            z, enc_std = self.reparameterize(enc_mu, enc_logvar)
            rec_mu, rec_logvar = self.decode(enc_mu)
            _, dec_std = self.reparameterize(rec_mu, rec_logvar)

            if torch.isnan(enc_mu).any():
                print("enc_logvar is nan", torch.isnan(enc_logvar).any())
                raise Exception('mu is nan')

            return dict(mu=enc_mu, logvar=enc_logvar, enc_st=enc_std, rec=rec_mu, rec_std=dec_std, z=z)

    def loss_fn(self, pred_result, x, return_loss_comps=False):

        ## Recon loss ##
        if self.config.use_transformer:
            if self.config.mode == 'VAE':
                recon_loss = reconstruction_loss(self.transformer[self.config.principal_elem], pred_result['rec'], x,
                                                 pred_result['rec_std'], self.config.loss_factor)
            else:
                recon_loss = reconstruction_loss(self.transformer[self.config.principal_elem], pred_result['rec'], x)
        else:
            recon_loss = self.config.loss_factor * mse_loss(pred_result['rec'], x)

        ## recon metric ##
        recon_metric = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])

        ## KL Divergence ##
        kl = 0.0

        if self.config.mode == 'VAE':
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

    def training_step(self, batch, batch_idx, *args, **kwargs):

        x = batch[self.config.datamodule_principal_elem_index()]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x)

        self.log(f"train_loss", pred_result['loss'], on_step=True, logger=True)

        logs = {"loss": pred_result['loss']}

        return logs

    def validation_step(self, batch, batch_idx):
        x = batch[self.config.datamodule_principal_elem_index()]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x)

        self.log('val_loss', pred_result['loss'], on_step=True, logger=True)

        val_test_loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])

        self.log('val_test_loss', val_test_loss, on_step=True, logger=True)

        return {"loss": pred_result['loss']}

    def test_step(self, batch, batch_idx):
        x = batch[self.config.datamodule_principal_elem_index()]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x)

        loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])
        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), weight_decay=self.config.l2scale)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["transformer"] = self.transformer

    def on_load_checkpoint(self, checkpoint) -> None:
        self.transformer = checkpoint["transformer"]

    def fit(self, vae_data):

        self.transformer = vae_data.transformer

        self.save_hyperparameters()

        model_path = artifacts_path.joinpath(self.model_impl)

        callbacks = self.callbacks(model_path)
        callbacks.append(pl_bar)

        trainer = pl.Trainer(max_epochs=self.config.epochs, log_every_n_steps=8, callbacks=callbacks)

        start_test_loss_log = trainer.test(self, vae_data)
        log_train = trainer.fit(self, vae_data)
        end_test_loss_log = trainer.test(self, vae_data)

        callbacks[2].plot('loss')

    def transform_encode(self, data):
        with torch.no_grad():
            return self.encode(torch.from_numpy(self.transformer['x'].transform(data)).float())[0]

    # def transform_decode(self, data):
    #     with torch.no_grad():
    #         return self.encode(torch.from_numpy(self.transformer['x'].transform(data)).float())



# class VAE2(VAEFrame):
#     model_impl = 'VAE2'
#
#     def __init__(self, ae_config: AEConfig, *args, **kwargs):
#         super(VAE2, self).__init__(ae_config, *args, **kwargs)
#
#         x_ae_config = copy.deepcopy(ae_config)
#         x_ae_config.principal_elem = 'x'
#
#         t_ae_config = copy.deepcopy(ae_config)
#         t_ae_config.principal_elem = 't'
#
#         self.x_vae = VAE(x_ae_config)
#         self.t_vae = VAE(t_ae_config)
#
#     @classmethod
#     def load_best(self, name, config=None):
#         if config is None:
#             config = AEConfig()
#
#         x_ae_config = copy.deepcopy(config)
#         x_ae_config.principal_elem = 'x'
#
#         t_ae_config = copy.deepcopy(config)
#         t_ae_config.principal_elem = 't'
#
#         vae = VAE2(config)
#
#         vae.t_vae = VAE.load_best(name, t_ae_config.principal_elem)
#         vae.x_vae = VAE.load_best(name, x_ae_config.principal_elem)
#
#         return vae
#
#     def fit(self, vae_data):
#         self.x_vae.fit(vae_data)
#         self.t_vae.fit(vae_data)


# if __name__ == '__main__':

"""
Tests
"""
# %% Data set
# db = Real.income()
# df = db.to_df()

# %%


# %%

#
# # %%
#
# config = AEConfig(x_data=VAEData(data=db.x.to_numpy()), name='vae', mode='VAE')
# vae = VAE(config)
# vae.fit(epochs=50, batch_size=500)

# %%
# from pytorch_lightning import Callback
#
# x = df[[col for col in df.columns if 'x' in col or 't' in col]]
#
# basic_model_config = {
#     'name': 'ae',
#     'latent_size': 128,
#     'compress_dims': (128, 128),
#     'decompress_dims': (128, 128),
#     'num_resamples': 8,
#     'device': 'cpu',
#     'lr': 1e-4,
#     'batch_size': 100,
#     'epochs': 200,
#     'no_progress_bar': False,
#     'steps_log_loss': 10,
#     'steps_log_norm_params': 10,
#     'weight_decay': 1e-5,
#     'loss_factor': 2,
#     'mode': 'AE',
#     'use_transformer': True,
#     'x_data': VAEData(data=db.x.to_numpy(), use_transformer=True),
#     'patience': 40
# }
#
#
# def generate_test_case(model_config):
#     model_path = artifacts_path.joinpath(model_config['name'])
#
#     early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.05, patience=100, verbose=False,
#                                         mode="min")
#
#     checkpoint_callback = ModelCheckpoint(
#         monitor="val_loss",
#         dirpath=model_path,
#         filename=model_config['name'] + "-{epoch:02d}-{val_loss:.2f}",
#         save_top_k=3,
#         mode="min",
#     )
#
#     vae_data = VAEDataModule(model_config['x_data'], batch_size=model_config['batch_size'])
#     vae_data.setup()
#
#     cb_log = MetricTracker()
#
#     vae = VAE(x_data=model_config['x_data'], embedding_dim=model_config['latent_size'],
#               compress_dims=model_config['compress_dims'], decompress_dims=model_config['decompress_dims'],
#               loss_factor=model_config['loss_factor'], l2scale=model_config['weight_decay'],
#               mode=model_config['mode'], transformer=model_config['x_data'].transformer)
#
#     callbacks = [checkpoint_callback, pl_bar, early_stop_callback, cb_log]
#
#     return vae, vae_data, callbacks
#
#
# # %%
# # if __name__ == '__main__':
# import matplotlib.pyplot as plt
#
# basic_model_config['epochs'] = 100
# basic_model_config['batch_size'] = 500
#
# vae, vae_data, callbacks = generate_test_case(basic_model_config)
# # logger = set_wandb(basic_model_config, project='leoleo', job_type='vae_leoleo')
#
# trainer = pl.Trainer(max_epochs=basic_model_config['epochs'], log_every_n_steps=8, callbacks=callbacks)
#
# start_test_loss_log = trainer.test(vae, vae_data)
# log_train = trainer.fit(vae, vae_data)
# end_test_loss_log = trainer.test(vae, vae_data)
#
# # diff_loss = start_test_loss_log[0]['test_loss'] - end_test_loss_log[0]['test_loss']
# callbacks[3].plot('loss')
#
# # %%VAE converges
#
# basic_model_config['mode'] = 'VAE'
# basic_model_config['name'] = 'vae'
# basic_model_config['epochs'] = 50
# basic_model_config['batch_size'] = 500
# vae, vae_data, callbacks = generate_test_case(basic_model_config)
#
# trainer = pl.Trainer(max_epochs=basic_model_config['epochs'], log_every_n_steps=8,
#                      callbacks=callbacks)
#
# start_test_loss = trainer.test(vae, vae_data)[0]
# trainer.fit(vae, vae_data)
#
# end_test_loss = trainer.test(vae, vae_data)[0]
