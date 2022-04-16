import torch
import torch.nn as nn
import torchmetrics
from Fdatasets.real import Real
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn.functional import mse_loss
from torch.optim import Adam

from fair_vae.datamodule import VAEDataModule, DataConfig
from fair_vae.losses import reconstruction_loss, kld_loss, MMD
from fair_vae.modules import Encoder, Decoder

import pytorch_lightning as pl

from fair_vae.util import artifacts_path, MetricTracker, pl_bar, AEConfig


class FVAE(LightningModule):
    def __init__(self, ae_config: AEConfig, *args, **kwargs):
        super(FVAE, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.config = ae_config

        self.vae_data = VAEDataModule(x_data=self.config.x_data, t_data=self.config.t_data, y_data=self.config.y_data,
                                 batch_size=self.config.batch_size, transform=self.config.use_transformer)

        encoder1_inp_size = self.vae_data.shape('x') + self.vae_data.shape('t')
        self.encoder1 = Encoder(encoder1_inp_size, self.config.compress_dims, self.config.embedding_dim,
                                self.config.mode)  # q(z1 | t, x)

        encoder2_inp_size = self.encoder1.output_size + self.vae_data.shape('y')
        self.encoder2 = Encoder(encoder2_inp_size, self.config.compress_dims, self.config.embedding_dim,
                                self.config.mode)  # q(z2 | y, z1)

        self.y_learner = nn.Linear(self.encoder1.output_size, self.vae_data.shape('y'))  # p(y | z1)

        decoder1_inp_size = self.encoder2.output_size + self.vae_data.shape('y')
        self.decoder1 = Decoder(self.encoder1.output_size, self.config.decompress_dims, decoder1_inp_size,
                                self.config.mode)  # p(z1 | z2, y)

        decoder2_inp_size = self.decoder1.output_size + self.vae_data.shape('t')
        self.decoder2 = Decoder(self.vae_data.shape('x'), self.config.decompress_dims, decoder2_inp_size,
                                self.config.mode)  # p(x | z1, t)

    def encode(self, x, t, y):
        z1 = self.encoder1(torch.cat([x, t], dim=-1))

        z1_mu = z1[0]

        z2 = self.encoder2(torch.cat([z1_mu, y], dim=-1))

        y_pred = self.y_learner(z1_mu)

        return z1, z2, y_pred

    def decode(self, z2, t, y):
        rec_z1 = self.decoder1(torch.cat([z2, y], dim=-1))

        rec_z1_mu = rec_z1[0]

        rec_x = self.decoder2(torch.cat([rec_z1_mu, t], dim=-1))

        return rec_z1, rec_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), std

    def predict(self, x, t, y):
        (z1_mu, z1_logvar), (z2_mu, z2_logvar), rec_y = self.encode(x, t, y)

        (rec_z1_mu, rec_z1_logvar), (rec_x_mu, rec_x_logvar) = self.decode(z2_mu, t, y)

        if self.config.mode == 'VAE':
            _, dec_z1_std = self.reparameterize(rec_z1_mu, rec_z1_logvar)
            _, dec_x_std = self.reparameterize(rec_x_mu, rec_x_logvar)
        else:
            dec_z1_std, dec_x_std = None, None

        return dict(z1_mu=z1_mu, z1_logvar=z1_logvar, z2_mu=z2_mu,
                    z2_logvar=z2_logvar, rec_y=rec_y, rec_z1_mu=rec_z1_mu, rec_z1_logvar=rec_z1_logvar,
                    rec_x_mu=rec_x_mu, rec_x_logvar=rec_x_logvar, dec_z1_std=dec_z1_std,
                    dec_x_std=dec_x_std, x=x, t=t, y=y)

    def recon_loss(self, rec_mu, ori, rec_std=None, transformer=None):
        if transformer is not None:
            if rec_std is not None:
                recon_loss = reconstruction_loss(transformer, rec_mu, ori, rec_std, self.confi.loss_factor)
            else:
                recon_loss = reconstruction_loss(transformer, rec_mu, ori)
        else:
            recon_loss = self.config.loss_factor * mse_loss(rec_mu, ori)

        return recon_loss

    def forward(self, x, t, y, global_step):
        pred = self.predict(x, t, y)

        total_loss, total_rec_loss, total_kl_loss, mmd_loss = self.loss_fn(pred, global_step)

        return dict(pred=pred, total_loss=total_loss, total_rec_loss=total_rec_loss,
                    total_kl_loss=total_kl_loss, mmd_loss=mmd_loss)

    def loss_fn(self, pred, global_step, beta=0.01):

        total_kl_loss, mmd_loss = 0.0, 0.0

        ## rec loss x

        rec_loss_x = self.recon_loss(pred['rec_x_mu'], pred['x'], pred.get('rec_x_std', None), self.vae_data.transformer['x'])

        ## rec loss z1

        rec_loss_z1 = self.recon_loss(pred['rec_z1_mu'], pred['z1_mu'], pred.get('rec_z1_std', None))

        ## rec loss Y

        rec_loss_y = self.recon_loss(pred['rec_y'], pred['y'], transformer=self.vae_data.transformer['y'])

        total_rec_loss = rec_loss_x + rec_loss_z1 + rec_loss_y

        if self.config.mode == 'VAE':
            ## kl loss z1

            kl_z1_loss = kld_loss(pred['z1_mu'], pred['z1_logvar'])

            # anneal_coef_z1 = torch.minimum(torch.tensor(1.0), torch.tensor(beta * global_step * self.encoder1.output_size/ self.encoder1.input_size))
            # kl_z1_loss = anneal_coef_z1 * kl_z1_loss

            ## kl loss z2

            kl_z2_loss = kld_loss(pred['z2_mu'], pred['z2_logvar'])

            # anneal_coef_z2 = torch.minimum(torch.tensor(1.0), torch.tensor(beta * global_step * self.encoder1.output_size/ self.encoder1.input_size))
            # kl_z2_loss = anneal_coef_z2 * kl_z2_loss

            total_kl_loss = kl_z1_loss + kl_z2_loss

        ## MMD loss
        mmd_loss = MMD(pred['z1_mu'], pred['rec_z1_mu'])

        total_loss = total_rec_loss + total_kl_loss + mmd_loss

        return total_loss, total_rec_loss, total_kl_loss, mmd_loss

    def training_step(self, batch, batch_idx, *args, **kwargs):

        x, t, y = batch
        if self._device:
            x = x.to(self._device)

        pred_result = self.forward(x, t, y, self.current_epoch * batch_idx)

        self.log(f"train_loss", pred_result['total_loss'], on_step=True, logger=True)

        logs = {"loss": pred_result['total_loss']}

        return logs

    def validation_step(self, batch, batch_idx):
        x, t, y = batch

        if self._device:
            x = x.to(self._device)

        pred_result = self.forward(x, t, y, self.current_epoch * batch_idx)

        self.log('val_loss', pred_result['total_loss'], on_step=True, logger=True)

        val_test_loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['pred']['rec_x_mu'])

        self.log('val_test_loss', val_test_loss, on_step=True, logger=True)

        return {"loss": pred_result['total_loss']}

    def test_step(self, batch, batch_idx):
        x, t, y = batch
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x, t, y, self.current_epoch * batch_idx)

        loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['pred']['rec_x_mu'])
        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), weight_decay=self.config.l2scale)

    def fit(self, epochs, batch_size):

        self.config.epochs = epochs
        self.config.batch_size = batch_size

        model_path = artifacts_path.joinpath(self.config.name)

        early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.05, patience=100, verbose=False,
                                            mode="min")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=model_path,
            filename=self.config.name + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )

        cb_log = MetricTracker()

        callbacks = [checkpoint_callback, pl_bar, early_stop_callback, cb_log]

        trainer = pl.Trainer(max_epochs=self.config.epochs, log_every_n_steps=8, callbacks=callbacks)

        start_test_loss_log = trainer.test(self, self.vae_data)
        log_train = trainer.fit(self, self.vae_data)
        end_test_loss_log = trainer.test(self, self.vae_data)

        # diff_loss = start_test_loss_log[0]['test_loss'] - end_test_loss_log[0]['test_loss']
        callbacks[3].plot('loss')

# %% Data set
# db = Real.income()
# df = db.to_df()
#
# # %%
# config = AEConfig(name='f_ae', x_data=VAEData(data=db.x.to_numpy()), t_data=VAEData(data=db.t.to_numpy()),
#                   y_data=VAEData(data=db.y.to_numpy()))
# fae = FVAE(config)
# fae.fit(epochs=100, batch_size=500)
#
# #%%
# config = AEConfig(x_data=VAEData(data=db.x.to_numpy()), t_data=VAEData(data=db.t.to_numpy()),
#                   y_data=VAEData(data=db.y.to_numpy()), name='f_vae', mode='VAE')
# fvae = FVAE(config)
# fvae.fit(epochs=50, batch_size=500)

# if __name__ == '__main__':
#     db = Real.income()
#
#     # %%
#
#     model_config = {
#         'name': 'f_ae',
#         'latent_size': 128,
#         'compress_dims': (128, 128),
#         'decompress_dims': (128, 128),
#         'num_resamples': 8,
#         'device': 'cpu',
#         'lr': 1e-4,
#         'batch_size': 100,
#         'epochs': 40,
#         'no_progress_bar': False,
#         'steps_log_loss': 10,
#         'steps_log_norm_params': 10,
#         'weight_decay': 1e-5,
#         'loss_factor': 2,
#         'mode': 'AE',
#         'use_transformer': True,
#         'x_data': VAEData(data=db.x.to_numpy(), use_transformer=True),
#         't_data': VAEData(data=db.t.to_numpy(), use_transformer=True),
#         'y_data': VAEData(data=db.y.to_numpy(), use_transformer=True),
#         'patience': 40
#     }
#
#
#     def generate_test_case(model_config):
#         model_path = artifacts_path.joinpath('fvae')
#
#         early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.05, patience=100, verbose=False,
#                                             mode="min")
#
#         checkpoint_callback = ModelCheckpoint(
#             monitor="val_loss",
#             dirpath=model_path,
#             filename=model_config['name'] + "-{epoch:02d}-{val_loss:.2f}",
#             save_top_k=3,
#             mode="min",
#         )
#
#         vae_data = VAEDataModule(x_data=model_config['x_data'],
#                                  t_data=model_config['t_data'],
#                                  y_data=model_config['y_data'],
#                                  batch_size=model_config['batch_size'])
#         vae_data.setup()
#
#         cb_log = MetricTracker()
#
#         vae = FVAE(x_size=model_config['x_data'].size, t_size=model_config['t_data'].size,
#                    y_size=model_config['y_data'].size, embedding_dim=model_config['latent_size'],
#                    compress_dims=model_config['compress_dims'], decompress_dims=model_config['decompress_dims'],
#                    loss_factor=model_config['loss_factor'], l2scale=model_config['weight_decay'],
#                    mode=model_config['mode'], transformer_x=model_config['x_data'].transformer,
#                    transformer_t=model_config['t_data'].transformer, transformer_y=model_config['y_data'].transformer)
#
#         callbacks = [checkpoint_callback, pl_bar, early_stop_callback, cb_log]
#
#         return vae, vae_data, callbacks
#
#
#     vae, vae_data, callbacks = generate_test_case(model_config)
#
#     trainer = pl.Trainer(max_epochs=model_config['epochs'], log_every_n_steps=8,
#                          callbacks=callbacks)
#
#     start_test_loss_log = trainer.test(vae, vae_data)
#     log_train = trainer.fit(vae, vae_data)
#     end_test_loss_log = trainer.test(vae, vae_data)
#
#     # diff_loss = start_test_loss_log[0]['test_loss'] - end_test_loss_log[0]['test_loss']
#     callbacks[3].plot('loss')
#
#     # %%
#     model_config['mode'] = 'VAE'
#     model_config['name'] = 'f_vae'
#     model_config['epochs'] = 10
#     model_config['batch_size'] = 500
#     vae, vae_data, callbacks = generate_test_case(model_config)
#
#     trainer = pl.Trainer(max_epochs=model_config['epochs'], log_every_n_steps=8,
#                          callbacks=callbacks)
#
#     start_test_loss_log = trainer.test(vae, vae_data)
#     log_train = trainer.fit(vae, vae_data)
#     end_test_loss_log = trainer.test(vae, vae_data)
#
#     # diff_loss = start_test_loss_log[0]['test_loss'] - end_test_loss_log[0]['test_loss']
#     callbacks[3].plot('loss')
