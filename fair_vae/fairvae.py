import torch
import torch.nn as nn
import torchmetrics
from Fdatasets.real import Real
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torchmetrics import MetricTracker

from fair_vae.losses import reconstruction_loss, kld_loss, MMD
from fair_vae.modules import Encoder, Decoder


class FVAE(nn.Module):
    def __init__(self, x_size, t_size, y_size, embedding_dim, compress_dims, decompress_dims, loss_factor, l2scale,
                 mode):
        super().__init__()

        encoder1_inp_size = x_size + t_size
        self.encoder1 = Encoder(encoder1_inp_size, compress_dims, embedding_dim, mode)  # q(z1 | t, x)

        encoder2_inp_size = self.encoder1.output_size + y_size
        self.encoder2 = Encoder(encoder2_inp_size, compress_dims, embedding_dim, mode)  # q(z2 | y, z1)

        self.y_learner = nn.Linear(self.encoder1.output_size, y_size)  # p(y | z1)

        decoder1_inp_size = self.encoder2.output_size + y_size
        self.decoder1 = Decoder(self.encoder1.output_size, decompress_dims, decoder1_inp_size)  # p(z1 | z2, y)

        decoder2_inp_size = self.encoder1.output_size + t_size
        self.decoder2 = Decoder(x_size, decompress_dims, decoder2_inp_size)  # p(x | z1, t)

    def encode(self, x, t, y):
        z1 = self.encoder1(torch.cat([x, t], dim=-1))

        z2 = self.encoder2(torch.cat([z1, y], dim=-1))

        y_pred = self.y_learner(z1)

        return z1, z2, y_pred

    def decode(self, z2, t, y):
        rec_z1 = self.decoder1(torch.cat([z2, t], dim=-1))

        rec_x = self.decoder2(torch.cat([rec_z1, y], dim=-1))

        return rec_z1, rec_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), std

    def predict(self, x, t, y):
        if self.mode == 'AE':
            z1, z2, rec_y = self.encode(x, t, y)
            rec_z1, rec_x = self.decode(z2, t, y)

            return dict(z1=z1, z2=z2, rec_y=rec_y, rec_z1=rec_z1, rec_x=rec_x, x=x, t=t, y=y)

        else:

            (z1, z1_logvar), (z2, z2_logvar), rec_y = self.encode(x, t, y)

            z1_mu, z1_std = self.reparameterize(z1, z1_logvar)
            z2_mu, z2_std = self.reparameterize(z2, z2_logvar)

            (rec_z1_mu, rec_z1_logvar), (rec_x_mu, rec_x_logvar) = self.decode(z2, t, y)

            _, dec_z1_std = self.reparameterize(rec_z1_mu, rec_z1_logvar)
            _, dec_x_std = self.reparameterize(rec_x_mu, rec_x_logvar)

            return dict(z1_mu=z1_mu, z1_logvar=z1_logvar, z2_mu=z2_mu,
                        z2_logvar=z2_logvar, rec_y=rec_y, z1=z1, z1_std=z1_std,
                        z2=z2, z2_std=z2_std, rec_z1_mu=rec_z1_mu, rec_z1_logvar=rec_z1_logvar,
                        rec_x_mu=rec_x_mu, rec_x_logvar=rec_x_logvar, dec_z1_std=dec_z1_std,
                        dec_x_std=dec_x_std, x=x, t=t, y=y)

    def recon_loss(self, rec_mu, ori, rec_std=None, transformer=None):
        if transformer is not None:
            if rec_std is not None:
                recon_loss = reconstruction_loss(transformer, rec_mu, ori, rec_std, self.loss_factor)
            else:
                recon_loss = reconstruction_loss(transformer, rec_mu, ori)
        else:
            recon_loss = self.loss_factor * mse_loss(rec_mu, ori)

        return recon_loss

    def forward(self, x, t, y):
        pred = self.predict(x, t, y)

        total_loss, total_rec_loss, total_kl_loss, mmd_loss = self.loss_fn(pred)

        return dict(pred=pred, total_loss=total_loss, total_rec_loss=total_rec_loss,
                    total_kl_loss=total_kl_loss, mmd_loss=mmd_loss)

    def loss_fn(self, pred):

        total_kl_loss, mmd_loss = 0.0, 0.0

        ## rec loss x

        rec_loss_x = self.recon_loss(pred['rec_x_mu'], pred['x'], pred.get('rec_x_std', None), self.tranformer_x)

        ## rec loss z1

        rec_loss_z1 = self.recon_loss(pred['rec_z1_mu'], pred['z1'], pred.get('rec_z1_std', None))

        ## rec loss Y

        rec_loss_y = self.recon_loss(pred['rec_y'], pred['y'], transformer=self.tranformer_y)

        total_rec_loss = rec_loss_x + rec_loss_z1 + rec_loss_y

        if self.mode == 'VAE':
            ## kl loss z1

            kl_z1_loss = kld_loss(pred['z1_mu'], pred['z1_logvar'])

            ## kl loss z2

            kl_z2_loss = kld_loss(pred['z2_mu'], pred['z2_logvar'])

            total_kl_loss = kl_z1_loss + kl_z2_loss

        ## MMD loss
        mmd_loss = MMD(pred['z1'], pred['rec_z1_mu'])

        total_loss = total_rec_loss + total_kl_loss + mmd_loss

        return total_loss, total_rec_loss, total_kl_loss, mmd_loss

    def training_step(self, batch, batch_idx, *args, **kwargs):

        x, t, y = batch[0]
        if self._device:
            x = x.to(self._device)

        pred_result = self.forward(x, t, y)

        self.log(f"train_loss", pred_result['loss'], on_step=True, logger=True)

        logs = {"loss": pred_result['loss']}

        return logs

    def validation_step(self, batch, batch_idx):
        x, t, y = batch[0]

        if self._device:
            x = x.to(self._device)

        pred_result = self.forward(x, t, y)

        self.log('val_loss', pred_result['loss'], on_step=True, logger=True)

        val_test_loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])

        self.log('val_test_loss', val_test_loss, on_step=True, logger=True)

        return {"loss": pred_result['loss']}

    def test_step(self, batch, batch_idx):
        x, t, y = batch[0]
        if self._device:
            x = x.to(self._device)
        pred_result = self.forward(x, t, y)

        loss = torchmetrics.functional.mean_absolute_percentage_error(x, pred_result['rec'])
        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), weight_decay=self.l2scale)

#%%

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

    vae = FVAE(input_size=model_config['input_size'], embedding_dim=model_config['latent_size'],
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