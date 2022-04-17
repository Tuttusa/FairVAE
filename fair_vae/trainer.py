from Fdatasets.real import Real, Dataset

from fair_vae.configs import AEConfig
from fair_vae.datamodule import VAEDataModule
from fair_vae.models.fairvae import FVAE
from fair_vae.models.vae import VAE


class VAETrainer:

    @classmethod
    def fit(self, data: Dataset, config: AEConfig):
        vae_data = VAEDataModule(config=config, data=data)

        if config.fair:
            config.x_data_shape = vae_data.shape('x')
            config.t_data_shape = vae_data.shape('t')
            config.y_data_shape = vae_data.shape('y')
            vae = FVAE(config)
        else:
            config.x_data_shape = vae_data.shape('x')
            vae = VAE(config)

        vae.fit(vae_data)

        return vae

    @classmethod
    def load(self, name, fair=False, config=None):
        if fair:
            vae = FVAE.load_best(name, config)
        else:
            vae = VAE.load_best(name, config)
        return vae


# %%
# db = Real.income()
#%%
# model_config = AEConfig()
#
# ae = VAETrainer.fit(db, model_config)
# aeb = VAETrainer.load(model_config.name)

# %%
# model_config = AEConfig(name='vae', mode='VAE')
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name)

# %%
# model_config = AEConfig(name='fae', mode='AE', fair=True)
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name, fair=True)
#
# #%%
#
# model_config = AEConfig(name='fvae', mode='VAE', fair=True)
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name, fair=True)
#
# print("fff")
