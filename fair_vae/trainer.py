from Fdatasets.real import Real, Dataset

from fair_vae.configs import AEConfig
from fair_vae.datamodule import VAEDataModule
from fair_vae.models.fairvae import FVAE
from fair_vae.models.vae import VAE


class VAETrainer:

    @classmethod
    def fit(self, data: Dataset, config: AEConfig):
        vae_data = VAEDataModule(config=config, data=data)
        config.datamodule_elem_index = vae_data.elem_batch_index()

        config.x_data_shape = vae_data.shape('x')
        config.t_data_shape = vae_data.shape('t')
        config.y_data_shape = vae_data.shape('y')

        if config.model == 'FVAE':
            vae = FVAE(config)

        else:
            vae = VAE(config)

        vae.fit(vae_data)

        return vae

    @classmethod
    def load(self, name, principal_elem, model, config=None):
        if model == 'FVAE':
            vae = FVAE.load_best(name, principal_elem, config)
        else:
            vae = VAE.load_best(name, principal_elem, config)
        return vae


# %%
db = Real.income()

#%%
model_config = AEConfig(name='ae', mode='AE', uncertainty_decoder=True)
model_config.principal_elem = 'x'

ae = VAETrainer.fit(db, model_config)
aeb = VAETrainer.load(model_config.name, model_config.principal_elem,  model='AE')

# %%
# model_config = AEConfig(name='vae', mode='VAE')
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name)

# %%
# model_config = AEConfig(name='fae', mode='AE', model='FVAE')
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name, model='FVAE')

# %%

# model_config = AEConfig(name='fvae', mode='VAE', model='VAE')
# model_config.principal_elem = 't'
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name, model_config.principal_elem, 'VAE')
# %%

# model_config = AEConfig(name='fvae', mode='VAE', fair=True)
#
# vae = VAETrainer.fit(db, model_config)
# vaeb = VAETrainer.load(model_config.name, fair=True)
# print("fff")
