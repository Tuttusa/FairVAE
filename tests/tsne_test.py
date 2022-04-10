import torch
from Fdatasets.real import Real
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fair_vae.util import artifacts_path
from fair_vae.vae import VAE
from fair_vae.fairvae import FVAE


def get_ae(case='vae'):
    if case == 'vae':
        vae_path = artifacts_path.joinpath('vae/vae-epoch=49-val_loss=8.73.ckpt')

        vae = VAE.load_from_checkpoint(vae_path)

        with torch.no_grad():
            ll = torch.from_numpy(vae.transformer.transform(db.x.to_numpy())).float()
            X = vae.encoder(ll)

    elif case == 'ae':
        vae_path = artifacts_path.joinpath('ae/ae-epoch=99-val_loss=0.14.ckpt')

        vae = VAE.load_from_checkpoint(vae_path)

        with torch.no_grad():
            ll = torch.from_numpy(vae.transformer.transform(db.x.to_numpy())).float()
            X = vae.encoder(ll)

    elif case == 'fae':
        vae_path = artifacts_path.joinpath('fvae/f_ae-epoch=39-val_loss=3.92.ckpt')

        vae = FVAE.load_from_checkpoint(vae_path)

        with torch.no_grad():
            ll = torch.cat([torch.from_numpy(vae.transformer_x.transform(db.x.to_numpy())).float(),
                            torch.from_numpy(vae.transformer_t.transform(db.t.to_numpy())).float()], dim=-1)
            X = vae.encoder1(ll)

    else:
        vae_path = artifacts_path.joinpath('fvae/f_vae-epoch=09-val_loss=13.83.ckpt')

        vae = FVAE.load_from_checkpoint(vae_path)

        with torch.no_grad():
            ll = torch.cat([torch.from_numpy(vae.transformer_x.transform(db.x.to_numpy())).float(),
                            torch.from_numpy(vae.transformer_t.transform(db.t.to_numpy())).float()], dim=-1)
            X = vae.encoder1(ll)

    return X


def plot_gender_tsne_distr(X, y, use_case):
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=20, palette="deep")
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("{} gender".format(use_case))
    plt.show()


# %%
db = Real.income()
df = db.to_df()

x = df[[col for col in df.columns if 'x' in col or 't' in col]]

#%%
y = df['t_2']

use_case = 'fvae'

X, _ = get_ae(use_case)
plot_gender_tsne_distr(X, y, use_case)
