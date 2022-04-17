import re
from pytorch_lightning import Callback, LightningModule
import abc

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from fair_vae.configs import AEConfig
from fair_vae.util import artifacts_path, MetricTracker


class VAEFrame(LightningModule, abc.ABC):
    def __init__(self, ae_config: AEConfig, *args, **kwargs):
        super(VAEFrame, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.config = ae_config

    @abc.abstractmethod
    def encode(self, x):
        pass

    @abc.abstractmethod
    def reparameterize(self, mu, logvar):
        pass

    @abc.abstractmethod
    def decode(self, z):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def loss_fn(self, pred_result, x, return_loss_comps=False):
        pass

    def forward(self, x):
        pass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def fit(self, vae_data):
        pass

    def callbacks(self, model_path):
        early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.05, patience=100, verbose=False,
                                            mode="min")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=model_path,
            filename=self.config.name + str(hash(self.config)) + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )

        cb_log = MetricTracker()

        callbacks = [checkpoint_callback, early_stop_callback, cb_log]

        return callbacks

    @classmethod
    def load_best(self, name, config=None):
        vae_paths = list(filter(lambda x:name in x.stem, artifacts_path.joinpath(self.model_impl).iterdir()))

        if config is not None:
            config_hash = str(hash(config))

            vae_paths = [e for e in vae_paths if config_hash in str(e)]

        vae_paths = sorted(vae_paths, key=lambda x: float(re.findall(r"\d+\.\d+", x.stem)[-1]))

        best = vae_paths[0]

        return self.load_from_checkpoint(best)