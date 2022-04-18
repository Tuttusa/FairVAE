import re
from pytorch_lightning import LightningModule
import abc

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from fair_vae.configs import AEConfig
from fair_vae.util import artifacts_path, MetricTracker


class VAEFrame(LightningModule, abc.ABC):
    def __init__(self, ae_config: AEConfig, *args, **kwargs):
        super(VAEFrame, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.config = ae_config

    @classmethod
    def saving_name(self, name, princ_elem):
        return name + princ_elem

    def callbacks(self, model_path):
        early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.05, patience=100, verbose=False,
                                            mode="min")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=model_path,
            filename=self.saving_name(self.config.name, self.config.principal_elem) + self.config.hash() + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )

        cb_log = MetricTracker()

        callbacks = [checkpoint_callback, early_stop_callback, cb_log]

        return callbacks

    @classmethod
    def load_best(self, name, princ_elem, config=None):
        vae_paths = list(
            filter(lambda x: self.saving_name(name, princ_elem) in x.stem, artifacts_path.joinpath(self.model_impl).iterdir()))

        if config is not None:
            config_hash = config.hash()

            vae_paths = [e for e in vae_paths if config_hash in str(e)]

        vae_paths = sorted(vae_paths, key=lambda x: float(re.findall(r"\d+\.\d+", x.stem)[-1]))

        best = vae_paths[0]

        return self.load_from_checkpoint(best)
