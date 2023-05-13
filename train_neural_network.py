import yaml
import argparse
from pathlib import Path
import logging
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
import torch.nn as nn

from _data_utils import DataModule
from _nn_config import DataConfig, ModelConfig, TrainingConfig
import _models


# record all info messages
logging.basicConfig(level=logging.INFO)

# load configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=Path, default="config.yaml",)
p = parser.parse_args()
with open(p.config_path, "r") as f:
    config = yaml.safe_load(f)

data_config  = DataConfig(**config["data"])
model_config = ModelConfig(**config["model"])
train_config = TrainingConfig(**config["training"])


def show_data_range(
    logging, 
    train_data, 
    test_data, 
    val_data,
):
    logging.info(f" FEATURES SUMMARY ")
    for feature in range(train_data.x.shape[-1]):
        logging.info(
            f"Training Feature {feature}: {train_data.x[...,feature].min():.5f} < x < {train_data.x[...,feature].max():.5f}"
        )
        logging.info(
            f"Validation Feature {feature}: {val_data.x[...,feature].min():.5f} < x < {val_data.x[...,feature].max():.5f}"
        )
        logging.info(
            f"Test Feature {feature}: {test_data.x[...,feature].min():.5f} < x < {test_data.x[...,feature].max():.5f}"
        )
    print(train_data.y.shape)
    logging.info(f" LABELS SUMMARY ")
    logging.info(f"Training Output: {train_data.y.min():.5f} < y < {train_data.y.max():.5f}")
    logging.info(f"Validation Output: {val_data.y.min():.5f} < y < {val_data.y.max():.5f}")
    logging.info(f"Test Output: {test_data.y.min():.5f} < y < {test_data.y.max():.5f}")





# load dataset module
data_module = DataModule(
    feature_columns=data_config.feature_columns,
    label_columns=data_config.label_columns,
    batch_size=data_config.batch_size,
    num_workers=data_config.num_workers,
    shuffle=data_config.shuffle,
    feature_scaler=data_config.feature_scaler,
    label_scaler=data_config.label_scaler,
)

data_module.setup(
    train_data_path=Path(data_config.train_data_path),
    val_data_path=Path(data_config.val_data_path),
    test_data_path=Path(data_config.test_data_path),
    autoencoder=model_config.autoencoder,
)
# Print some summaries of the data
logging.info(f"N features = {data_module.train_data.x.shape[-1]}")
logging.info(f"N outputs = {data_module.train_data.y.shape[-1]}")
logging.info(f"N training samples = {data_module.train_data.x.shape[0]}")
logging.info(f"*" * 20 + " beforing scaling " + "*" * 20)
show_data_range(
    logging=logging,
    train_data=data_module.train_data,
    test_data=data_module.test_data,
    val_data=data_module.val_data,
)
# Apply scalings before training
data_module.transform_datasets()

logging.info(f"*" * 20 + " after scaling " + "*" * 20)
show_data_range(
    logging=logging,
    train_data=data_module.train_data,
    test_data=data_module.test_data,
    val_data=data_module.val_data,
)

# load model
output_dim=data_module.train_data.y.shape[-1]
if model_config.loss == 'moment_loss':
    output_dim *= 2
model = getattr(_models, model_config.type)(
    n_features=data_module.train_data.x.shape[-1],
    output_dim=output_dim,
    **dict(model_config),
)
# Change weight init
def init_weights_small_variance(m,):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=1.e-3)
        m.bias.data.fill_(0)

model.apply(init_weights_small_variance)
# Changed patience from 100 to 30
callbacks = [EarlyStopping(
    monitor="loss/val", patience=30, mode="min", verbose=False,
)]
'''
checkpoint_callback = ModelCheckpoint(
            monitor='loss/val',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            dirpath=train_config.default_root_dir,
        )

callbacks.append(checkpoint_callback)
'''
if train_config.stochastic_weight_avg:
    callbacks.append(
        StochasticWeightAveraging(
            swa_epoch_start = 110,
        )
    )
trainer = pl.Trainer(
    callbacks=callbacks,
    gpus=train_config.gpus,
    max_epochs=train_config.max_epochs,
    default_root_dir=train_config.default_root_dir,
    gradient_clip_val=0.5,
)
# Store dictionary with scalers
scalers_path = Path(trainer.log_dir)
scalers_path.mkdir(parents=True, exist_ok=True)
data_module.dump_scalers(path=scalers_path / "scalers.pkl")
# Store config file used to run it
with open(scalers_path / "config.yaml", "w") as f:
    yaml.dump(config, f)
t0 = time.time()
trainer.fit(model=model, datamodule=data_module)
print(f"Model took {time.time() - t0} seconds to train")
t0 = time.time()
# Testing model
result = trainer.test(model=model, datamodule=data_module)


