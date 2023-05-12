import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple, Optional, List, Type
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from semu.models.moment_loss import moment_loss
from semu.models.activations import Speculator


class CPower(pl.LightningModule):
    """ Fully connected network with forward pass.
    """

    def __init__(
        self,
        n_features: int,
        output_dim: int,
        hidden_dims: List[int] = [250, 250],
        loss: str = "MSELoss",
        learning_rate: float = 1.0e-2,
        **kwargs,
    ):

        super(CPower, self).__init__()
        self.save_hyperparameters(
            "n_features", "output_dim", "hidden_dims", "learning_rate", "loss",
        )
        self.learning_rate = learning_rate

        self.n_features = n_features
        self.output_dim = output_dim
        # Stack of hidden layers
        self.n_hidden_layers = len(hidden_dims)
        self.hidden_layers = nn.ModuleList()
        self.activations = []
        for i in range(self.n_hidden_layers):
            layer = nn.Linear(
                in_features=n_features if i == 0 else hidden_dims[i - 1],
                out_features=hidden_dims[i],
            )
            self.hidden_layers.append(layer)
            self.activations.append(Speculator(dim_input=hidden_dims[i]))
        self.activations = nn.ModuleList(self.activations)
        self.output_layer = nn.Linear(
            in_features=hidden_dims[-1], out_features=self.output_dim
        )
        self.loss = loss
        self.loss_fct = getattr(torch.nn, loss)()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning_rate", type=float, default=1.0e-3)
        parser.add_argument("--weight_decay", type=float, default=1.0e-4)
        parser.add_argument("--n_hidden", type=int, default=100)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="gelu")
        parser.add_argument("--batch_norm", type=int, default=0)
        return parent_parser

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer.forward(x)
            x = self.activations[i](x)
        return self.output_layer.forward(x)

    def custom_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x,)
        return self.loss_fct(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/train", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/test", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/val", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,)
        # TODO: move this to config file
        # scheduler = ExponentialLR(optimizer, gamma=0.97)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.1,
            min_lr=self.learning_rate * 1.0e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": "loss/val",
        }
