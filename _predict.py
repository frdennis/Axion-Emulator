from pathlib import Path
import pickle
from typing import Union, List, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import torch
import yaml
from flax.core.frozen_dict import freeze, FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict


import _scalers as scalers
import _models as models


def convert_state_dict_from_pt(
    model_class: ABC, state: Dict,
):
    """
    Converts a PyTorch parameter state dict to an equivalent Flax parameter state dict
    """
    state = {k: v.numpy() for k, v in state.items()}
    state = model_class.convert_from_pytorch(state,)
    state = unflatten_dict({tuple(k.split(".")): v for k, v in state.items()})
    return state


class Predictor:
    def __init__(
        self,
        feature_scaler: "Scaler",
        label_scaler: "Scaler",
        model: "nn.Module",
        device: int = -1,
        flax: bool = False,
        params: dict = None,
    ):
        if model is not None and not flax:
            model.eval()
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.model = model
        self.device = device
        self.flax = flax
        if self.flax:
            self.params = params

    @classmethod
    def from_path(cls, path: Union[Path, str], device: int = -1, flax: bool = False):
        if type(path) is str:
            path = Path(path)
        if not flax:
            model = cls.load_model_from_path(path,)
            params = None
        else:
            model, params = cls.load_flax_model_from_path(path,)
        feature_scaler, label_scaler = cls.load_scalers_from_path(path=path, flax=flax)
        return cls(
            feature_scaler=feature_scaler,
            label_scaler=label_scaler,
            model=model,
            device=device,
            flax=flax,
            params=params,
        )

    @classmethod
    def get_path_to_checkpoint(cls, path):
        path_to_checkpoints = path / "checkpoints"
        return [f for f in path_to_checkpoints.glob("*.ckpt")][0]

    @classmethod
    def load_hyperparams(cls, path):
        with open(path / "hparams.yaml", "r") as fp:
            hparams = yaml.safe_load(fp)
        return hparams

    @classmethod
    def load_config(cls, path):
        with open(path / "config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        return config

    @classmethod
    def load_model_from_path(
        cls, path,
    ):
        path_to_checkpoint = cls.get_path_to_checkpoint(path=path)
        hparams = cls.load_hyperparams(path=path)
        config = cls.load_config(path=path)
        model = getattr(models, config["model"]["type"])
        return model.load_from_checkpoint(path_to_checkpoint, **hparams)

    @classmethod
    def load_flax_model_from_path(
        cls, path,
    ):
        path_to_checkpoint = cls.get_path_to_checkpoint(path=path)
        hparams = cls.load_hyperparams(path=path)
        model = models.FlaxFullyConnected(
            output_dim=hparams["output_dim"],
            hidden_dims=hparams["hidden_dims"],
            activation=hparams["activation"],
        )
        pt_state = torch.load(path_to_checkpoint, map_location="cpu")["state_dict"]
        params = convert_state_dict_from_pt(model_class=model, state=pt_state,)
        return model, params

    @classmethod
    def load_scalers_from_path(
        cls, path: Path, flax: bool = False
    ) -> Tuple["Scaler", "Scaler"]:
        with open(path / "scalers.pkl", "rb") as fp:
            scaler_dict = pickle.load(fp)
            scaler_dict["features"]["attrs"] = {
                k: v for k, v in scaler_dict["features"]["attrs"].items() if k != "flax"
            }
            scaler_dict["labels"]["attrs"] = {
                k: v for k, v in scaler_dict["labels"]["attrs"].items() if k != "flax"
            }
        feature_scaler = getattr(scalers, scaler_dict["features"]["type"])(
            flax=flax, **scaler_dict["features"]["attrs"]
        )
        label_scaler = getattr(scalers, scaler_dict["labels"]["type"])(
            flax=flax, **scaler_dict["labels"]["attrs"]
        )
        return feature_scaler, label_scaler

    def __call__(self, inputs: Union[np.array, torch.tensor], return_tensors=False):
        if not torch.is_tensor(inputs) and not self.flax:
            inputs = torch.from_numpy(inputs.astype(np.float32))
        inputs = self.feature_scaler.transform(inputs)
        if not self.flax:
            with torch.no_grad():
                predictions = self.model(inputs)
        else:
            predictions = self.model.apply(freeze({"params": self.params}), inputs)
        predictions = self.label_scaler.inverse_transform(predictions)
        if not return_tensors:
            if not self.flax:
                return predictions.detach().numpy()
            else:
                return jnp.asarray(predictions)
        return predictions
