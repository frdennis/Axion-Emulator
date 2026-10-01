from typing import Tuple, Optional, List, Type, Any
import pytorch_lightning as pl
import torch


class EnsembleVotingModel(pl.LightningModule):
    def __init__(
        self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str], loss_fct
    ):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )
        self.loss_fct = loss_fct

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        # TODO CHANGE LOSS
        loss = self.loss_fct(logits, batch[1])
        self.log("test_loss", loss)
