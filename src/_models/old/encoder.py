import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim, kernel_size, **kwargs):
        super().__init__()
        ### Convolutional section
        input_kernel_size = kernel_size
        kernel_size = (input_kernel_size[0], 1)
        output_before_flatten = (70 - (kernel_size[0] - 1) * 3) * 32
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, input_kernel_size,),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size,),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size,),
            nn.ReLU(True),
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(output_before_flatten, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, kernel_size, **kwargs):
        super().__init__()
        input_kernel_size = kernel_size
        kernel_size = (input_kernel_size[0], 1)
        output_before_flatten = (70 - (kernel_size[0] - 1) * 3) * 32
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, output_before_flatten),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(
            dim=1, unflattened_size=(32, 70 - (kernel_size[0] - 1) * 3, 1)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size,),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, input_kernel_size),
        )
        #self.smoothen = nn.Conv2d(8,8,kernel_size,padding='same',groups=8)

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        #x = self.smoothen(torch.transpose(x,-1, -3))
        #x = torch.transpose(x,-3,-1)
        return x


class AutoEncoder(LightningModule):
    """Standard AE.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
    """

    def __init__(
        self,
        input_height: int,
        first_conv: bool = False,
        latent_dim: int = 12,
        learning_rate: float = 1e-4,
        enc_out_dim=126,
        kernel_size_left=16,
        kernel_size_right=16,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()
        self.save_hyperparameters(
            "input_height", "latent_dim", "kernel_size_left", "kernel_size_right",
        )
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.kernel_size = (kernel_size_left, kernel_size_right)

        self.encoder = Encoder(latent_dim=self.latent_dim, kernel_size=self.kernel_size)
        self.decoder = Decoder(latent_dim=self.latent_dim, kernel_size=self.kernel_size)

        # self.enc_out_dim = enc_out_dim
        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        z = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        self.log("loss/val", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        self.log("loss/test", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate,  # weight_decay=self.weight_decay
        )
        # TODO: move this to config file
        # scheduler = ExponentialLR(optimizer, gamma=0.97)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=30, factor=0.1, min_lr=1.0e-6, verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val",
                "interval": "epoch",
                "frequency": 1,
            },
        }
