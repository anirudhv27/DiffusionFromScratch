import os
from typing import Any, Sequence, Union

import lightning as L
import torch
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchvision.datasets.cifar import CIFAR10

from models.unet import DiffusionUNet

"""
Training algorithm step:
- Sample a random data point, and sample T
- Produce a random noise
- Make gradient descent step (Loss function is equal to the MSE between the actual and predicted noise, using computed alphas and betas)

Sampling:
- Sample a random noise, then iteratively predict previous noise until the initial image is found
"""


class LightingDiffusion(L.LightningModule):
    def __init__(
        self,
        model: DiffusionUNet,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model: DiffusionUNet = model
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T).to("mps")
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(
            self.alphas, dim=0
        )  # zero indexed: just remember to subtract one before indexing to get value!

    def training_step(self, batch, batch_idx):
        batch = batch[0]  # don't need label information
        batch = batch.to("mps")

        # Sample random values from 0 to T - 1
        batch_size = batch.shape[0]
        # print('batch shape', batch.shape)
        t_batch = torch.randint(0, self.T, (batch_size,), device=batch.device)
        # print('t batch shape', t_batch.shape)

        assert batch.shape[0] == t_batch.shape[0]

        noise = torch.randn_like(batch).to("mps")
        alpha_bars_batch = self.alpha_bars[t_batch].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bars_batch = torch.sqrt(1.0 - self.alpha_bars)[
            t_batch
        ].view(batch_size, 1, 1, 1)

        noised_batch = (
            alpha_bars_batch * batch + sqrt_one_minus_alpha_bars_batch * noise
        )

        pred_noise = self.model(
            noised_batch,
            t_batch,
        )

        loss = torch.nn.functional.mse_loss(noise, pred_noise)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


if __name__ == "__main__":
    model = DiffusionUNet()
    diffusion = LightingDiffusion(model)

    # setup data for MNIST
    torchvision.datasets.CIFAR10.url = torchvision.datasets.CIFAR10.url.replace(
        "https://", "http://"
    )
    dataset = torchvision.datasets.CIFAR10(
        os.path.dirname(os.getcwd()) + "/data",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(max_epochs=10, default_root_dir=os.getcwd() + '/../results')
    trainer.fit(model=diffusion, train_dataloaders=train_loader)
