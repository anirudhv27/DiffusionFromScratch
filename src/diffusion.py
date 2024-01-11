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
        self.model.to('cpu')
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T).to('mps')
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(
            self.alphas, dim=0
        )  # zero indexed: just remember to subtract one before indexing to get value!

    def training_step(self, batch, batch_idx):
        batch = batch[0] # don't need label information        
        batch = batch.to('mps')
        
        # Sample random values from 0 to T - 1
        batch_size = batch.shape[0]
        t_batch = torch.randint(1, self.T, (batch_size,)).to('mps')
        assert batch.shape[0] == t_batch.shape[0]

        gt_noise = torch.randn_like(batch).to('mps')
        
        pred_noise = self.model(
            self.alpha_bars[t_batch - 1] * batch
            + torch.sqrt(1 - self.alpha_bars)[t_batch - 1] * gt_noise,
            t_batch,
        )
        loss_fn = torch.nn.functional.mse_loss(pred_noise, gt_noise)
        return loss_fn

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = DiffusionUNet()
    diffusion = LightingDiffusion(model)

    # setup data for MNIST
    torchvision.datasets.CIFAR10.url = torchvision.datasets.CIFAR10.url.replace(
        "https://", "http://"
    )
    dataset = torchvision.datasets.CIFAR10(
        os.path.dirname(os.getcwd()) + '/data', download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(dataset)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=diffusion, train_dataloaders=train_loader)
