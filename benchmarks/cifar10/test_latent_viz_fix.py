"""Quick test of the fixed LatentViz WandB logging."""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_ssl as ssl
from stable_ssl.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

# Import the fixed LatentViz callback
from latent_viz_fixed import LatentVizFixed

# Simple transforms for quick testing
simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**ssl.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**ssl.data.static.CIFAR10),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**ssl.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = ssl.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=simclr_transform,
)
val_dataset = ssl.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

# Small batch size for quick testing
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=64,  # Smaller batch size
    num_workers=4,
    drop_last=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=4,
)

data = ssl.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    if self.training:
        proj = self.projector(out["embedding"])
        views = ssl.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.simclr_loss(views[0], views[1])
    return out


backbone = ssl.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 128),
)

module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {
            "type": "Adam",
            "lr": 1e-3,
        },
        "interval": "epoch",
    },
)

# Test the fixed LatentViz callback
latent_viz = LatentVizFixed(
    name="test_viz",
    input="embedding",
    target="label",
    projection=nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    ),
    queue_length=512,  # Smaller queue for testing
    k_neighbors=10,
    n_negatives=3,
    optimizer={
        "type": "Adam",
        "lr": 1e-3,
    },
    accumulate_grad_batches=1,
    update_interval=5,  # More frequent updates for testing
    distance_metric="euclidean",
    plot_interval=1,  # Plot every epoch for testing
    save_dir="test_latent_viz",
    input_dim=512,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-simclr-test-viz",
    log_model=False,
    name="test-latent-viz-fix",
)

trainer = pl.Trainer(
    max_epochs=2,  # Just 2 epochs for testing
    num_sanity_val_steps=0,
    callbacks=[latent_viz],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    limit_train_batches=20,  # Only run 20 batches per epoch
    limit_val_batches=10,  # Only validate on 10 batches
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()

print("\n" + "="*60)
print("TEST COMPLETE!")
print("Check WandB project 'cifar10-simclr-test-viz' for visualizations")
print("Look for 'test_viz/2d_latent_space' in the Media or Custom Charts section")
print("="*60)