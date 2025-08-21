"""SimCLR training on CIFAR10 with Latent Visualization."""

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

# Use core LatentViz callback with W&B fix
from stable_ssl.callbacks.latent_viz import LatentViz

simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.0),
            transforms.ToImage(**ssl.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
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

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=256,
    num_workers=8,
    drop_last=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
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
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

linear_probe = ssl.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = ssl.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

# LatentViz callback for 2D visualization of embeddings (clean with only interactive plot)
latent_viz = LatentViz(
    name="embedding_viz",
    input="embedding",  # Visualize the backbone embeddings
    target="label",  # Color points by class labels
    projection=nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),  # Project to 2D
    ),
    queue_length=2048,  # Number of samples to keep in memory
    k_neighbors=15,  # Number of neighbors for high-D graph
    n_negatives=5,  # Number of negative samples per positive
    optimizer={
        "type": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-2,
    },
    scheduler={
        "type": "CosineAnnealingLR",
        "T_max": 1000,
    },
    accumulate_grad_batches=1,
    update_interval=10,  # Update projection every 10 batches
    warmup_epochs=20,  # Wait 20 epochs before starting projection training
    distance_metric="euclidean",
    plot_interval=10,  # Save visualization every 10 epochs
    save_dir="latent_viz_simclr_cifar10",
    input_dim=512,  # Dimension of ResNet18 embeddings
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-simclr-latent-viz",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=1000,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, latent_viz],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()