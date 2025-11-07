import pytest
import torch
import torch.nn as nn
from lightning.pytorch import Trainer

from stable_pretraining import Module, forward
from stable_pretraining.data import DataModule
from stable_pretraining.losses import NTXEntLoss


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_module_amp_multiple_optimizers_gpu():
    """Test AMP (fp16) with multiple optimizers on GPU.

    This test verifies that the precision plugin properly handles GradScaler
    when using multiple optimizers in manual optimization mode with fp16.

    Issue: #337 - Without using trainer.precision_plugin.optimizer_step(),
    the GradScaler's inf check tracking fails with:
    "AssertionError: No inf checks were recorded for this optimizer."

    Fix: PR #356 - Use trainer.precision_plugin.optimizer_step() instead of
    direct optimizer.step() to properly handle AMP with multiple optimizers.
    """
    # Define simple backbone and projector
    backbone = nn.Linear(128, 64)
    projector = nn.Linear(64, 32)

    # Define the module with multiple optimizers
    module = Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr_forward,
        simclr_loss=NTXEntLoss(temperature=0.5),
        optim={
            "backbone_opt": {
                "modules": "backbone",
                "optimizer": {"type": "AdamW", "lr": 1e-3},
            },
            "projector_opt": {
                "modules": "projector",
                "optimizer": {"type": "AdamW", "lr": 1e-3},
            },
        },
    )

    # Define dummy data loaders
    train_data = [
        {"image": torch.randn(128), "label": torch.tensor([0])} for _ in range(4)
    ]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2)

    val_data = [
        {"image": torch.randn(128), "label": torch.tensor([0])} for _ in range(2)
    ]
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=2)

    data = DataModule(train=train_loader, val=val_loader)

    # Define the trainer with GPU and fp16 (not bf16)
    trainer = Trainer(
        max_epochs=1,
        num_sanity_val_steps=1,
        callbacks=[],
        precision="16-mixed",  # This forces fp16 with GradScaler on GPU
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # This should work with the fix, fail without it on main branch
    trainer.fit(module, datamodule=data, ckpt_path=None)
