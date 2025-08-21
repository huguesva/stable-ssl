"""Generate and launch SLURM jobs for LatentViz parameter sweep experiments on CIFAR10."""

import json
import itertools
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import time

def generate_parameter_grid():
    """Generate all LatentViz parameter combinations for the full sweep.
    
    This generates ALL combinations: 3×3×3×1×1×1 = 27 experiments total.
    """
    param_grid = {
        'queue_length': [2048, 4096, 16384],    # 3 values
        'k_neighbors': [5, 10, 20],             # 3 values  
        'n_negatives': [1, 5, 10],              # 3 values
        'update_interval': [10],                    # FIXED at 10
        'lr': [1e-3],                               # FIXED at 1e-3
        'warmup_epochs': [20],                      # FIXED at 20
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    experiments = []
    for combination in itertools.product(*values):
        config_dict = dict(zip(keys, combination))
        exp_id = (f"q{config_dict['queue_length']}_k{config_dict['k_neighbors']}_"
                 f"n{config_dict['n_negatives']}_ui{config_dict['update_interval']}_"
                 f"lr{config_dict['lr']:.0e}_w{config_dict['warmup_epochs']}")
        config_dict['exp_id'] = exp_id
        experiments.append(config_dict)
    
    return experiments


def generate_subset_experiments():
    """Generate a comprehensive grid of experiments for the paper.
    
    This generates ALL combinations: 3×3×3 = 27 experiments total.
    """
    
    # Grid search over key parameters with fixed update_interval, lr, and warmup
    param_grid = {
        'queue_length': [2048, 4096, 16384],        # 3 values (larger queues)
        'k_neighbors': [5, 10, 20],                 # 3 values
        'n_negatives': [1, 5, 10],                  # 3 values
    }
    
    # Fixed parameters (good defaults)
    fixed_params = {
        'update_interval': 10,
        'lr': 1e-3,
        'warmup_epochs': 20,
    }
    
    # Generate all combinations: 3×3×3 = 27 experiments
    all_configs = []
    for ql in param_grid['queue_length']:
        for kn in param_grid['k_neighbors']:
            for nn in param_grid['n_negatives']:
                config = {
                    'queue_length': ql,
                    'k_neighbors': kn,
                    'n_negatives': nn,
                    **fixed_params
                }
                all_configs.append(config)
    
    # Add exp_id to each config
    for config in all_configs:
        config['exp_id'] = (f"q{config['queue_length']}_k{config['k_neighbors']}_"
                          f"n{config['n_negatives']}_ui{config['update_interval']}_"
                          f"lr{config['lr']:.0e}_w{config['warmup_epochs']}")
    
    return all_configs


def create_experiment_script(config: Dict[str, Any], exp_dir: Path) -> Path:
    """Create a Python script for a single experiment."""
    
    script_content = f'''"""CIFAR10 SimCLR training with LatentViz - Experiment {config['exp_id']}."""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('/home/vanasseh/stable-ssl/benchmarks')
from utils import get_data_dir

import stable_ssl as ssl
from stable_ssl.data import transforms
from stable_ssl.callbacks.latent_viz import LatentViz
from stable_ssl.callbacks.rankme import RankMe

# ====== DATA SETUP (Same as simclr-resnet18.py) ======
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

# ====== MODEL SETUP (Same as simclr-resnet18.py) ======
def forward(self, batch, stage):
    out = {{}}
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
    optim={{
        "optimizer": {{
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
        }},
        "scheduler": {{
            "type": "LinearWarmupCosineAnnealing",
        }},
        "interval": "epoch",
    }},
)

# ====== CALLBACKS (Standard + LatentViz with experimental params) ======
linear_probe = ssl.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={{
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    }},
)

knn_probe = ssl.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length={config['queue_length']},  # Match LatentViz queue
    metrics={{"accuracy": torchmetrics.classification.MulticlassAccuracy(10)}},
    input_dim=512,
    k=10,
)

# LatentViz with experimental parameters
latent_viz = LatentViz(
    name="embedding_viz",
    input="embedding",
    target="label",
    projection=nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    ),
    queue_length={config['queue_length']},
    k_neighbors={config['k_neighbors']},
    n_negatives={config['n_negatives']},
    optimizer={{
        "type": "AdamW",
        "lr": {config['lr']},
        "weight_decay": 1e-2,
    }},
    scheduler={{
        "type": "CosineAnnealingLR",
        "T_max": 1000,
    }},
    accumulate_grad_batches=1,
    update_interval={config['update_interval']},
    warmup_epochs={config['warmup_epochs']},
    distance_metric="euclidean",
    plot_interval=10,
    save_dir="experiments/cifar10_latent_viz_sweep/{config['exp_id']}/latent_viz",
    input_dim=512,
)

# RankMe for comparison
rankme = RankMe(
    name="rankme",
    target="embedding",
    queue_length={config['queue_length']},
    target_shape=512,
)

# ====== TRAINING ======
wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-latent-viz-sweep",
    name="cifar10_{config['exp_id']}",
    log_model=False,
    config={repr(config)},
)

csv_logger = CSVLogger(
    save_dir="experiments/cifar10_latent_viz_sweep/{config['exp_id']}",
    name="metrics"
)

trainer = pl.Trainer(
    max_epochs=1000,  # Reduced for parameter sweep
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, latent_viz, rankme],
    precision="16-mixed",
    logger=[wandb_logger, csv_logger],
    enable_checkpointing=False,
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
'''
    
    script_path = exp_dir / f"train_{config['exp_id']}.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def create_slurm_script(config: Dict[str, Any], script_path: Path, exp_dir: Path) -> Path:
    """Create SLURM submission script for a single experiment."""
    
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name=c10_lv_{config['exp_id'][:17]}
#SBATCH --output={exp_dir}/slurm_%j.out
#SBATCH --error={exp_dir}/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=owner_gred_braid_gpu
#SBATCH --partition=owner_gred_braid_gpu

# Load the conda module and activate environment
module load conda-forge-bundle
source /apps/rocs/2020.08/cascadelake/software/conda-forge-bundle/2021.05/etc/profile.d/conda.sh
conda activate /home/vanasseh/scratch/conda/envs/stable-ssl-env

# Run experiment
cd {Path.cwd()}
python {script_path}

# Save completion marker
echo "Experiment completed at $(date)" > {exp_dir}/completed.txt
'''
    
    slurm_path = exp_dir / f"submit_{config['exp_id']}.sh"
    with open(slurm_path, 'w') as f:
        f.write(slurm_content)
    
    return slurm_path


def launch_experiments(experiments: List[Dict[str, Any]], 
                       base_dir: Path = Path("experiments/cifar10_latent_viz_sweep"),
                       dry_run: bool = False):
    """Create and launch all experiment jobs."""
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment index
    with open(base_dir / "experiment_index.json", 'w') as f:
        json.dump(experiments, f, indent=2)
    
    job_ids = []
    for i, config in enumerate(experiments):
        exp_dir = base_dir / config['exp_id']
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create experiment script
        script_path = create_experiment_script(config, exp_dir)
        
        # Create SLURM script
        slurm_path = create_slurm_script(config, script_path, exp_dir)
        
        if not dry_run:
            # Submit job (Python 3.6 compatible)
            result = subprocess.run(
                ['sbatch', str(slurm_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"Submitted CIFAR10 job {job_id} for experiment {config['exp_id']}")
            else:
                print(f"Failed to submit {config['exp_id']}: {result.stderr}")
        else:
            print(f"[DRY RUN] Would submit CIFAR10 experiment: {config['exp_id']}")
        
        # Small delay to avoid overwhelming scheduler
        if i % 10 == 0 and not dry_run:
            time.sleep(1)
    
    # Save job IDs
    if job_ids and not dry_run:
        with open(base_dir / "job_ids.txt", 'w') as f:
            for job_id in job_ids:
                f.write(f"{job_id}\n")
        
        print(f"\nSubmitted {len(job_ids)} CIFAR10 jobs")
        print(f"Monitor with: squeue -u $USER")
        print(f"Job IDs saved to: {base_dir}/job_ids.txt")
    
    return job_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch CIFAR10 LatentViz parameter sweep experiments")
    parser.add_argument("--mode", choices=["subset", "full"], default="subset",
                       help="Run subset (focused) or full parameter sweep")
    parser.add_argument("--dry-run", action="store_true",
                       help="Create scripts without submitting jobs")
    parser.add_argument("--base-dir", type=str, default="experiments/cifar10_latent_viz_sweep",
                       help="Base directory for experiments")
    
    args = parser.parse_args()
    
    if args.mode == "subset":
        experiments = generate_subset_experiments()
        print(f"Generated {len(experiments)} focused CIFAR10 experiments")
    else:
        experiments = generate_parameter_grid()
        print(f"Generated {len(experiments)} total CIFAR10 experiments (full grid)")
    
    launch_experiments(
        experiments,
        base_dir=Path(args.base_dir),
        dry_run=args.dry_run
    )