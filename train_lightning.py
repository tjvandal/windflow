import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import (
    SLURMEnvironment,
)

_IN_SLURM = 'SLURM_NODEID' in os.environ

import torch

torch.cuda.empty_cache()

from torch.utils import data

from windflow.datasets import get_dataset
from windflow.networks.models import get_flow_model

os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"


def train_net(params, rank=0):
    # set device
    # if not device:
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = rank  # % N_DEVICES
    print(f"in train_net rank {rank} on device {device}")

    dataset_train, dataset_valid = get_dataset(
        params["dataset"],
        params["data_path"],
        scale_factor=params["scale_input"],
        frames=params["input_frames"],
        levels=params["levels"],
    )

    data_params = {
        "batch_size": params["batch_size"],
        "shuffle": True,
        "num_workers": 8,
    }
    training_generator = data.DataLoader(dataset_train, **data_params)
    val_generator = data.DataLoader(dataset_valid, **data_params)

    model = get_flow_model(
        params["model_name"], small=False,
        scheduler_total_steps=params["max_iterations"],
    )

    logger = pl.loggers.WandbLogger(
        project=params["project"],
        offline=False,
        name=params["model_name"],
        # **config["logger"],
        save_dir=params["model_path"],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=params["model_path"], every_n_train_steps=1000
    )

    train_iters = 1000
    trainer = pl.Trainer(
        max_epochs=params["max_iterations"] // train_iters,
        logger=logger,
        accelerator="gpu",
        gradient_clip_val=1.0,
        devices=params["n_gpus"],
        num_nodes=params["n_nodes"],
        limit_val_batches=1,
        limit_train_batches=train_iters,
        log_every_n_steps=1,
        default_root_dir=params["model_path"],
        callbacks=[checkpoint_callback],
        detect_anomaly=True,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",
        # auto_lr_find=True,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        plugins=[SLURMEnvironment(auto_requeue=False)] if _IN_SLURM else [],
        precision="bf16-mixed",
        # accumulate_grad_batches=conf.trainer.accumulate_grad_batches,
    )

    print(f'Start Training {params["model_name"]} on {params["dataset"]}')
    trainer.fit(
        model=model,
        train_dataloaders=training_generator,
        val_dataloaders=val_generator,
        ckpt_path=params["load_ckpt_path"],
    )


if __name__ == "__main__":
    # Feb 1 2020, Band 1 hyper-parameter search
    # {'w': 0.6490224421024322, 's': 0.22545622639358046, 'batch_size': 128}
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/raft-size_512/", type=str)
    parser.add_argument("--dataset", default="g5nr", type=str)
    parser.add_argument(
        "--data_path",
        default="/explore/nobackup/people/tvandal/data/windflow/G5NR_7km_patches",
        type=str,
    )
    parser.add_argument("--input_frames", default=2, type=int)
    parser.add_argument("--load_ckpt_path", default=None, type=str)
    parser.add_argument("--model_name", default="raft", type=str)
    parser.add_argument("--project", default="windflow", type=str)
    # parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument(
        "--scale_input",
        default=None,
        type=float,
        help="Bilinear interpolation on input image.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000000,
        help="Number of training iterations",
    )
    parser.add_argument("--log_step", default=500, type=int)
    parser.add_argument("--checkpoint_step", default=5000, type=int)
    parser.add_argument("--n_gpus", default=1, type=int)
    parser.add_argument("--n_nodes", default=1, type=int)
    parser.add_argument("--loss", default="L1", type=str)
    parser.add_argument(
        "--levels",
        default=None,
        type=lambda s: [int(x) for x in s.split(",")] if s else None,
        help="Comma-separated vertical level indices (e.g. '64' or '60,64,68'). Default: all levels.",
    )

    args = parser.parse_args()
    train_net(vars(args))
