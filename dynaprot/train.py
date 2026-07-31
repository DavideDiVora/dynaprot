"""Train DynaProt.

    python -m dynaprot.train --data_config configs/data/atlas_config_randomcropping.yaml \
                             --model_config configs/model/dynaprot_marginal_nosinusoidal_1layer.yaml \
                             --name my-run

``--logger neptune`` needs NEPTUNE_API_TOKEN; ``--logger csv`` and ``--logger none`` do not.
"""
import argparse
import os
import random
import re

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dynaprot.config import load_config
from dynaprot.data.datasets import DynaProtDataset, OpenFoldBatchCollator
from dynaprot.model.architecture import DynaProt


def parse_args():
    parser = argparse.ArgumentParser(description="Training dynaprot")
    parser.add_argument(
        "--data_config",
        default="configs/data/atlas_config_randomcropping.yaml",
        type=str,
        help="Path to the data config YAML file",
    )
    parser.add_argument(
        "--model_config",
        default="configs/model/dynaprot_marginal_nosinusoidal_1layer.yaml",
        type=str,
        help="Path to the model config YAML file",
    )
    parser.add_argument("--name", type=str, required=True, help="Run name identifier")
    parser.add_argument(
        "--logger",
        default="neptune",
        choices=["neptune", "csv", "none"],
        help="Experiment logger. neptune needs NEPTUNE_API_TOKEN in the environment.",
    )
    parser.add_argument("--ckpt_dir", default="ckpt/", help="Where to write checkpoints")
    parser.add_argument("--num_workers", type=int, default=12, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable autograd anomaly detection (slow)",
    )
    return parser.parse_args()


def build_logger(kind, args, model_config, ckpt_path):
    """Build the requested logger, resuming the Neptune run when restarting from a ckpt."""
    if kind == "none":
        return None
    if kind == "csv":
        return CSVLogger(save_dir="logs", name=args.name)

    import neptune
    from pytorch_lightning.loggers import NeptuneLogger

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        raise SystemExit(
            "NEPTUNE_API_TOKEN is not set. Export it, or train with --logger csv / --logger none."
        )
    project = model_config["train_params"]["project"]
    tags = model_config["train_params"].get("tags", [])
    log_ckpts = model_config["train_params"].get("log_model_checkpoints", True)

    # Only checkpoints written under .neptune/ carry a resumable run id in their path.
    run_id = None
    if ckpt_path:
        match = re.search(r"\.neptune/([^/]+)", ckpt_path)
        run_id = match.group(1) if match else None
        if run_id is None:
            print(f"[train] {ckpt_path} has no Neptune run id in its path; starting a new run.")

    if run_id is not None:
        run = neptune.init_run(with_id=run_id, project=project, api_token=api_token, tags=tags)
        return NeptuneLogger(run=run, log_model_checkpoints=log_ckpts)

    return NeptuneLogger(
        project=project,
        name=args.name,
        api_key=api_token,
        tags=tags,
        log_model_checkpoints=log_ckpts,
    )


def main():
    args = parse_args()

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    model_config["data_config"] = data_config

    data_config["seed"] = args.seed
    set_seed(args.seed)

    train_dataset = DynaProtDataset(data_config, split="train")
    val_dataset = DynaProtDataset(data_config, split="val")
    test_dataset = DynaProtDataset(data_config, split="test")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    batch_size = model_config["train_params"]["batch_size"]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
        shuffle=False,
    )

    ckpt_path = model_config.get("checkpoint_path", "")

    if ckpt_path:
        model = DynaProt.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=model_config)
        print(f"[train] resumed weights from {ckpt_path}")
    else:
        model = DynaProt(model_config)

    logger = build_logger(args.logger, args, model_config, ckpt_path)
    print(model)

    trainer = pl.Trainer(
        max_epochs=model_config["train_params"]["epochs"],
        logger=logger if logger is not None else False,
        accelerator=model_config["train_params"]["accelerator"],
        strategy=model_config["train_params"]["strategy"],
        devices=model_config["train_params"]["num_devices"],
        num_nodes=model_config["train_params"].get("num_nodes", 1),
        precision=model_config["train_params"].get("precision", 32),
        callbacks=[
            ModelCheckpoint(
                dirpath=args.ckpt_dir,
                filename=args.name + "-{step:06d}",
                save_top_k=-1,
                every_n_train_steps=500,
            )
        ],
    )

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


if __name__ == "__main__":
    main()
