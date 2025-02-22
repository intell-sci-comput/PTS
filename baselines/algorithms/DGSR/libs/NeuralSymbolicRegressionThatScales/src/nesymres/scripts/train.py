import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from dataclasses import dataclass
from typing import Tuple
from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.utils import load_metadata_hdf5
import wandb
from dataclass_dict_convert import dataclass_dict_convert
from pytorch_lightning.loggers import WandbLogger
import hydra
from pathlib import Path


@hydra.main(config_name="config")
def main(cfg):
    seed_everything(9)
    train_path = Path(hydra.utils.to_absolute_path(cfg.train_path))
    val_path = Path(hydra.utils.to_absolute_path(cfg.val_path))
    data = DataModule(
        train_path,
        val_path,
        None,
        cfg
    )
    model = Model(cfg=cfg.architecture)
    if cfg.wandb:
        wandb.init(project="ICML")
        config = wandb.config
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # /dataloader_idx_0",
        dirpath="Exp_weights/",
        filename=train_path.stem + "_log_" + "-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    trainer = pl.Trainer(
        distributed_backend="ddp",
        gpus=cfg.gpu,
        max_epochs=cfg.epochs,
        # val_check_interval=cfg.val_check_interval,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        #      resume_from_checkpoint='/local/home/lbiggio/NeuralSymbolicRegressionThatScales/run/True/2021-05-27/12-18-49/Exp_weights/100000000_log_-epoch=00-val_loss=0.81.ckpt'
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    # ,1,2,3,4,5,6,7"  # ,1,2,4,5,6,7" Change Me
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
