import os.path
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
import sys

import numpy as np
import random
import torch
import os
import pytorch_lightning as pl
import timm
import wandb

from dataset import LitDataModule
from model import LitModule


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def train(cfg, debug, full, device):
    seed_everything(cfg['seed'])
    datamodule = LitDataModule(**cfg, full=full)
    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())
    cfg['len_train_dl'] = len_train_dl
    module = eval(cfg['model_type'])(**cfg)
    model_checkpoint = ModelCheckpoint(
        cfg['output_dir'],
        filename=f"model",
        # monitor="val_loss",
        save_last=True,
    )

    cfg['max_epochs'] = 2 if debug else cfg['max_epochs']
    cfg['limit_train_batches'] = 0.05 if debug else 1.0
    cfg['limit_val_batches'] = 0.1 if debug else 1.0
    print('-----------------------------------------------------------')
    print(f"Init WANDB project={cfg['project']}, name={cfg['name']}")
    print('-----------------------------------------------------------')
    loggers = [WandbLogger(project=cfg['project'], name=cfg['name'])]
    trainer = Trainer(
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
        auto_lr_find=cfg['auto_lr_find'],
        auto_scale_batch_size=cfg['auto_scale_batch_size'],
        benchmark=True,
        callbacks=[model_checkpoint],
        deterministic=True,
        fast_dev_run=cfg['fast_dev_run'],
        gpus=device,
        max_epochs=cfg['max_epochs'],
        precision=cfg['precision'],
        stochastic_weight_avg=cfg['stochastic_weight_avg'],
        limit_train_batches=cfg['limit_train_batches'],
        limit_val_batches=0.5 if full else cfg['limit_val_batches'],
        logger=loggers
    )
    with open(os.path.join(cfg['output_dir'], "cfg.yml"), 'w') as f:
        yaml.safe_dump(cfg, f)
    trainer.tune(module, datamodule=datamodule)
    trainer.fit(module, datamodule=datamodule)
    print(f"Finish fit model")
    print(f"Save model and config file in {cfg['output_dir']}")


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c")
    parser.add_argument("-device", "-d", default=0, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--full", default=False, action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    # pred_cfg = cfg.pop('predictor')

    print("--------------------- config ----------------------------")
    print(cfg)
    print("--------------------- ------ ----------------------------")

    if args.full:
        exp_name = f"full_{cfg['data']}_{cfg['model_name']}_ep{cfg['max_epochs']}_img{cfg['image_size']}"
    else:
        exp_name = f"{cfg['data']}_{cfg['model_name']}_ep{cfg['max_epochs']}_img{cfg['image_size']}"
    if args.debug:
        exp_name = "debug_" + exp_name

    if cfg['bnneck']:
        print("USE BNNECK")
        exp_name += "_bnneck"

    cfg['name'] = exp_name
    exp_dir = Path(cfg['output_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(exp_dir)

    if args.debug:
        print("-------------------- DEBUG MODE ------------------------")
    print(f"device {args.device}")
    print(cfg["project"], cfg['name'])
    train(cfg, debug=args.debug, full=args.full, device=[args.device])


# python ./fg_identify/train.py -c /home/kky/project/BoolArtProject/fg_identify/config.yml -d 0
