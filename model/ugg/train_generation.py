import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', '..'))
sys.path.append(pjoin(base_dir, '../../LION/'))

import time
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import build_model
from model.ugg.callback import CustomDDPStrategy, DiffusionVisCallback
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.utils import load_model

NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


torch.set_float32_matmul_precision('medium')

def train_model(cfg):
    # initialize dataloader
    train_loader, val_loader = build_dataloader(cfg)
    
    # initialize model
    model = build_model(cfg)

    # device
    if len(cfg.GPUS) > 1:
        print("multiple gpus, use ddp training")

    # The result folder is cfg.OUTPUT_PATH
    # The model save folder is cfg.MODEL_SAVE_PATH
    model_save_path = cfg.MODEL_SAVE_PATH
    output_path = cfg.OUTPUT_PATH

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="model_{epoch:03d}",
        save_top_k=-1,
        monitor=cfg.CALLBACK.CHECKPOINT_MONITOR,
        mode=cfg.CALLBACK.CHECKPOINT_MODE,
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=cfg.TRAIN.VAL_EVERY,
    )
    callbacks = [
        LearningRateMonitor("epoch"),
        checkpoint_callback,
    ]
    
    visualize_callback = DiffusionVisCallback(
        cfg=cfg,
        sample_num=cfg.TRAIN.VAL_SAMPLE_VIS,
        train_loader=train_loader,
        test_loader=val_loader,
        task=cfg.MODEL.diffusion.task
    )
    callbacks.append(visualize_callback)

    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        logger_suffix = cfg.LOG_FILE_NAME
    else:
        logger_suffix = NOW_TIME
    logger_name = f"{cfg.MODEL_NAME}_{logger_suffix}"
    logger_id = None
    logger = WandbLogger(
        project=cfg.PROJECT,
        name=logger_name,
        id=logger_id,
        save_dir=output_path
    )

    all_gpus = list(range(torch.cuda.device_count()))
    
    try:
        precision = int(cfg.TRAIN.PRECISION)
    except:
        precision = cfg.TRAIN.PRECISION
        
    training_log_dict = dict(
        logger=logger,
        accelerator='gpu',
        devices=all_gpus,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        callbacks=callbacks,
        benchmark=cfg.CUDNN,
        check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY,
        log_every_n_steps=5,
        profiler='simple',
        detect_anomaly=True,
        precision=precision,
    )
    if len(all_gpus) > 1:
        training_log_dict.update({
            "strategy": CustomDDPStrategy()
        })

    trainer = pl.Trainer(**training_log_dict)
    
    ckp_path = load_model(model, cfg.WEIGHT_FILE, model_save_path)

    print("finish setting -----")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckp_path)

    print("Done training.")
        
    

if __name__ == '__main__':
    from dataset import build_dataloader
    from utils.config import cfg
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args("Diffusion")

    pl.seed_everything(cfg.RANDOM_SEED)

    file_suffix = NOW_TIME
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    full_log_name = f"train_log_{file_suffix}"
    
    with DupStdoutFileManager(os.path.join(cfg.OUTPUT_PATH, f"{full_log_name}.log")) as _:

        print_easydict(cfg)

        train_model(cfg)
    
