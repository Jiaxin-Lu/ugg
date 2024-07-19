import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_model
from model.modules.base_lightning import BaseLightningModel
from utils.distribution import sample_normal
from utils.hand_helper import *
from utils.hand_model import HandModel
from utils.loss import kl_divergence


class HandEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_dims = self.cfg.MODEL.hand_vae.encoder_dims
        self.encoder_list = []
        for i in range(len(self.encoder_dims) - 2):
            self.encoder_list.extend([
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                nn.BatchNorm1d(self.encoder_dims[i+1]),
                nn.ReLU(inplace=True),
                ])
        self.encoder = nn.Sequential(*self.encoder_list)
        self.encoder_mu = nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1])
        self.encoder_logvar = nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1])

    def forward(self, data_dict):
        x = data_dict['x']
        B = x.shape[0]
        x = x.reshape(B, -1)
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        mu, logvar = mu.reshape(B, -1).contiguous(), logvar.reshape(B, -1).contiguous()
        return dict(mu=mu, logvar=logvar)

    def sample(self, mu, logvar):
        z = sample_normal(mu, logvar)[0]
        return z


class HandDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.decoder_dims = self.cfg.MODEL.hand_vae.decoder_dims
        self.decoder_list = []
        for i in range(len(self.decoder_dims) - 2):
            self.decoder_list.extend([
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
                nn.BatchNorm1d(self.decoder_dims[i+1]),
                nn.ReLU(inplace=True),
            ])
        self.decoder_list.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
        self.decoder = nn.Sequential(*self.decoder_list)
        self.activation = nn.Tanh()

    def forward(self, data_dict):
        z = data_dict['z']
        B = z.shape[0]
        z = z.reshape(B, -1)
        h = self.decoder(z)
        x = self.activation(h)
        return x


class HandVAE(nn.Module):
    def __init__(self, cfg):
        super(HandVAE, self).__init__()
        self.cfg = cfg
        self.model_cfg = self.cfg.MODEL.hand_vae
        self.latent_dim = self.cfg.MODEL.hand_vae.encoder_dims[-1]
        assert self.latent_dim == self.cfg.MODEL.hand_vae.decoder_dims[0], "latent dim must match"
        self.encoder = HandEncoder(cfg)
        self.decoder = HandDecoder(cfg)

    def encode(self, data_dict):
        x = data_dict['x']
        B = x.shape[0]
        out_dict = dict()
        encoder_dict = self.encoder(dict(x=x))
        out_dict.update(encoder_dict)
        mu = out_dict['mu']
        logvar = out_dict['logvar']
        z = self.encoder.sample(mu, logvar)
        out_dict['z'] = z
        return out_dict
    
    def decode(self, data_dict):
        x_recon = self.decoder(dict(z=data_dict['z']))
        out_dict = {'x_recon': x_recon}
        return out_dict

    def forward(self, data_dict, eval=False):
        out_dict = dict()
        if eval:
            z = data_dict['z']
            B = z.shape[0]
        else:
            x = data_dict['x']
            B = x.shape[0]
            encoder_dict = self.encoder(dict(x=x))
            out_dict.update(encoder_dict)
            mu = out_dict['mu']
            logvar = out_dict['logvar']
            z = self.encoder.sample(mu, logvar)
        out_dict['z'] = z
        x_recon = self.decoder(dict(z=z))
        out_dict['x_recon'] = x_recon
        out_dict['batch_size'] = B
        return out_dict
    

class HandTrainer(BaseLightningModel):
    def __init__(self, cfg):
        super(HandTrainer, self).__init__(cfg)
        self.model_cfg = self.cfg.MODEL.hand_vae
        self.hand_model = HandVAE(cfg)
        self.kl_weight = self.model_cfg.kl_weight
        self.z = torch.randn(cfg.BATCH_SIZE, self.hand_model.latent_dim)
        try:
            self.hand_rot_type = self.cfg.DATA.ROT_TYPE
        except:
            self.hand_rot_type = 'ax'
        
        if self.hand_rot_type in ['ax', 'euler']:
            self.hand_rot_dim = 3
        elif self.hand_rot_type in ['quat']:
            self.hand_rot_dim = 4
        else:
            self.hand_rot_dim = 6
        
        self.hand_pose_mean = torch.tensor(HAND_POSE_MEAN)
        self.hand_pose_std = torch.tensor(HAND_POSE_STD)
        
        self.hand = None

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        hand_pose = batch['hand_pose']
        x = hand_pose[:, 3 + self.hand_rot_dim:]
        batch['x'] = x
        return batch
    
    def forward(self, data_dict):
        out_dict = self.hand_model(data_dict)
        return out_dict

    def eval_z(self, data_dict):
        data_dict['z'] = self.z.to(self.device)
        out_dict_z = self.hand_model(data_dict, eval=True)
        out_dict_z = {
                k+'_on_z': v for k, v in out_dict_z.items()
            }
        return out_dict_z
    
    def calc_spen(self, hand_recon):
        B = hand_recon.shape[0]
        trans = torch.zeros(B, 3, device=self.device)
        rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
        hand_pose = compose_hand_param(hand_t=trans, hand_r=rot, hand_param=hand_recon, rot_type='mat')
        if self.hand is None:
            self.hand = HandModel(
                mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
                mesh_path='hand_model_mjcf/meshes',
                contact_points_path='hand_model_mjcf/contact_points.json',
                penetration_points_path='hand_model_mjcf/penetration_points.json',
                device=self.device
            )
        self.hand.set_parameters(hand_pose)
        loss_spen = self.hand.self_penetration()
        return loss_spen.mean()

    def loss_function(self, data_dict, out_dict, optimizer_idx=-1):
        loss_dict = dict()
        if self.model_cfg.loss == 'mse':
            mse_loss = F.mse_loss(out_dict['x_recon'], data_dict['x'])
            loss = mse_loss
            loss_dict['mse_loss'] = mse_loss
        else:
            raise NotImplementedError("loss function not implemented")
            
        loss_spen = self.calc_spen(out_dict['x_recon'])
        loss += 0.01 * loss_spen
        loss_dict['spen_loss'] = loss_spen
        
        kl_div = kl_divergence(out_dict['mu'], out_dict['logvar'])
        loss = loss + self.kl_weight * kl_div
        loss_dict['kl_div'] = kl_div
        loss_dict['loss'] = loss
        return loss_dict
    

if __name__ == '__main__':
    import os
    from datetime import datetime

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (LearningRateMonitor,
                                             ModelCheckpoint)
    from pytorch_lightning.loggers import TensorBoardLogger

    from dataset import build_dataloader
    from utils.config import cfg
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    from utils.utils import load_model
    NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    args = parse_args("hand_vae")

    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    file_suffix = NOW_TIME

    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    full_log_name = f"train_log_{file_suffix}"
    
    model_save_path = cfg.MODEL_SAVE_PATH
    with DupStdoutFileManager(os.path.join(cfg.OUTPUT_PATH, f"{full_log_name}.log")) as _:
        print_easydict(cfg)
    
        model = build_model(cfg)
        
        ckp_path = load_model(model, cfg.WEIGHT_FILE, model_save_path)

        # device
        if len(cfg.GPUS) > 1:
            parallel_strategy = 'ddp'
            print("multiple gpus, use ddp training")

        train_loader, val_loader = build_dataloader(cfg)

        model_save_path = cfg.MODEL_SAVE_PATH
        output_path = cfg.OUTPUT_PATH

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_save_path,
            filename="model_{epoch:03d}",
            monitor=cfg.CALLBACK.CHECKPOINT_MONITOR,
            save_top_k=-1,
            mode=cfg.CALLBACK.CHECKPOINT_MODE,
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.TRAIN.VAL_EVERY,            
        )

        callbacks = [
            LearningRateMonitor("epoch"),
            checkpoint_callback,
        ]
        logger = TensorBoardLogger(
            save_dir=output_path, 
            name="tensorboard",
            )
        
        all_gpus = list(cfg.GPUS)

        training_log_dict = dict(
            logger=logger,
            accelerator='gpu',
            devices=all_gpus,
            max_epochs=cfg.TRAIN.NUM_EPOCHS,
            callbacks=callbacks,
            benchmark=cfg.CUDNN,
            check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY,
            log_every_n_steps=10,
            profiler='simple',
            detect_anomaly=True,
            precision=32
        )
        if len(all_gpus) > 1:
            training_log_dict.update({
                "strategy": parallel_strategy,
            })
        trainer = pl.Trainer(**training_log_dict)

        print("finish setting -----")
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckp_path)

        print("Done training.")

    
    
