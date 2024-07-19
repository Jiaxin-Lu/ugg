import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', '..'))
sys.path.append(pjoin(base_dir, '..', '..', '..'))

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from model.modules.base_lightning import BaseLightningModel
from model.ugg.UViT.pn2_layer import TransitionDown, TransitionUp
from model.ugg.UViT.timm import DropPath, Mlp, trunc_normal_
from utils.hand_helper import ROT_DIM_DICT
from utils.utils import load_model

torch.set_float32_matmul_precision('high')

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        from xformers.ops import memory_efficient_attention
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            # with torch.autocast(device_type='cuda', enabled=False):
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim) if skip else None
        self.norm2 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
            x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))
        x = self.norm3(x)

        return x


class LatentDiscriminatorOriPC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg = cfg.MODEL.diffusion

        self.num_features = self.embed_dim = embed_dim = model_cfg.disc.embed_dim

        self.norm_layer = nn.LayerNorm

        self.hand_param_dim = hand_param_dim = model_cfg.disc.hand_param_dim

        self.pc_local_pts = model_cfg.disc.pc_local_pts
        self.pc_local_pts_dim = model_cfg.disc.pc_local_pts_dim

        assert self.pc_local_pts == self.cfg.DATA.PC_NUM_POINTS
        self.pc_local_token_pts = int(self.pc_local_pts * 0.25)
        self.pc_local_patch = TransitionDown(in_channels=self.pc_local_pts_dim, out_channels=embed_dim, k=8)
        self.pc_local_unpatch = TransitionUp(in_channels=embed_dim, out_channels=self.pc_local_pts_dim)

        self.hand_param_embed = nn.Linear(hand_param_dim, embed_dim)
        self.hand_param_out = nn.Linear(embed_dim, hand_param_dim)

        self._set_hand_rot()
        self.hand_rot_dim=6
        self.hand_R_embed = nn.Linear(self.hand_rot_dim, self.embed_dim)
        self.hand_R_out = nn.Linear(self.embed_dim, self.hand_rot_dim)

        self.hand_t_embed = nn.Linear(3, self.embed_dim)
        self.hand_t_out = nn.Linear(self.embed_dim, 3)

        self.num_tokens = self.pc_local_token_pts + 1 + 1 + 1  # t_pc, t_hand, pc_global, pc_local, hand_param, handR, handt

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=model_cfg.disc.pos_drop_rate)

        self.norm = self.norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.trans_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=model_cfg.disc.num_heads, 
                mlp_ratio=model_cfg.disc.mlp_ratio, qkv_bias=model_cfg.disc.qkv_bias, qk_scale=model_cfg.disc.qk_scale, 
                drop=model_cfg.disc.drop_rate, attn_drop=model_cfg.disc.attn_drop_rate, norm_layer=self.norm_layer,
                use_checkpoint=model_cfg.disc.use_checkpoint) 
            for _ in range(model_cfg.disc.depth)
        ])

        self.pc_token_mlp = nn.Linear(2 * embed_dim, embed_dim)
        self.hand_token_mlp = nn.Linear(2 * embed_dim, embed_dim)

        self._set_classifier()

    def _set_classifier(self):
        self.cls_mode = mode = self.model_cfg.disc.cls_mode
        if mode == "cat":
            self.classifier = nn.Sequential(
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, 1)
            )
        elif mode in ['add', 'avg']:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.BatchNorm1d(self.embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim // 2, 1)
            )
        else:
            raise NotImplementedError(f"{mode} not implemented")

    def classify(self, pc_feat, hand_feat):
        if self.cls_mode == 'cat':
            feat = torch.cat([pc_feat, hand_feat], dim=-1)
        elif self.cls_mode == 'add':
            feat = pc_feat + hand_feat
        elif self.cls_mode == 'avg':
            feat = (pc_feat + hand_feat) / 2.
        else:
            raise NotImplementedError(f"{self.cls_mode} not implemented")

        res = self.classifier(feat)
        return res
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def _set_hand_rot(self):
        try:
            self.hand_rot_type = self.cfg.DATA.ROT_TYPE
        except:
            self.hand_rot_type = 'ax'
        
        self.hand_rot_dim = ROT_DIM_DICT[self.hand_rot_type]

        self.hand_trans_slice = slice(0, 3)
        self.hand_rot_slice = slice(3, 3+self.hand_rot_dim)
        self.hand_param_slice = slice(3+self.hand_rot_dim, None)

        self.hand_encode_rot = False
    
    def forward(self, data_dict):
        pc_local = data_dict['object_pc']  # [B, N, 3]
        hand_param = data_dict['hand_param']  # [B, Dh]
        hand_R = data_dict['hand_R']  # [B, 4]
        hand_t = data_dict['hand_t']  # [B, 3]

        device = hand_param.device

        B, N, _ = pc_local.shape

        N = self.pc_local_pts
        pc_local_feat = pc_local.reshape(B * N, self.pc_local_pts_dim)
        pc_local_pos = pc_local_feat[:, :3]
        pc_local_batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=device), N)
        pc_local_token, sub_pc_local_pos, sub_pc_local_batch = self.pc_local_patch(pc_local_feat, pc_local_pos, pc_local_batch)

        pc_local_token = pc_local_token.reshape(B, -1, self.embed_dim)  # [B, N_sub, embed_dim]

        hand_param_token = self.hand_param_embed(hand_param)  # [B, embed_dim]
        hand_param_token = hand_param_token.unsqueeze(1)

        hand_R_token = self.hand_R_embed(hand_R)
        hand_R_token = hand_R_token.unsqueeze(1)

        hand_t_token = self.hand_t_embed(hand_t)
        hand_t_token = hand_t_token.unsqueeze(1)

        x = torch.cat([pc_local_token, hand_param_token, hand_R_token, hand_t_token], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.trans_blocks:
            x = blk(x)

        x = self.norm(x)

        pc_token, hand_token = x.split((self.pc_local_token_pts, 3), dim=1)

        pc_token_avg = F.adaptive_avg_pool1d(pc_token.transpose(1, 2), 1).view(B, -1)
        pc_token_max = F.adaptive_max_pool1d(pc_token.transpose(1, 2), 1).view(B, -1)
        pc_token = torch.cat([pc_token_avg, pc_token_max], dim=1)
        pc_token = self.pc_token_mlp(pc_token)

        hand_token_avg = F.adaptive_avg_pool1d(hand_token.transpose(1, 2), 1).view(B, -1)
        hand_token_max = F.adaptive_max_pool1d(hand_token.transpose(1, 2), 1).view(B, -1)
        hand_token = torch.cat([hand_token_avg, hand_token_max], dim=1)
        hand_token = self.hand_token_mlp(hand_token)

        pred = self.classify(pc_feat=pc_token, hand_feat=hand_token)

        out_dict = {
            'pred': pred
        }
        return out_dict


class DiscriminatorTrainer(BaseLightningModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_cfg = cfg.MODEL
        self.diffusion_cfg = cfg.MODEL.diffusion
        if self.diffusion_cfg.disc.type == 'ori_pc':
            self.discriminator = LatentDiscriminatorOriPC(cfg)
        else:
            raise NotImplementedError(f"{self.diffusion_cfg.disc.type} not implemented")

    def forward(self, data_dict):
        if self.diffusion_cfg.disc.type == 'ori_pc':
            assert 'object_pc' in data_dict
        
        out_dict = self.discriminator(data_dict)
        return out_dict

    def loss_function(self, data_dict, out_dict, optimizer_idx=-1):
        gt = data_dict['valid'].reshape(-1, 1)
        pred = out_dict['pred']
        B = pred.shape[0]
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean')
        cls_acc = torchmetrics.functional.accuracy(pred, gt, task='binary')
        cls_precision = torchmetrics.functional.precision(pred, gt, task='binary')
        cls_recall = torchmetrics.functional.recall(pred, gt, task='binary')
        cls_f1_score = torchmetrics.functional.f1_score(pred, gt, task='binary')

        loss_dict = {
            'loss': loss,
            'acc': cls_acc,
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1_score,
            'batch_size': B
        }
        return loss_dict
    
        
if __name__ == '__main__':
    from datetime import datetime

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (LearningRateMonitor,
                                             ModelCheckpoint)
    from pytorch_lightning.loggers import WandbLogger

    from dataset import build_dataloader
    from model.ugg.callback import CustomDDPStrategy
    from utils.config import cfg
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') 
    
    args = parse_args("Diffusion")

    pl.seed_everything(cfg.RANDOM_SEED)

    file_suffix = NOW_TIME
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    full_log_name = f"train_log_{file_suffix}"
    
    with DupStdoutFileManager(os.path.join(cfg.OUTPUT_PATH, f"{full_log_name}.log")) as _:

        print_easydict(cfg)

        train_loader, val_loader = build_dataloader(cfg)

        model_save_path = cfg.MODEL_SAVE_PATH
        output_path = cfg.OUTPUT_PATH

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_save_path,
            filename="model_{epoch:03d}",
            save_top_k=-1,
            monitor=cfg.CALLBACK.CHECKPOINT_MONITOR,
            mode=cfg.CALLBACK.CHECKPOINT_MODE,
            save_last=True,
            # save_on_train_epoch_end=True,
        )
        callbacks = [
            LearningRateMonitor("epoch"),
            checkpoint_callback,
        ]
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
        
        all_gpus = list(cfg.GPUS)

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
            precision=cfg.TRAIN.PRECISION,
        )
        if len(all_gpus) > 1:
            training_log_dict.update({
                "strategy": CustomDDPStrategy()
            })
        trainer = pl.Trainer(**training_log_dict)
        model = DiscriminatorTrainer(cfg)

        ckp_path = load_model(model, cfg.WEIGHT_FILE, model_save_path)
        
        print("finish setting")
        
        trainer.fit(model, train_loader, val_loader, ckp_path=ckp_path)
        
        print("Done train.")

    