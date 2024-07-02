import math
import torch
import torch.nn as nn

from utils.hand_helper import ROT_DIM_DICT
from .timm import trunc_normal_, DropPath, Mlp
from .pn2_layer import TransitionDown, TransitionUp
import einops
import torch.nn.functional as F

from loguru import logger

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
    logger.warning("Provided checkpoint only supports xformers.")
else:
    try:
        from xformers.ops import memory_efficient_attention
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
        logger.warning("Provided checkpoint only supports xformers.")
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


class UViTContact(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg = cfg.MODEL.diffusion

        self.num_features = self.embed_dim = embed_dim = model_cfg.embed_dim

        self.norm_layer = nn.LayerNorm

        self.time_hand_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if model_cfg.mlp_time_embed else nn.Identity()

        self.time_pc_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if model_cfg.mlp_time_embed else nn.Identity()

        self.pc_global_dim = pc_global_dim = model_cfg.pc_global_dim
        self.hand_param_dim = hand_param_dim = model_cfg.hand_param_dim

        self.pc_global_embed = nn.Linear(pc_global_dim, embed_dim)
        self.pc_global_out = nn.Linear(embed_dim, pc_global_dim)

        self.pc_local_pts = model_cfg.pc_local_pts
        self.pc_local_pts_dim = model_cfg.pc_local_pts_dim

        self.gen_contact = model_cfg.gen_contact
        if self.gen_contact:
            self.contact_num = 5
            self.time_contact_embed = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            ) if model_cfg.mlp_time_embed else nn.Identity()
            self.contact_token_pts = self.contact_num
            self.contact_embed = nn.Linear(3, embed_dim)
            self.contact_out = nn.Linear(embed_dim, 3)
        else:
            self.time_contact_embed = None

        assert self.pc_local_pts == self.cfg.DATA.PC_NUM_POINTS

        self.pc_local_token_pts = int(self.pc_local_pts * 0.25)
        self.pc_local_patch = TransitionDown(in_channels=self.pc_local_pts_dim, out_channels=embed_dim, k=8)
        self.pc_local_unpatch = TransitionUp(in_channels=embed_dim, out_channels=self.pc_local_pts_dim)

        self.hand_param_embed = nn.Linear(hand_param_dim, embed_dim)
        self.hand_param_out = nn.Linear(embed_dim, hand_param_dim)

        self._set_hand_rot()
        self.hand_R_embed = nn.Linear(self.hand_rot_dim, self.embed_dim)
        self.hand_R_out = nn.Linear(self.embed_dim, self.hand_rot_dim)

        self.hand_t_embed = nn.Linear(3, self.embed_dim)
        self.hand_t_out = nn.Linear(self.embed_dim, 3)

        self.num_tokens = 2 + 1 + self.pc_local_token_pts + 1 + 1 + 1  # t_pc, t_hand, pc_global, pc_local, hand_param, handR, handt

        if self.gen_contact:
            self.num_tokens += 1 + self.contact_token_pts # add t_contact

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=model_cfg.pos_drop_rate)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=model_cfg.num_heads, 
                mlp_ratio=model_cfg.mlp_ratio, qkv_bias=model_cfg.qkv_bias, qk_scale=model_cfg.qk_scale,
                drop=model_cfg.drop_rate, attn_drop=model_cfg.attn_drop_rate, norm_layer=self.norm_layer, use_checkpoint=model_cfg.use_checkpoint)
            for _ in range(model_cfg.depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=model_cfg.num_heads, 
                mlp_ratio=model_cfg.mlp_ratio, qkv_bias=model_cfg.qkv_bias, qk_scale=model_cfg.qk_scale,
                drop=model_cfg.drop_rate, attn_drop=model_cfg.attn_drop_rate, norm_layer=self.norm_layer, use_checkpoint=model_cfg.use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=model_cfg.num_heads, 
                mlp_ratio=model_cfg.mlp_ratio, qkv_bias=model_cfg.qkv_bias, qk_scale=model_cfg.qk_scale,
                drop=model_cfg.drop_rate, attn_drop=model_cfg.attn_drop_rate, norm_layer=self.norm_layer, skip=True, use_checkpoint=model_cfg.use_checkpoint)
            for _ in range(model_cfg.depth // 2)])
        
        self.norm = self.norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

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
    
    def forward(self, data_dict):
        pc_global = data_dict['pc_global']  # [B, D1]
        pc_local = data_dict['pc_local']  # [B, ND2]
        hand_param = data_dict['hand_param']  # [B, Dh]
        hand_R = data_dict['hand_R']  # [B, 4]
        hand_t = data_dict['hand_t']  # [B, 3]

        t_pc = data_dict['t_pc']
        t_hand = data_dict['t_hand']

        t_pc_token = self.time_pc_embed(timestep_embedding(t_pc, self.embed_dim))
        t_pc_token = t_pc_token.unsqueeze(1)  # [B, 1, embed_dim]
        t_hand_token = self.time_hand_embed(timestep_embedding(t_hand, self.embed_dim))
        t_hand_token = t_hand_token.unsqueeze(1)  # [B, 1, embed_dim]

        B, ND2 = pc_local.shape
        N = self.pc_local_pts
        device = pc_global.device

        if self.gen_contact:
            t_contact = data_dict['t_contact']
            t_contact_token = self.time_contact_embed(timestep_embedding(t_contact, self.embed_dim))
            t_contact_token = t_contact_token.unsqueeze(1)
            contact_map = data_dict['contact_map'].reshape(B, self.contact_num, 3)
            contact_token = self.contact_embed(contact_map)

        pc_global_token = self.pc_global_embed(pc_global)  # [B, embed_dim]
        pc_global_token = pc_global_token.unsqueeze(1)  # [B, 1, embed_dim]
        pc_local_feat = pc_local.reshape(B * N, self.pc_local_pts_dim)
        pc_local_pos = pc_local_feat[:, :3]
        pc_local_batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=device), N)
        pc_local_token, sub_pc_local_pos, sub_pc_local_batch = self.pc_local_patch(pc_local_feat, pc_local_pos, pc_local_batch)

        pc_local_token = pc_local_token.reshape(B, -1, self.embed_dim)  # [B, N_sub, embed_dim]
        N_sub = pc_local_token.shape[1]

        hand_param_token = self.hand_param_embed(hand_param)  # [B, embed_dim]
        hand_param_token = hand_param_token.unsqueeze(1)

        hand_R_token = self.hand_R_embed(hand_R)
        hand_R_token = hand_R_token.unsqueeze(1)

        hand_t_token = self.hand_t_embed(hand_t)
        hand_t_token = hand_t_token.unsqueeze(1)

        if self.gen_contact:
            x = torch.cat((t_pc_token, t_hand_token, t_contact_token, pc_global_token, pc_local_token, hand_param_token, hand_R_token, hand_t_token, contact_token), dim=1)
        else:
            x = torch.cat((t_pc_token, t_hand_token, pc_global_token, pc_local_token, hand_param_token, hand_R_token, hand_t_token), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skip=skips.pop())

        x = self.norm(x)

        if self.gen_contact:
            t_pc_token_out, t_hand_token_out, t_contact_token_out, pc_global_out, pc_local_out, hand_param_out, hand_R_out, hand_t_out, contact_out = x.split((1, 1, 1, 1, self.pc_local_token_pts, 1, 1, 1, self.contact_token_pts), dim=1)
        else:
            t_pc_token_out, t_hand_token_out, pc_global_out, pc_local_out, hand_param_out, hand_R_out, hand_t_out = x.split((1, 1, 1, self.pc_local_token_pts, 1, 1, 1), dim=1)

        pc_global_out = self.pc_global_out(pc_global_out).squeeze(1)
        pc_local_out = pc_local_out.reshape(B, N_sub, -1)
        pc_local_pos = pc_local_pos.reshape(B, N, -1)
        sub_pc_local_pos = sub_pc_local_pos.reshape(B, N_sub, -1)
        pc_local_out = self.pc_local_unpatch(None, pc_local_out, pc_local_pos, sub_pc_local_pos, pc_local_batch, sub_pc_local_batch)

        if self.gen_contact:
            contact_map_out = self.contact_out(contact_out).reshape(B, self.contact_num * 3)
        else:
            contact_map_out = None
        
        pc_local_out = pc_local_out.reshape(B, ND2)

        hand_param_out = self.hand_param_out(hand_param_out).squeeze(1)
        hand_R_out = self.hand_R_out(hand_R_out).squeeze(1)
        hand_t_out = self.hand_t_out(hand_t_out).squeeze(1)

        out_dict = {
            'pc_global_eps_pred': pc_global_out,
            'pc_local_eps_pred': pc_local_out,
            'hand_t_eps_pred': hand_t_out,
            'hand_R_eps_pred': hand_R_out,
            'hand_param_eps_pred': hand_param_out,
            'contact_map_eps_pred': contact_map_out,
        }

        return out_dict


