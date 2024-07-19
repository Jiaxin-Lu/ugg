from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

__C.UGG = edict()

# hand_vae
__C.UGG.hand_vae = edict()
__C.UGG.hand_vae.encoder_dims = [22, 32, 64]
__C.UGG.hand_vae.decoder_dims = [64, 32, 22]

__C.UGG.hand_vae.kl_weight = 0.00001
__C.UGG.hand_vae.loss = 'mse'

# generation
__C.UGG.diffusion = edict()
__C.UGG.diffusion.pc_global_dim = 128  # lion: latent_pts.style_dimpc_checkpoint
__C.UGG.diffusion.hand_param_dim = 64
__C.UGG.diffusion.pc_local_pts = 2048
__C.UGG.diffusion.pc_local_pts_dim = 4

__C.UGG.diffusion.n_timestep = 200
__C.UGG.diffusion.beta_start = 0.0001
__C.UGG.diffusion.beta_end = 0.02
__C.UGG.diffusion.beta_schedule = 'quad'

# hand
__C.UGG.diffusion.gen_hand = True
__C.UGG.diffusion.encode_hand = True
__C.UGG.diffusion.loss_hand = 'l2'
__C.UGG.diffusion.hand_checkpoint = './checkpoints/hand_vae/hand_vae_bs1024_lr004_kl00001_22_32_64_mse_step/model_epoch=009.ckpt'

# pc
__C.UGG.diffusion.gen_pc = True
__C.UGG.diffusion.loss_pc = 'l2'
__C.UGG.diffusion.pc_checkpoint = './checkpoints/lion/aeb159h_hvae_lion_B32/checkpoints/epoch_5999_iters_1667999.pt'
__C.UGG.diffusion.pc_args = './checkpoints/lion/aeb159h_hvae_lion_B32/cfg.yml'

# contact
__C.UGG.diffusion.gen_contact = True
__C.UGG.diffusion.loss_contact = 'l2'
__C.UGG.diffusion.contact_map_normalize_factor = 200.


# UViT
__C.UGG.diffusion.embed_dim = 512
__C.UGG.diffusion.pos_drop_rate = 0.
__C.UGG.diffusion.num_heads = 8
__C.UGG.diffusion.mlp_ratio = 4
__C.UGG.diffusion.qkv_bias = False
__C.UGG.diffusion.qk_scale = None
__C.UGG.diffusion.drop_rate = 0.
__C.UGG.diffusion.attn_drop_rate = 0.
__C.UGG.diffusion.use_checkpoint = True
__C.UGG.diffusion.depth = 12
__C.UGG.diffusion.mlp_time_embed = False

# test
__C.UGG.test = edict()
__C.UGG.test.thres_contact = 0.005
__C.UGG.test.n_cpu = 10
__C.UGG.test.m = 8
__C.UGG.test.mu = 1.
__C.UGG.test.lambda_torque = 10
__C.UGG.test.max_contact = 20
__C.UGG.test.nms = True
__C.UGG.test.thres_pen = 0.005

__C.UGG.test.w_contact = 10.0
__C.UGG.test.w_spen = 1.0
__C.UGG.test.w_pen = 100.0
__C.UGG.test.w_joints = 0.1
__C.UGG.test.opt_lr = 0.001
__C.UGG.test.opt_iter = 100

__C.UGG.diffusion.task = ['joint', 'obj2hand', 'hand2obj']

# discriminator
__C.UGG.diffusion.with_discriminator = False
__C.UGG.diffusion.disc = edict()
__C.UGG.diffusion.disc.type = 'ori_pc'

__C.UGG.diffusion.disc.embed_dim = 128
__C.UGG.diffusion.disc.num_heads = 4
__C.UGG.diffusion.disc.mlp_ratio = 2
__C.UGG.diffusion.disc.pos_drop_rate = 0.
__C.UGG.diffusion.disc.hand_param_dim = 22
__C.UGG.diffusion.disc.pc_global_dim = 128
__C.UGG.diffusion.disc.pc_local_pts = 2048
__C.UGG.diffusion.disc.pc_local_pts_dim = 4

__C.UGG.diffusion.disc.qkv_bias = False
__C.UGG.diffusion.disc.qk_scale = None
__C.UGG.diffusion.disc.drop_rate = 0.
__C.UGG.diffusion.disc.attn_drop_rate = 0.
__C.UGG.diffusion.disc.use_checkpoint = False
__C.UGG.diffusion.disc.depth = 2
__C.UGG.diffusion.disc.mlp_time_embed = False
__C.UGG.diffusion.disc.cls_mode = 'cat'
__C.UGG.diffusion.disc.decoded_hand = False
__C.UGG.diffusion.disc.discriminator_checkpoint = ''


def get_model_cfg():
    return model_cfg.UGG
