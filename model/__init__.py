from .modules import *

def build_model(cfg, ckpt=None):
    module_list = cfg.MODULE.lower().split('.')
    if ckpt is None:
        if module_list[0] == 'ugg':
            if module_list[1] == 'hand_vae':
                from .ugg.hand_vae import HandTrainer
                return HandTrainer(cfg)
            elif module_list[1] == 'generation':
                from .ugg.generation_contact_trainer import UGGGenerationTrainer
                return UGGGenerationTrainer(cfg)
    else:
        if module_list[0] == 'ugg':
            if module_list[1] == 'hand_vae':
                from .ugg.hand_vae import HandTrainer
                return HandTrainer.load_from_checkpoint(ckpt, cfg=cfg)
            elif module_list[1] == 'generation':
                from .ugg.generation_contact_trainer import UGGGenerationTrainer
                return UGGGenerationTrainer.load_from_checkpoint(ckpt, cfg=cfg)
    raise NotImplementedError(f"Module {cfg.MODULE} not implemented")
        