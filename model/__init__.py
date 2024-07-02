from .modules import *

def build_model(cfg):
    module_list = cfg.MODULE.lower().split('.')
    if module_list[0] == 'ugg':
        if module_list[1] == 'hand_vae':
            from .ugg.hand_vae import HandTrainer
            return HandTrainer(cfg)
        elif module_list[1] == 'generation':
            from .ugg.generation_contact_trainer import UGGGenerationTrainer
            return UGGGenerationTrainer(cfg)
        else:
            raise NotImplementedError(f"Module {cfg.MODULE} not implemented")
    else:
        raise NotImplementedError(f"Module {cfg.MODULE} not implemented")
        