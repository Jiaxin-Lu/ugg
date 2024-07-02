from .dataset_config import dataset_cfg


def build_dataloader(cfg):
    if cfg.DATASET.lower() == "dexgraspnet":
        from .dexgraspnet_dataset import build_dexgraspnet_dataloader
        return build_dexgraspnet_dataloader(cfg)
    else:
        raise NotImplementedError(f"Dataloader {cfg.DATASET.lower()} not implemented")
    
