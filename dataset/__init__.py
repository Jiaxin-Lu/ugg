from .dataset_config import dataset_cfg


def build_dataloader(cfg, **kwargs):
    if cfg.DATASET.lower() == "dexgraspnet":
        from .dexgraspnet_dataset import build_dexgraspnet_dataloader
        return build_dexgraspnet_dataloader(cfg)
    elif cfg.DATASET.lower() == 'discriminator':
        from .discriminator_dataset import build_discriminator_dataset
        return build_discriminator_dataset(cfg)
    elif cfg.DATASET.lower() == 'dexgraspobject':
        from .dexgraspobject_dataset import build_dexgraspobject_dataloader
        return build_dexgraspobject_dataloader(cfg)
    elif cfg.DATASET.lower() == 'h2o':
        from .h2o_dataset import build_h2o_dataloader
        return build_h2o_dataloader(cfg, **kwargs)
    else:
        raise NotImplementedError(f"Dataloader {cfg.DATASET.lower()} not implemented")
    
