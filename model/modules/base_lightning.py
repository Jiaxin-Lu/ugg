import pytorch_lightning
import torch
from torch import optim
from torch.optim import lr_scheduler

from utils.lr import CosineAnnealingWarmupRestarts

# torch.set_float32_matmul_precision('medium')

class BaseLightningModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.validation_loss_dict_list = []
        self.save_hyperparameters()

    def forward(self, data_dict):
        raise NotImplementedError
    
    def loss_function(self, data_dict, out_dict, optimizer_idx=-1):
        raise NotImplementedError 
    

    def training_step(self, data_dict, batch_idx, optimizer_idx=-1, *args, **kwargs):
        loss_dict = self.forward_pass(
            data_dict, mode='train', optimizer_idx=optimizer_idx
        )
        self.log('loss', loss_dict['loss'], prog_bar=True, rank_zero_only=True)
        return loss_dict['loss']

    def validation_step(self, data_dict, batch_idx, *args, **kwargs):
        loss_dict = self.forward_pass(
            data_dict, mode='val', optimizer_idx=-1
        )
        self.validation_loss_dict_list.append(loss_dict)
        return loss_dict

    def test_step(self, data_dict, batch_idx, *args, **kwargs):
        loss_dict = self.forward_pass(
            data_dict, mode='test', optimizer_idx=-1
        )
        self.validation_loss_dict_list.append(loss_dict)
        return loss_dict
    
    def on_validation_epoch_end(self):
        outputs = self.validation_loss_dict_list
        func = torch.tensor if \
            isinstance(outputs[0]['batch_size'], int) else torch.stack
        # print('validation.outputs[0]', outputs[0])
        batch_sizes = func([output.pop('batch_size') for output in outputs
                            ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'val/{k}': torch.stack([output[k] for output in outputs]).reshape(-1)
            for k in outputs[0].keys()
        }  # each is [1], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        self.log_dict(avg_loss, sync_dist=True)
        self.validation_loss_dict_list.clear()

    def on_test_epoch_end(self):
        outputs = self.validation_loss_dict_list
        # avg_loss among all data
        # we need to consider different batch_size
        if isinstance(outputs[0]['batch_size'], int):
            func_bs = torch.tensor
            func_loss = torch.stack
        else:
            func_bs = torch.cat
            func_loss = torch.cat
        batch_sizes = func_bs([output.pop('batch_size') for output in outputs
                               ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'test/{k}': func_loss([output[k] for output in outputs]).reshape(-1)
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        print(';\n'.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))
        # this is a hack to get results outside `Trainer.test()` function
        self.test_results = avg_loss
        self.log_dict(avg_loss, sync_dist=True)

    def forward_pass(self, data_dict, mode, optimizer_idx=-1):
        out_dict = self.forward(data_dict)
        loss_dict = self.loss_function(data_dict, out_dict, optimizer_idx)
        total_loss = loss_dict['loss']
        if total_loss.numel() != 1:
            loss_dict.update({
                k: v.mean() if ('loss' in k.lower()) and (v.numel() != 1) else v
                for k, v in loss_dict.items()
            })
            loss_dict['loss'] = total_loss.mean()

        # val/test will need batch information
        if not self.training:
            if 'batch_size' not in loss_dict:
                loss_dict['batch_size'] = out_dict['batch_size']

        # log every step in train
        if mode == 'train' and self.local_rank == 0:
            log_dict = {f"{mode}/{k}": v
                        for k, v in loss_dict.items()}
            data_name = [
                k for k in self.trainer.profiler.recorded_durations.keys()
                if 'prepare_data' in k
            ][0]
            log_dict[f'{mode}/data_time'] = \
                self.trainer.profiler.recorded_durations[data_name][-1]
            self.log_dict(log_dict, logger=True, sync_dist=False, rank_zero_only=True)

        return loss_dict
    
    def configure_optimizers(self):
        lr = self.cfg.TRAIN.LR
        optim_name = self.cfg.TRAIN.OPTIMIZER
        wd = self.cfg.TRAIN.WEIGHT_DECAY
        non_frozen_parameters = [p for p in self.parameters() if p.requires_grad]
        # non_frozen_name = [name for name, param in self.named_parameters() if param.requires_grad]
        if optim_name.lower() == 'adam':
            optimizer = optim.Adam([dict(params=non_frozen_parameters, initial_lr=lr)],
                                        lr=lr, 
                                        weight_decay=0.)
        elif optim_name.lower() == 'sgd':
            optimizer = optim.SGD(non_frozen_parameters,
                                       lr=lr,
                                       momentum=0.98,
                                       weight_decay=wd)

        if len(self.cfg.TRAIN.LR_SCHEDULER):
            if self.cfg.TRAIN.LR_SCHEDULER.lower() == 'multistep':
                scheduler = lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self.cfg.TRAIN.SCHEDULER.milestones,
                    gamma=self.cfg.TRAIN.SCHEDULER.gamma,
                    last_epoch=self.cfg.TRAIN.START_EPOCH)
            elif self.cfg.TRAIN.LR_SCHEDULER.lower() == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.TRAIN.NUM_EPOCHS,
                    eta_min=lr / self.cfg.TRAIN.SCHEDULER.lr_decay_factor)
            elif self.cfg.TRAIN.LR_SCHEDULER.lower() == 'cosine_warmup':
                scheduler = CosineAnnealingWarmupRestarts(
                    optimizer,
                    first_cycle_steps=self.cfg.TRAIN.NUM_EPOCHS+100,
                    max_lr=lr,
                    min_lr=lr / self.cfg.TRAIN.SCHEDULER.lr_decay_factor,
                    warmup_steps=0)
            elif self.cfg.TRAIN.LR_SCHEDULER.lower() == 'exponential':
                scheduler = lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=self.cfg.TRAIN.SCHEDULER.gamma)
            else:
                raise NotImplementedError(f"scheduler {self.cfg.TRAIN.LR_SCHEDULER} not implemented")
            return (
                [optimizer],
                [{
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }],
            )
        return optimizer
    