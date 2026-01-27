import torch
import torch.nn as nn

from pathlib import Path
from typing import Union, List, Tuple, Optional

def load_net(
    net: nn.Module,
    checkpoints_file: Union[str, Path],
    device: torch.device
    ):
    
    if checkpoints_file is None:
        epoch = 0
        optim_dict = None
        sched_dict = None
    else:
        print('Restoring checkpoint: ', checkpoints_file)
        checkpoint_dict = torch.load(checkpoints_file, map_location=device)
        # Remove "module." from DataParallel, if present.
        checkpoint_dict['state_dict'] = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in
                                            checkpoint_dict['state_dict'].items()}
        try:
            load_result = net.load_state_dict(checkpoint_dict['state_dict'], strict=False)
            if load_result.missing_keys:
                print(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys: {load_result.unexpected_keys}")
        except RuntimeError as e:
            print(f"State dict loading issues:\n{e}")

        epoch = checkpoint_dict.get('epoch', 0)
        optim_dict = checkpoint_dict.get('optimizer', None)
        sched_dict = checkpoint_dict.get('scheduler_state_dict', None)
        
    net = net.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    return net, epoch, optim_dict, sched_dict

def update_optimizer(
        net: nn.Module,
        optim,
        decay,
        lr
    ):

    if optim == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=decay)

    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=decay)

    elif optim == "RMSProp":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=decay)
        
    else:
        raise NotImplementedError(f"Optimizer: {optim} is not valid.")

    return optimizer

class ModelUtilizer(object):
    """Module utility class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.

    """
    def __init__(self, configer):
        """Class constructor for Module utility"""
        self.configer = configer
        self.device = torch.device(self.configer.device)
        print(f"Device (model_utilizer.py): {self.device}")
        self.output_file_name = self.configer.output_file_name
        self.save_policy = self.configer.model_config.get("checkpoints_save_policy")
        if self.save_policy == "all":
            self.save = self.save_all
        elif self.save_policy == "best":
            if self.configer.model_config.get("early_stop_number") > 0:
                self.save = self.early_stop
            else:
                self.save = self.save_best
        else:
            raise ValueError(f'Policy "{self.save_policy}" is unknown.')

        self.best_metric = self.configer.model_config.get("checkpoints_metric")
        self.best_metric_value = 0
        self.last_improvement_cnt = 0

    def _save_net(self, net, optimizer, epoch, scheduler=None):
        """Saving net state method.

            Args:
                net (torch.nn.Module): Module in use
                optimizer (torch.nn.optimizer): Optimizer state to save
                epoch (int): Current epoch number to save
        """
        
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }
        
        checkpoints_dir = Path(self.configer.general_config.get('checkpoints_dir'))
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if self.save_policy == "all":
            latest_name = '{}_epoch_{}.pth'.format(self.output_file_name, epoch)
        elif self.save_policy == "best":
            latest_name = 'best_{}.pth'.format(self.output_file_name)
        else:
            raise ValueError(f'Policy {self.save_policy} is unknown.')
   
        torch.save(state, checkpoints_dir / latest_name)

    def save_all(self, metric_value, net, optimizer, epoch, scheduler=None):
        self._save_net(net, optimizer, epoch, scheduler)
        return metric_value

    def save_best(self, metric_value, net, optimizer, epoch, scheduler=None):
        if metric_value > self.best_metric_value:
            self.best_metric_value = metric_value
            self._save_net(net, optimizer, epoch, scheduler)
            return self.best_metric_value
        else:
            return 0

    def early_stop(self, metric_value, net, optimizer, epoch, scheduler=None):
        ret = self.save_best(metric_value, net, optimizer, epoch, scheduler)
        if ret > 0:
            self.last_improvement_cnt = 0
        else:
            self.last_improvement_cnt += 1
        if self.last_improvement_cnt >= self.configer.model_config.get("early_stop_number"):
            return -1
        else:
            return ret