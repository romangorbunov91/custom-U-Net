import os
import torch
import torch.nn as nn

from pathlib import Path

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
        self.save_policy = self.configer.get("checkpoints", "save_policy")
        if self.save_policy == "all":
            self.save = self.save_all
        elif self.save_policy == "best":
            if self.configer.get("checkpoints", "early_stop_number") > 0:
                self.save = self.early_stop
            else:
                self.save = self.save_best
        else:
            raise ValueError(f'Policy "{self.save_policy}" is unknown.')

        self.best_accuracy = 0
        self.last_improvement = 0

    def update_optimizer(self, net):
        """Load optimizer and adjust learning rate during training.

            Args:
                net (torch.nn.Module): Module in use.

            Returns:
                optimizer (torch.optim.optimizer): PyTorch Optimizer.
                lr (float): Learning rate for training procedure.

        """
        optim = self.configer.get('solver', 'type')
        decay = self.configer.get('solver', 'weight_decay')
        lr = self.configer.get('solver', 'base_lr')
        
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

        return optimizer, lr

    def load_net(self, net):
        """Loading net method. If resume is True load from provided checkpoint, if False load new DataParallel

            Args:
                net (torch.nn.Module): Module in use.

            Returns:
                net (torch.nn.DataParallel): Loaded Network module.
                epoch (int): Loaded current epoch number, 0 if Resume is False.
                optimizer (torch.nn.optimizer): Loaded optimizer state, None if Resume is False.
        """
        
        if self.configer.get('resume') is None:
            epoch = 0
            optim_dict = None
        else:
            print('Restoring checkpoint: ', self.configer.get('resume'))
            checkpoint_dict = torch.load(self.configer.get('resume'), map_location=self.device)
            # Remove "module." from DataParallel, if present.
            checkpoint_dict['state_dict'] = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in
                                             checkpoint_dict['state_dict'].items()}
            try:
                net.load_state_dict(checkpoint_dict['state_dict'], strict=False)
            except RuntimeError as e:
                print(f"State dict loading issues:\n{e}")

            epoch = checkpoint_dict.get('epoch', 0)
            optim_dict = checkpoint_dict.get('optimizer', None)
            
        net = net.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        return net, epoch, optim_dict

    def _save_net(self, net, optimizer, epoch):
        """Saving net state method.

            Args:
                net (torch.nn.Module): Module in use
                optimizer (torch.nn.optimizer): Optimizer state to save
                epoch (int): Current epoch number to save
        """
        
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        checkpoints_dir = Path(self.configer.get('checkpoints', 'save_dir')) / self.configer.get("dataset", "name")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if self.save_policy == "all":
            latest_name = '{}_epoch_{}.pth'.format(self.output_file_name, epoch)
        elif self.save_policy == "best":
            latest_name = 'best_{}.pth'.format(self.output_file_name)
        else:
            raise ValueError(f'Policy {self.save_policy} is unknown.')
   
        torch.save(state, checkpoints_dir / latest_name)

    def save_all(self, accuracy, net, optimizer, epoch):
        self._save_net(net, optimizer, epoch)
        return accuracy

    def save_best(self, accuracy, net, optimizer, epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_net(net, optimizer, epoch)
            return self.best_accuracy
        else:
            return 0

    def early_stop(self, accuracy, net, optimizer, epoch):
        ret = self.save_best(accuracy, net, optimizer, epoch)
        if ret > 0:
            self.last_improvement = 0
        else:
            self.last_improvement += 1
        if self.last_improvement >= self.configer.get("checkpoints", "early_stop_number"):
            return -1
        else:
            return ret