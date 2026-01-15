from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Import Utils.
from torch.utils.data import DataLoader
from utils.average_meter import AverageMeter

# Import Datasets.
from datasets.TinyImageNetDataset import TinyImageNetDataset

# Import Model.
from models.model_utilizer import ModelUtilizer
from models.backbone import customResNet18

# Setting seeds.
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class ResNet18Trainer(object):

    def __init__(self, configer):
        self.configer = configer

        #: str: Type of dataset.
        self.dataset = self.configer.get("dataset", "name").lower()
        self.data_path = Path(self.configer.get("data", "data_path")) / self.configer.get("dataset", "name")
        
        # DataLoaders.
        self.train_loader = None
        self.val_loader = None

        # Module load and save utility.
        self.device = torch.device(self.configer.device)
        print(f"Device (train.py): {self.device}")
        self.model_utility = ModelUtilizer(self.configer) #: Model utility for load, save and update optimizer
        self.net = None
        self.lr = None

        # Training procedure.
        self.epoch = None
        self.optimizer = None
        self.loss = None
        self.train_transforms = None
        self.val_transforms = None
        
        #: int: Chosen classes to work with.
        self.selected_classes = self.configer.get("dataset", "selected_classes")
        self.n_classes = len(self.selected_classes)
        
        # Train and val losses.
        self.losses = {
            'train': AverageMeter(),
            'val': AverageMeter()
        }

        # Train and val accuracy.
        self.accuracy = {
            'train': AverageMeter(),
            'val': AverageMeter()
        }
        
        self.train_history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": []
        }
        
    def init_model(self):
        """Initialize model and other data for procedure"""
        
        self.loss = nn.CrossEntropyLoss().to(self.device)
        
        img_size = self.configer.get('dataset', 'img_size')
        self.net = customResNet18(
            num_classes = self.n_classes,
            layers_config = self.configer.get("model", "layers_num")*[self.configer.get("model", "block_size")],
            in_channels = img_size[0],
            layer0_channels = 256 // 2**(self.configer.get("model", "layers_num") - 1)
        )

        # Initializing training.
        self.net, self.epoch_init, optim_dict = self.model_utility.load_net(self.net)
        self.epoch = self.epoch_init
        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net)

        # Resuming training, restoring optimizer value.
        if optim_dict is None:
            print(f"Starting training from scratch using {self.configer.get('solver', 'type')}.")
        else:
            print(f"Resuming training from epoch {self.epoch} using {self.configer.get('solver', 'type')}.")
            self.optimizer.load_state_dict(optim_dict)
        
        self.model_size = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Model size: {self.model_size}")
        mdl_img_size = self.configer.get('model', 'img_size')
        # Selecting Dataset and DataLoader
        if self.dataset == "tiny-imagenet-200":
            Dataset = TinyImageNetDataset
            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            self.train_transforms = transforms.Compose([
                transforms.Resize(tuple([int(img_size[1] * 1.125)]*2)),
                transforms.RandomResizedCrop(mdl_img_size[1], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ])

            self.val_transforms = transforms.Compose([
                transforms.Resize(tuple(mdl_img_size[1:])),
                transforms.ToTensor(),
                normalize,
            ])
            
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset', 'name')}")

        # Setting Dataloaders.
        self.train_loader = DataLoader(
            Dataset(
                self.data_path,
                split="train",
                transform=self.train_transforms,
                selected_classes=self.selected_classes
                ),
            batch_size=self.configer.get("data", "batch_size"),
            shuffle=True,
            num_workers=self.configer.get("data", "workers"))
        
        self.val_loader = DataLoader(
            Dataset(
                self.data_path,
                split="val",
                transform=self.val_transforms,
                selected_classes=self.selected_classes
                ),
            batch_size=self.configer.get("data", "batch_size"),
            shuffle=False,
            num_workers=self.configer.get("data", "workers"))

        print(f"Train size: {len(self.train_loader.dataset)}")
        print(f"Val size: {len(self.val_loader.dataset)}")
        print(f"Number of classes: {len(self.train_loader.dataset.class_names)}")
              
    def __train(self):
        """Train function for every epoch."""
        self.net.train()
        for data_tuple in tqdm(self.train_loader, desc="Train"):

            inputs, gt = data_tuple[0].to(self.device), data_tuple[1].to(self.device)

            output = self.net(inputs)

            self.optimizer.zero_grad()
            loss = self.loss(output, gt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            predicted = torch.argmax(output.detach(), dim=1)
            correct = gt.detach()

            self.update_metrics("train", loss.item(), inputs.size(0),
                                float((predicted==correct).sum()) / len(correct))

    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            for data_tuple in tqdm(self.val_loader, desc="Val"):

                inputs, gt = data_tuple[0].to(self.device), data_tuple[1].to(self.device)

                output = self.net(inputs)
                
                loss = self.loss(output, gt)

                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach()

                self.update_metrics("val", loss.item(), inputs.size(0),
                                    float((predicted == correct).sum()) / len(correct))
        
        ret = self.model_utility.save(
            self.accuracy["val"].avg,
            self.net,
            self.optimizer,
            self.epoch + 1)

        if ret < 0:
            return -1
        return ret

    def train(self):        
        for n in range(self.configer.get("epochs")):
            print("Starting epoch {} of {}.".format(self.epoch + 1, self.configer.get("epochs") + self.epoch_init))
            self.__train()
            ret = self.__val()
            
            self.train_history["epoch"].append(self.epoch + 1)
            self.train_history["train_loss"].append(self.losses["train"].avg)
            self.train_history["train_accuracy"].append(self.accuracy["train"].avg)
            self.train_history["val_loss"].append(self.losses["val"].avg)
            self.train_history["val_accuracy"].append(self.accuracy["val"].avg)
            self.train_history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            prefix = f"Epoch {self.train_history['epoch'][-1]:2d} | "
            print(f"{prefix}Train Loss: {self.train_history['train_loss'][-1]:.4f}, Accuracy: {self.train_history['train_accuracy'][-1]:.4f}")
            print(f"{' ' * len(prefix)}Val   Loss: {self.train_history['val_loss'][-1]:.4f}, Accuracy: {self.train_history['val_accuracy'][-1]:.4f}")
            
            self.losses["train"].reset()
            self.accuracy["train"].reset()
            self.losses["val"].reset()
            self.accuracy["val"].reset()

            if ret < 0:
                print("Got no improvement for {} subsequent epochs. Finished epoch {}, than stopped."
                      .format(self.configer.get("checkpoints", "early_stop_number"), self.epoch_init + n+1))
                break
            
            self.epoch += 1
        
        return self.train_history, \
            len(self.train_loader.dataset), \
            len(self.val_loader.dataset), \
            len(self.train_loader.dataset.class_names), \
            self.model_size, \
            str(self.net)
    
    def update_metrics(self, split: str, loss, bs, accuracy):
        self.losses[split].update(loss, bs)
        self.accuracy[split].update(accuracy, bs)