import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import Utils.
from torch.utils.data import DataLoader
from utils.metrics import AverageMeter, CombinedLoss, dice_coefficient, iou_score, pixel_accuracy

# Import Datasets.
from datasets.TinyImageNetDataset import TinyImageNetDataset
from datasets.MoonSegmentBinaryDataset import MoonSegmentationDataset

# Import Model.
from models.model_utilizer import ModelUtilizer
from models.backbone import customResNet18
from models.customUNet import customUNet

# Setting seeds.
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]

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
        
        # Selecting Dataset and DataLoader.        
        if self.dataset == "tiny-imagenet-200":
            self.train_transforms = transforms.Compose([
                transforms.Resize(tuple([int(img_size[1] * 1.125)]*2)),
                transforms.RandomResizedCrop(mdl_img_size[1], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_norm, std=std_norm)
            ])

            self.val_transforms = transforms.Compose([
                transforms.Resize(tuple(mdl_img_size[1:])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_norm, std=std_norm)
            ])
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset', 'name')}")

        # Setting Dataloaders.
        if self.dataset == "tiny-imagenet-200":
            self.train_loader = DataLoader(
                TinyImageNetDataset(
                    data_path = self.data_path,
                    split = "train",
                    transform = self.train_transforms,
                    selected_classes = self.selected_classes
                    ),
                batch_size=self.configer.get("data", "batch_size"),
                shuffle=True,
                num_workers=self.configer.get("data", "workers"))
            
            self.val_loader = DataLoader(
                TinyImageNetDataset(
                    data_path = self.data_path,
                    split = "val",
                    transform = self.val_transforms,
                    selected_classes = self.selected_classes
                    ),
                batch_size=self.configer.get("data", "batch_size"),
                shuffle=False,
                num_workers=self.configer.get("data", "workers"))
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset', 'name')}")
        
        print(f"Train. size: {len(self.train_loader.dataset)}")
        print(f"Valid. size: {len(self.val_loader.dataset)}")
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

            self.update_metrics("train", inputs.size(0), loss.item(),
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

                self.update_metrics("val", inputs.size(0), loss.item(),
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
            self.model_size
    
    def update_metrics(self, split: str, bs: int, loss, accuracy):
        self.losses[split].update(loss, bs)
        self.accuracy[split].update(accuracy, bs)
        
class UNetTrainer(object):

    def __init__(self, configer):
        self.configer = configer

        #: str: Type of dataset.
        self.dataset = self.configer.get("dataset", "name").lower()
        self.data_path = Path(self.configer.get("data", "data_path")) / (self.configer.get("dataset", "name") + '/images')
        
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
        self.train_augmentation = None
        self.val_augmentation = None
        self.preprocessing = None
        
        # Train and val losses.
        self.losses = {
            'train': AverageMeter(),
            'val': AverageMeter()
        }
        # Train and val dice.
        self.dice = {
            'train': AverageMeter(),
            'val': AverageMeter()
        }
        # Train and val iou.
        self.iou = {
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
            "train_dice": [],
            "train_iou": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_accuracy": [],
            "lr": []
        }
        
    def init_model(self):
        """Initialize model and other data for procedure"""
        
        self.loss = CombinedLoss(bce_weight=0.5, dice_weight=0.5).to(self.device)
        
        img_size = self.configer.get('dataset', 'img_size')
        self.net = customUNet(
            in_channels = img_size[0],
            out_channels = 1,
            features = self.configer.get("model", "feature_list")
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
        
        # Selecting Dataset and DataLoader.
        if self.dataset == "moon-segmentation-binary":
            self.train_augmentation = A.Compose([
                A.Resize(*mdl_img_size[1:]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.9, 1.1),
                    rotate=(-45, 45),
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MedianBlur(blur_limit=5),
                ], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])

            self.val_augmentation = A.Compose([
                A.Resize(*mdl_img_size[1:]),
            ])

            self.preprocessing = A.Compose([
                A.Normalize(mean=mean_norm, std=std_norm),
                ToTensorV2(),
            ])  
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset', 'name')}")

        # Setting Dataloaders.
        if self.dataset == "moon-segmentation-binary":
            img_prefix = self.configer.get('dataset', 'img_prefix')
            mask_prefix = self.configer.get('dataset', 'mask_prefix')
            img_folder = img_prefix + '/'
            all_images = [img_no_ext.replace(img_prefix, '') for img_no_ext in
                [img.replace('.png', '') for img in os.listdir(self.data_path / img_folder) if img.endswith('.png')]
            ]
            train_images, val_images = train_test_split(
                all_images,
                test_size=0.2,
                random_state = self.configer.seed)
            
            self.train_loader = DataLoader(
                MoonSegmentationDataset(
                    data_path = self.data_path,
                    samples = train_images,
                    img_prefix = img_prefix,
                    mask_prefix = mask_prefix,
                    augmentation = self.train_augmentation,
                    preprocessing = self.preprocessing
                    ), 
                batch_size=self.configer.get("data", "batch_size"),
                shuffle=True,
                num_workers=self.configer.get("data", "workers"),
                pin_memory=True)

            self.val_loader = DataLoader(
                MoonSegmentationDataset(
                    data_path = self.data_path,
                    samples = val_images,
                    img_prefix = img_prefix,
                    mask_prefix = mask_prefix,
                    augmentation = self.val_augmentation,
                    preprocessing = self.preprocessing
                    ), 
                batch_size=self.configer.get("data", "batch_size"),
                shuffle=False,
                num_workers=self.configer.get("data", "workers"),
                pin_memory=True)
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset', 'name')}")
        
        print(f"Train. size: {len(self.train_loader.dataset)}")
        print(f"Valid. size: {len(self.val_loader.dataset)}")
              
    def __train(self):
        """Train function for every epoch."""
        self.net.train()
        for data_tuple in tqdm(self.train_loader, desc="Train"):

            inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)

            outputs = self.net(inputs)

            self.optimizer.zero_grad()
            loss = self.loss(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            dice = dice_coefficient(outputs.detach(), masks.detach())
            iou = iou_score(outputs.detach(), masks.detach())
            accuracy = pixel_accuracy(outputs.detach(), masks.detach())

            self.update_metrics(
                "train",
                inputs.size(0),
                loss.item(),
                dice,
                iou,
                accuracy)

    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            for data_tuple in tqdm(self.val_loader, desc="Val"):
                
                inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)
                
                outputs = self.net(inputs)
                
                loss = self.loss(outputs, masks)

                dice = dice_coefficient(outputs.detach(), masks.detach())
                iou = iou_score(outputs.detach(), masks.detach())
                accuracy = pixel_accuracy(outputs.detach(), masks.detach())

                self.update_metrics(
                    "val",
                    inputs.size(0),
                    loss.item(),
                    dice,
                    iou,
                    accuracy)
                
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
            self.train_history["train_dice"].append(self.dice["train"].avg)
            self.train_history["train_iou"].append(self.iou["train"].avg)
            self.train_history["train_accuracy"].append(self.accuracy["train"].avg)
            self.train_history["val_loss"].append(self.losses["val"].avg)
            self.train_history["val_loss"].append(self.losses["val"].avg)
            self.train_history["val_dice"].append(self.dice["val"].avg)
            self.train_history["val_iou"].append(self.iou["val"].avg)
            self.train_history["val_accuracy"].append(self.accuracy["val"].avg)
            self.train_history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            prefix = f"Epoch {self.train_history['epoch'][-1]:2d} | "
            print(f"{prefix}Train Loss: {self.train_history['train_loss'][-1]:.4f}, Accuracy: {self.train_history['train_accuracy'][-1]:.4f}")
            print(f"{' ' * len(prefix)}Val   Loss: {self.train_history['val_loss'][-1]:.4f}, Accuracy: {self.train_history['val_accuracy'][-1]:.4f}")
            
            self.losses["train"].reset()
            self.dice["train"].reset()
            self.iou["train"].reset()
            self.accuracy["train"].reset()
            self.losses["val"].reset()
            self.dice["val"].reset()
            self.iou["val"].reset()
            self.accuracy["val"].reset()

            if ret < 0:
                print("Got no improvement for {} subsequent epochs. Finished epoch {}, than stopped."
                      .format(self.configer.get("checkpoints", "early_stop_number"), self.epoch_init + n+1))
                break
            
            self.epoch += 1
        
        return self.train_history, \
            len(self.train_loader.dataset), \
            len(self.val_loader.dataset), \
            self.model_size
    
    def update_metrics(self, split: str, bs: int, loss, dice, iou, accuracy):
        self.losses[split].update(loss, bs)
        self.dice[split].update(dice, bs)
        self.iou[split].update(iou, bs)
        self.accuracy[split].update(accuracy, bs)