import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional

class MoonSegmentationDataset(Dataset):
    def __init__(
        self,
        data_path,
        split: str = 'train',
        img_prefix: str = 'render',
        mask_prefix: str = 'ground',
        augmentation: Optional[A.Compose] = None,
        preprocessing: Optional[A.Compose] = None,
        seed = None
        ):
        super().__init__()
        
        self.data_path = data_path
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.img_folder = self.img_prefix + '/'
        self.mask_folder = self.mask_prefix + '/'
        
        all_images = [img_no_ext.replace(self.img_prefix, '') for img_no_ext in
            [img.replace('.png', '') for img in os.listdir(self.data_path / self.img_folder) if img.endswith('.png')]
        ]
        train_images, val_images = train_test_split(
            all_images,
            test_size=0.2,
            random_state = seed)
        
        if split == 'train':
            self.samples = train_images
        elif split == 'val':
            self.samples = val_images
        else:
            raise ValueError("'split' must be 'train' or 'val'")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        
        img_id = self.samples[idx]
        
        img_path = self.data_path / (self.img_folder + self.img_prefix + f"{img_id}.png")
        mask_path = self.data_path / (self.mask_folder + self.mask_prefix + f"{img_id}.png")
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)      
        # Бинарная маска [0, 1]: 0 = фон, 1 = камни.
        mask = (mask > 0).astype(np.float32)
        
        # Аугментации.
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        
        # Предобработка.
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']        
        
        return img, mask