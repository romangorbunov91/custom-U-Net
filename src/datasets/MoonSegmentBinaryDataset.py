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
        img_folder: str = 'render',
        mask_folder: str = 'ground',
        augmentation: Optional[A.Compose] = None,
        preprocessing: Optional[A.Compose] = None,
        seed = None
        ):
        super().__init__()
        
        self.data_path = data_path
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        all_images = os.listdir(self.data_path / self.img_folder)
        train_images, val_images = train_test_split(
            all_images,
            test_size=0.2,
            random_state = seed)
        
        if split == 'train':
            self.img_ids = [img.replace('.png', '') for img in train_images if img.endswith('.png')]
        elif split == 'val':
            self.img_ids = [img.replace('.png', '') for img in val_images if img.endswith('.png')]
        else:
            raise ValueError("'split' must be 'train' or 'val'")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        
        img_id = self.img_ids[idx]
        
        img_path = self.root_dir / ('images' + self.img_folder + f"{img_id}.png")
        
        # Для масок убираем префикс "render" если он есть.
        # Например: render0001 - 0001.
        mask_id = img_id.replace('render', '') if 'render' in img_id else img_id
        mask_path = self.root_dir / ('images' + self.mask_folder + f"ground{mask_id}.png")
        
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