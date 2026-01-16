import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class MoonSegmentationDataset(Dataset):
    def __init__(
        self,
        root_path,
        img_folder = 'render',
        mask_folder = 'ground',
        img_ids = None,
        augmentation=None,
        preprocessing=None
        ):
        super().__init__()
        
        self.root_path = root_path
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        if img_ids is None:
            all_images = os.listdir(self.root_path / self.img_folder)
            self.img_ids = [img.replace('.png', '') for img in all_images if img.endswith('.png')]
        else:
            self.img_ids = img_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        
        img_id = self.img_ids[idx]
        
        img_path = os.path.join(self.root_dir, 'images', self.img_folder, f"{img_id}.png")
        
        # Для масок убираем префикс "render" если он есть.
        # Например: render0001 - 0001.
        mask_id = img_id.replace('render', '') if 'render' in img_id else img_id
        mask_path = os.path.join(self.root_dir, 'images', self.mask_folder, f"ground{mask_id}.png")
        
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