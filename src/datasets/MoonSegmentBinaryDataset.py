import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from typing import Union, List, Tuple, Optional

class MoonSegmentationDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        samples: List[str],
        img_prefix: str,
        mask_prefix: str,
        augmentation: Optional[A.Compose] = None,
        preprocessing: Optional[A.Compose] = None
        ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.img_dir = self.data_path / self.img_prefix
        self.mask_dir = self.data_path / self.mask_prefix
        
        self.samples = samples
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        img_id = self.samples[idx]
        
        img_path = self.img_dir / (self.img_prefix + f"{img_id}.png")
        mask_path = self.mask_dir / (self.mask_prefix + f"{img_id}.png")

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        assert np.all((mask == 0) | (mask == 255)), "Mask must be binary (0 or 255)"
        # Бинарная маска [0, 1]: 0 = фон, 1 = камни.
        mask = (mask == 255).astype(np.float32)
        
        # Аугментации.
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        
        # Предобработка.
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']        
        
        # Ensure channels-first format for PyTorch
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # (H, W, C) → (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)           # (H, W) → (1, H, W)

        return img, mask