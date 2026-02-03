import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from pathlib import Path
from typing import Union, List, Tuple, Callable, Optional

class MoonSegmentationDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        samples: List[str],
        img_prefix: str,
        mask_prefix: str,
        geometric_augmentations: Optional[transforms_v2.Compose] = None,
        photometric_augmentations: Optional[List] = None,
        postprocessing: Optional[transforms_v2.Compose] = None
        ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.geometric_augmentations = geometric_augmentations
        self.photometric_augmentations = photometric_augmentations
        self.postprocessing = postprocessing

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
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Validate mask is binary (0 or 255)
        mask_np = np.array(mask)
        if not np.all((mask_np == 0) | (mask_np == 255)):
            raise ValueError(f"Mask must be binary (0 or 255), got values: {np.unique(mask_np)}")

        if self.geometric_augmentations is not None:
            img, mask = self.geometric_augmentations(img, mask)
        
        img = transforms_v2.ToImage()(img)
        img = transforms_v2.ToDtype(torch.float32, scale=True)(img)
        
        if self.photometric_augmentations is not None:
            idx = torch.randint(0, len(self.photometric_augmentations), (1,)).item()
            img = self.photometric_augmentations[idx](img)
        print(img.shape, img.min(), img.max(), img.mean())
        
        if self.postprocessing is not None:
            img = self.postprocessing(img)

        mask = transforms_v2.ToImage()(mask)
        mask = transforms_v2.ToDtype(torch.long, scale=False)(mask)
        mask = (mask.squeeze(0) // 255).long()

        print('image:', img.shape)
        print('mask:', mask.shape)
        return img.squeeze(0), mask.squeeze(0)