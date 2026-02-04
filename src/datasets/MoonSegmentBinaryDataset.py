import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from pathlib import Path
from typing import Union, List, Tuple, Callable, Optional

def validate_binary(input, eps = 1e-3):
    # Validate 0 or 1.
    input_np = np.array(input)
    valid_mask = (np.abs(input_np) < eps) | (np.abs(input_np - 1) < eps)
    if not np.all(valid_mask):
        invalid_values = np.unique(input_np[~valid_mask])
        raise ValueError(f"Must be binary (0 or 1), got invalid values: {invalid_values}")

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
        
        img = transforms_v2.ToImage()(img)
        img = transforms_v2.ToDtype(torch.float32, scale=True)(img)

        mask = (np.array(mask) > 0).astype(np.float32)
        
        mask = transforms_v2.ToImage()(mask)
        # Convert mask to uint, otherwise augmentations make it smooth.
        # After augmentations convert back to float32.
        mask = mask.to(torch.uint8)
        
        if self.geometric_augmentations is not None:
            img, mask = self.geometric_augmentations(img, mask)
        
        if self.photometric_augmentations is not None:
            aug_idx = torch.randint(0, len(self.photometric_augmentations), (1,)).item()
            img = self.photometric_augmentations[aug_idx](img)
        
        if self.postprocessing is not None:
            img = self.postprocessing(img)
        
        mask = mask.to(torch.float32)
        validate_binary(mask)
        
        return img, mask