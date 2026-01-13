import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Tuple, Optional

class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        data_path,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        selected_classes: Optional[List[str]] = None
        ):
        super().__init__()
        
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.selected_classes = selected_classes
        
        # Read all class names.
        annotations_path = self.data_path / 'wnids.txt'
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations has not been found: {annotations_path}")
        with open(annotations_path, 'r') as f:
            all_class_names = [line.strip() for line in f.readlines()]
        
        self.class_names: List[str] = []
        self.class_to_idx: dict = {}

        # Select only specified classes, or all if None
        if self.selected_classes is not None:
            # Validate that all selected classes exist.
            invalid = set(self.selected_classes) - set(all_class_names)
            if invalid:
                raise ValueError(f"Selected classes not in dataset: {invalid}")
            self.class_names = self.selected_classes
        else:
            self.class_names = all_class_names

        # Create class_to_idx mapping for selected classes only.
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # Load data.
        data_dir = self.data_path / split
        self.samples: List[Tuple[str, int]] = []  # (img_path, class_id)
        if split == 'train':
            for cls_name in self.class_names:
                img_dir = data_dir / cls_name / 'images'
                for img_name in os.listdir(img_dir):
                    img_path = img_dir / img_name
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        elif split == 'val':
            val_annotations_path = data_dir / 'val_annotations.txt'
            if not os.path.exists(val_annotations_path):
                raise FileNotFoundError(f"{val_annotations_path} not found.")
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split('\t')
                    # Skip malformed lines.
                    if len(line_parts) < 2:
                        continue  
                    img_name, cls_name = line_parts[0], line_parts[1]                    
                    if cls_name in self.class_names:
                        if cls_name not in self.class_to_idx:
                            raise ValueError(f"Unknown class_name '{cls_name}' in {val_annotations_path}")
                        img_path = data_dir / 'images' / img_name
                        self.samples.append((img_path, self.class_to_idx[cls_name]))
        
        else:
            raise ValueError("'split' must be 'train' or 'val'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label