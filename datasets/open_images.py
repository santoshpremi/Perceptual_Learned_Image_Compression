#!/usr/bin/env python3
"""Open Images V7 Dataset loader for image compression training."""
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class OpenImagesDataset(Dataset):
    """Dataset wrapper for Open Images V7.
    
    Expects images to be downloaded using the Open Images downloader script.
    The dataset loads images from a directory structure:
    
    .. code-block::
    
        root/
          ├── train/
          │     ├── <image_id_1>.jpg
          │     ├── <image_id_2>.jpg
          │     └── ...
          ├── validation/
          │     ├── <image_id_1>.jpg
          │     └── ...
          └── test/
                ├── <image_id_1>.jpg
                └── ...
    
    Args:
        root: Root directory containing the Open Images dataset
        split: Dataset split ('train', 'validation', or 'test')
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Open Images stores images in split-specific directories
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise RuntimeError(f"Split directory not found: {split_dir}")
        
        # Collect all image files (supports common image formats)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.samples = [
            f for f in split_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}")
        
        print(f"Found {len(self.samples)} images in {split_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        img_path = self.samples[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        if self.transform:
            img = self.transform(img)
        
        return img



