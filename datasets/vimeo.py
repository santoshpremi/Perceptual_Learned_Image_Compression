#!/usr/bin/env python3
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class Vimeo90kTestDataset(Dataset):
    """Dataset wrapper around the Vimeo-90K test set.

    Expects the following directory structure (matching the official release):

    .. code-block::

        root/
          ├── sequences/
          │     ├── 00001/0001/im1.png ... im7.png
          │     ├── ...
          └── sep_testlist.txt

    By default the dataset will load the centre frame (``im4.png``) for each
    sequence listed in ``sep_testlist.txt``.
    """

    def __init__(
        self,
        root: str,
        list_file: Optional[str] = None,
        frame_index: int = 4,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sequences_dir = self.root / "sequences"
        if not self.sequences_dir.is_dir():
            raise RuntimeError(f"Sequences directory not found: {self.sequences_dir}")

        if list_file is None:
            list_path = self.root / "sep_testlist.txt"
        else:
            list_path = Path(list_file)
            if not list_path.is_absolute():
                list_path = self.root / list_path

        if not list_path.is_file():
            raise RuntimeError(f"List file not found: {list_path}")

        if not 1 <= frame_index <= 7:
            raise ValueError("frame_index must be between 1 and 7 for Vimeo-90K sequences")

        self.frame_name = f"im{frame_index}.png"
        self.transform = transform

        with list_path.open("r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.readlines() if line.strip()]

        self.samples = []
        for seq in lines:
            frame_path = self.sequences_dir / seq / self.frame_name
            if not frame_path.is_file():
                raise RuntimeError(f"Missing frame {frame_path}")
            self.samples.append(frame_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img
