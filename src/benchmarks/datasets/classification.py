from PIL import Image
from typing import List
import numpy as np
from src.benchmarks.datasets.base import BaseImageDataset

class ImageClassificationDataset(BaseImageDataset):
    def __init__(self, root_dir: str, img_size: int, image_paths: List[str], labels: List[int]):
        super().__init__(root_dir, img_size, image_paths)
        self.targets = np.array(labels, dtype=np.int64)
    
    def __getitem__(self, idx):
        image = self._open_and_transform_image(idx)
        label = self.targets[idx]
        return image, label

