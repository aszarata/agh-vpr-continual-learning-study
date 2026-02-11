from PIL import Image
from typing import List
import numpy as np
from src.benchmarks.datasets.base import BaseImageDataset

class ImageRegressionDataset(BaseImageDataset):
    def __init__(self, root_dir: str, img_size: int, image_paths: List[str], labels: List[int], coordinates: list[tuple[float]] = None):
        super().__init__(root_dir, img_size, image_paths)
        self.targets = np.array(coordinates)
        self.labels = np.array(labels, dtype=np.int64)
        

    def __getitem__(self, idx):
        image = self._open_and_transform_image(idx)
        coords = self.targets[idx]
        return image, coords