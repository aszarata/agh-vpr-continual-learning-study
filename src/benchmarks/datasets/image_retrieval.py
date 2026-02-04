from PIL import Image
from typing import List, Literal
import numpy as np
from src.benchmarks.datasets.base import BaseImageDataset

StrategyType = Literal["distance", "class"]

class ImageRetrievalDataset(BaseImageDataset):
    def __init__(
        self, 
        root_dir: str,
        img_size: int, 
        image_paths: List[str], 
        labels: list[int] = None, 
        coordinates: list[tuple[float]] = None,
        strategy: StrategyType = "class"
    ):
        super().__init__(root_dir, img_size, image_paths)
        self.targets = labels
        self.coordinates = coordinates
        self.strategy = strategy

        self.class_indices = {
            c: np.where(self.targets == c)[0] 
            for c in np.unique(self.targets)
        }

        if strategy == "class" and labels is None:
            raise ValueError("Labels must be provided when using 'class' strategy")
        elif strategy == "distance" and coordinates is None:
            raise ValueError("Coordinates must be provided when using 'distance' strategy")
    
    def __getitem__(self, idx):
        if self.strategy == "class":
            anchor, positive, negative = self.__get_triplet_class_strategy(idx)
        elif self.strategy == "distance":
            anchor, positive, negative = self.__get_triplet_distance_strategy(idx)

        return (anchor, positive, negative), -1

    def __get_triplet_class_strategy(self, idx):
        anchor_label = self.targets[idx]
        positive_ids = self.class_indices[anchor_label]
        positive_ids_filtered = positive_ids[positive_ids != idx]

        if len(positive_ids_filtered) > 0:
            positive_idx = np.random.choice(positive_ids_filtered)
        else:
            positive_idx = idx

        negative_ids = [c for c in self.class_indices.keys() if c != anchor_label]
        negative_label = np.random.choice(negative_ids)
        negative_idx = np.random.choice(self.class_indices[negative_label])

        anchor = self._open_and_transform_image(idx)
        positive = self._open_and_transform_image(positive_idx)
        negative = self._open_and_transform_image(negative_idx)

        return anchor, positive, negative

    def __get_triplet_distance_strategy(self, idx):
        raise NotImplementedError()