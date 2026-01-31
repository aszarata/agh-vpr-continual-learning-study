from typing import List, Tuple
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset

from src.benchmarks.datasets.classification import ImageClassificationDataset
from src.benchmarks.datasets.image_retrieval import ImageRetrievalDataset

class BenchmarkFactory:
    def __init__(self, root_dir: str, image_size: int):
        self.root_dir = root_dir
        self.image_size = image_size

    def build_img_classification_benchmark(
        self, 
        train_paths: List[List[str]], 
        train_labels: List[List[int]],
        test_paths: List[List[str]], 
        test_labels: List[List[int]],
    ):
        
        train_configs = [{"image_paths": p, "labels": l} for p, l in zip(train_paths, train_labels)]
        test_configs = [{"image_paths": p, "labels": l} for p, l in zip(test_paths, test_labels)]
        
        return self.__build_generic_benchmark(ImageClassificationDataset, train_configs, test_configs)
    
    def build_img_retrieval_benchmark(
        self, 
        train_paths: List[List[str]], 
        train_labels: List[List[int]],
        test_paths: List[List[str]], 
        test_labels: List[List[int]],
        train_coords: List[List[Tuple[float]]] = None,
        test_coords: List[List[Tuple[float]]] = None,
        simmilarity_strategy: str = "class"
    ):
        train_configs = [
            {
                "image_paths": p, 
                "labels": l, 
                "coordinates": c, 
                "strategy": simmilarity_strategy
            } for p, l, c in zip(train_paths, train_labels, train_coords or [None]*len(train_paths))
        ]
        test_configs = [
            {
                "image_paths": p, 
                "labels": l, 
                "coordinates": c, 
                "strategy": simmilarity_strategy
            } for p, l, c in zip(test_paths, test_labels, test_coords or [None]*len(test_paths))
        ]
        
        return self.__build_generic_benchmark(ImageRetrievalDataset, train_configs, test_configs)
    
    def __build_generic_benchmark(self, dataset_class, train_configs, test_configs):
        train_datasets = []
        test_datasets = []

        for train_cfg, test_cfg in zip(train_configs, test_configs):
            
            train_ds = dataset_class(
                root_dir=self.root_dir,
                img_size=self.image_size,
                **train_cfg
            )
            train_datasets.append(AvalancheDataset(train_ds))

            test_ds = dataset_class(
                root_dir=self.root_dir,
                img_size=self.image_size,
                **test_cfg
            )
            test_datasets.append(AvalancheDataset(test_ds))

        return benchmark_from_datasets(
            train_datasets=train_datasets,
            test_datasets=test_datasets,
        )