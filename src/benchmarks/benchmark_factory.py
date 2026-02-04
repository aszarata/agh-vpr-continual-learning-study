from typing import List, Tuple
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset

from src.benchmarks.datasets.classification import ImageClassificationDataset
from src.benchmarks.datasets.image_retrieval import ImageRetrievalDataset
from src.scenarios.task_splitters import TaskConfig
from src.settings import *

class BenchmarkFactory:
    def __init__(self, root_dir: str, image_size: int):
        self.root_dir = root_dir
        self.image_size = image_size

    def build_img_classification_benchmark(self, configs: List[TaskConfig]):
        train_configs = [
            {
                "image_paths": task.train_paths, 
                "labels": task.train_labels, 
            } for task in configs
        ]

        val_configs = [
            {
                "image_paths": task.val_paths, 
                "labels": task.val_labels
            } for task in configs
        ]

        test_configs = [
            {
                "image_paths": task.test_paths, 
                "labels": task.test_labels
            } for task in configs
        ]
        
        return self.__build_generic_benchmark(ImageClassificationDataset, train_configs, val_configs, test_configs)
    
    def build_img_retrieval_benchmark(self, configs: List[TaskConfig], simmilarity_strategy: str = "class"):
        train_configs = [
            {
                "image_paths": task.train_paths, 
                "labels": task.train_labels, 
                "coordinates": task.train_coords, 
                "strategy": simmilarity_strategy,
            } for task in configs
        ]

        val_configs = [
            {
                "image_paths": task.val_paths, 
                "labels": task.val_labels, 
                "coordinates": task.val_coords, 
                "strategy": simmilarity_strategy
            } for task in configs
        ]

        test_configs = [
            {
                "image_paths": task.test_paths, 
                "labels": task.test_labels, 
                "coordinates": task.test_coords, 
                "strategy": simmilarity_strategy
            } for task in configs
        ]
        
        return self.__build_generic_benchmark(ImageRetrievalDataset, train_configs, val_configs, test_configs)
    
    def __build_generic_benchmark(self, dataset_class, train_configs, val_configs, test_configs):
        train_datasets = []
        valid_datasets = []
        test_datasets = []

        for train_cfg, val_cfg, test_cfg in zip(train_configs, val_configs, test_configs):
            
            train_ds = dataset_class(
                root_dir=self.root_dir,
                img_size=self.image_size,
                **train_cfg
            )

            train_datasets.append(as_classification_dataset(train_ds))

            val_ds = dataset_class(
                root_dir=self.root_dir,
                img_size=self.image_size,
                **val_cfg
            )
            valid_datasets.append(as_classification_dataset(val_ds))

            test_ds = dataset_class(
                root_dir=self.root_dir,
                img_size=self.image_size,
                **test_cfg
            )
            test_datasets.append(as_classification_dataset(test_ds))

        return benchmark_from_datasets(
            train=train_datasets,
            test=test_datasets,
            valid=valid_datasets,
        )