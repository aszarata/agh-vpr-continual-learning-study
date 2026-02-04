from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from src.settings import *

@dataclass
class TaskConfig:
    train_paths: List[str]
    val_paths: List[str]
    test_paths: List[str]
    train_labels: List[int]
    val_labels: List[int]
    test_labels: List[int]
    train_coords: List[Tuple[float]]
    val_coords: List[Tuple[float]]
    test_coords: List[Tuple[float]]

class TaskSplitter:
    def __init__(
        self, 
        group_by: str, 
        keys_per_task: List[List[str|int]] = None, 
        split_ratios: List[float] = [0.7, 0.15, 0.15],
        shuffle: bool = True
    ):
        self.group_by = group_by
        self.keys_per_task = keys_per_task
        self.split_ratios = split_ratios
        self.shuffle = shuffle

    def split(self, df: pd.DataFrame) -> List[TaskConfig]:
        if self.keys_per_task is None:
            unique_keys = df[self.group_by].unique()
            if self.shuffle:
                np.random.shuffle(unique_keys)
            self.keys_per_task = [[key] for key in unique_keys]
            
        return [self._create_task_for_keys(df, keys) for keys in self.keys_per_task]

    def _create_task_for_keys(self, df: pd.DataFrame, keys: List[str|int]) -> TaskConfig:
        task_df = df[df[self.group_by].isin(keys)]
        
        if self.shuffle:
            task_df = task_df.sample(frac=1).reset_index(drop=True)
        
        n = len(task_df)
        train_end = int(n * self.split_ratios[0])
        val_end = int(n * (self.split_ratios[0] + self.split_ratios[1]))

        train_df = task_df.iloc[:train_end]
        val_df = task_df.iloc[train_end:val_end]
        test_df = task_df.iloc[val_end:]
        
        return TaskConfig(
            train_paths=train_df['img_path'].tolist(),
            val_paths=val_df['img_path'].tolist(),
            test_paths=test_df['img_path'].tolist(),
            train_labels=train_df[CLASS_IDX_COLUMN_NAME].tolist(),
            val_labels=val_df[CLASS_IDX_COLUMN_NAME].tolist(),
            test_labels=test_df[CLASS_IDX_COLUMN_NAME].tolist(),
            train_coords=train_df['location'].tolist(),
            val_coords=val_df['location'].tolist(),
            test_coords=test_df['location'].tolist()
        )