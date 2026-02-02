from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.settings import *

@dataclass
class TaskConfig:
    train_paths: List[int]
    test_paths: List[int]
    train_labels: List[int]
    test_labels: List[int]
    train_coords: List[int]
    test_coords: List[int]


class TaskSplitter:
    def __init__(
        self, 
        group_by: str, 
        keys_per_task: List[List[str|int]] = None, 
        train_test_split: List[float] | List[str] = [0.8, 0.2],
        shuffle = True
    ):
        self.group_by = group_by
        self.keys_per_task = keys_per_task
        self.train_test_split = train_test_split
        self.shuffle = shuffle

    def split(self, df: pd.DataFrame) -> List[TaskConfig]:
        if self.keys_per_task is None:
            return self._create_tasks_for_all_keys(df)
        return self._create_multiple_tasks(df)

    def _create_tasks_for_all_keys(self, df: pd.DataFrame) -> List[TaskConfig]:
        unique_keys = df[self.group_by].unique()
        if self.shuffle:
            np.random.shuffle(unique_keys)
        
        task_configs = []
        for key in unique_keys:
            task_config = self._create_task_for_keys(df, [key])
            task_configs.append(task_config)
        return task_configs

    def _create_multiple_tasks(self, df: pd.DataFrame) -> List[TaskConfig]:
        task_configs = []
        for keys in self.keys_per_task:
            task_config = self._create_task_for_keys(df, keys)
            task_configs.append(task_config)
        return task_configs

    def _create_task_for_keys(self, df: pd.DataFrame, keys: List[str|int]) -> TaskConfig:
        task_df = df[df[self.group_by].isin(keys)]
        
        if self.shuffle:
            task_df = task_df.sample(frac=1).reset_index(drop=True)
        
        split_idx = self._calculate_split_point(len(task_df))
        
        train_df = task_df.iloc[:split_idx]
        test_df = task_df.iloc[split_idx:]
        
        return TaskConfig(
            train_paths=train_df['img_path'].tolist(),
            test_paths=test_df['img_path'].tolist(),
            train_labels=train_df[CLASS_IDX_COLUMN_NAME].tolist(),
            test_labels=test_df[CLASS_IDX_COLUMN_NAME].tolist(),
            train_coords=train_df['location'].tolist(),
            test_coords=test_df['location'].tolist()
        )

    def _calculate_split_point(self, n: int) -> int:
        split_ratio = float(self.train_test_split[0]) if isinstance(self.train_test_split[0], str) else self.train_test_split[0]
        return int(n * split_ratio)