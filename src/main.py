import os
from datetime import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.supervised import Naive, Replay, EWC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger

from src.settings import *
from src.utils.data import read_and_prepare_metadata
from src.models.torch_models import get_resnet18_for_cl
from src.scenarios.task_splitters import TaskSplitter
from src.benchmarks.benchmark_factory import BenchmarkFactory

EXPERIMENT_NAME = "camera-domain-all-separate-01"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/{EXPERIMENT_NAME}/{timestamp}"
os.makedirs(log_dir, exist_ok=True)

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[
        InteractiveLogger(), 
        TensorboardLogger(log_dir)
    ],
)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'


df = read_and_prepare_metadata()
splitter = TaskSplitter(
    group_by="camera_type"
)
benchmark_factory = BenchmarkFactory(DATASET_ROOT, IMG_SIZE)

configs = splitter.split(df)
benchmark = benchmark_factory.build_img_classification_benchmark(configs)

model = get_resnet18_for_cl(9, False)
criterion = CrossEntropyLoss()
num_epochs = 20

strategies = {
    "Naive": Naive(
        model, SGD(model.parameters(), lr=0.01, momentum=0.9,), criterion,
        train_mb_size=32, train_epochs=num_epochs, eval_mb_size=32, evaluator=eval_plugin
    ),
    "Replay": Replay(
        model, SGD(model.parameters(), lr=0.01, momentum=0.9,), criterion,
        mem_size=500, train_mb_size=32, train_epochs=num_epochs, evaluator=eval_plugin
    ),
    "EWC": EWC(
        model, SGD(model.parameters(), lr=0.01, momentum=0.9,), criterion,
        ewc_lambda=0.4, train_mb_size=32, train_epochs=num_epochs, evaluator=eval_plugin
    )
}

for i, config in enumerate(configs):
    print(f"Task {i}: train={len(config.train_paths)}, test={len(config.test_paths)}")

for name, strategy in strategies.items():
    print(f"Method: {name}")
    
    for experience in benchmark.train_datasets_stream:
        print(f"Training on domain: {experience.current_experience}")
        strategy.train(experience)
        
        print("Evaluation")
        strategy.eval(benchmark.test_datasets_stream)

print(f"Finished. Saved in: {log_dir}")