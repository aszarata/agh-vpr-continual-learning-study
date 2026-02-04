import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.supervised import Naive, Replay, EWC
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EarlyStoppingPlugin, LRSchedulerPlugin

from src.settings import *
from src.utils.data import read_and_prepare_metadata
from src.models.torch_models import get_resnet18_for_cl
from src.scenarios.task_splitters import TaskSplitter
from src.benchmarks.benchmark_factory import BenchmarkFactory

# === CONFIGURATION ===
EXPERIMENT_NAME = "camera-domain-all-separate-naive-01"
strategy_name = "Naive"
num_epochs = 20
batch_size = 16
starting_lr = 0.01
momentum = 0.7

# === INIT ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/{EXPERIMENT_NAME}/{timestamp}"
os.makedirs(log_dir, exist_ok=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'

# === TASKS ===

df = read_and_prepare_metadata()
splitter = TaskSplitter(
    group_by="camera_type"
)
benchmark_factory = BenchmarkFactory(DATASET_ROOT, IMG_SIZE)

configs = splitter.split(df)
benchmark = benchmark_factory.build_img_classification_benchmark(configs)

model = get_resnet18_for_cl(9, False)
criterion = nn.CrossEntropyLoss()

# === STRATEGIES AND PLUGINS ===
optimizer = optim.SGD(model.parameters(), lr=starting_lr, momentum=momentum,)

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[
        InteractiveLogger(), 
        TensorboardLogger(log_dir)
    ],
)

early_stopping = EarlyStoppingPlugin(
    patience=6, 
    val_stream_name='valid_stream', 
    metric_name='Top1_Acc_Epoch/eval_phase/valid_stream'
)

scheduler_plugin = LRSchedulerPlugin(
    scheduler.ReduceLROnPlateau(optimizer, patience=3),
    metric="val_loss"
)

strategies = {
    "Naive": Naive(
        model, 
        optimizer, 
        criterion,
        train_mb_size=batch_size, 
        train_epochs=num_epochs, 
        eval_mb_size=batch_size, 
        evaluator=eval_plugin,
        plugins=[early_stopping, scheduler_plugin],
        device=device,
    ),
}

# === EXPERIMENTS ===

for i, config in enumerate(configs):
    print(f"Task {i}: train={len(config.train_paths)}, test={len(config.test_paths)}")

strategy: SupervisedTemplate = strategies[strategy_name]
    
for experience in benchmark.train_stream:
    if experience.current_experience > 0:
        for g in strategy.optimizer.param_groups:
            g['lr'] = 0.001
    print(f"Training on domain: {experience.current_experience}")
    strategy.train(experience, eval_streams=[benchmark.valid_stream])
    
    print("Evaluation")
    strategy.eval(benchmark.test_stream)

print(f"Finished. Saved in: {log_dir}")