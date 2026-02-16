import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics, bwt_metrics, MAC_metrics, timing_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EarlyStoppingPlugin, LRSchedulerPlugin

from src.settings import *
from src.configuration.parser import get_config
from src.utils.data import read_and_prepare_metadata
from src.models.torch_models import get_resnet18_for_cl, get_resnet34_for_cl
from src.scenarios.task_splitters import TaskSplitter
from src.benchmarks.benchmark_factory import BenchmarkFactory
from src.configuration.strategies import get_strategy

def run_experiment(cfg):
    # === CONFIGURATION ===
    EXPERIMENT_NAME = cfg["EXPERIMENT_NAME"]
    strategy_name = cfg["strategy_name"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    starting_lr = cfg["starting_lr"]
    next_lr = cfg["next_lr"]
    momentum = cfg["momentum"]
    train_val_test_split = cfg["train_val_test_split"]

    # === INIT ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/{EXPERIMENT_NAME}/{timestamp}"
    weights_dir = log_dir + "/weights"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'

    print(f"Using device: {device}")

    # === TASKS ===

    df = read_and_prepare_metadata()
    splitter = TaskSplitter(
        group_by="camera_type",
        split_ratios=train_val_test_split,
    )
    benchmark_factory = BenchmarkFactory(DATASET_ROOT, IMG_SIZE)

    configs = splitter.split(df)
    benchmark = benchmark_factory.build_img_classification_benchmark(configs)

    model = get_resnet34_for_cl(9, False)
    criterion = nn.CrossEntropyLoss()

    # === STRATEGIES AND PLUGINS ===
    optimizer = optim.SGD(model.parameters(), lr=starting_lr, momentum=momentum,)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        MAC_metrics(minibatch=True, epoch=True, experience=True),
        timing_metrics(minibatch=True, experience=True, stream=True),
        loggers=[
            InteractiveLogger(), 
            TensorboardLogger(log_dir)
        ],
    )

    early_stopping = EarlyStoppingPlugin(
        patience=6, 
        val_stream_name='valid_stream', 
        metric_name='Top1_Acc_Epoch/eval_phase/valid_stream/Exp000'
    )

    scheduler_plugin = LRSchedulerPlugin(
        scheduler.ReduceLROnPlateau(optimizer, patience=3),
        metric="val_loss"
    )

    # === EXPERIMENTS ===

    for i, config in enumerate(configs):
        print(f"Task {splitter.task_id_to_name[i]}: train={len(config.train_paths)}, test={len(config.test_paths)}")

    strategy: SupervisedTemplate = get_strategy(
        name=strategy_name, 
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        evaluation_plugin=eval_plugin, 
        plugins=[early_stopping, scheduler_plugin], 
        device=device,
    )
        
    for experience in benchmark.train_stream:

        if experience.current_experience > 0:
            for g in strategy.optimizer.param_groups:
                g['lr'] = next_lr
        
        print(f"Training on domain: {experience.current_experience}")
        strategy.train(experience, eval_streams=[benchmark.valid_stream[experience.current_experience]])
        
        print("Evaluation")
        strategy.eval(benchmark.test_stream)

        torch.save(
            strategy.model.state_dict(), 
            os.path.join(weights_dir, f"model_weights_exp{experience.current_experience}.pth")
        )

    print(f"Finished. Saved in: {log_dir}")

if __name__ == "__main__":
    config_data = get_config()
    run_experiment(config_data)