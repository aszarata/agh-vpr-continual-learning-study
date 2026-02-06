from avalanche.training.supervised import Naive, Replay, EWC, GenerativeReplay

def get_strategy(
    name: str, 
    model, 
    optimizer, 
    criterion,
    batch_size,
    num_epochs,
    evaluation_plugin, 
    plugins, 
    device
):
    strategies = {
        "Naive": Naive(
            model, 
            optimizer, 
            criterion,
            train_mb_size=batch_size, 
            train_epochs=num_epochs, 
            eval_mb_size=batch_size, 
            evaluator=evaluation_plugin,
            plugins=plugins,
            device=device,
            eval_every=1,
        ),
        "Replay": Replay(
            model, 
            optimizer, 
            criterion,
            mem_size=500, 
            train_mb_size=batch_size, 
            train_epochs=num_epochs, 
            evaluator=evaluation_plugin,
            plugins=plugins,
            device=device,
            eval_every=1,
        ),
        "EWC": EWC(
            model, 
            optimizer, 
            criterion,
            ewc_lambda=0.4, 
            train_mb_size=batch_size, 
            train_epochs=num_epochs, 
            evaluator=evaluation_plugin,
            plugins=plugins,
            device=device,
            eval_every=1,
        )
    }

    return strategies[name]