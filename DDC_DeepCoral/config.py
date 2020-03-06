CFG = {
    "data_path": "../cmfd_forge_train/",
    "kwargs": {"num_workers": 4},
    "batch_size": 128,
    "epoch": 25,
    "lr": 1e-3,
    "momentum": 0.9,
    "log_interval": 10,
    "l2_decay": 0,
    "lambda": 10,
    "backbone": "alexnet",
    "n_class": 2,
}
