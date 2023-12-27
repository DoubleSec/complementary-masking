import argparse

train_parser = argparse.ArgumentParser(
    description="Train a model", argument_default=argparse.SUPPRESS
)

# Specify a base config file.
train_parser.add_argument("--base_config", type=str, default="./config.yaml")

# Hyperparameters that can be controlled from a command line. Not all of them.
train_parser.add_argument("--bt_lambda", type=float, required=False)
train_parser.add_argument("--projection_size", type=int, required=False)


def update_config(args, config: dict):
    for k in config:
        if k in args:
            config[k] = vars(args)[k]

    return config
