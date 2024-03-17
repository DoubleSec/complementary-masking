import argparse

# Parser for training

train_parser = argparse.ArgumentParser(
    description="Train a model", argument_default=argparse.SUPPRESS
)
# Specify a base config file.
train_parser.add_argument("--base_config", type=str, default="./config.yaml")
# Hyperparameters that can be controlled from a command line. Not all of them.
train_parser.add_argument("--bt_lambda", type=float, required=False)
train_parser.add_argument("--projection_size", type=int, required=False)

# Parser for linear probing

lp_parser = argparse.ArgumentParser(
    description="Linear probe a model", argument_default=argparse.SUPPRESS
)
# Specify a base config file.
lp_parser.add_argument("--base_config", type=str, default="./lp_config.yaml")
# Hyperparameters that can be controlled from a command line. Not all of them.
lp_parser.add_argument("--run_id", type=str, required=False)
lp_parser.add_argument("--embedding_subset", type=int, required=False, default=None)


def update_config(args, config: dict):
    """For updating variables in config based on arguments."""
    for k in config:
        if k in args:
            config[k] = vars(args)[k]

    return config
