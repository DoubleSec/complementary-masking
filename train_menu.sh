#!/bin/bash

# Lazy temporary solution for multiple runs with different hyperparameters.
python 02_train_model.py --bt_lambda 0.75
python 02_train_model.py --bt_lambda 1.0
python 02_train_model.py --bt_lambda 1.25
python 02_train_model.py --bt_lambda 1.5
