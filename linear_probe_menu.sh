#!/bin/bash

# Lazy temporary solution for multiple runs with different hyperparameters.
python 03_linear_probe.py --embedding_subset 2
python 03_linear_probe.py --embedding_subset 4
python 03_linear_probe.py --embedding_subset 8
python 03_linear_probe.py --embedding_subset 16
python 03_linear_probe.py --embedding_subset 32
