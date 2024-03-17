#!/bin/bash

# Lazy temporary solution for multiple runs with different hyperparameters.
python 03_linear_probe.py --run_id f356669850844891a15e40120585b568 --embedding_subset 16
python 03_linear_probe.py --run_id 8b69f950dceb4cb5b7b8f5a5a8e83cc2 --embedding_subset 16

python 03_linear_probe.py --run_id f356669850844891a15e40120585b568 --embedding_subset 32
python 03_linear_probe.py --run_id 8b69f950dceb4cb5b7b8f5a5a8e83cc2 --embedding_subset 32
