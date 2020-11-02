#!/bin/bash
PYTHONPATH='.' pipenv run python3 training/run_experiment.py --save '{"dataset": "ColorsDataset", "model": "ColorModel", "network": "distilbert"}'