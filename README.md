# Color generator

Using DistilBERT to predict color (rgb scale) given its text description

### Sample results after minimal training:

![Screenshot 2020-11-03 at 12 48 05](https://user-images.githubusercontent.com/41793223/97981657-ebb86500-1dd2-11eb-8803-43c7f76ebf91.png)

## Repository tree
```
├─ color_generator
│  ├─ datasets
│  │  ├─ __init__.py
│  │  ├─ colors_dataset.py
│  │  └─ dataset.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  └─ color_model.py
│  ├─ networks
│  │  ├─ __init__.py
│  │  └─ distilbert.py
│  ├─ __init__.py
│  └─ color_predictor.py
├─ data
│  └─ raw
│     └─ colors.csv
├─ notebooks
│  └─ color_generator.ipynb
├─ tasks
│  └─ train_color_predictor.sh
├─ training
│  ├─ run_experiment.py
│  └─ util.py
├─ .gitattributes
├─ .gitignore
├─ Pipfile
├─ Pipfile.lock
├─ README.md
├─ requirements-dev.in
├─ requirements-dev.txt
├─ requirements.in
└─ requirements.txt
```

## Setup
```zsh
pip install -r requirements.txt -r requirements-dev.txt
pipenv lock
```

## Training
 - Load and preprocess data
 - Fine-tune DistilBERT on batch size = 32 for two epochs (keras and transformers)
 - Evaluate on test set
 - Save weights to .h5 (weights folder)
```zsh
PYTHONPATH='.' pipenv run python3 training/run_experiment.py --save '{"dataset": "ColorsDataset", "model": "ColorModel", "network": "distilbert"}'      
```

To modify batch size and number of epochs add train_args
```
PYTHONPATH='.' pipenv run python3 training/run_experiment.py --save '{"dataset": "ColorsDataset", "model": "ColorModel", "network": "distilbert", "train_args":{"batch_size": 32,"epochs": 2}}'      
```
