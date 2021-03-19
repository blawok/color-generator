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
|  |  ├─ dataloaders.py
│  │  └─ dataset.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  ├─ color_model.py
|  |  └─ early_stopping.py
│  ├─ networks
│  │  ├─ __init__.py
│  │  └─ distilbert.py
│  ├─ __init__.py
│  └─ color_predictor.py
├─ data
│  └─ raw
│     └─ colors.csv
├─ tasks
|  ├─ config.json
|  ├─ predict_color.sh
|  ├─ run_tests.sh
│  └─ train_color_predictor.sh
├─ tests
|  ├─ test_ColorModel.py
|  ├─ test_ColorsDataset.py
|  ├─ test_Dataloaders.py
|  └─ test_Distilbert.py
├─ training
│  └─ run_experiment.py
├─ weights
|  └─ ColorModel_ColorsDataset_Distilbert_weights.py
├─ .gitattributes
├─ .gitignore
├─ Pipfile
├─ Pipfile.lock
└─ README.md
```

## Setup
```zsh
pipenv sync
```
Add flag -d to also sync development packages.

## Training
 - Load and preprocess data
 - Fine-tune DistilBERT based on config.json
 - Evaluate on test set
 - Save weights to .pt (weights folder)
```zsh
chmod +x tasks/train_color_predictor.sh
tasks/train_color_predictor.sh \
    _PATH_TO_JSON_FILE_WITH_EXPERIMENT_CONFIG_
```

## Predict
 - Load model and its weights
 - Print RGB for specified color name and display its sample
 - To suppress plotting a sample use --no_plot flag
 - To predict on GPU use --gpu flag
``` zsh
chmod +x tasks/predict_color.sh
tasks/predict_color.sh \
    _PATH_TO_JSON_FILE_WITH_EXPERIMENT_CONFIG_ \
    _PATH_TO_FILE_WITH_MODEL_WEIGHTS_ \
    _COLOR_NAME_
```
