#!/usr/bin/env python
"""Script to run an experiment."""
import argparse
import json
import importlib


conf = {
    "dataset": "ColorsDataset",
    "model": "ColorModel",
    "network": "Distilbert",
    "device": "cpu",
    "dataset_args": {"batch_size": 1},
    "train_args": {"epochs": 2},
}


def run_experiment(experiment_config):
    """
    Run a training experiment.

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "datasets": "ColorsDataset",
            "model": "ColorModel",
            "network": "Distilbert",
            "train_args": {
                "batch_size": 32,
                "epochs": 2
            }
        }
    """
    print(f"Running experiment with config {experiment_config}")

    datasets_module = importlib.import_module("color_generator.datasets")

    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataloader_class_ = getattr(datasets_module, "DataLoaders")

    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)

    dataloaders = dataloader_class_(dataset)

    networks_module = importlib.import_module("color_generator.networks")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    network_args = experiment_config.get("network_args", {})
    network = network_fn_(**network_args)

    models_module = importlib.import_module("color_generator.models")
    model_class_ = getattr(models_module, experiment_config["model"])
    model = model_class_(
        dataloaders=dataloaders, network_fn=network, device=experiment_config["device"]
    )

    experiment_config["train_args"] = {**experiment_config.get("train_args", {})}
    model.fit(epochs=experiment_config["train_args"]["epochs"])

    score = model.evaluate()
    print(f"Test evaluation: {score}")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "ColorDataset", "model": "ColorModel", "network": "Distilbert"}\'',
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()

    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config)


if __name__ == "__main__":
    main()
