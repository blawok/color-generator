"""Script to run an experiment."""
import argparse
import json
import importlib
import time

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


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
            "network": {
                "args": {
                    "architecture": "distilbert-base-uncased",
                    "freeze": true
                }
            },
            "train_args": {
                "epochs": 2
            }
        }
    """
    print(f"Running experiment with config {experiment_config}")

    # dataset
    datasets_module = importlib.import_module("color_generator.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    try:
        architecture = experiment_config["network"]["args"]["architecture"]
        dataset_args["architecture"] = architecture
    except KeyError:
        pass
    dataset = dataset_class_(**dataset_args)

    # network
    networks_module = importlib.import_module("color_generator.networks")
    network_class_ = getattr(networks_module, experiment_config["network"]["name"])
    network_args = experiment_config["network"].get("args", {})
    network = network_class_(**network_args)

    # model
    models_module = importlib.import_module("color_generator.models")
    model_class_ = getattr(models_module, experiment_config["model"])
    model_args = experiment_config.get("train_args", {})
    model = model_class_(network, experiment_config["device"], **model_args)

    t = time.monotonic()
    model.fit(dataset=dataset)
    duration = int(time.monotonic() - t)
    print(
        f"Training took {duration//86400} days "
        f"{duration % 86400 // 3600} hours "
        f"{duration % 86400 % 3600 // 60} minutes.\n"
    )

    _, score = model.evaluate(model._dataloaders.test_loader)
    print(f"Test evaluation (cosine similarity): {score:.5f}")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Path to experiment JSON like: \'{"dataset": "ColorDataset", "model": "ColorModel", "network": "Distilbert"}\'',
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()

    with open(args.experiment_config, "r") as f:
        config = f.read()
    experiment_config = json.loads(config)
    run_experiment(experiment_config)


if __name__ == "__main__":
    main()
