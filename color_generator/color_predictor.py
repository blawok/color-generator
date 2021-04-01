"""ColorPredictor class"""
import matplotlib.pyplot as plt
import importlib
import argparse
import json


class ColorPredictor:
    """Recognize color based on text, returns color in RGB"""

    def __init__(self, model, path_to_weights):
        self.model = model
        self.model.load_weights(path_to_weights)

    def predict_color(self, input_text, plot=True):
        """Predict on a single text."""
        rgb_list = self.model.predict_on_text(input_text)
        # print(f"RGB={rgb_list}")
        if plot:
            plt.figure(figsize=(2, 2))
            plt.imshow([[rgb_list]], interpolation="none")
            plt.axis("off")
            plt.show()
        return rgb_list


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Path to experiment JSON like: \'{"dataset": "ColorDataset", "model": "ColorModel", "network": "Distilbert"}\''
    )
    parser.add_argument("weights", type=str, help="Path to file with weights")
    parser.add_argument(
        "color",
        nargs="+",
        type=str,
        help="Colorname to predict"
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Whether to plot predicted color"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether to use GPU"
    )
    args = parser.parse_args()

    args.device = "cuda" if args.gpu else "cpu"

    return args


def main():

    args = _parse_args()
    with open(args.experiment_config, "r") as f:
        config = f.read()
    experiment_config = json.loads(config)

    networks_module = importlib.import_module("color_generator.networks")
    network_class_ = getattr(networks_module, experiment_config["network"]["name"])
    network_args = experiment_config["network"].get("network_args", {})
    network = network_class_(**network_args)

    models_module = importlib.import_module("color_generator.models")
    model_class_ = getattr(models_module, experiment_config["model"])
    model = model_class_(network_fn=network, device=args.device)

    predictor = ColorPredictor(model, args.weights)
    predictor.predict_color(" ".join(args.color), not args.no_plot)


if __name__ == "__main__":
    main()
