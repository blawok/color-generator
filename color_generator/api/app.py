#!/usr/bin/env python3

import os, random, importlib
from flask import Flask, request, render_template
from color_generator.color_predictor import ColorPredictor
from color_generator.datasets import ColorsDataset, DataLoaders

app = Flask(__name__)

example_colors = ["Ferrari Red",
                  "Ugly Yellow",
                  "British Racing Green",
                  "Salmon",
                  "Kowalski's Hair Color"]

dataset_class_ = ColorsDataset
dataloader_class_ = DataLoaders

dataset_args = {}
dataset = dataset_class_(**dataset_args)
dataloaders = dataloader_class_(dataset)

networks_module = importlib.import_module("color_generator.networks")
network_fn_ = getattr(networks_module, 'Distilbert')
network_args = {}
network = network_fn_(**network_args)

models_module = importlib.import_module("color_generator.models")
model_class_ = getattr(models_module, 'ColorModel')

model = model_class_(
    dataloaders=dataloaders, network_fn=network, device='cpu'
)
model = ColorPredictor(model)


@app.route("/test")
def index():
    """Provide simple health check route."""
    return "It works!"


@app.route("/", methods=["GET", "POST"])
def send_color():
    if request.method == 'POST':
        color_desc = request.form.get('color_desc')
        prediction = predict(color_desc)
        color = pred_to_rgb(prediction)
        darker_col = darker_color(prediction)
        hex_code = pred_to_hex(prediction)
        print (color_desc, color, darker_col)
        return render_template("submitted.html",
                               color=color,
                               darker_col=darker_col,
                               hex_code=hex_code,
                               color_desc=color_desc)
    example_color = draw_color()
    return render_template("index.html", example_color=example_color)


def predict(color_desc):
    input_text = color_desc.lower()
    pred = model.predict_color(input_text, plot=False)
    return pred


def draw_color():
    example = random.choice(example_colors)
    return example


def pred_to_rgb(pred):
    color = f"rgb({pred[0]}, {pred[1]}, {pred[2]})"
    return color


def pred_to_hex(pred):
    hex_code = '#%02x%02x%02x' % (pred[0], pred[1], pred[2])
    return hex_code


def darker_color(pred):
    color = f"rgb({int(pred[0]*0.90)}, {int(pred[1]*0.90)}, {int(pred[2]*0.90)})"
    return color


def main():
    """Run the app."""
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()