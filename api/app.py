#!/usr/bin/env python3

import os, random, importlib
from flask import Flask, request, render_template
# from color_generator.color_predictor import ColorPredictor
# from color_generator.datasets import ColorsDataset, DataLoaders

# from memory_profiler import profile
import torch

app = Flask(__name__)

EXAMPLE_COLORS = ["Ferrari Red",
                  "Ugly Yellow",
                  "British Racing Green",
                  "Salmon",
                  "Kowalski's Hair Color"]


@app.route("/test")
def index():
    """Provide simple health check route."""
    return """<h1> It works! </h1>
    <form>
      <button formaction="./">Now let me invent some colors</button>
    </form>"""


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
    example = random.choice(EXAMPLE_COLORS)
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


# @profile
def load_all():
    model = torch.load('./model_save_test.pt')
    return model


if __name__ == "__main__":
    model = load_all()
    main()
