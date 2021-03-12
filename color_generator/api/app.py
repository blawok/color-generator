#!/usr/bin/env python3

import os, random
from flask import Flask, request, jsonify, render_template
from color_generator.models.color_model import ColorModel
from color_generator.color_predictor import ColorPredictor

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

example_colors = ["Ferrari Red",
                  "Ugly Yellow",
                  "British Racing Green",
                  "Salmon",
                  "Kowalski's Hair Color"]


@app.route("/")
def index():
    """Provide simple health check route."""
    return "It works!"


@app.route("/show_color", methods=["GET", "POST"])
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
    predictor = ColorPredictor()
    pred = predictor.predict(input_text)
    print(input_text, pred)
    pred = [int(x*255) for x in pred[0]]
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
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
