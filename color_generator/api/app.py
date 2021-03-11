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
        color = predict(color_desc)
        print ("POST", color_desc, color)
        return render_template("submitted.html", color=color)
    print ("GET")
    color = "rgb(255, 0, 0)"
    example_color = draw_color()
    return render_template("index.html", color=color, example_color=example_color)


def predict(color_desc):
    input_text = color_desc.lower()
    predictor = ColorPredictor()
    pred = predictor.predict(input_text)
    print(input_text, pred)
    pred = [int(x*255) for x in pred[0]]
    color = f"rgb({pred[0]}, {pred[1]}, {pred[2]})"
    return color


def draw_color():
    example = random.choice(example_colors)
    return example


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
