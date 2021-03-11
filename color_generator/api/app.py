#!/usr/bin/env python3

import os, random
from flask import Flask, request, jsonify, render_template
from color_generator.models.color_model import ColorModel

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

example_colors = ["Ferrari Red",
                  "Ugly Yellow",
                  "British Racing Green",
                  "Salmon",
                  "Kowalski's Hair Color"]


def plot_rgb(rgb):
    # data = [[rgb]]
    plt.figure(figsize=(2,2))
    plt.imshow(rgb, interpolation='nearest')
    plt.show()


@app.route("/")
def index():
    """Provide simple health check route."""
    return "It works!"


@app.route("/show_color", methods=["GET", "POST"])
def send_color():
    # handle the POST request
    if request.method == 'POST':
        print ("POST")
        color_desc = request.form.get('color_desc')
        color = predict(color_desc)
        return render_template("submitted.html", color=color)
    print ("GET")
    color = "rgb(255, 0, 0)"
    example_color = draw_color()
    return render_template("index.html", color=color, example_color=example_color)


def predict(color_desc):
    if color_desc is None:
        print("Color description was not valid.")
        return None
    input_text = color_desc
    predictor = ColorModel()
    predictor.load_weights()
    pred = predictor.predict_on_text(input_text.lower())
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
