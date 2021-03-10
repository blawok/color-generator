#!/usr/bin/env python3

import os
from flask import Flask, request, jsonify
from color_generator.models.color_model import ColorModel

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))


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
def predict():
    predictor = ColorModel()
    predictor.load_weights()
    input_text = request.form.get('description')
    pred = predictor.predict_on_text(input_text)
    print("RGB for desired color is {}".format(pred))
    answer = "RGB for desired color is {}".format(pred)


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
