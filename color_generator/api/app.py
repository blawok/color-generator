#!/usr/bin/env python3

import os

from flask import Flask, request, jsonify

from color_generator.models.color_model import ColorModel

app = Flask(__name__)


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    predictor = ColorModel()
    predictor.load_weights()
    input_text = request.form.get('description')
    pred = predictor.predict_on_text(input_text)
    print("RGB for desired color is {}".format(pred))
    answer = "RGB for desired color is {}".format(pred)

    return answer


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=True)  # nosec


if __name__ == "__main__":
    main()
