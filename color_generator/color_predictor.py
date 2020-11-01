"""ColorPredictor class"""
from typing import Tuple, Union

import numpy as np

from color_generator.models import ColorModel


class ColorPredictor:
    """Recognize color based on text, returns color in RGB"""

    def __init__(self):
        self.model = ColorModel()
        self.model.load_weights()

    def predict(self, input_text: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single text."""
        return self.model.predict_on_text(input_text)

    def evaluate(self, dataset):
        """Evaluate on a datasets."""
        return self.model.evaluate(dataset)
