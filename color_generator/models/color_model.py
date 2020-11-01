"""ColorModel class."""
from typing import Callable, Dict, Tuple
import numpy as np

from color_generator.models.base import Model
from color_generator.datasets.colors_dataset import ColorsDataset
from color_generator.networks.distilbert import mlp

from transformers import DistilBertTokenizer
from tensorflow import convert_to_tensor


class ColorModel(Model):
    """ColorModel works on datasets providing single text sequence up to 32 tokens, with RGB labels."""

    def __init__(
        self,
        dataset_cls: type = ColorsDataset,
        network_fn: Callable = mlp,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def predict_on_text(self, input_text: str) -> np.array:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        input_text = tokenizer(input_text, max_length=32)
        predictions = self.network.predict({
            'input_ids': convert_to_tensor([input_text['input_ids']], np.int32),
            'attention_mask': convert_to_tensor([input_text['attention_mask']], np.int32)
        })

        return predictions
