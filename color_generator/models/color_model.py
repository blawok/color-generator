"""ColorModel class."""
from os import truncate
from color_generator.datasets.colors_dataset import ColorsDataset

from color_generator.models.base import Model
from color_generator.datasets.dataloaders import DataLoaders
from color_generator.networks.distilbert import Distilbert

from transformers import DistilBertTokenizer


class ColorModel(Model):
    """ColorModel works on datasets providing single text sequence up to 32 tokens, with RGB labels."""

    def __init__(self, dataloaders, network_fn, device):
        super().__init__(dataloaders, network_fn, device)

    def predict_on_text(self, input_text):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        input_text = tokenizer(input_text, truncation=True, return_tensors="pt")
        self.load_weights()
        self.network.eval()
        predictions = (
            (self.network(input_text["input_ids"], input_text["attention_mask"]))
            .squeeze()
            .tolist()
        )

        return self._unnorm(predictions)

    @staticmethod
    def _unnorm(predictions):
        return [int(x) for x in predictions * 255]
