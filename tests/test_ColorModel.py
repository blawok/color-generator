import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from color_generator.datasets import ColorsDataset, DataLoaders
from color_generator.networks import Distilbert
from color_generator.models import ColorModel

path_to_dataset = "raw/colors.csv"


@pytest.fixture
def model():
    torch.manual_seed(2137)
    network = Distilbert()
    model = ColorModel(DataLoaders(ColorsDataset(path=path_to_dataset)), network, "cpu")
    return model


def test_fit(model):
    loss, cos_sim = model.fit(testing=True)
    assert loss < 0.1
    assert cos_sim > 0.9


def test_evaluate(model):
    _, cos_sim = model.evaluate(model._dataloaders.test_loader)
    assert 0 < cos_sim <= 1


def test_predict_on_text(model):
    rgb_list = model.predict_on_text("red")
    assert isinstance(rgb_list, list)
    assert len(rgb_list) == 3
    assert min(rgb_list) >= 0
    assert max(rgb_list) <= 255
