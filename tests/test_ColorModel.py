import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from color_generator.datasets import ColorsDataset
from color_generator.networks import Distilbert
from color_generator.models import ColorModel

path_to_dataset = "raw/colors.csv"


@pytest.fixture
def model():
    torch.manual_seed(2137)
    network = Distilbert()
    model = ColorModel(network, "cpu")
    return model


def test_fit_evaluate(model):
    loss, cos_sim = model.fit(
        ColorsDataset(path=path_to_dataset), epochs=50, testing=True
    )
    _, cos_sim = model.evaluate(model._dataloaders.test_loader)
    assert loss < 0.1
    assert cos_sim > 0.9
    assert 0 < cos_sim <= 1


def test_predict_on_text(model):
    rgb_list = model.predict_on_text("red")
    assert isinstance(rgb_list, list)
    assert len(rgb_list) == 3
    assert min(rgb_list) >= 0
    assert max(rgb_list) <= 255
