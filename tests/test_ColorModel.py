import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from color_generator.datasets import ColorsDataset, DataLoaders
from color_generator.networks import Distilbert
from color_generator.models import ColorModel


class AnyModel(ColorModel):
    def __init__(self, dataloaders, network):
        super().__init__(dataloaders, network, "cpu")

    def optimizer(self):
        return optim.AdamW(self.network.parameters(), lr=3e-4)

    def fit_one_batch(self, epochs=50):

        criterion = self.criterion()
        cs = nn.CosineSimilarity(dim=1)
        batch = {
            "input_ids": torch.tensor(
                [
                    [101, 8215, 4135, 5358, 3123, 102, 0, 0, 0],
                    [101, 9388, 4115, 102, 0, 0, 0, 0, 0],
                    [101, 2350, 2630, 102, 0, 0, 0, 0, 0],
                    [101, 6187, 20808, 102, 0, 0, 0, 0, 0],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
            "target": torch.tensor(
                [
                    [0.6510, 0.4549, 0.4706],
                    [0.1647, 0.0784, 0.0549],
                    [0.2627, 0.5961, 0.7843],
                    [0.4118, 0.3412, 0.2784],
                ]
            ),
        }
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        targets = batch["target"].to(self.device)

        for _ in range(epochs):
            self.network.train()
            running_loss = 0.0
            running_cs = 0.0

            outputs = self.network(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            self.optimizer().zero_grad()
            loss.backward()
            self.optimizer().step()

            running_loss += loss.item()
            running_cs += cs(targets, outputs).mean().item()

        return running_loss, running_cs


@pytest.fixture
def model():
    torch.manual_seed(2137)
    network = Distilbert()
    model = AnyModel(DataLoaders(ColorsDataset()), network)
    return model


def test_fit(model):
    loss, cos_sim = model.fit_one_batch()
    assert loss < 0.02
    assert cos_sim > 0.95


def test_evaluate(model):
    _, cos_sim = model.evaluate(model._dataloaders.test_loader)
    assert 0 < cos_sim <= 1


def test_predict_on_text(model):
    rgb_list = model.predict_on_text("red")
    assert isinstance(rgb_list, list)
    assert len(rgb_list) == 3
    assert min(rgb_list) >= 0
    assert max(rgb_list) <= 255
