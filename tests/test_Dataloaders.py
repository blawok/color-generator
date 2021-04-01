import pytest
from torch import Tensor
from color_generator.datasets import ColorsDataset, DataLoaders

path_to_dataset = "raw/colors.csv"


@pytest.fixture
def dataloader():
    return DataLoaders(ColorsDataset(path=path_to_dataset))


def test_train_loader(dataloader):
    batch = next(iter(dataloader.train_loader))
    assert len(batch["input_ids"].shape) == 2
    assert len(batch["attention_mask"].shape) == 2
    assert len(batch["target"].shape) == 2
    assert isinstance(batch["input_ids"], Tensor)
    assert isinstance(batch["attention_mask"], Tensor)
    assert isinstance(batch["target"], Tensor)


def test_valid_loader(dataloader):
    batch = next(iter(dataloader.valid_loader))
    assert len(batch["input_ids"].shape) == 2
    assert len(batch["attention_mask"].shape) == 2
    assert len(batch["target"].shape) == 2
    assert isinstance(batch["input_ids"], Tensor)
    assert isinstance(batch["attention_mask"], Tensor)
    assert isinstance(batch["target"], Tensor)


def test_test_loader(dataloader):
    batch = next(iter(dataloader.test_loader))
    assert len(batch["input_ids"].shape) == 2
    assert len(batch["attention_mask"].shape) == 2
    assert len(batch["target"].shape) == 2
    assert isinstance(batch["input_ids"], Tensor)
    assert isinstance(batch["attention_mask"], Tensor)
    assert isinstance(batch["target"], Tensor)
