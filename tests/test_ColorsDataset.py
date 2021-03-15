import pytest
from torch import Tensor
from color_generator.datasets import ColorsDataset


@pytest.fixture
def dataset():
    return ColorsDataset()


def test_dataset_len(dataset):
    assert len(dataset) > 0


def test_dataset_structure(dataset):
    test_item = dataset[0]
    assert len(test_item) == 3
    assert isinstance(test_item["input_ids"], Tensor)
    assert isinstance(test_item["attention_mask"], Tensor)
    assert isinstance(test_item["target"], Tensor)
