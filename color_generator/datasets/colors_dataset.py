import numpy as np
import pandas as pd
from color_generator.datasets.dataset import DefaultDataset, _parse_args
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset, DataLoader
from transformers import DistilBertTokenizer


class ColorsDataset(DefaultDataset):
    def __init__(self, test_size, val_size, batch_size, num_workers):
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.test = None
        self.val = None
        self._tokenizer = None
        self._inputs = None
        self._targets = None

    def load_and_generate_data(self):
        """Generate preprocessed data from a file"""
        self._tokenizer, self._inputs, self._targets = _load_and_process_colors()

    def __getitem__(self, index):
        """Get item"""
        item = {
            k: torch.tensor(v)
            for k, v in self._tokenizer(
                self._inputs[index], truncation=True, max_length=32
            ).items()
        }
        item["target"] = torch.tensor(_norm(self._targets[index]))
        return item

    def __len__(self):
        return len(self._inputs)


def _norm(rgb_list):
    return [value / 255.0 for value in rgb_list]


def _load_and_process_colors():
    def rgb_to_list(x):
        return [x["red"], x["green"], x["blue"]]

    path_to_data = ColorsDataset.data_dirname() / "raw/colors.csv"
    dataset = pd.read_csv(path_to_data)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    names = dataset["name"].tolist()
    rgb = dataset.apply(rgb_to_list, axis=1).tolist()

    return tokenizer, names, rgb


def main():
    """
    Load and preprocess colors.
    Make dataloaders.
    """
    args = _parse_args()

    # dataset
    dataset = ColorsDataset(
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataset.load_and_generate_data()

    # split indices: train, val and test set
    indexes = list(range(len(dataset)))
    split_point = int(
        np.floor((1 - (dataset.val_size + dataset.test_size)) * len(dataset))
    )
    np.random.seed(2137)
    np.random.shuffle(indexes)
    train_indexes, rest_indexes = indexes[:split_point], indexes[split_point:]
    val_test_split_point = int(
        np.floor(
            (dataset.val_size / (dataset.val_size + dataset.test_size))
            * len(rest_indexes)
        )
    )
    valid_indexes, test_indexes = (
        rest_indexes[:val_test_split_point],
        rest_indexes[val_test_split_point:],
    )

    # make dataset samplers
    train_sampler = SubsetRandomSampler(train_indexes)
    valid_sampler = SubsetRandomSampler(valid_indexes)
    test_sampler = SubsetRandomSampler(test_indexes)

    # datasets: train, valid and test
    train_dataset = Subset(dataset=dataset, indices=train_sampler.indices)
    valid_dataset = Subset(dataset=dataset, indices=valid_sampler.indices)
    test_dataset = Subset(dataset=dataset, indices=test_sampler.indices)

    # dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=dataset.batch_size,
        shuffle=True,
        num_workers=dataset.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=dataset.batch_size,
        shuffle=True,
        num_workers=dataset.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=dataset.batch_size,
        shuffle=False,
        num_workers=dataset.num_workers,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    main()
