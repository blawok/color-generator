import pandas as pd
import torch
from color_generator.datasets.dataset import DefaultDataset, _parse_args
from transformers import DistilBertTokenizer


class ColorsDataset(DefaultDataset):
    def __init__(
        self, path, val_size=0.1, test_size=0.15, batch_size=32, num_workers=4
    ):
        self.path = path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.test = None
        self.val = None
        self._color_names = None
        self._inputs = None
        self._targets = None

        self.load_and_generate_data(self.path)

    def load_and_generate_data(self, path):
        """Generate preprocessed data from a file"""
        self._color_names, self._inputs, self._targets = _load_and_process_colors(path)

    def __getitem__(self, index):
        """Get item"""
        item = dict.fromkeys(self._inputs, {})
        item["input_ids"] = self._inputs["input_ids"][index]
        item["attention_mask"] = self._inputs["attention_mask"][index]
        item["target"] = _norm(self._targets[index])

        return item

    def __len__(self):
        return len(self._color_names)


def _norm(rgb_list):
    return torch.tensor([value / 255.0 for value in rgb_list])


def _load_and_process_colors(path):
    def rgb_to_list(x):
        return [x["red"], x["green"], x["blue"]]

    path_to_data = ColorsDataset.data_dirname() / path
    dataset = pd.read_csv(path_to_data)

    names = dataset["name"].tolist()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = tokenizer(names, padding=True, return_tensors="pt")
    rgb = dataset.apply(rgb_to_list, axis=1).tolist()

    return names, tokenized, rgb


def main():
    """
    Load and preprocess colors.
    """

    args = _parse_args()

    # dataset
    dataset = ColorsDataset(
        path=args.path,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
