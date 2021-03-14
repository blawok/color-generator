"""Dataset class to be extended by datasets-specific classes."""
from torch.utils.data import Dataset
from pathlib import Path
import argparse


class DefaultDataset(Dataset):
    """Simple abstract class for datasets."""

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    def load_and_generate_data(self):
        pass


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="If given, is used as the fraction of data to test on.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.10,
        help="If given, is used as the fraction of data to validate on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="If given, is used as the batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="If given, is used as the number of workers to use.",
    )
    return parser.parse_args()
