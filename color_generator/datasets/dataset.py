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
        "--subsample_fraction",
        type=float,
        default=None,
        help="If given, is used as the fraction of data to expose.",
    )
    return parser.parse_args()
