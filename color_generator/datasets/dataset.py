"""Dataset class to be extended by datasets-specific classes."""
from pathlib import Path
import argparse
import os


class Dataset:
    """Simple abstract class for datasets."""

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"

    def load_or_generate_data(self):
        pass


def _download_raw_dataset(metadata):
    pass


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsample_fraction", type=float, default=None, help="If given, is used as the fraction of data to expose.",
    )
    return parser.parse_args()
