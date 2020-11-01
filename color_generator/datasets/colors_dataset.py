from color_generator.datasets.dataset import _download_raw_dataset, Dataset, _parse_args
from pathlib import Path
import numpy as np

import tensorflow as tf
from datasets import load_dataset
from transformers import DistilBertTokenizer


RAW_DATA_DIRNAME = Dataset.data_dirname() / "raw" / "colors.csv"


class ColorsDataset(Dataset):

    def __init__(self, test_size=0.15):
        self.test_size = test_size
        self.train = None
        self.test = None

    def load_or_generate_data(self):
        """Generate preprocessed data from a file"""
        self.train, self.test = _load_and_process_colors(self.test_size)



def _load_and_process_colors(test_size=0.15):
    """
    Preprocess dataset file:
        1. Load Tokenizer from hub
        2. Tokenize dataset in 1000 (dafault) batches
        3. Transform into tensorflow tensors
        4. Extract features
        5. Transform into tf.data.Dataset

    Returns: tf.data.Dataset
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    dataset = load_dataset('csv', data_files=RAW_DATA_DIRNAME)
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    encoded_dataset = train_test_dataset.map(lambda examples: tokenizer(examples['name'], max_length=32), batched=True)

    # features
    encoded_dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask'])
    train_features = {x: encoded_dataset['train'][x].to_tensor(default_value=0, shape=[None, 32]) for x in
                      ['input_ids', 'attention_mask']}
    test_features = {x: encoded_dataset['test'][x].to_tensor(default_value=0, shape=[None, 32]) for x in
                     ['input_ids', 'attention_mask']}

    # labels
    encoded_dataset.set_format(type='numpy', columns=['red', 'green', 'blue'])
    train_labels = np.column_stack([_norm(encoded_dataset['train'][:]['red']),
                                    _norm(encoded_dataset['train'][:]['green']),
                                    _norm(encoded_dataset['train'][:]['blue'])])

    test_labels = np.column_stack([_norm(encoded_dataset['test'][:]['red']),
                                   _norm(encoded_dataset['test'][:]['green']),
                                   _norm(encoded_dataset['test'][:]['blue'])])

    train_tfdataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels
                                                          )).batch(32)

    test_tfdataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels
                                                         )).batch(32)

    return train_tfdataset, test_tfdataset


def _norm(value):
    return value / 255.0


def main():
    """Load and preprocess colors datasets and print info."""
    args = _parse_args()
    dataset = ColorsDataset()
    dataset.load_or_generate_data()

    print(dataset)


if __name__ == "__main__":
    main()
