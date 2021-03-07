import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler


class ColorsDataLoaders:
    def __init__(self, dataset):

        self._dataset = dataset
        self.val_size = self._dataset.val_size
        self.test_size = self._dataset.test_size
        self.batch_size = self._dataset.batch_size
        self.num_workers = self._dataset.num_workers
        self._train_indexes = None
        self._valid_indexes = None
        self._test_indexes = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self._split_indices()
        self._split_dataset()
        self._make_dataloaders()

    def _split_indices(self):
        # randomly split to train, valid and test set
        indexes = list(range(len(self._dataset)))
        split_point = int(
            np.floor((1 - (self.val_size + self.test_size)) * len(self._dataset))
        )
        np.random.seed(2137)
        np.random.shuffle(indexes)
        self._train_indexes, rest_indexes = indexes[:split_point], indexes[split_point:]
        val_test_split_point = int(
            np.floor(
                (self.val_size / (self.val_size + self.test_size)) * len(rest_indexes)
            )
        )
        self._valid_indexes, self._test_indexes = (
            rest_indexes[:val_test_split_point],
            rest_indexes[val_test_split_point:],
        )

    def _split_dataset(self):
        # make dataset samplers
        train_sampler = SubsetRandomSampler(self._train_indexes)
        valid_sampler = SubsetRandomSampler(self._valid_indexes)
        test_sampler = SubsetRandomSampler(self._test_indexes)

        # datasets: train, valid and test
        self.train_dataset = Subset(
            dataset=self._dataset, indices=train_sampler.indices
        )
        self.valid_dataset = Subset(
            dataset=self._dataset, indices=valid_sampler.indices
        )
        self.test_dataset = Subset(dataset=self._dataset, indices=test_sampler.indices)

    def _make_dataloaders(self):
        # dataloaders
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return self.train_loader, self.valid_loader, self.test_loader
