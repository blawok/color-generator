"""Model class, to be extended by specific types of models."""
# pylint: disable=missing-function-docstring
from pathlib import Path
from typing import Callable, Dict, Optional

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam


DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(**network_args)
        self.network.summary()

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self, dataset, batch_size: int = 32, epochs: int = 10, callbacks: list = None,
    ):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer())

        self.network.fit(
            dataset,
            epochs=epochs,
            callbacks=callbacks,
            # validation_data=test_sequence,
            use_multiprocessing=False,
            workers=1,
            shuffle=True,
        )

    def loss(self):
        # pylint: disable=no-self-use
        return "mse"

    def optimizer(self):  # pylint: disable=no-self-use
        return Adam(learning_rate=0.005)

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)

    def evaluate(self, dataset):
        return self.network.evaluate(dataset)
