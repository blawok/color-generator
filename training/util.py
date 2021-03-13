"""Function to train a model."""
from time import time

from tensorflow.keras.callbacks import EarlyStopping, Callback


from color_generator.preprocessing.dataset import Dataset
from color_generator.models.base import Model
import tensorflow.keras.backend as K


EARLY_STOPPING = True


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, use_wandb: bool = False) -> Model:
    """Train model."""
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto")
        callbacks.append(early_stopping)

    model.network.summary()

    t = time()
    _history = model.fit(dataset=dataset, epochs=epochs, callbacks=callbacks)
    print("Training took {:2f} s".format(time() - t))

    return model
