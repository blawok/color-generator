"""Model class, to be extended by specific types of models."""
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
import torch


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self, dataloaders, network_fn, device="cpu"):

        self.device = device
        self._dataloaders = dataloaders
        self.dataset = self._dataloaders._dataset
        self.network = network_fn.to(self.device)

        self.name = f"{self.__class__.__name__}_{self.dataset.__class__.__name__}_{self.network.__class__.__name__}"

    @property
    def weights_filename(self):
        p = Path(__file__).resolve().parents[2] / "weights"
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"{self.name}_weights.pt")

    def fit(self, epochs=10):

        criterion = self.criterion()

        for epoch in range(epochs):
            self.network.train()
            running_loss = 0.0
            for i, batch in enumerate(self._dataloaders.train_loader):
                # forward and backward propagation
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)
                outputs = self.network(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                self.optimizer().zero_grad()
                loss.backward()
                self.optimizer().step()

                # save results
                running_loss += loss.item()
                if i > 0 and i % 100 == 0:
                    stats = (
                        f"Epoch: {epoch+1}/{epochs}, batch: {i}/{len(self._dataloaders.train_loader)}, "
                        f"train_loss: {running_loss/i:.5f}"
                    )
                    print(stats, flush=True)
                    with open("stats.log", "a") as f:
                        print(stats, file=f)

            # calculate loss and accuracy on validation dataset
            with torch.no_grad():
                val_loss = self.validate()
            stats = f"Epoch: {epoch+1}/{epochs}, train_loss: {running_loss/i:.5f}, valid_loss: {val_loss:.5f}"
            print(stats)
            with open("stats.log", "a") as f:
                print(stats, file=f)

            # save after each epoch
            self.save_weights()

            # check for early stopping
            self.early_stopping(val_loss, self.network)
            if self.early_stopping.early_stop:
                print("Early stopping.")
                break

        self.load_weights(early_stopping=True)
        self.save_weights()
        print("\nFinished training")

    def criterion(self):
        return nn.MSELoss(reduction="mean").to(self.device)

    def optimizer(self):
        return optim.AdamW(self.network.parameters())

    def load_weights(self, early_stopping=False):
        if early_stopping:
            f = "early_stopping_checkpoint.pt"
        else:
            f = self.weights_filename
        self.network.load_state_dict(torch.load(f))

    def save_weights(self):
        torch.save(self.network.state_dict(), self.weights_filename)

    def validate(self):
        pass

    def evaluate(self):
        pass

    def early_stopping(self):
        pass

