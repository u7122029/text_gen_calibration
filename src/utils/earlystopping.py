from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

TEMP_DIR = "temp"


class EarlyStopping:
    def __init__(self, patience=4, tolerance=0.005, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.tolerance = tolerance
        self.counter = 0
        self.best_train_loss = torch.inf

    def __call__(self, train_loss, model):
        """

        @param train_loss:
        @param model:
        @return: True if the model should stop being trained, and False otherwise.
        """
        temp_dir = Path(TEMP_DIR)
        if not temp_dir.exists():
            temp_dir.mkdir(exist_ok=True, parents=True)

        if train_loss + self.tolerance >= self.best_train_loss:
            self.counter += 1
            if self.verbose:
                tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                return True
        else:
            if self.verbose:
                tqdm.write(f'Train loss decreased ({self.best_train_loss:.6f} --> {train_loss:.6f}).  Saving model.')
            self.best_train_loss = train_loss

            torch.save(model.state_dict(), str(temp_dir / "es_checkpoint.pt"))
            self.counter = 0
        return False

    def load_checkpoint(self, model: nn.Module):
        """
        Loads a model with the best performing state dict.
        @param model:
        @return:
        """
        if self.verbose:
            tqdm.write("Loading checkpoint model weights.")
        model.load_state_dict(torch.load(f"{TEMP_DIR}/es_checkpoint.pt"))


"""
# Initialize the early_stopping object
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(1, n_epochs + 1):
    train(...)
    val_loss = validate(...)

    # Early stopping
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))"""