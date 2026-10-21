import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from src.classification.modules import MLP
from src.utils import SketchClassifierDataset, get_train_val_records


def train(
    model: MLP,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.SGD,
    device: torch.device,
    epochs: int = 100,
    log_every: int = 10,
) -> float:
    model.train()

    print(f'TRAIN STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    avg_loss = float('nan')
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_samples = 0

        for sketches, labels in loader:
            sketches = sketches.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            logits = model(sketches)

            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * sketches.size(0)
            epoch_samples += sketches.size(0)

        avg_loss = epoch_loss / epoch_samples
        if epoch % log_every == 0 or epoch == 1:
            print(f'epoch: {epoch}/{epochs} - loss: {avg_loss:.6f}')

    print(f'TRAIN FIINSHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    return avg_loss


def main():
    ROOT = Path('C:\\Users\\Utente\\Drake\\GENSKETCH\\data')
    TEST_FOLD = 5
    CAT_SUBSET = 10

    selected = set(
        [f for f in os.listdir(ROOT / 'photo')[:CAT_SUBSET]]
    )

    train_records, val_records, test_records = get_train_val_records(Path(ROOT / 'index.json'), selected, TEST_FOLD)

    train_dataset = SketchClassifierDataset(ROOT, train_records)
    val_dataset = SketchClassifierDataset(ROOT, val_records)
    test_dataset = SketchClassifierDataset(ROOT, test_records)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(selected)

if __name__ == '__main__':
    main()