import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from src.classification.modules import MLP
from src.utils import (
    SketchClassifierDataset,
    get_train_val_records,
    MODELS_DIR,
    DATA
)


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
    TEST_FOLD = 5
    CAT_SUBSET = 10

    selected = set(
        [f for f in os.listdir(DATA / 'photo')[:CAT_SUBSET]]
    )

    train_records, val_records, test_records = get_train_val_records(Path(DATA / 'index.json'), selected, TEST_FOLD)

    train_dataset = SketchClassifierDataset(DATA, train_records)
    # val_dataset = SketchClassifierDataset(DATA, val_records)
    # test_dataset = SketchClassifierDataset(DATA, test_records)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f'Using device {device}')

    loss_fn = nn.CrossEntropyLoss()
    model = MLP().to(device)
    lr = 1e-3
    optim = torch.optim.SGD(model.parameters(), lr)
    epochs = 500

    avg_loss = train(
        model,
        train_loader,
        loss_fn,
        optim,
        device,
        epochs,
        100
    )

    torch.save(
        model.state_dict(),
        f'{MODELS_DIR}/mlp_classifier.pt'
    )

    print(f'AVG TRAIN LOSS: {avg_loss}')

if __name__ == '__main__':
    main()