import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from .modules import MLP
from ..utils import (
    SketchClassifierDataset,
    get_train_val_records,
    write_log,
    MODELS_DIR,
    DATA,
    LOG_WARN,
)
FILE_NAME = Path(__file__).stem

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
    write_log(f'TRAIN STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}', FILE_NAME)
    avg_loss = float('nan')
    last_epoch_avg_loss = float('nan')
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
        if avg_loss >= last_epoch_avg_loss:
            write_log(f'Loss function raised on epoch {epoch}, delta: {(avg_loss - last_epoch_avg_loss):.6f}', FILE_NAME, lvl=LOG_WARN)
        
        if epoch % log_every == 0 or epoch == 1:
            print(f'epoch: {epoch}/{epochs} - loss: {avg_loss:.6f}')
            write_log(f'epoch: {epoch}/{epochs} - loss: {avg_loss:.6f}', FILE_NAME)

    print(f'TRAIN FINSHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    write_log(f'TRAIN FINSHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}', FILE_NAME)
    return avg_loss


def main():
    TEST_FOLD = 4
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
    epochs = 300

    write_log(f'Starting training on {len(train_records)} records', FILE_NAME)
    write_log(f'Current configuration: LOSS: {loss_fn.__str__()}, LR: {lr}, OPTIM: {optim.__str__()}, EPOCHS: {epochs}', FILE_NAME)
    avg_loss = train( # prima prova di train su 500 epoche avg_loss: ~0.56
        model,
        train_loader,
        loss_fn,
        optim,
        device,
        epochs,
        epochs//10
    )

    torch.save(
        model.state_dict(),
        f'{MODELS_DIR}/mlp_classifier.pt'
    )

    print(f'AVG TRAIN LOSS: {avg_loss}')

if __name__ == '__main__':
    main()