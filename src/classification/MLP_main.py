import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from .modules import MLP
from ..utils import (
    SketchClassifierDataset,
    get_split_records,
    write_log,
    MODELS_DIR,
    DATA,
    LOG_WARN,
)
FILE_NAME = Path(__file__).stem

def train_one_epoch(
    epoch: int,
    model: MLP,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    print(f'TRAIN ON EPOCH {epoch} STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    write_log(f'TRAIN ON EPOCH {epoch} STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}', FILE_NAME)
    
    epoch_avg_loss = float('nan')
    epoch_accuracy = float('nan')
    epoch_loss = 0.0
    epoch_samples = 0
    epoch_corrects = 0
    epoch_incorrects = 0

    # FOREACH MINIBATCH
    for sketches, labels in loader:
        sketches = sketches.to(device)
        labels = labels.to(device)

        optim.zero_grad()
        logits = model(sketches)

        predicted_labels = logits.argmax(dim=1)
        
        loss = loss_fn(logits, labels)
        loss.backward()
        optim.step()

        epoch_loss += loss.item() * sketches.size(0)
        epoch_corrects += (predicted_labels == labels).sum().item()
        epoch_incorrects += (predicted_labels != labels).sum().item()
        epoch_samples += sketches.size(0)

    epoch_avg_loss = epoch_loss / epoch_samples
    epoch_accuracy = epoch_corrects / epoch_samples

    print(f'TRAIN ON EPOCH {epoch} FINSHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    write_log(f'TRAIN ON EPOCH {epoch} FINSHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}', FILE_NAME)
    return epoch_avg_loss, epoch_accuracy


def validate(
    model,
    loader,
    loss_fn,
    device
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.inference_mode():
        for sketches, labels in loader:
            sketches = sketches.to(device)
            labels = labels.to(device)

            logits = model(sketches)

            loss = loss_fn(logits, labels)

            batch_size = logits.size(0)
            total_loss += loss.item() * batch_size

            predicted_labels = logits.argmax(dim=1)

            correct += (predicted_labels == labels).sum().item()

            total_samples += batch_size


    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def main():
    CAT_SUBSET = 10

    selected = set(
        [f for f in sorted(os.listdir(DATA / 'photo')[:CAT_SUBSET])]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device {device}')
    loss_fn = nn.CrossEntropyLoss()
    lr = 1e-3
    epochs = 100

    write_log(f'Current configuration: DEVICE: {device} LOSS: {loss_fn.__str__()}, EPOCHS: {epochs}', FILE_NAME)

    fold_results = []
    for test_fold in range(5):
        model = MLP().to(device)
        # optim = torch.optim.SGD(model.parameters(), lr)
        optim = torch.optim.Adam(
            model.parameters(),
            lr,
            weight_decay=0
        )
        # optim = torch.optim.AdamW(
        #     model.parameters(),
        #     lr,
        #     weight_decay=1e-4
        # )

        train_records, val_records, test_records = get_split_records(Path(DATA / 'index.json'), selected, test_fold)

        train_dataset = SketchClassifierDataset(DATA, train_records)
        val_dataset = SketchClassifierDataset(DATA, val_records)
        test_dataset = SketchClassifierDataset(DATA, test_records)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        best_val_loss = float('inf')
        checkpoint_path = f'{MODELS_DIR}/mlp_fold_{test_fold}_best.pt'

        for epoch in range(1, epochs + 1):
            train_one_epoch(
                epoch,
                model,
                train_loader,
                loss_fn,
                optim,
                device,
            )

            val_loss, val_acc = validate(
                model,
                val_loader,
                loss_fn,
                device
            )

            print(
                f'VALIDATION:\n'
                f'\tLOSS: {val_loss}\n'
                f'\tACC: {val_acc}\n'
            )
            write_log(
                f'VALIDATION:\n'
                f'\tLOSS: {val_loss}\n'
                f'\tACC: {val_acc}\n',
                FILE_NAME
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'fold': test_fold,
                }, checkpoint_path)

        best_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(best_checkpoint['model_state'])
        model.eval().to(device)

        test_loss, test_acc = validate(
            model,
            test_loader,
            loss_fn,
            device
        )

        fold_results.append({
            'fold': test_fold,
            'best_epoch': best_checkpoint['epoch'],
            'val_loss': best_checkpoint['val_loss'],
            'val_accuracy': best_checkpoint['val_acc'],
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        })

        print(
            f'FOLD: {test_fold}\n'
            f'TEST:\n'
            f'\tLOSS: {test_loss}\n'
            f'\tACC: {test_acc}\n'
        )
        write_log(
            f'FOLD: {test_fold}\n'
            f'TEST:\n'
            f'\tLOSS: {test_loss}\n'
            f'\tACC: {test_acc}\n',
            FILE_NAME
        )

    mean_loss = (
        sum(r['test_loss'] for r in fold_results)
        /
        len(fold_results)
    )
    mean_accuracy = (
        sum(r['test_accuracy'] for r in fold_results)
        /
        len(fold_results)
    )
    
    print(
        f'MEAN LOSS IN FOLDS: {mean_loss}'
    )
    write_log(
        f'MEAN ACCURACY IN FOLDS: {mean_accuracy}',
        FILE_NAME
    )

if __name__ == '__main__':
    main()