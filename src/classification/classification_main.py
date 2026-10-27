import torch
import torch.nn as nn
import os
import json
import hashlib
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from .modules import CNN_classification
from ..utils import (
    SketchClassifierDataset,
    get_split_records,
    write_log,
    MODELS_DIR,
    DATA,
)
FILE_NAME = Path(__file__).stem

def train_one_epoch(
    model: CNN_classification,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    
    epoch_loss = 0.0
    epoch_samples = 0
    epoch_corrects = 0

    # FOREACH MINIBATCH
    for sketches, labels in loader:
        sketches = sketches.to(device)
        labels = labels.to(device)

        optim.zero_grad()
        logits = model(sketches)
        if not torch.isfinite(logits).all():
            raise RuntimeError('logits are not finite')

        predicted_labels = logits.argmax(dim=1)
        
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('loss is not finite')
        loss.backward()
        optim.step()

        epoch_loss += loss.item() * sketches.size(0)
        epoch_corrects += (predicted_labels == labels).sum().item()
        epoch_samples += sketches.size(0)

    train_loss = epoch_loss / epoch_samples
    train_accuracy = epoch_corrects / epoch_samples
    return train_loss, train_accuracy


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
            if not torch.isfinite(loss):
                raise RuntimeError('loss is not finite')

            batch_size = logits.size(0)
            total_loss += loss.item() * batch_size

            predicted_labels = logits.argmax(dim=1)

            correct += (predicted_labels == labels).sum().item()

            total_samples += batch_size


    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def main():
    BASE_SEED = 42
    
    CAT_SUBSET = 10
    BATCH_SIZE = 32
    PATIENCE = 20
    # MIN_DELTA = 1e-3
    IN_CHANNELS = 1
    HIDDEN_LAYERS = 4 # 3
    OUT_CLASSES = 125
    STARTING_HIDDEN_CHANNELS = 16
    WEIGHT_DECAY = 1e-4

    selected = set(
        sorted(
        f for f in os.listdir(DATA / 'photo')
        )[:CAT_SUBSET]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device {device}')
    loss_fn = nn.CrossEntropyLoss()
    lr = 3e-4
    epochs = 225

    config = {
        "base_seed": BASE_SEED,
        "model": "CNN",
        "hidden_layers": HIDDEN_LAYERS,
        "starting_hidden_channels": STARTING_HIDDEN_CHANNELS,
        "normalization": "GroupNorm",
        "groups": 4,
        "categories": sorted(selected),
        "image_size": 256,
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "patience": PATIENCE,
        "min_delta": None, # MIN_DELTA,
    }

    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_id = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:10]
    run_id = datetime.now().strftime("%Y%m%d")

    LOG_FILE = f'{FILE_NAME}_{config_id}_{run_id}'
    write_log(
        f'CONFIG_ID: {config_id}\n'
        f'CONFIG: {config_json}',
        LOG_FILE
    )

    print(f'RUN STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    write_log(f'RUN STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}', LOG_FILE)

    fold_results = []
    for test_fold in range(5):
        fold_seed = BASE_SEED + test_fold
        torch.manual_seed(fold_seed)
        
        print(
            f'=====Fold {test_fold}=====\n'
            f'=====[Seed] {fold_seed}=====\n'
        )
        write_log(
            f'=====Fold {test_fold}=====\n'
            f'=====[Seed] {fold_seed}=====\n',
            LOG_FILE
        )
        # model = CNN_classification(1, 3, 125).to(device)
        model = CNN_classification(
            in_channels=IN_CHANNELS,
            hidden_layers=HIDDEN_LAYERS,
            out_classes=OUT_CLASSES,
            starting_hidden_channels=STARTING_HIDDEN_CHANNELS
        ).to(device)
        # === OPTIM. BASELINE 1 ===
        # optim = torch.optim.SGD(
        #     model.parameters(),
        #     lr,
        #     momentum=0.9,
        #     weight_decay=WEIGHT_DECAY,
        # )
        # === OPTIM. BASELINE 2 ===
        # optim = torch.optim.Adam(
        #     model.parameters(),
        #     lr,
        #     weight_decay=0
        # )
        # === OPTIM. BASELINE 3 ===
        optim = torch.optim.AdamW(
            model.parameters(),
            lr,
            weight_decay=WEIGHT_DECAY
        )

        train_records, val_records, test_records = get_split_records(Path(DATA / 'index.json'), selected, test_fold)

        train_dataset = SketchClassifierDataset(DATA, train_records)
        val_dataset = SketchClassifierDataset(DATA, val_records)
        test_dataset = SketchClassifierDataset(DATA, test_records)

        shuffle_generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_generator)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')
        best_epoch = None
        checkpoint_path = f'{MODELS_DIR}/cnn_fold_{test_fold}_best.pt'

        straight_nonimproving_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optim,
                device,
            )

            # if epoch % 15 == 0:
            #     train_eval_loss, train_eval_acc = validate(
            #         model,
            #         train_loader,
            #         loss_fn,
            #         device
            #     )

            val_loss, val_acc = validate(
                model,
                val_loader,
                loss_fn,
                device
            )

            if val_loss < best_val_loss: # - MIN_DELTA:
                best_val_loss = val_loss
                best_epoch = epoch
                straight_nonimproving_epochs = 0
                print(
                    f'NEW BEST VAL LOSS: {best_val_loss}'
                )
                write_log(
                    f'NEW BEST VAL LOSS: {best_val_loss}',
                    LOG_FILE
                )

                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'fold': test_fold,
                }, checkpoint_path)
            else:
                straight_nonimproving_epochs += 1

            if straight_nonimproving_epochs >= PATIENCE:
                break # early stop

            # if epoch % 15 == 0:
            #     print(
            #         f'[FOLD {test_fold}]'
            #         f'[EPOCH {epoch}]\n'
            #         f'TRAIN:\n'
            #         f'\tLOSS: {train_loss}\n'
            #         f'\tACC: {train_acc}\n'
            #         f'EVAL TRAIN:\n'
            #         f'\tLOSS: {train_eval_loss}\n'
            #         f'\tACC: {train_eval_acc}\n'
            #         f'VALIDATION:\n'
            #         f'\tLOSS: {val_loss}\n'
            #         f'\t(BEST): {best_val_loss}\n'
            #         f'\tACC: {val_acc}\n'
            #     )
            #     write_log(
            #         f'[FOLD {test_fold}]'
            #         f'[EPOCH {epoch}]\n'
            #         f'TRAIN:\n'
            #         f'\tLOSS: {train_loss}\n'
            #         f'\tACC: {train_acc}\n'
            #         f'EVAL TRAIN:\n'
            #         f'\tLOSS: {train_eval_loss}\n'
            #         f'\tACC: {train_eval_acc}\n'
            #         f'VALIDATION:\n'
            #         f'\tLOSS: {val_loss}\n'
            #         f'\t(BEST): {best_val_loss}\n'
            #         f'\tACC: {val_acc}\n',
            #         LOG_FILE
            #     )
            # else:
            print(
                f'[FOLD {test_fold}]'
                f'[EPOCH {epoch}]\n'
                f'TRAIN:\n'
                f'\tLOSS: {train_loss}\n'
                f'\tACC: {train_acc}\n'
                f'VALIDATION:\n'
                f'\tLOSS: {val_loss}\n'
                f'\t(BEST [{best_epoch}]): {best_val_loss}\n'
                f'\tACC: {val_acc}\n'
            )
            write_log(
                f'[FOLD {test_fold}]'
                f'[EPOCH {epoch}]\n'
                f'TRAIN:\n'
                f'\tLOSS: {train_loss}\n'
                f'\tACC: {train_acc}\n'
                f'VALIDATION:\n'
                f'\tLOSS: {val_loss}\n'
                f'\t(BEST [{best_epoch}]): {best_val_loss}\n'
                f'\tACC: {val_acc}\n',
                LOG_FILE
            )

            if straight_nonimproving_epochs > 0:
                print(f'STREAK OF NON-IMPROVING EPOCHS: {straight_nonimproving_epochs}\n')

        best_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(best_checkpoint['model_state'])
        model.to(device)

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
            LOG_FILE
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

    print(f'RUN FINISHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    write_log(f'RUN FINISHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}', LOG_FILE)
    
    print(
        f'MEAN LOSS IN FOLDS: {mean_loss}'
    )
    print(
        f'MEAN ACCURACY IN FOLDS: {mean_accuracy}'
    )
    write_log(
        f'MEAN LOSS IN FOLDS: {mean_loss}',
        LOG_FILE
    )
    write_log(
        f'MEAN ACCURACY IN FOLDS: {mean_accuracy}',
        LOG_FILE
    )

    best_fold = min(fold_results, key=lambda r: r['val_loss'])
    print(
        f'BEST FOLD: {best_fold["fold"]}\n'
        f'\tEPOCH: {best_fold["best_epoch"]}\n'
        f'\tVAL LOSS: {best_fold["val_loss"]}\n'
        f'\tVAL ACCURACY: {best_fold["val_accuracy"]}\n'
        f'\tTEST LOSS: {best_fold["test_loss"]}\n'
        f'\tTEST ACCURACY: {best_fold["test_accuracy"]}\n'
    )
    write_log(
        f'BEST FOLD: {best_fold["fold"]}\n'
        f'\tEPOCH: {best_fold["best_epoch"]}\n'
        f'\tVAL LOSS: {best_fold["val_loss"]}\n'
        f'\tVAL ACCURACY: {best_fold["val_accuracy"]}\n'
        f'\tTEST LOSS: {best_fold["test_loss"]}\n'
        f'\tTEST ACCURACY: {best_fold["test_accuracy"]}\n',
        LOG_FILE
    )

if __name__ == '__main__':
    main()