import torch
import torch.nn as nn
import json
import hashlib
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from .modules import CNN_classification
from ..utils import config as cfg
from ..utils import (
    SketchClassifierDataset,
    get_split_records,
    write_log,
    MODELS_DIR,
    DATA,
    print_losses_plot,
    print_confusion_matrix
)
from typing import Literal, overload

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


@overload
def validate(
    model,
    loader,
    loss_fn,
    device,
    is_test: Literal[False] = False
) -> tuple[float, float]: ...


@overload
def validate(
    model,
    loader,
    loss_fn,
    device,
    is_test: Literal[True]
) -> tuple[float, float, list[int], list[int]]: ...


def validate(
    model,
    loader,
    loss_fn,
    device,
    is_test: bool = False
) -> tuple[float, float, list[int], list[int]] | tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    if is_test:
        y_true, y_pred = [], []

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

            if is_test:
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted_labels.cpu().tolist())

            correct += (predicted_labels == labels).sum().item()

            total_samples += batch_size


    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    if is_test:
        r = (avg_loss, accuracy, y_true, y_pred)
    else:
        r = (avg_loss, accuracy)
    
    return r


def main():
    selected = set(cfg.SELECTED_CATEGORIES)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device {device}')
    loss_fn = nn.CrossEntropyLoss()
    config = {
        "base_seed": cfg.BASE_SEED,
        "k_folds": cfg.K_FOLDS,
        "model": cfg.MODEL_NAME,
        "in_channels": cfg.IN_CHANNELS,
        "out_classes": cfg.OUT_CLASSES,
        "hidden_layers": cfg.HIDDEN_LAYERS,
        "convs_per_block": cfg.CONVS_PER_BLOCK,
        "residual_blocks": cfg.RESIDUAL_BLOCKS,
        "starting_hidden_channels": cfg.STARTING_HIDDEN_CHANNELS,
        "dropout": cfg.DROPOUT,
        "normalization": cfg.NORMALIZATION,
        "groups": cfg.GROUPS,
        "categories": sorted(selected),
        "image_size": cfg.IMAGE_SIZE,
        "optimizer": cfg.OPTIMIZER_CLASS.__name__,
        "lr": cfg.LR,
        "scheduler": cfg.SCHEDULER_CLASS.__name__,
        "sched_mode": cfg.SCHED_MODE,
        "sched_patience": cfg.SCHED_PATIENCE,
        "sched_factor": cfg.SCHED_FACTOR,
        "sched_threshold": cfg.SCHED_THRESHOLD,
        "sched_threshold_mode": cfg.SCHED_THRESHOLD_MODE,
        "sched_min_lr": cfg.SCHED_MIN_LR,
        "train_augmentation": cfg.TRAIN_AUGMENTATION,
        "weight_decay": cfg.WEIGHT_DECAY,
        "batch_size": cfg.BATCH_SIZE,
        "epochs": cfg.EPOCHS,
        "patience": cfg.PATIENCE,
        "min_delta": cfg.MIN_DELTA,
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

    start = datetime.now()

    print(f'RUN STARTED AT {start.astimezone().strftime("%H:%M:%S")}')
    write_log(f'RUN STARTED AT {start.astimezone().strftime("%H:%M:%S")}', LOG_FILE)

    fold_results = []
    fold_metrics ={
        "train_loss_curve": ([], []),
        "val_loss_curve": ([], []),
    }
    fold_boundaries = []
    for test_fold in range(cfg.K_FOLDS):
        if test_fold > 0:
            fold_boundaries.append(len(fold_metrics['train_loss_curve'][0]) + 0.5)
        
        fold_seed = cfg.BASE_SEED + test_fold
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
        model = CNN_classification(
            in_channels=cfg.IN_CHANNELS,
            hidden_layers=cfg.HIDDEN_LAYERS,
            out_classes=cfg.OUT_CLASSES,
            starting_hidden_channels=cfg.STARTING_HIDDEN_CHANNELS
        ).to(device)
        optim = cfg.OPTIMIZER_CLASS(
            model.parameters(),
            cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        scheduler = cfg.SCHEDULER_CLASS(
            optim,
            mode=cfg.SCHED_MODE,
            factor=cfg.SCHED_FACTOR,
            patience=cfg.SCHED_PATIENCE,
            threshold=cfg.SCHED_THRESHOLD,
            threshold_mode=cfg.SCHED_THRESHOLD_MODE,
            min_lr=cfg.SCHED_MIN_LR,
        )

        train_records, val_records, test_records = get_split_records(
            Path(DATA / 'index.json'),
            selected,
            test_fold
        )

        train_dataset = SketchClassifierDataset(DATA, train_records, size=cfg.IMAGE_SIZE, augment=True)
        val_dataset = SketchClassifierDataset(DATA, val_records, size=cfg.IMAGE_SIZE)
        test_dataset = SketchClassifierDataset(DATA, test_records, size=cfg.IMAGE_SIZE)

        shuffle_generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, generator=shuffle_generator)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')
        best_epoch = None
        checkpoint_path = f'{MODELS_DIR}/cnn_fold_{test_fold}_best.pt'

        straight_nonimproving_epochs = 0

        for epoch in range(1, cfg.EPOCHS + 1):
            plot_epoch = len(fold_metrics['train_loss_curve'][0]) + 1
            lr_this_epoch = optim.param_groups[0]['lr']

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optim,
                device,
            )
            fold_metrics["train_loss_curve"][0].append(plot_epoch)
            fold_metrics["train_loss_curve"][1].append(train_loss)

            val_loss, val_acc= validate(
                model,
                val_loader,
                loss_fn,
                device,
                is_test=False
            )
            fold_metrics["val_loss_curve"][0].append(plot_epoch)
            fold_metrics["val_loss_curve"][1].append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss - (cfg.MIN_DELTA or 0.0):
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

            if straight_nonimproving_epochs >= cfg.PATIENCE:
                break # early stop

            print(
                f'[FOLD {test_fold}]'
                f'[EPOCH {epoch}]\n'
                f'TRAIN:\n'
                f'\tLOSS: {train_loss}\n'
                f'\tACC: {train_acc}\n'
                f'\tLR: {lr_this_epoch}\n'
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
                f'\tLR: {lr_this_epoch}\n'
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

        test_loss, test_acc, y_true, y_pred = validate(
            model,
            test_loader,
            loss_fn,
            device,
            is_test=True
        )

        print_confusion_matrix(
            y_true,
            y_pred,
            {record['category']: record['label'] for record in test_records},
            filename=f'confusion_matrix_fold_{test_fold}_{config_id}_{run_id}.png'
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

    end = datetime.now()
    elapsed = end-start
    hours, remainder = divmod(elapsed.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # alternativa: str(elapsed).split('.')[0]
    elapsed_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    print(f'RUN FINISHED AT {end.astimezone().strftime("%H:%M:%S")}')
    write_log(f'RUN FINISHED AT {end.astimezone().strftime("%H:%M:%S")}', LOG_FILE)

    print(f'ELAPSED {elapsed_str}')
    write_log(f'ELAPSED {elapsed_str}', LOG_FILE)
    
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
    print_losses_plot(fold_metrics, fold_boundaries, run_id)

if __name__ == '__main__':
    main()
