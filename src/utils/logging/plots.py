import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .writing import LOG_DIR

def print_losses_plot(
    metrics: dict[str, tuple[list[int], list[float]]],
    fold_boundaries,
    run_id,
    save_fig: bool = True
) -> None:
    x_train: list[int] = metrics['train_loss_curve'][0]
    y_train: list[float] = metrics['train_loss_curve'][1]
    x_val: list[int] = metrics['val_loss_curve'][0]
    y_val: list[float] = metrics['val_loss_curve'][1]
    plt.plot(
        x_train, y_train, 'r--',
        x_val, y_val, 'b--',
    )
    for boundary in fold_boundaries:
        plt.axvline(boundary, color='gray', linestyle='-.', alpha=0.6, linewidth=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.7)
    if save_fig:
        plt.savefig(
            LOG_DIR / f'train_val_loss_curves_{run_id}.png'
        )
    plt.show()

def print_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    categories: dict[str, int],
    filename: str = 'confusion_matrix.png',
    show_img: bool = False
) -> None:
    ordered = sorted(categories.items(), key=lambda item: item[1])
    class_ids = [class_id for _, class_id in ordered]
    other_id = max(class_ids) + 1
    labels = class_ids + [other_id]
    names = [name for name, _ in ordered] + ['altre classi']
    selected_ids = set(class_ids)
    
    true_labels = [label if label in selected_ids else other_id for label in y_true]
    pred_labels = [label if label in selected_ids else other_id for label in y_pred]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    display = ConfusionMatrixDisplay(cm, display_labels=names)

    fig, ax = plt.subplots(figsize=(20, 20))
    display.plot(ax=ax, xticks_rotation=90, colorbar=True, values_format='d')
    fig.tight_layout()
    fig.savefig(LOG_DIR / filename, dpi=200)
    if show_img:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    import numpy as np
    m = {
        'train_loss_curve': (list(range(10)), list(np.random.randn(10))),
        'val_loss_curve': (list(range(10)), list(np.random.randn(10)))
    }

    m['train_loss_curve'][0].extend(list(range(11, 20)))
    m['train_loss_curve'][1].extend(list(np.random.randn(9)))
    m['val_loss_curve'][0].extend(list(range(11, 20)))
    m['val_loss_curve'][1].extend(list(np.random.randn(9)))

    b = [10.5, 19.5]
    
    print_losses_plot(
        m,
        b,
        42,
        save_fig=False
    )
