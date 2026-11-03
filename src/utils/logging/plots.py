import matplotlib.pyplot as plt
from .writing import LOG_DIR

def print_losses_plot(metrics: dict[str, tuple[list[int], list[float]]], fold_boundaries, run_id) -> None:
    x_train: list[int] = metrics['train_loss_curve'][0]
    y_train: list[float] = metrics['train_loss_curve'][1]
    x_val: list[int] = metrics['val_loss_curve'][0]
    y_val: list[float] = metrics['val_loss_curve'][1]
    plt.plot(
        x_train, y_train, 'r--',
        x_val, y_val, 'b--',
    )
    for boundary in fold_boundaries:
        plt.axvline(boundary, color='gray', linestyle='--', alpha=0.6)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(
        LOG_DIR / f'train_val_loss_curves_{run_id}.png'
    )
    plt.show()

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
        42
    )