"""Training-curve and prediction visualizations, saved alongside each fold's results."""

import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


class TrainingPlotter:
    """Tracks per-epoch loss/metric history for one fold and plots it."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.metrics_history = []

    def update(self, train_loss: float, val_loss: float, lr: float, metrics: Dict = None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        if metrics:
            self.metrics_history.append(metrics)

    def plot_losses(self, fold: int = None, save: bool = True):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Training Progress{f" - Fold {fold}" if fold else ""}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.learning_rates, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            suffix = f'_fold{fold}' if fold else ''
            plt.savefig(os.path.join(self.save_dir, f'training_curves{suffix}.png'), dpi=150)
        plt.show()
        plt.close()

    def plot_metrics(self, fold: int = None, save: bool = True):
        if not self.metrics_history:
            return

        metric_names = list(self.metrics_history[0].keys())
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metric_names[:6]):
            values = [m[metric] for m in self.metrics_history]
            axes[i].plot(values)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} Over Training')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            suffix = f'_fold{fold}' if fold else ''
            plt.savefig(os.path.join(self.save_dir, f'metrics{suffix}.png'), dpi=150)
        plt.show()
        plt.close()

    def plot_predictions(self, inputs: np.ndarray, targets: np.ndarray,
                          predictions: np.ndarray, n_samples: int = 3,
                          fold: int = None, save: bool = True):
        """Show input / target / prediction / absolute-difference for a few samples.
        For 3D volumes, the middle slice is shown."""
        n_samples = min(n_samples, len(inputs))
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            if inputs[i].ndim == 3:
                mid = inputs[i].shape[0] // 2
                inp, tgt, pred = inputs[i][mid], targets[i][mid], predictions[i][mid]
            else:
                inp, tgt, pred = inputs[i], targets[i], predictions[i]

            diff = np.abs(tgt - pred)

            axes[i, 0].imshow(inp, cmap='gray')
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(tgt, cmap='gray')
            axes[i, 1].set_title('Target')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

            im = axes[i, 3].imshow(diff, cmap='hot')
            axes[i, 3].set_title('Absolute Difference')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

        plt.tight_layout()
        if save:
            suffix = f'_fold{fold}' if fold else ''
            plt.savefig(os.path.join(self.save_dir, f'predictions{suffix}.png'), dpi=150)
        plt.show()
        plt.close()
