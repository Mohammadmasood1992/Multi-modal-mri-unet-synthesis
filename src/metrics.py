"""Image-quality metrics for comparing predicted vs. target images."""

from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """MAE, MSE, RMSE, PSNR, Pearson correlation, and SSIM between two
    same-shaped arrays (a single image, or a batch/volume of them)."""

    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)

    max_val = max(target.max() - target.min(), 1e-8)
    psnr = 20 * np.log10(max_val / (rmse + 1e-8))

    corr = np.corrcoef(pred.flatten(), target.flatten())[0, 1]

    # SSIM uses a 2D sliding window, so 3D volumes are scored slice-wise and averaged.
    if pred.ndim == 3:
        ssim_vals = []
        for i in range(pred.shape[0]):
            data_range = target[i].max() - target[i].min()
            if data_range > 0:
                ssim_vals.append(ssim_metric(target[i], pred[i], data_range=data_range))
        ssim_val = np.mean(ssim_vals) if ssim_vals else 0
    else:
        data_range = target.max() - target.min()
        ssim_val = ssim_metric(target, pred, data_range=data_range) if data_range > 0 else 0

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'correlation': corr,
        'ssim': ssim_val,
    }
