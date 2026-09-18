"""Loss functions for image-to-image synthesis.

`combined` (L1 + SSIM, weighted) is the loss used for the reported results;
the others are kept because the trainer lets you swap losses via config
without touching code.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """Mean absolute error."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class MSELoss(nn.Module):
    """Mean squared error."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """1 - Structural Similarity Index. Computed slice-wise for 3D volumes,
    since SSIM as defined here uses 2D Gaussian windows."""

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1d_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        if pred.dim() == 5:  # 3D: (B, C, D, H, W) -> average SSIM over depth slices
            ssim_vals = [
                self._compute_ssim_2d(pred[:, :, d, :, :], target[:, :, d, :, :])
                for d in range(pred.shape[2])
            ]
            return 1 - torch.stack(ssim_vals).mean()
        return 1 - self._compute_ssim_2d(pred, target)

    def _compute_ssim_2d(self, pred, target):
        channel = pred.size(1)
        if channel != self.channel or self.window.device != pred.device:
            self.window = self._create_window(self.window_size, channel).to(pred.device)
            self.channel = channel
        return self._ssim(pred, target, self.window, self.window_size, channel, self.size_average)


class PerceptualLoss(nn.Module):
    """VGG16-feature L1 loss (2D only; applied slice-wise for 3D volumes).
    Falls back to plain L1 if torchvision/VGG weights aren't available."""

    def __init__(self, device='cuda'):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
            self.vgg = vgg.eval().to(device)
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.available = True
        except Exception:
            self.available = False
            print("Warning: VGG not available for perceptual loss. Using L1 instead.")

        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        if not self.available:
            return self.l1(pred, target)

        if pred.dim() == 5:  # 3D: (B, C, D, H, W)
            losses = [
                self._compute_perceptual_2d(pred[:, :, d, :, :], target[:, :, d, :, :])
                for d in range(pred.shape[2])
            ]
            return torch.stack(losses).mean()
        return self._compute_perceptual_2d(pred, target)

    def _compute_perceptual_2d(self, pred, target):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.l1(self.vgg(pred), self.vgg(target))


class CombinedLoss(nn.Module):
    """Weighted sum of any subset of {l1, mse, ssim, perceptual}.

    Returns (total_loss, per-component dict) so the trainer can log each
    term separately during training.
    """

    def __init__(self, weights: Dict[str, float] = None, device='cuda'):
        super().__init__()

        if weights is None:
            weights = {'l1': 1.0, 'ssim': 0.5}
        self.weights = weights

        self.losses = {}
        if 'l1' in weights:
            self.losses['l1'] = L1Loss()
        if 'mse' in weights:
            self.losses['mse'] = MSELoss()
        if 'ssim' in weights:
            self.losses['ssim'] = SSIMLoss()
        if 'perceptual' in weights:
            self.losses['perceptual'] = PerceptualLoss(device)

    def forward(self, pred, target):
        total_loss = 0
        loss_components = {}
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(pred, target)
            loss_components[name] = loss_val.item()
            total_loss += self.weights[name] * loss_val
        return total_loss, loss_components


def get_loss_function(loss_type: str, device='cuda', **kwargs):
    """Factory: build a loss module by name ('l1', 'mse', 'ssim', 'perceptual', 'combined')."""
    loss_functions = {
        'l1': L1Loss,
        'mse': MSELoss,
        'ssim': SSIMLoss,
        'perceptual': lambda: PerceptualLoss(device),
        'combined': lambda: CombinedLoss(kwargs.get('weights'), device),
    }
    if loss_type.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_type}")
    return loss_functions[loss_type.lower()]()
