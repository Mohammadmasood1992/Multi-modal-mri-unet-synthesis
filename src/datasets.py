"""PyTorch Dataset wrappers around the preloaded input/output image arrays."""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import Augmentation2D, Augmentation3D


class Dataset3D(Dataset):
    """Whole 3D volumes, one input/output pair per sample."""

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray,
                 augmentation: Augmentation3D = None, train: bool = True):
        self.inputs = inputs
        self.outputs = outputs
        self.augmentation = augmentation if train else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_in = self.inputs[idx].copy()
        img_out = self.outputs[idx].copy()

        if self.augmentation is not None:
            img_in, img_out = self.augmentation(img_in, img_out)

        img_in = np.expand_dims(img_in, axis=0).astype(np.float32)   # (D, H, W) -> (1, D, H, W)
        img_out = np.expand_dims(img_out, axis=0).astype(np.float32)
        return torch.from_numpy(img_in), torch.from_numpy(img_out)


class Dataset2D(Dataset):
    """2D slices extracted from 3D volumes along a chosen axis (or all three)."""

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray,
                 slice_axis: str = 'axial', skip_background: int = 2,
                 augmentation: Augmentation2D = None, train: bool = True):
        """
        Args:
            inputs, outputs: 3D volumes, shape (n_samples, D, H, W).
            slice_axis: 'axial', 'sagittal', 'coronal', or 'all'.
            skip_background: slices to skip at each edge of the volume
                (the outermost slices are mostly empty background).
        """
        self.augmentation = augmentation if train else None
        self.slice_axis = slice_axis
        self.skip_background = skip_background
        self.slices_in, self.slices_out, self.slice_info = self._extract_slices(
            inputs, outputs, slice_axis, skip_background
        )

    def _extract_slices(self, inputs: np.ndarray, outputs: np.ndarray,
                         slice_axis: str, skip: int) -> Tuple[List, List, List]:
        slices_in, slices_out, slice_info = [], [], []
        axes = ['axial', 'sagittal', 'coronal'] if slice_axis == 'all' else [slice_axis]
        axis_map = {'axial': 0, 'sagittal': 1, 'coronal': 2}

        for sample_idx in range(len(inputs)):
            vol_in = inputs[sample_idx]
            vol_out = outputs[sample_idx]

            for axis_name in axes:
                axis = axis_map[axis_name]
                n_slices = vol_in.shape[axis]

                for slice_idx in range(skip, n_slices - skip):
                    if axis == 0:
                        sl_in, sl_out = vol_in[slice_idx, :, :], vol_out[slice_idx, :, :]
                    elif axis == 1:
                        sl_in, sl_out = vol_in[:, slice_idx, :], vol_out[:, slice_idx, :]
                    else:
                        sl_in, sl_out = vol_in[:, :, slice_idx], vol_out[:, :, slice_idx]

                    slices_in.append(sl_in)
                    slices_out.append(sl_out)
                    slice_info.append((sample_idx, axis_name, slice_idx))

        return slices_in, slices_out, slice_info

    def __len__(self):
        return len(self.slices_in)

    def __getitem__(self, idx):
        img_in = self.slices_in[idx].copy()
        img_out = self.slices_out[idx].copy()

        if self.augmentation is not None:
            img_in, img_out = self.augmentation(img_in, img_out)

        img_in = np.expand_dims(img_in, axis=0).astype(np.float32)   # (H, W) -> (1, H, W)
        img_out = np.expand_dims(img_out, axis=0).astype(np.float32)
        return torch.from_numpy(img_in), torch.from_numpy(img_out)
