"""On-the-fly data augmentation for paired input/output medical images.

The same random transform is applied to both the input and target image so
that spatial correspondence between them is preserved; intensity jitter is
applied to the input only, since it simulates acquisition/scanner noise
rather than a change in anatomy.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates


class Augmentation3D:
    """Rotation, scaling, elastic deformation, and intensity jitter for 3D volumes."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rotation_range = self.config.get('rotation_range', 15)
        self.scale_range = self.config.get('scale_range', (0.9, 1.1))
        self.intensity_shift_range = self.config.get('intensity_shift_range', 0.1)
        self.intensity_scale_range = self.config.get('intensity_scale_range', (0.9, 1.1))
        self.elastic_alpha = self.config.get('elastic_alpha', 50)
        self.elastic_sigma = self.config.get('elastic_sigma', 5)
        self.prob = self.config.get('prob', 0.5)

    def __call__(self, image_in: np.ndarray, image_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            axes = [(1, 2), (0, 2), (0, 1)][np.random.choice([0, 1, 2])]
            image_in = rotate(image_in, angle, axes=axes, reshape=False, order=1, mode='constant')
            image_out = rotate(image_out, angle, axes=axes, reshape=False, order=1, mode='constant')

        if np.random.random() < self.prob:
            scale = np.random.uniform(*self.scale_range)
            image_in = self._scale_image(image_in, scale)
            image_out = self._scale_image(image_out, scale)

        if np.random.random() < self.prob:
            image_in, image_out = self._elastic_deform(image_in, image_out)

        if np.random.random() < self.prob:
            shift = np.random.uniform(-self.intensity_shift_range, self.intensity_shift_range)
            scale = np.random.uniform(*self.intensity_scale_range)
            image_in = image_in * scale + shift

        return image_in, image_out

    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Zoom then crop/pad back to the original shape so tensor sizes stay fixed."""
        original_shape = image.shape
        scaled = zoom(image, scale, order=1)
        result = np.zeros(original_shape)

        if scale > 1:
            starts = [(s - o) // 2 for s, o in zip(scaled.shape, original_shape)]
            slices = [slice(s, s + o) for s, o in zip(starts, original_shape)]
            result = scaled[tuple(slices)]
        else:
            starts = [(o - s) // 2 for s, o in zip(scaled.shape, original_shape)]
            slices = [slice(s, s + sc) for s, sc in zip(starts, scaled.shape)]
            result[tuple(slices)] = scaled

        return result

    def _elastic_deform(self, image_in: np.ndarray, image_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shape = image_in.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = [
            np.clip(x + dx, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(z + dz, 0, shape[2] - 1),
        ]

        image_in = map_coordinates(image_in, indices, order=1, mode='constant').reshape(shape)
        image_out = map_coordinates(image_out, indices, order=1, mode='constant').reshape(shape)
        return image_in, image_out


class Augmentation2D:
    """Rotation, scaling, elastic deformation, and intensity jitter for 2D slices."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rotation_range = self.config.get('rotation_range', 15)
        self.scale_range = self.config.get('scale_range', (0.9, 1.1))
        self.intensity_shift_range = self.config.get('intensity_shift_range', 0.1)
        self.intensity_scale_range = self.config.get('intensity_scale_range', (0.9, 1.1))
        self.elastic_alpha = self.config.get('elastic_alpha', 50)
        self.elastic_sigma = self.config.get('elastic_sigma', 5)
        self.prob = self.config.get('prob', 0.5)

    def __call__(self, image_in: np.ndarray, image_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image_in = rotate(image_in, angle, reshape=False, order=1, mode='constant')
            image_out = rotate(image_out, angle, reshape=False, order=1, mode='constant')

        if np.random.random() < self.prob:
            scale = np.random.uniform(*self.scale_range)
            image_in = self._scale_image(image_in, scale)
            image_out = self._scale_image(image_out, scale)

        if np.random.random() < self.prob:
            image_in, image_out = self._elastic_deform(image_in, image_out)

        if np.random.random() < self.prob:
            shift = np.random.uniform(-self.intensity_shift_range, self.intensity_shift_range)
            scale = np.random.uniform(*self.intensity_scale_range)
            image_in = image_in * scale + shift

        return image_in, image_out

    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        original_shape = image.shape
        scaled = zoom(image, scale, order=1)
        result = np.zeros(original_shape)

        if scale > 1:
            starts = [(s - o) // 2 for s, o in zip(scaled.shape, original_shape)]
            slices = [slice(s, s + o) for s, o in zip(starts, original_shape)]
            result = scaled[tuple(slices)]
        else:
            starts = [(o - s) // 2 for s, o in zip(scaled.shape, original_shape)]
            slices = [slice(s, s + sc) for s, sc in zip(starts, scaled.shape)]
            result[tuple(slices)] = scaled

        return result

    def _elastic_deform(self, image_in: np.ndarray, image_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shape = image_in.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [
            np.clip(x + dx, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
        ]

        image_in = map_coordinates(image_in, indices, order=1, mode='constant').reshape(shape)
        image_out = map_coordinates(image_out, indices, order=1, mode='constant').reshape(shape)
        return image_in, image_out
