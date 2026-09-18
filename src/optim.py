"""Optimizer / LR-scheduler factories, and early stopping."""

import copy
from typing import Dict

import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, ExponentialLR,
)


def get_optimizer(model, optimizer_type: str, learning_rate: float, weight_decay: float):
    """Build an optimizer by name ('adam', 'adamw', 'sgd', 'rmsprop')."""
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
    }

    if optimizer_type.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    if optimizer_type.lower() == 'sgd':
        return optimizers[optimizer_type.lower()](
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9
        )
    return optimizers[optimizer_type.lower()](
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def get_scheduler(optimizer, scheduler_type: str, scheduler_params: Dict = None):
    """Build an LR scheduler by name, or None for a constant learning rate."""
    if scheduler_type is None or scheduler_type.lower() == 'none':
        return None

    params = scheduler_params or {}
    schedulers = {
        'step': lambda: StepLR(optimizer, step_size=params.get('step_size', 30),
                                gamma=params.get('gamma', 0.1)),
        'cosine': lambda: CosineAnnealingLR(optimizer, T_max=params.get('T_max', 100)),
        'plateau': lambda: ReduceLROnPlateau(optimizer, mode='min',
                                              factor=params.get('factor', 0.5),
                                              patience=params.get('patience', 10)),
        'exponential': lambda: ExponentialLR(optimizer, gamma=params.get('gamma', 0.95)),
        'cyclic': lambda: CyclicLR(optimizer,
                                    base_lr=params.get('base_lr', 1e-5),
                                    max_lr=params.get('max_lr', 1e-3),
                                    step_size_up=params.get('step_size_up', 2000)),
        'onecycle': lambda: OneCycleLR(optimizer,
                                        max_lr=params.get('max_lr', 1e-3),
                                        total_steps=params.get('total_steps', 1000)),
    }

    if scheduler_type.lower() not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    return schedulers[scheduler_type.lower()]()


class EarlyStopping:
    """Stops training when a monitored score stops improving, and keeps a copy
    of the best-scoring model weights seen so far."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        current_score = -score if self.mode == 'min' else score

        if self.best_score is not None and current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
