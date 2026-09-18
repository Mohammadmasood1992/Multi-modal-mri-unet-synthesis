"""K-fold cross-validation trainer for the U-Net, split into three phases so
each one can run as its own SLURM job across multiple GPUs:

    Phase 1  preprocess()      -- split the data and save fold indices to disk
                                   (CPU only, run once)
    Phase 2  train(fold_id=N)  -- train a single fold on its own GPU
                                   (run once per fold, in parallel)
    Phase 3  aggregate()       -- load all trained fold models and evaluate
                                   the external test set as an ensemble

train(fold_id=None) also supports running all folds sequentially in a single
job, for smaller experiments that don't need a multi-GPU cluster.

The model architecture, loss functions, metrics, and data pipeline are
unchanged from the single-job version -- only the orchestration (splitting
work across SLURM jobs, checkpointing, resuming) is new.
"""

import copy
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .augmentation import Augmentation2D, Augmentation3D
from .datasets import Dataset2D, Dataset3D
from .losses import CombinedLoss, get_loss_function
from .metrics import compute_metrics
from .optim import EarlyStopping, get_optimizer, get_scheduler
from .plotting import TrainingPlotter


class UNetTrainer:
    """Complete U-Net training pipeline with grouped k-fold cross-validation
    and a held-out external test set, evaluated as a 5-model ensemble."""

    def __init__(
        self,
        model: nn.Module,
        input_modality: np.ndarray,
        output_modality: np.ndarray,
        samples_reserved_to_external_test: int,
        groups: pd.DataFrame,
        iid: pd.Series,
        eid: pd.Series,
        fid: pd.Series,
        n_folds: int = 5,
        fold_strategy: str = 'family',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 8,
        n_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer_type: str = 'adam',
        loss_function: str = 'l1',
        scheduler_type: str = 'plateau',
        scheduler_params: Dict = None,
        clip_grad: float = None,
        early_stopping_patience: int = 15,
        augmentation_config: Dict = None,
        device: str = 'cuda',
        save_dir: str = './experiments',
        experiment_name: str = None,
        random_state: int = 42,
        verbose: bool = True,
        save_all_fold_models: bool = True,
        save_predictions: bool = True,
        slice_axis: str = 'axial',
        skip_background_slices: int = 2,
        loss_weights: Dict = None,
        enable_gradient_checkpointing: bool = True,
    ):
        self.model_template = model
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.samples_reserved_to_external_test = samples_reserved_to_external_test
        self.iid = iid.reset_index(drop=True)
        self.eid = eid.reset_index(drop=True)
        self.fid = fid.reset_index(drop=True)
        self.groups = groups
        self.n_folds = n_folds
        self.fold_strategy = fold_strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        self.clip_grad = clip_grad
        self.early_stopping_patience = early_stopping_patience
        self.augmentation_config = augmentation_config or {}
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        self.save_all_fold_models = save_all_fold_models
        self.save_predictions = save_predictions
        self.slice_axis = slice_axis
        self.skip_background_slices = skip_background_slices
        self.loss_weights = loss_weights
        self.enable_gradient_checkpointing = enable_gradient_checkpointing

        self.dimensions = model.dimensions

        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Shared between all three phases/jobs of the same experiment.
        self.splits_dir = os.path.join(self.save_dir, 'fold_splits')
        os.makedirs(self.splits_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        self.fold_models = []
        self.fold_results = []
        self.external_test_results = None

        if self.verbose:
            self._print_config()

    # =========================================================================
    # PHASE 1 -- split data, save fold indices (run once, CPU only)
    # =========================================================================

    def preprocess(self):
        print("\n" + "=" * 70)
        print("PHASE 1 -- PREPROCESSING (split + save fold indices)")
        print("=" * 70)

        (external_input, external_output,
         remaining_input, remaining_output,
         external_mask, remaining_mask,
         remaining_groups) = self._split_external_test()

        self._check_overlap(external_mask, remaining_mask)

        fold_splits = self._create_fold_splits(remaining_input, remaining_output, remaining_groups)
        self._save_fold_splits(fold_splits, remaining_mask)

        print("\nPreprocessing complete.")
        print(f"  Fold index files saved to: {self.splits_dir}")
        print("  Submit the training jobs now.\n")

    # =========================================================================
    # PHASE 2 -- train one fold (run in parallel, one GPU each), or all folds
    #            sequentially in a single job when fold_id is None
    # =========================================================================

    def train(self, fold_id: Optional[int] = None) -> Dict:
        if fold_id is not None:
            return self._train_single_fold_job(fold_id)

        print("\n" + "=" * 70)
        print("STARTING U-NET TRAINING PIPELINE (single-job mode)")
        print("=" * 70)

        (external_input, external_output,
         remaining_input, remaining_output,
         external_mask, remaining_mask,
         remaining_groups) = self._split_external_test()

        self._check_overlap(external_mask, remaining_mask)

        fold_splits = self._create_fold_splits(remaining_input, remaining_output, remaining_groups)

        for fold_idx, fold_data in enumerate(fold_splits):
            model, fold_result = self._train_fold(fold_data, fold_idx)
            self.fold_models.append(model)
            self.fold_results.append(fold_result)

        ensemble_preds, ensemble_metrics = self._evaluate_external_test(external_input, external_output)

        self._print_summary()
        self._save_results()

        return {
            'fold_results': self.fold_results,
            'external_test_results': self.external_test_results,
            'fold_models': self.fold_models,
        }

    # =========================================================================
    # PHASE 3 -- load all fold models, evaluate the external test set
    # =========================================================================

    def aggregate(self):
        print("\n" + "=" * 70)
        print("PHASE 3 -- AGGREGATION (load models + external test evaluation)")
        print("=" * 70)

        external_mask_path = os.path.join(self.splits_dir, 'external_mask.npy')
        if not os.path.exists(external_mask_path):
            raise FileNotFoundError(
                f"External mask not found at {external_mask_path}. Run preprocess() first."
            )
        external_mask = np.load(external_mask_path)
        external_input = self.input_modality[external_mask]
        external_output = self.output_modality[external_mask]

        for fold_idx in range(self.n_folds):
            model_path = os.path.join(self.save_dir, f'model_fold_{fold_idx + 1}.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model for fold {fold_idx + 1} not found at {model_path}. "
                    "Make sure all training jobs completed successfully."
                )
            model = copy.deepcopy(self.model_template).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.fold_models.append(model)
            print(f"  Loaded model: fold {fold_idx + 1}")

        for fold_idx in range(self.n_folds):
            result_path = os.path.join(self.save_dir, f'fold_{fold_idx + 1}', 'fold_result.json')
            if os.path.exists(result_path):
                with open(result_path) as f:
                    self.fold_results.append(json.load(f))

        ensemble_preds, ensemble_metrics = self._evaluate_external_test(external_input, external_output)

        self._print_summary()
        self._save_results()

        print("\nAggregation complete.\n")
        return {
            'fold_results': self.fold_results,
            'external_test_results': self.external_test_results,
        }

    # =========================================================================
    # Internal: single-fold job (called by train(fold_id=N))
    # =========================================================================

    def _train_single_fold_job(self, fold_id: int) -> Dict:
        print("\n" + "=" * 70)
        print(f"PHASE 2 -- TRAINING (fold {fold_id + 1}/{self.n_folds})")
        print(f"Device : {self.device}")
        print("=" * 70)

        fold_splits = self._load_fold_splits()
        fold_data = fold_splits[fold_id]

        model, fold_result = self._train_fold(fold_data, fold_id)

        fold_result_dir = os.path.join(self.save_dir, f'fold_{fold_id + 1}')
        os.makedirs(fold_result_dir, exist_ok=True)

        # Convert numpy scalars to native Python types for JSON serialization.
        serialisable = {}
        for k, v in fold_result.items():
            if k == 'test_metrics':
                serialisable[k] = {mk: float(mv) for mk, mv in v.items()}
            elif isinstance(v, (np.integer, np.floating)):
                serialisable[k] = v.item()
            else:
                serialisable[k] = v

        with open(os.path.join(fold_result_dir, 'fold_result.json'), 'w') as f:
            json.dump(serialisable, f, indent=2)

        print(f"\nFold {fold_id + 1} complete. Result saved.\n")
        return {'fold_result': fold_result}

    # =========================================================================
    # Internal: save / load fold splits (shared across the 3 SLURM phases)
    # =========================================================================

    def _save_fold_splits(self, fold_splits: List[Dict], remaining_mask: np.ndarray):
        np.save(os.path.join(self.splits_dir, 'external_mask.npy'), remaining_mask == False)

        for fs in fold_splits:
            n = fs['fold']  # 1-based
            np.save(os.path.join(self.splits_dir, f'fold_{n}_train_idx.npy'), fs['train_idx'])
            np.save(os.path.join(self.splits_dir, f'fold_{n}_val_idx.npy'), fs['val_idx'])
            np.save(os.path.join(self.splits_dir, f'fold_{n}_test_idx.npy'), fs['test_idx'])

        if self.verbose:
            print(f"\n  Fold split files written to: {self.splits_dir}")
            for fs in fold_splits:
                print(f"    Fold {fs['fold']}: "
                      f"train={len(fs['train_idx'])}, val={len(fs['val_idx'])}, test={len(fs['test_idx'])}")

    def _load_fold_splits(self) -> List[Dict]:
        fold_splits = []
        for fold_idx in range(self.n_folds):
            n = fold_idx + 1
            train_path = os.path.join(self.splits_dir, f'fold_{n}_train_idx.npy')
            if not os.path.exists(train_path):
                raise FileNotFoundError(
                    f"Fold split file not found: {train_path}\n"
                    "Run preprocess() (Phase 1) before the training jobs."
                )
            fold_splits.append({
                'fold': n,
                'train_idx': np.load(os.path.join(self.splits_dir, f'fold_{n}_train_idx.npy')),
                'val_idx': np.load(os.path.join(self.splits_dir, f'fold_{n}_val_idx.npy')),
                'test_idx': np.load(os.path.join(self.splits_dir, f'fold_{n}_test_idx.npy')),
            })

        if self.verbose:
            print(f"  Loaded fold splits from: {self.splits_dir}")
        return fold_splits

    # =========================================================================
    # Internal: checkpointing, so a fold's job can resume after a SLURM
    # time-limit kill instead of restarting from epoch 0
    # =========================================================================

    def _checkpoint_path(self, fold_idx: int) -> str:
        return os.path.join(self.checkpoints_dir, f'fold_{fold_idx + 1}_checkpoint.pth')

    def _save_checkpoint(self, fold_idx: int, epoch: int, model: nn.Module,
                          optimizer, scheduler, best_val_loss: float):
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if scheduler is not None:
            ckpt['scheduler_state'] = scheduler.state_dict()
        torch.save(ckpt, self._checkpoint_path(fold_idx))

    def _load_checkpoint(self, fold_idx: int, model: nn.Module, optimizer, scheduler):
        """Returns (start_epoch, best_val_loss); (0, inf) if no checkpoint exists."""
        path = self._checkpoint_path(fold_idx)
        if not os.path.exists(path):
            return 0, float('inf')

        print(f"  Resuming fold {fold_idx + 1} from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler is not None and 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])

        return ckpt['epoch'] + 1, ckpt['best_val_loss']

    # =========================================================================
    # Internal: train one fold, with epoch-level checkpointing
    # =========================================================================

    def _train_fold(self, fold_data: Dict, fold_idx: int) -> Tuple[nn.Module, Dict]:
        print(f"\n{'=' * 70}")
        print(f"TRAINING FOLD {fold_idx + 1}/{self.n_folds}")
        print(f"{'=' * 70}")

        model = copy.deepcopy(self.model_template).to(self.device)
        if self.enable_gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

        train_input = self.input_modality[fold_data['train_idx']]
        train_output = self.output_modality[fold_data['train_idx']]
        val_input = self.input_modality[fold_data['val_idx']]
        val_output = self.output_modality[fold_data['val_idx']]
        test_input = self.input_modality[fold_data['test_idx']]
        test_output = self.output_modality[fold_data['test_idx']]

        train_loader, val_loader, test_loader = self._create_dataloaders(
            train_input, train_output, val_input, val_output, test_input, test_output,
        )

        criterion = get_loss_function(self.loss_function, self.device, weights=self.loss_weights)
        optimizer = get_optimizer(model, self.optimizer_type, self.learning_rate, self.weight_decay)

        if self.scheduler_type and self.scheduler_type.lower() == 'onecycle':
            self.scheduler_params['total_steps'] = len(train_loader) * self.n_epochs

        scheduler = get_scheduler(optimizer, self.scheduler_type, self.scheduler_params)
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        plotter = TrainingPlotter(os.path.join(self.save_dir, f'fold_{fold_idx + 1}'))

        # Resume from checkpoint if this fold's job was previously killed
        # (e.g. by a SLURM time limit) and restarted.
        start_epoch, best_val_loss = self._load_checkpoint(fold_idx, model, optimizer, scheduler)
        if start_epoch > 0:
            print(f"  Resuming from epoch {start_epoch} (best val loss so far: {best_val_loss:.6f})")

        last_epoch = start_epoch  # in case the loop body never executes

        for epoch in range(start_epoch, self.n_epochs):
            last_epoch = epoch

            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{self.n_epochs}")

            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_metrics = self._validate(model, val_loader, criterion)

            current_lr = optimizer.param_groups[0]['lr']
            plotter.update(train_loss, val_loss, current_lr, val_metrics)

            if self.verbose:
                print(f"  Train Loss : {train_loss:.6f}")
                print(f"  Val Loss   : {val_loss:.6f}")
                print(f"  Val MAE={val_metrics['mae']:.4f}  "
                      f"SSIM={val_metrics['ssim']:.4f}  Corr={val_metrics['correlation']:.4f}")

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Overwrites the previous checkpoint -- only the latest epoch is kept.
            self._save_checkpoint(fold_idx, epoch, model, optimizer, scheduler, best_val_loss)

        early_stopping.load_best_model(model)

        test_loss, test_metrics = self._validate(model, test_loader, criterion)

        print(f"\nFold {fold_idx + 1} test results:")
        print(f"  Test Loss : {test_loss:.6f}")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        plotter.plot_losses(fold=fold_idx + 1)
        plotter.plot_metrics(fold=fold_idx + 1)

        test_preds = self._predict_3d(model, test_input[:3])
        plotter.plot_predictions(test_input[:3], test_output[:3], test_preds, fold=fold_idx + 1)

        if self.save_all_fold_models:
            torch.save(model.state_dict(), os.path.join(self.save_dir, f'model_fold_{fold_idx + 1}.pth'))

        del train_input, train_output, val_input, val_output, test_input, test_output
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

        ckpt_path = self._checkpoint_path(fold_idx)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            if self.verbose:
                print(f"  Checkpoint removed: {ckpt_path}")

        fold_result = {
            'fold': fold_idx + 1,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'n_epochs_trained': last_epoch + 1,
        }
        return model, fold_result

    # =========================================================================
    # Internal: configuration and reporting
    # =========================================================================

    def _print_config(self):
        print("=" * 70)
        print("U-NET TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model dimensions: {'2D' if self.dimensions == 2 else '3D'}")
        print(f"Input shape: {self.input_modality.shape}")
        print(f"Output shape: {self.output_modality.shape}")
        print(f"Device: {self.device}")
        print("-" * 70)
        print(f"External test samples: {self.samples_reserved_to_external_test}")
        print(f"Fold strategy: {self.fold_strategy}")
        print(f"Number of folds: {self.n_folds}")
        print(f"Train/Val/Test ratio: {self.train_ratio}/{self.val_ratio}/{self.test_ratio}")
        print("-" * 70)
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.n_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"Loss function: {self.loss_function}")
        print(f"Scheduler: {self.scheduler_type}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Gradient checkpointing: {self.enable_gradient_checkpointing}")
        print("=" * 70)

    def _get_group_column(self) -> str:
        """Pick the grouping column matching `fold_strategy` ('family' or
        'site') so that no family/site appears in both train and test."""
        if self.fold_strategy.lower() == 'family':
            for col in self.groups.columns:
                if 'family' in col.lower() or 'fam' in col.lower():
                    return col
        elif self.fold_strategy.lower() == 'site':
            for col in self.groups.columns:
                if 'site' in col.lower():
                    return col
        return self.groups.columns[0]

    def _split_external_test(self) -> Tuple:
        """Hold out whole groups (families/sites) for the external test set,
        so no group is split across the external set and cross-validation."""
        group_col = self._get_group_column()
        groups_array = self.groups[group_col].values
        unique_groups = np.unique(groups_array)
        np.random.shuffle(unique_groups)

        group_to_indices = {g: np.where(groups_array == g)[0] for g in unique_groups}
        group_sizes = np.array([len(group_to_indices[g]) for g in unique_groups])
        cumsum_sizes = np.cumsum(group_sizes)
        n_ext = np.searchsorted(cumsum_sizes, self.samples_reserved_to_external_test, side='left') + 1
        n_ext = min(n_ext, len(unique_groups))

        external_groups = set(unique_groups[:n_ext])
        external_mask = np.array([g in external_groups for g in groups_array])
        remaining_mask = ~external_mask
        external_indices = np.where(external_mask)[0]

        external_test_df = pd.DataFrame({
            'index': external_indices,
            'iid': self.iid.iloc[external_indices].values,
            'eid': self.eid.iloc[external_indices].values,
            'fid': self.fid.iloc[external_indices].values,
        })
        external_test_df.to_csv(os.path.join(self.save_dir, 'external_test_indices.csv'), index=False)

        self.external_indices = external_indices
        self.external_test_df = external_test_df

        external_input = self.input_modality[external_mask]
        external_output = self.output_modality[external_mask]
        remaining_input = self.input_modality[remaining_mask]
        remaining_output = self.output_modality[remaining_mask]
        remaining_groups_df = self.groups[remaining_mask].reset_index(drop=True)

        if self.verbose:
            print("\nExternal test split:")
            print(f"  External: {len(external_input)}, Remaining: {len(remaining_input)}")

        return (external_input, external_output,
                remaining_input, remaining_output,
                external_mask, remaining_mask, remaining_groups_df)

    def _check_overlap(self, external_mask: np.ndarray, remaining_mask: np.ndarray):
        """Confirm the external test set and the CV pool share no samples or
        groups -- a hard guarantee against data leakage."""
        overlap = np.sum(external_mask & remaining_mask)
        print("\n" + "=" * 70)
        print("DATA OVERLAP CHECK")
        print("=" * 70)
        print(f"External test: {np.sum(external_mask)}, Remaining: {np.sum(remaining_mask)}")
        print(f"Overlap: {overlap}")
        print("No sample overlap" if overlap == 0 else "WARNING: sample overlap detected!")

        group_col = self._get_group_column()
        ext_grps = set(self.groups[external_mask][group_col].unique())
        rem_grps = set(self.groups[remaining_mask][group_col].unique())
        grp_overlap = ext_grps & rem_grps
        if grp_overlap:
            print(f"WARNING: overlapping groups between external test and CV pool: {grp_overlap}")
        else:
            print("No group overlap")
        print("=" * 70)

    def _create_fold_splits(self, remaining_input, remaining_output, remaining_groups) -> List[Dict]:
        """Grouped k-fold: GroupKFold provides the test fold per group, then
        the remaining groups are split into train/val by ratio."""
        group_col = self._get_group_column()
        groups_array = remaining_groups[group_col].values
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_splits = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(remaining_input, groups=groups_array)):
            train_val_groups = groups_array[train_val_idx]
            unique_tv_groups = np.unique(train_val_groups)
            np.random.shuffle(unique_tv_groups)

            val_ratio_adj = self.val_ratio / (self.train_ratio + self.val_ratio)
            n_val_groups = max(1, int(len(unique_tv_groups) * val_ratio_adj))

            val_groups_set = set(unique_tv_groups[:n_val_groups])
            train_groups_set = set(unique_tv_groups[n_val_groups:])

            train_mask = np.array([g in train_groups_set for g in train_val_groups])
            val_mask = np.array([g in val_groups_set for g in train_val_groups])

            train_idx = train_val_idx[train_mask]
            val_idx = train_val_idx[val_mask]

            fold_splits.append({
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
            })

            if self.verbose:
                print(f"\nFold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return fold_splits

    def _create_dataloaders(self, train_input, train_output, val_input, val_output, test_input, test_output):
        if self.dimensions == 3:
            aug = Augmentation3D(self.augmentation_config)
            train_dataset = Dataset3D(train_input, train_output, augmentation=aug, train=True)
            val_dataset = Dataset3D(val_input, val_output, augmentation=None, train=False)
            test_dataset = Dataset3D(test_input, test_output, augmentation=None, train=False)
        else:
            aug = Augmentation2D(self.augmentation_config)
            train_dataset = Dataset2D(train_input, train_output, slice_axis=self.slice_axis,
                                       skip_background=self.skip_background_slices, augmentation=aug, train=True)
            val_dataset = Dataset2D(val_input, val_output, slice_axis=self.slice_axis,
                                     skip_background=self.skip_background_slices, augmentation=None, train=False)
            test_dataset = Dataset2D(test_input, test_output, slice_axis=self.slice_axis,
                                      skip_background=self.skip_background_slices, augmentation=None, train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader

    def _train_epoch(self, model, train_loader, optimizer, criterion) -> float:
        model.train()
        total_loss = 0
        n_batches = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)[0] if isinstance(criterion, CombinedLoss) else criterion(outputs, targets)
            loss.backward()

            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

            if self.verbose and (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(f"    Batch {batch_idx + 1}/{n_batches}, Loss: {loss.item():.6f}")

        torch.cuda.empty_cache()
        return total_loss / n_batches

    def _validate(self, model, val_loader, criterion) -> Tuple[float, Dict]:
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = model(inputs)

                loss = criterion(outputs, targets)[0] if isinstance(criterion, CombinedLoss) else criterion(outputs, targets)

                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        torch.cuda.empty_cache()
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = compute_metrics(all_preds.squeeze(), all_targets.squeeze())
        return total_loss / len(val_loader), metrics

    def _predict_3d(self, model, inputs_3d):
        """Run inference on whole volumes, slice-by-slice if the model is 2D."""
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(inputs_3d)):
                if self.dimensions == 3:
                    inp = np.expand_dims(inputs_3d[i:i + 1], axis=1)
                    inp = torch.from_numpy(inp.astype(np.float32)).to(self.device, non_blocking=True)
                    pred = model(inp)
                    predictions.append(pred.cpu().numpy().squeeze())
                else:
                    volume_shape = inputs_3d[i].shape
                    pred_volume = np.zeros(volume_shape)
                    axis_map = {'axial': 0, 'sagittal': 1, 'coronal': 2}
                    axis = axis_map.get(self.slice_axis, 0)

                    for slice_idx in range(volume_shape[axis]):
                        if axis == 0:
                            sl = inputs_3d[i][slice_idx:slice_idx + 1, :, :]
                        elif axis == 1:
                            sl = inputs_3d[i][:, slice_idx:slice_idx + 1, :].transpose(1, 0, 2)
                        else:
                            sl = inputs_3d[i][:, :, slice_idx:slice_idx + 1].transpose(2, 0, 1)

                        sl = np.expand_dims(sl, axis=0)
                        sl = torch.from_numpy(sl.astype(np.float32)).to(self.device, non_blocking=True)
                        pred_sl = model(sl).cpu().numpy().squeeze()

                        if axis == 0:
                            pred_volume[slice_idx, :, :] = pred_sl
                        elif axis == 1:
                            pred_volume[:, slice_idx, :] = pred_sl
                        else:
                            pred_volume[:, :, slice_idx] = pred_sl

                    predictions.append(pred_volume)

        torch.cuda.empty_cache()
        return np.array(predictions)

    def _save_nifti(self, data: np.ndarray, filename: str):
        import nibabel as nib
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine), filename)

    def _evaluate_external_test(self, external_input, external_output):
        print(f"\n{'=' * 70}")
        print("EXTERNAL TEST SET EVALUATION")
        print(f"{'=' * 70}")

        all_predictions = []
        for fold_idx, model in enumerate(self.fold_models):
            if self.verbose:
                print(f"\nPredicting with model {fold_idx + 1}/{len(self.fold_models)}...")
            model.eval()
            preds = self._predict_3d(model, external_input)
            all_predictions.append(preds)
            metrics = compute_metrics(preds, external_output)
            print(f"\nFold {fold_idx + 1} on external test:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")

        all_predictions = np.array(all_predictions)
        ensemble_preds = np.mean(all_predictions, axis=0)
        ensemble_metrics = compute_metrics(ensemble_preds, external_output)

        print("\nEnsemble on external test:")
        for metric, value in ensemble_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        if self.save_predictions:
            self._save_external_predictions(external_input, external_output, all_predictions, ensemble_preds)

        plotter = TrainingPlotter(self.save_dir)
        plotter.plot_predictions(external_input[:5], external_output[:5], ensemble_preds[:5], n_samples=5, save=True)
        plt.savefig(os.path.join(self.save_dir, 'external_test_predictions.png'), dpi=150)

        self.external_test_results = {
            'fold_predictions': all_predictions,
            'ensemble_predictions': ensemble_preds,
            'fold_metrics': [compute_metrics(p, external_output) for p in all_predictions],
            'ensemble_metrics': ensemble_metrics,
        }
        return ensemble_preds, ensemble_metrics

    def _save_external_predictions(self, external_input, external_output, all_predictions, ensemble_preds):
        external_dir = os.path.join(self.save_dir, 'external_test')
        os.makedirs(external_dir, exist_ok=True)

        np.save(os.path.join(external_dir, 'input.npy'), external_input)
        np.save(os.path.join(external_dir, 'target.npy'), external_output)
        np.save(os.path.join(external_dir, 'predictions_all_folds.npy'), all_predictions)
        np.save(os.path.join(external_dir, 'predictions_ensemble.npy'), ensemble_preds)

        input_4d = np.transpose(external_input, (1, 2, 3, 0))
        target_4d = np.transpose(external_output, (1, 2, 3, 0))
        ensemble_4d = np.transpose(ensemble_preds, (1, 2, 3, 0))

        self._save_nifti(input_4d, os.path.join(external_dir, 'input.nii.gz'))
        self._save_nifti(target_4d, os.path.join(external_dir, 'target.nii.gz'))
        self._save_nifti(ensemble_4d, os.path.join(external_dir, 'predictions_ensemble.nii.gz'))

        for fold_idx in range(len(all_predictions)):
            fold_preds_4d = np.transpose(all_predictions[fold_idx], (1, 2, 3, 0))
            self._save_nifti(fold_preds_4d, os.path.join(external_dir, f'predictions_fold_{fold_idx + 1}.nii.gz'))

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)

        if self.fold_results:
            metrics_names = list(self.fold_results[0]['test_metrics'].keys())
            print("\nFold results:")
            for fr in self.fold_results:
                print(f"Fold {fr['fold']}: val={fr['best_val_loss']:.6f}, "
                      f"test={fr['test_loss']:.6f}, epochs={fr['n_epochs_trained']}")
            print("\nAverage test metrics (mean +/- std across folds):")
            for metric in metrics_names:
                vals = [f['test_metrics'][metric] for f in self.fold_results]
                print(f"  {metric.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        if self.external_test_results:
            print("\nExternal test (ensemble):")
            for metric, value in self.external_test_results['ensemble_metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
        print("=" * 70)

    def _save_results(self):
        if self.fold_results:
            results_df = pd.DataFrame(self.fold_results)
            results_df.to_csv(os.path.join(self.save_dir, 'fold_results.csv'), index=False)

        config = {
            'n_folds': self.n_folds, 'fold_strategy': self.fold_strategy,
            'batch_size': self.batch_size, 'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate, 'optimizer': self.optimizer_type,
            'loss_function': self.loss_function, 'scheduler': self.scheduler_type,
            'early_stopping_patience': self.early_stopping_patience,
            'dimensions': self.dimensions,
            'gradient_checkpointing': self.enable_gradient_checkpointing,
        }
        with open(os.path.join(self.save_dir, 'config.txt'), 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        print(f"\nResults saved to {self.save_dir}")


def train_Unet(
    model: nn.Module,
    input_modality: np.ndarray,
    output_modality: np.ndarray,
    samples_reserved_to_external_test: int,
    groups: pd.DataFrame,
    iid: pd.Series,
    eid: pd.Series,
    fid: pd.Series,
    fold_id: Optional[int] = None,       # None = all folds, 0-4 = a single fold
    preprocess_only: bool = False,       # Phase 1: split + save indices
    aggregate_only: bool = False,        # Phase 3: load models + external test
    n_folds: int = 5,
    fold_strategy: str = 'family',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 8,
    n_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    optimizer_type: str = 'adam',
    loss_function: str = 'l1',
    scheduler_type: str = 'plateau',
    scheduler_params: Dict = None,
    clip_grad: float = None,
    early_stopping_patience: int = 15,
    augmentation_config: Dict = None,
    device: str = 'cuda',
    save_dir: str = './experiments',
    experiment_name: str = None,
    random_state: int = 42,
    verbose: bool = True,
    save_all_fold_models: bool = True,
    save_predictions: bool = True,
    slice_axis: str = 'axial',
    skip_background_slices: int = 2,
    loss_weights: Dict = None,
    enable_gradient_checkpointing: bool = True,
) -> Dict:
    """Entry point used by the notebook. Three mutually exclusive modes,
    selected by the SLURM phase (see the routing cell in the notebook):

        preprocess_only=True   -> Phase 1: split data, save fold indices to disk
        fold_id=N               -> Phase 2: train fold N (0-based) on one GPU
        aggregate_only=True     -> Phase 3: load all models, evaluate external test
        (none of the above)     -> train all folds sequentially in one job
    """
    trainer = UNetTrainer(
        model=model,
        input_modality=input_modality,
        output_modality=output_modality,
        samples_reserved_to_external_test=samples_reserved_to_external_test,
        groups=groups, iid=iid, eid=eid, fid=fid,
        n_folds=n_folds, fold_strategy=fold_strategy,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        batch_size=batch_size, n_epochs=n_epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        optimizer_type=optimizer_type, loss_function=loss_function,
        scheduler_type=scheduler_type, scheduler_params=scheduler_params,
        clip_grad=clip_grad, early_stopping_patience=early_stopping_patience,
        augmentation_config=augmentation_config,
        device=device, save_dir=save_dir, experiment_name=experiment_name,
        random_state=random_state, verbose=verbose,
        save_all_fold_models=save_all_fold_models, save_predictions=save_predictions,
        slice_axis=slice_axis, skip_background_slices=skip_background_slices,
        loss_weights=loss_weights,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )

    if preprocess_only:
        trainer.preprocess()
        return {}

    if aggregate_only:
        return trainer.aggregate()

    return trainer.train(fold_id=fold_id)
