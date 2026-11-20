"""
dl_model.py — Scikit-learn compatible Deep Learning classifiers for tabular credit risk data.

Three architectures implemented and benchmarked for credit card default prediction:

  1. DeepMLPClassifier       — 4-layer MLP with BatchNorm + Dropout (strong baseline)
  2. ResNetTabularClassifier — Residual blocks (deeper networks without degradation)
  3. AttentionMLPClassifier  — Feature-attention gate + GELU MLP (sparse feature emphasis)

Why these work well for credit default prediction:
  - Class imbalance handled via BCEWithLogitsLoss(pos_weight) — no SMOTE needed
  - BatchNorm stabilizes training on mixed-scale financial features (NTD amounts vs ratios)
  - Dropout prevents overfitting on the 30k-row dataset
  - Custom threshold_ attribute aligns with the ML pipeline's threshold calibration

All classes implement the full sklearn estimator interface:
  fit / predict / predict_proba / get_params / set_params

Requirements:
    pip install torch

Optional TabNet (best-in-class tabular DL, install separately):
    pip install pytorch-tabnet
"""

import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin


# ─────────────────────────────────────────────────────────────────────────────
# Guard: fail with helpful message if torch not installed
# ─────────────────────────────────────────────────────────────────────────────
def _require_torch(cls_name: str):
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"{cls_name} requires PyTorch. Install it with:\n"
            "    pip install torch\n"
            "or visit https://pytorch.org/get-started/locally/ for CUDA builds."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deep MLP Classifier
# ─────────────────────────────────────────────────────────────────────────────
class DeepMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Deep Multi-Layer Perceptron for binary tabular classification.

    Architecture:
        Input
          → [Linear(in → h) → BatchNorm1d → ReLU → Dropout] × len(hidden_dims)
          → Linear(h → 1)
          → BCEWithLogitsLoss

    Regularization:
        - Dropout (default 0.3) for variance reduction
        - L2 weight decay via Adam optimizer
        - ReduceLROnPlateau scheduler halves LR on plateau

    Imbalance handling:
        - pos_weight = n_negative / n_positive passed to BCEWithLogitsLoss
          (equivalent to oversampling the minority class in loss space)

    Best suited for:
        General-purpose deep baseline on tabular financial data.
        Trains in ~10–30s on 24k rows × 34 features (CPU).

    Hyperparameters to tune:
        hidden_dims : tuple of ints   — layer widths, e.g. (256, 128, 64, 32)
        dropout     : float           — 0.2 (low reg) to 0.5 (high reg)
        lr          : float           — 1e-3 to 5e-4
        epochs      : int             — 50–150
        batch_size  : int             — 128, 256, 512
    """

    def __init__(
        self,
        hidden_dims=(256, 128, 64, 32),
        dropout=0.3,
        lr=1e-3,
        epochs=60,
        batch_size=256,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    # ── Internal network builder ──────────────────────────────────────────────
    def _build_net(self, input_dim):
        layers = []
        in_dim = input_dim
        for h in self.hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        _require_torch("DeepMLPClassifier")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_t = torch.tensor(np.array(X, dtype=np.float32))
        y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)

        self.classes_ = np.unique(y)
        n_pos = float(y_t.sum())
        n_neg = float(len(y_t) - n_pos)
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)])

        self.net_ = self._build_net(X_t.shape[1])
        optimizer = optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, verbose=False
        )

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        self.net_.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net_(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)

        return self

    def predict_proba(self, X):
        _require_torch("DeepMLPClassifier")
        self.net_.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32))
        with torch.no_grad():
            probs = torch.sigmoid(self.net_(X_t).squeeze(1)).numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        threshold = getattr(self, "custom_threshold_", 0.40)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tabular ResNet Classifier
# ─────────────────────────────────────────────────────────────────────────────
class _ResidualBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Linear → BN → ReLU → Linear + skip."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)  # residual / skip connection


class ResNetTabularClassifier(BaseEstimator, ClassifierMixin):
    """
    Tabular ResNet — residual connections for deep tabular networks.

    Architecture:
        Input
          → Linear(input_dim → hidden_dim)   [embedding / projection]
          → [ResidualBlock(hidden_dim)] × n_blocks
          → BatchNorm1d → ReLU
          → Linear(hidden_dim → 1)

    Why residual connections for tabular data:
        In deep plain MLPs, gradient signal degrades across many layers.
        Residual connections provide a "gradient highway" that lets earlier
        layers continue to receive useful updates — enabling networks of
        4–8 blocks without accuracy degradation.
        (Reference: "Revisiting Deep Learning Models for Tabular Data", Gorishniy 2021)

    Best suited for:
        When plain MLP plateaus and you want to go deeper.
        Typically outperforms plain MLP on datasets with complex non-linear
        feature interactions (e.g., bill trend × delinquency streak).

    Hyperparameters to tune:
        hidden_dim : int   — 64, 128, 256 (single hidden dimension for all blocks)
        n_blocks   : int   — 2–8 residual blocks
        dropout    : float — 0.1–0.4
    """

    def __init__(
        self,
        hidden_dim=128,
        n_blocks=4,
        dropout=0.25,
        lr=1e-3,
        epochs=60,
        batch_size=256,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _build_net(self, input_dim):
        layers = [nn.Linear(input_dim, self.hidden_dim)]
        for _ in range(self.n_blocks):
            layers.append(_ResidualBlock(self.hidden_dim, self.dropout))
        layers += [nn.BatchNorm1d(self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)]
        return nn.Sequential(*layers)

    def fit(self, X, y):
        _require_torch("ResNetTabularClassifier")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_t = torch.tensor(np.array(X, dtype=np.float32))
        y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)

        self.classes_ = np.unique(y)
        n_pos = float(y_t.sum())
        n_neg = float(len(y_t) - n_pos)
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)])

        self.net_ = self._build_net(X_t.shape[1])
        optimizer = optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        self.net_.train()
        for _ in range(self.epochs):
            for Xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.net_(Xb), yb).backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X):
        _require_torch("ResNetTabularClassifier")
        self.net_.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32))
        with torch.no_grad():
            probs = torch.sigmoid(self.net_(X_t).squeeze(1)).numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        threshold = getattr(self, "custom_threshold_", 0.40)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention MLP Classifier
# ─────────────────────────────────────────────────────────────────────────────
class _AttentionMLP(nn.Module):
    """
    Feature-attention gate followed by a GELU MLP.
    The attention layer learns a soft per-feature importance weight (via sigmoid)
    and gates the input before passing to the MLP — acting as a learned feature selector.
    """
    def __init__(self, input_dim: int, hidden_dims: tuple, dropout: float):
        super().__init__()
        # Soft attention gate: learns importance weight per feature
        self.attention_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),   # outputs in (0, 1) per feature — acts as feature mask
        )
        # Main MLP with GELU activations
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        gated = self.attention_gate(x) * x  # element-wise feature gating
        return self.mlp(gated)


class AttentionMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Attention-gated MLP — lightweight feature attention + deep GELU MLP.

    Architecture:
        Input
          → Feature Attention Gate [Linear → Sigmoid] ⊙ Input  (element-wise gate)
          → [Linear → BatchNorm1d → GELU → Dropout] × len(hidden_dims)
          → Linear(h → 1)

    Why attention for credit risk:
        Not all 34 engineered features are equally predictive for every applicant.
        The attention gate learns per-feature importance weights end-to-end, letting
        the model automatically focus on (e.g.) PAY_0 and CREDIT_UTILIZATION_RECENT
        for high-risk profiles while downweighting demographic noise.

    Why GELU instead of ReLU:
        GELU (Gaussian Error Linear Unit) is smoother than ReLU and tends to produce
        better results on small-to-medium tabular datasets — the same activation used
        in BERT, GPT, and modern transformers.

    Best suited for:
        High-dimensional feature spaces where manual feature selection is impractical.
        The attention gate provides implicit feature importance (interpretable via
        the gate's weight magnitudes).

    Hyperparameters to tune:
        hidden_dims : tuple — (128, 64, 32) baseline; wider = more capacity
        dropout     : float — 0.2–0.4
        lr          : float — 5e-4 to 2e-3
    """

    def __init__(
        self,
        hidden_dims=(128, 64, 32),
        dropout=0.3,
        lr=1e-3,
        epochs=60,
        batch_size=256,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        _require_torch("AttentionMLPClassifier")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_t = torch.tensor(np.array(X, dtype=np.float32))
        y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)

        self.classes_ = np.unique(y)
        n_pos = float(y_t.sum())
        n_neg = float(len(y_t) - n_pos)
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)])

        self.net_ = _AttentionMLP(X_t.shape[1], self.hidden_dims, self.dropout)
        optimizer = optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            epochs=self.epochs,
            steps_per_epoch=max(1, len(X_t) // self.batch_size),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        self.net_.train()
        for _ in range(self.epochs):
            for Xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.net_(Xb), yb).backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        return self

    def predict_proba(self, X):
        _require_torch("AttentionMLPClassifier")
        self.net_.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32))
        with torch.no_grad():
            probs = torch.sigmoid(self.net_(X_t).squeeze(1)).numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        threshold = getattr(self, "custom_threshold_", 0.40)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: TabNet (best-in-class DL for tabular data)
# ─────────────────────────────────────────────────────────────────────────────
# TabNet uses sequential attention to select features at each decision step
# (like a differentiable decision tree). It achieves state-of-the-art results
# on many tabular benchmarks and natively outputs feature importances.
#
# To use TabNet:
#     pip install pytorch-tabnet
#
# Then in model_trainer.py:
#     from pytorch_tabnet.tab_model import TabNetClassifier
#     tabnet = TabNetClassifier(
#         n_d=32, n_a=32, n_steps=5,
#         gamma=1.5, n_independent=2, n_shared=2,
#         cat_idxs=[], cat_dims=[], cat_emb_dim=1,
#         optimizer_fn=torch.optim.Adam,
#         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
#         scheduler_params={"step_size":10, "gamma":0.9},
#         scheduler_fn=torch.optim.lr_scheduler.StepLR,
#         mask_type='entmax',
#         verbose=0,
#     )
#
# Note: TabNetClassifier has its own fit() with eval_set support — not directly
# compatible with GridSearchCV. Train and evaluate it separately like:
#     tabnet.fit(X_train, y_train, eval_set=[(X_test, y_test)], patience=15)
#     tabnet_proba = tabnet.predict_proba(X_test)[:, 1]
#     tabnet_roc   = roc_auc_score(y_test, tabnet_proba)
# ─────────────────────────────────────────────────────────────────────────────
