"""
深度学习通用工具函数
  - 数据集：截面回归 Dataset
  - 训练器：通用的 epoch 训练 + 早停
  - 标签处理：市值中性化 / 原始
  - 预测：批量推理

设计原则：
  1. 与模型解耦，只依赖 torch / numpy / pandas
  2. 供 MLP / LSTM / Transformer 等不同模型复用
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Tuple


# =====================================================================
# 数据集
# =====================================================================

class CrossSectionDataset(Dataset):
    """截面回归数据集：每个样本为 (特征向量, 标签标量)。
    输入为 numpy 数组，内部转为 float32 tensor。
    """

    def __init__(self, X: "np.ndarray", y: "np.ndarray"):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# 标签处理
# =====================================================================

def neutralize_label(df: "pd.DataFrame", label_col: str = "c_pct_5") -> "np.ndarray":
    """市值中性化标签：按日期分组，对 label 做 log(mv) 回归取残差。
    返回与 df 等长的 float32 数组。
    """
    dates = df["date"].values
    y = df[label_col].values.astype(np.float64)
    mv = np.log(df["market_value"].values.astype(np.float64).clip(1))
    result = np.empty_like(y, dtype=np.float32)

    for dt in np.unique(dates):
        mask = dates == dt
        y_d = y[mask]
        mv_d = mv[mask]
        X_d = np.column_stack([mv_d, np.ones(mask.sum())])
        beta, _ = np.linalg.lstsq(X_d, y_d, rcond=None)[:2]
        result[mask] = (y_d - X_d @ beta).astype(np.float32)

    return result


def neutralize_label_rank(df: "pd.DataFrame", label_col: str = "c_pct_5") -> "np.ndarray":
    """排名回归中性化：按日期分组，将 y 和 log(mv) 都转截面排名后回归取残差。
    对异常值更鲁棒。返回与 df 等长的 float32 数组。
    """
    from scipy.stats import rankdata

    dates = df["date"].values
    y = df[label_col].values.astype(np.float64)
    mv = np.log(df["market_value"].values.astype(np.float64).clip(1))
    result = np.empty_like(y, dtype=np.float32)

    for dt in np.unique(dates):
        mask = dates == dt
        y_d = y[mask]
        mv_d = mv[mask]
        y_rank = rankdata(y_d).astype(np.float64)
        mv_rank = rankdata(mv_d).astype(np.float64)
        X_d = np.column_stack([mv_rank, np.ones(mask.sum())])
        beta, _ = np.linalg.lstsq(X_d, y_rank, rcond=None)[:2]
        result[mask] = (y_rank - X_d @ beta).astype(np.float32)

    return result


def raw_label(df: "pd.DataFrame", label_col: str = "c_pct_5") -> "np.ndarray":
    """原始标签：直接返回 label 列。"""
    return df[label_col].values.astype(np.float32)


# =====================================================================
# 早停
# =====================================================================

class EarlyStopper:
    """早停器：验证集 loss 连续 patience 轮不降则停止。"""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: "nn.Module") -> bool:
        """返回 True 表示应该停止。"""
        if val_loss < self.best_loss - 1e-06:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: "nn.Module"):
        if self.best_state:
            model.load_state_dict(self.best_state)


# =====================================================================
# 训练
# =====================================================================

def train_model(
    model: "nn.Module",
    train_X: "np.ndarray",
    train_y: "np.ndarray",
    val_X: "np.ndarray",
    val_y: "np.ndarray",
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 4096,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
    verbose: bool = False,
) -> "Dict[str, list]":
    """
    通用模型训练函数。
    返回 {"train_loss": [...], "val_loss": [...], "best_epoch": int}。
    训练结束后 model 权重为验证集最优。
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_ds = CrossSectionDataset(train_X, train_y)
    val_ds = CrossSectionDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=2)

    stopper = EarlyStopper(patience=patience)
    history = {"train_loss": [], "val_loss": [], "best_epoch": -1}

    for epoch in range(max_epochs):
        # 训练
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
        train_loss = total_loss / total_n

        # 验证
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                val_total += loss.item() * len(yb)
                val_n += len(yb)
        val_loss = val_total / val_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if stopper.step(val_loss, model):
            history["best_epoch"] = epoch - stopper.counter
            break
    else:
        history["best_epoch"] = epoch - stopper.counter

    stopper.restore(model)
    return history


# =====================================================================
# 推理
# =====================================================================

@torch.no_grad()
def predict_model(
    model: "nn.Module",
    X: "np.ndarray",
    batch_size: int = 8192,
    device: str = "cpu",
) -> "np.ndarray":
    """批量推理，返回 float32 numpy 数组。"""
    model.to(device)
    model.eval()
    ds = CrossSectionDataset(X, np.zeros(len(X), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb).squeeze(-1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds).astype(np.float32)


# =====================================================================
# 截面标准化
# =====================================================================

def zscore_by_date(values: "np.ndarray", dates: "np.ndarray") -> "np.ndarray":
    """按日期做截面 zscore 标准化。"""
    result = np.empty_like(values, dtype=np.float32)
    for dt in np.unique(dates):
        mask = dates == dt
        v = values[mask].astype(np.float64)
        std = v.std()
        if std > 1e-12:
            result[mask] = ((v - v.mean()) / std).astype(np.float32)
        else:
            result[mask] = 0.0
    return result
