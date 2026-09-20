#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2026-04-13 MLP 回归实验
  - 直接用 MLP 预测截面收益，不做 fusion 三模型融合
  - all panel 训练，融券池尾组
  - 参照 BL-01 设置（全市场 MV 前 75% 尾组约束）
  - 包含 M-01~M-11（网络/标签/标准化对比）和 W-01~W-06（滚动窗口）
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "func"))
os.chdir("/root/quant")

import experiment_engine as ree
from func.dl_utils import (
    CrossSectionDataset, EarlyStopper,
    neutralize_label, neutralize_label_rank, raw_label,
    train_model, predict_model, zscore_by_date,
)

np.random.seed(42)
torch.manual_seed(42)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_print_lock = threading.Lock()

# ── 常量 ──
TRAIN_START = "2020-01-01"
BACKTEST_START = "2024-01-01"
DATA_END = "2026-03-01"
WINDOW_SIZE = 4          # 默认半年切分，4 段 = 2 年
ALL_DATA_DIR = "Data/all"
OUTPUT_DIR = "output/0413_mlp"
SHORT_LIST_PATH = "Data/short/short_list.pkl"
PRICE_PATH = "Data/all/panel_trade.pkl"
AVG_RETURN_PATH = "Data/all/avg_return_daily.pkl"
MV_TAIL_QUANTILE = 0.75  # 全市场 MV 前 75%
N_GROUPS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 日志 ──
LOG_PATH = os.path.join("output", "0413_mlp_run.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

class _Logger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "a", encoding="utf-8")
    def write(self, msg):
        try: self.terminal.write(msg)
        except UnicodeEncodeError: self.terminal.write(msg.encode("ascii","replace").decode("ascii"))
        self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

sys.stdout = _Logger(LOG_PATH)
sys.stderr = _Logger(LOG_PATH)


# =====================================================================
# 模型
# =====================================================================

class MLPRegressor(nn.Module):
    """通用 MLP 回归模型。hidden_dims 为隐藏层维度列表，如 [512, 256]。"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================================
# 信号构建
# =====================================================================

def build_signals_from_factor(
    factor_df: pd.DataFrame,
    short_codes: set,
    n_groups_head: int = 10,
    n_groups_tail: int = 10,
    mv_quantile: float = 0.75,
) -> Tuple[pd.Series, pd.Series]:
    """
    头组: 全市场 factor 最高的 1/n_groups_head
    尾组: 融券池中 factor 最低的 1/n_groups_tail，且市值前 mv_quantile
    返回 (head_signal, tail_signal)，每个为 Series[list[str]]
    """
    head_records, tail_records = [], []

    for dt, grp in factor_df.groupby("date"):
        # 头组: 全市场
        grp_h = grp.dropna(subset=["factor"]).copy()
        if len(grp_h) < n_groups_head * 2:
            continue
        grp_h["group_h"] = pd.qcut(
            grp_h["factor"].rank(method="first"),
            q=n_groups_head, labels=False, duplicates="drop",
        )
        head_codes = (
            grp_h.loc[grp_h["group_h"] == n_groups_head - 1, "stock_code"]
            .dropna().astype(str).tolist()
        )
        head_records.append((dt, head_codes))

        # 尾组: 融券池 + MV 过滤
        grp_t = grp_h[grp_h["stock_code"].isin(short_codes)].copy()
        if "market_value" in grp_t.columns:
            mv_thresh = grp_h["market_value"].quantile(1 - mv_quantile)
            grp_t = grp_t[grp_t["market_value"] >= mv_thresh]
        if len(grp_t) < n_groups_tail * 2:
            tail_records.append((dt, []))
            continue
        grp_t["group_t"] = pd.qcut(
            grp_t["factor"].rank(method="first"),
            q=n_groups_tail, labels=False, duplicates="drop",
        )
        tail_codes = (
            grp_t.loc[grp_t["group_t"] == 0, "stock_code"]
            .dropna().astype(str).tolist()
        )
        tail_records.append((dt, tail_codes))

    def _to_series(records):
        if not records:
            return pd.Series(dtype=object)
        idx, vals = zip(*records)
        return pd.Series(list(vals), index=pd.DatetimeIndex(idx))

    return _to_series(head_records), _to_series(tail_records)


# =====================================================================
# 数据切分
# =====================================================================

def quarter_split(df: pd.DataFrame, start: str, end: str) -> List[pd.DataFrame]:
    """按季度（3 个月）切分 DataFrame，返回列表。"""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    min_dt = pd.to_datetime(df["date"].min())
    start_dt = max(start_dt, min_dt)

    dfs = []
    t = start_dt.normalize()
    while t < end_dt:
        t_next = t + pd.DateOffset(months=3)
        chunk = df[(df["date"] >= t) & (df["date"] < t_next)]
        if len(chunk) > 0:
            dfs.append(chunk)
        t = t_next
    return dfs


def load_panel_split(panel_path: str, split_months: int,
                     start: str, end: str) -> List[pd.DataFrame]:
    """加载 panel 并按指定粒度切分。split_months=6 用 half_year，3 用 quarter_split。"""
    panel = pd.read_pickle(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    if split_months == 6:
        dfs = ree.half_year(panel, start, end)
    elif split_months == 3:
        dfs = quarter_split(panel, start, end)
    else:
        raise ValueError(f"不支持的 split_months: {split_months}")
    del panel
    gc.collect()
    return dfs


# =====================================================================
# 单窗口处理
# =====================================================================

def _process_one_window(
    window_idx: int,
    total_windows: int,
    tr: pd.DataFrame,
    test_raw: pd.DataFrame,
    fcols: List[str],
    cfg: Dict,
) -> Optional[pd.DataFrame]:
    """训练一个窗口的 MLP 并返回 test 上的 factor_df。"""
    print(f"    窗口 {window_idx}/{total_windows} 开始 ...")

    hidden_dims = cfg["hidden_dims"]
    label_fn = cfg["label_fn"]
    use_zscore = cfg.get("use_zscore", False)
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    batch_size = cfg.get("batch_size", 4096)
    max_epochs = cfg.get("max_epochs", 100)
    patience = cfg.get("patience", 10)
    dropout = cfg.get("dropout", 0.3)

    # 标签
    y_tr = label_fn(tr, "c_pct_5")
    valid_mask = np.isfinite(y_tr)
    tr_clean = tr[valid_mask].reset_index(drop=True)
    y_tr = y_tr[valid_mask]

    # 特征
    X_tr = tr_clean[fcols].values.astype(np.float32)
    nan_mask = np.isnan(X_tr)
    if nan_mask.any():
        col_mean = np.nanmean(X_tr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        inds = np.where(nan_mask)
        X_tr[inds] = col_mean[inds[1]]

    # zscore 标准化（按训练集统计量）
    if use_zscore:
        tr_mean = X_tr.mean(axis=0)
        tr_std = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - tr_mean) / tr_std

    # train/val 划分（随机 80/20）
    n = len(X_tr)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X_tr[train_idx], y_tr[train_idx]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]

    # 模型
    input_dim = X_train.shape[1]
    model = MLPRegressor(input_dim, hidden_dims, dropout=dropout)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, device="cpu", verbose=False,
    )
    best_epoch = history.get("best_epoch", -1)

    # 预测
    X_test = test_raw[fcols].values.astype(np.float32)
    nan_mask_test = np.isnan(X_test)
    if nan_mask_test.any():
        col_mean_test = np.nanmean(X_test, axis=0)
        col_mean_test = np.where(np.isnan(col_mean_test), 0.0, col_mean_test)
        inds_t = np.where(nan_mask_test)
        X_test[inds_t] = col_mean_test[inds_t[1]]

    if use_zscore:
        X_test = (X_test - tr_mean) / tr_std

    preds = predict_model(model, X_test, batch_size=8192, device="cpu")

    # 截面 zscore
    dates_test = test_raw["date"].values
    preds_z = zscore_by_date(preds, dates_test)

    # 构建 factor_df
    out = test_raw[["date", "stock_code", "market_value"]].copy()
    out["factor"] = preds_z

    # return_neutral（compute_rank_ic 需要）
    out["return_neutral"] = neutralize_label(test_raw, "c_pct_5")

    print(f"    窗口 {window_idx}/{total_windows} 完成, "
          f"样本 {n}, best_epoch {best_epoch}")

    del model, X_tr, X_train, X_val, X_test
    gc.collect()
    return out


# =====================================================================
# 滚动训练
# =====================================================================

def rolling_train_predict(
    dfs: List[pd.DataFrame],
    fcols: List[str],
    cfg: Dict,
    window_size: int = 4,
) -> pd.DataFrame:
    """滚动窗口训练 + 预测，返回拼接后的 factor_df。"""
    total = len(dfs)
    n_windows = total - window_size
    if n_windows <= 0:
        raise ValueError(f"数据段数 {total} <= 窗口大小 {window_size}，无法滚动训练")

    print(f"    共 {n_windows} 窗口, 顺序执行")
    results = []

    for i in range(window_size, total):
        tr = pd.concat(dfs[i - window_size: i], ignore_index=True)
        test_raw = dfs[i]
        out = _process_one_window(
            window_idx=i, total_windows=total,
            tr=tr, test_raw=test_raw, fcols=fcols, cfg=cfg,
        )
        if out is not None:
            results.append(out)
        del tr, test_raw
        gc.collect()

    if not results:
        raise RuntimeError("所有窗口均失败")
    return pd.concat(results, ignore_index=True)


# =====================================================================
# 多空回测图
# =====================================================================

def _save_yearly_longshort_figure(nav_df: pd.DataFrame, save_path: str,
                                  exp_name: str):
    """分年多空净值曲线。"""
    from func.model_backtest_framework_longshort import _calc_stats

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左图：全量净值
    ax = axes[0]
    ax.plot(nav_df.index, nav_df["head"], lw=2, color="red", label="头组")
    ax.plot(nav_df.index, nav_df["tail_raw"], lw=2, color="green", label="尾组(实际)")
    ax.plot(nav_df.index, nav_df["longshort"], lw=2, color="blue", label="多空")
    ax.plot(nav_df.index, nav_df["benchmark"], lw=1.5, color="gray",
            alpha=0.7, label="基准")
    ax.set_title(f"{exp_name} 净值曲线", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：分年
    ax2 = axes[1]
    ls_nav = nav_df["longshort"]
    years = sorted(set(ls_nav.index.year))
    for yr in years:
        yr_nav = ls_nav[ls_nav.index.year == yr]
        if len(yr_nav) < 2:
            continue
        yr_nav = yr_nav / yr_nav.iloc[0]
        ax2.plot(range(len(yr_nav)), yr_nav, label=str(yr), lw=1.5)
    ax2.set_title(f"{exp_name} 分年多空净值", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# =====================================================================
# 实验运行
# =====================================================================

def run_experiment(
    exp: Dict,
    dfs: List[pd.DataFrame],
    fcols: List[str],
    short_codes: set,
    price_data: Dict,
    window_size: int = 4,
):
    """运行单个实验。"""
    exp_id = exp["name"]
    exp_dir = os.path.join(OUTPUT_DIR, exp_id)
    metrics_path = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"\n[{exp_id}] 已有 metrics.json，跳过")
        return

    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n{'─'*50}")
    t0 = time.time()

    ws = exp.get("window_size", window_size)
    desc_parts = [f"网络: {exp['hidden_dims']}", f"标签: {exp['label_name']}"]
    if ws != WINDOW_SIZE:
        desc_parts.append(f"窗口={ws}段")
    print(f"[{exp_id}] 开始  {time.strftime('%H:%M:%S')}")
    print(f"  {', '.join(desc_parts)}")

    cfg = {
        "hidden_dims": exp["hidden_dims"],
        "label_fn": exp["label_fn"],
        "use_zscore": exp.get("use_zscore", False),
        "lr": exp.get("lr", 1e-3),
        "weight_decay": exp.get("weight_decay", 1e-4),
        "batch_size": exp.get("batch_size", 4096),
        "max_epochs": exp.get("max_epochs", 100),
        "patience": exp.get("patience", 10),
        "dropout": exp.get("dropout", 0.3),
    }

    try:
        factor_df = rolling_train_predict(dfs, fcols, cfg, window_size=ws)
    except Exception as e:
        print(f"  [{exp_id}] 训练失败: {e}")
        return

    elapsed = time.time() - t0

    # 合并 MV
    mv_all = pd.read_pickle(os.path.join(ALL_DATA_DIR, "panel_origin.pkl"))
    mv_all["date"] = pd.to_datetime(mv_all["date"])
    mv_cols = mv_all[["date", "stock_code", "market_value"]].drop_duplicates(
        ["date", "stock_code"], keep="last"
    )
    factor_df = factor_df.drop(columns=["market_value"], errors="ignore")
    factor_df = factor_df.merge(mv_cols, on=["date", "stock_code"], how="left")
    del mv_all, mv_cols
    gc.collect()

    # 回测期过滤
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    factor_df = factor_df[factor_df["date"] >= BACKTEST_START].copy()

    # 计算 RankIC
    ic_vals = []
    for _, g in factor_df.groupby("date"):
        if "return_neutral" in g.columns:
            ic = ree.calculate_rank_ic(g["factor"], g["return_neutral"])
            if not np.isnan(ic):
                ic_vals.append(ic)
    ic_mean = np.mean(ic_vals) if ic_vals else np.nan
    ic_std = np.std(ic_vals) if ic_vals else np.nan
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan

    # 信号构建
    head_signal, tail_signal = build_signals_from_factor(
        factor_df, short_codes, n_groups_head=10, n_groups_tail=10,
        mv_quantile=MV_TAIL_QUANTILE,
    )

    # 头尾去重
    overlap_removed = 0
    if len(head_signal) > 0 and len(tail_signal) > 0:
        common_dates = sorted(set(head_signal.index) & set(tail_signal.index))
        for dt in common_dates:
            h_set = set(head_signal.loc[dt])
            t_set = set(tail_signal.loc[dt])
            overlap = h_set & t_set
            if overlap:
                overlap_removed += len(overlap)
                head_signal.loc[dt] = [s for s in head_signal.loc[dt] if s not in overlap]
                tail_signal.loc[dt] = [s for s in tail_signal.loc[dt] if s not in overlap]
    print(f"  [{exp_id}] 头尾去重 {overlap_removed} 只")

    # 多空回测
    from func.model_backtest_framework_longshort import analyze_longshort_holdings

    ls_results = analyze_longshort_holdings(
        head_holdings_data=head_signal,
        tail_holdings_data=tail_signal,
        open_prices=price_data["open"],
        close_prices=price_data["close"],
        benchmark_data=price_data.get("benchmark"),
        status_data=price_data.get("status"),
        method="daily_rebalance",
        holding_period=1,
        commission_rate=0.0007,
        verbose=False,
    )

    stats = ls_results["statistics"]

    metrics = {
        "experiment": exp_id,
        "rank_ic_mean": round(ic_mean, 6) if not np.isnan(ic_mean) else ic_mean,
        "rank_ic_std": round(ic_std, 6) if not np.isnan(ic_std) else ic_std,
        "rank_ic_ir": round(ic_ir, 6) if not np.isnan(ic_ir) else ic_ir,
        "elapsed_sec": round(elapsed, 1),
        "overlap_removed": overlap_removed,
        "hidden_dims": exp["hidden_dims"],
        "label": exp["label_name"],
        "head": stats["head"],
        "tail": stats["tail"],
        "tail_raw": stats["tail_raw"],
        "longshort": stats["longshort"],
        "benchmark": stats["benchmark"],
        "head_excess": stats["head_excess"],
        "tail_excess": stats["tail_excess"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # 保存 factor_df
    factor_df.to_pickle(os.path.join(exp_dir, "factor_df.pkl"))

    # 保存图
    nav_df = ls_results["nav_df"]
    _save_yearly_longshort_figure(
        nav_df, os.path.join(exp_dir, "yearly_longshort.png"), exp_id
    )

    ls = stats["longshort"]
    he = stats["head_excess"]
    print(f"  [{exp_id}] 训练完成, {int(elapsed)}s, "
          f"{len(factor_df)} 样本")
    print(f"  [{exp_id}] IC={ic_mean:.4f}  "
          f"多空年化={ls['annual_return']:.1f}%  "
          f"夏普={ls['sharpe_ratio']:.3f}  "
          f"回撤={ls['max_drawdown']:.1f}%  "
          f"头超={he['annual_return']:.1f}%")

    del factor_df, ls_results, nav_df
    gc.collect()


# =====================================================================
# 实验列表
# =====================================================================

EXPERIMENTS = [
    # ── 第 1 批: 网络容量 × 标签（无 zscore）──
    {"name": "M-01", "hidden_dims": [256, 128],       "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": False},
    {"name": "M-02", "hidden_dims": [512, 256],       "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": False},
    {"name": "M-03", "hidden_dims": [1024, 512, 256], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": False},
    {"name": "M-04", "hidden_dims": [256, 128],       "label_fn": raw_label,        "label_name": "原始",   "use_zscore": False},
    {"name": "M-05", "hidden_dims": [512, 256],       "label_fn": raw_label,        "label_name": "原始",   "use_zscore": False},
    {"name": "M-06", "hidden_dims": [1024, 512, 256], "label_fn": raw_label,        "label_name": "原始",   "use_zscore": False},

    # ── 第 2 批: zscore 标准化 ──
    {"name": "M-07", "hidden_dims": [256, 128],       "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True},
    {"name": "M-08", "hidden_dims": [256, 128],       "label_fn": raw_label,        "label_name": "原始",   "use_zscore": True},
    {"name": "M-09", "hidden_dims": [512, 256],       "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True},
    {"name": "M-10", "hidden_dims": [1024, 512, 256], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True},
    {"name": "M-11", "hidden_dims": [256, 128],       "label_fn": neutralize_label_rank, "label_name": "中性化", "use_zscore": True},

    # ── 第 3 批: 滚动窗口（基于 M-07 配置）──
    {"name": "W-01", "hidden_dims": [256, 128], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True, "split_months": 6, "window_size": 3},
    {"name": "W-02", "hidden_dims": [256, 128], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True, "split_months": 6, "window_size": 4},
    {"name": "W-03", "hidden_dims": [256, 128], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True, "split_months": 6, "window_size": 6},
    {"name": "W-04", "hidden_dims": [256, 128], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True, "split_months": 3, "window_size": 4},
    {"name": "W-05", "hidden_dims": [256, 128], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True, "split_months": 3, "window_size": 6},
    {"name": "W-06", "hidden_dims": [256, 128], "label_fn": neutralize_label, "label_name": "中性化", "use_zscore": True, "split_months": 3, "window_size": 8},
]


# =====================================================================
# 主函数
# =====================================================================

def main():
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"2026-04-13 MLP 实验  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  回测: {BACKTEST_START[:4]}-{DATA_END[:4]}, "
          f"尾组: 融券池 MV 前 {int(MV_TAIL_QUANTILE*100)}%")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  实验数: {len(EXPERIMENTS)}")
    print(f"  PyTorch: {torch.__version__}, CPU")
    print(f"{'='*60}")

    # 融券池
    print("  加载融券标的列表 ...")
    short_list = pd.read_pickle(SHORT_LIST_PATH)
    if isinstance(short_list, pd.DataFrame):
        short_codes = set(short_list["stock_code"].astype(str).unique())
    else:
        short_codes = set(str(x) for x in short_list)
    print(f"  融券池: {len(short_codes)} 只")

    # 价格数据
    print("  加载价格数据 ...")
    pt = pd.read_pickle(PRICE_PATH)
    pt["date"] = pd.to_datetime(pt["date"])
    price_data = {
        "open": pt.pivot(index="date", columns="stock_code", values="open_price").sort_index(),
        "close": pt.pivot(index="date", columns="stock_code", values="close_price").sort_index(),
        "status": pt.pivot(index="date", columns="stock_code", values="status").sort_index(),
    }
    # 基准
    avg_ret = pd.read_pickle(AVG_RETURN_PATH)
    if isinstance(avg_ret, pd.DataFrame):
        if "date" not in avg_ret.columns:
            avg_ret = avg_ret.reset_index()
        avg_ret["date"] = pd.to_datetime(avg_ret["date"])
        pcol = "close_price" if "close_price" in avg_ret.columns else "open_price"
        bm = avg_ret.set_index("date")[pcol].sort_index().pct_change().fillna(0)
        price_data["benchmark"] = bm
    del pt
    gc.collect()

    # 数据缓存：按 split_months 缓存，避免重复加载
    data_cache: Dict[int, List[pd.DataFrame]] = {}
    panel_path = os.path.join(ALL_DATA_DIR, "panel_origin.pkl")

    # 特征列
    fcols = None

    for exp in EXPERIMENTS:
        sm = exp.get("split_months", None)
        if sm is None:
            sm = 6  # M-01~M-11 默认半年切分

        if sm not in data_cache:
            print(f"  加载 all ({sm}月切分): {panel_path} ...")
            dfs = load_panel_split(panel_path, sm, TRAIN_START, DATA_END)
            data_cache[sm] = dfs
            print(f"  all: {sum(len(d) for d in dfs):,} 行, "
                  f"{len(dfs[0].columns) - 4 if dfs else '?'} 特征, "
                  f"{len(dfs)} 区间")

            if fcols is None and dfs:
                exclude = {"date", "stock_code", "market_value", "c_pct_5",
                           "return_neutral", "status", "open_price", "close_price",
                           "industry", "industry_code"}
                fcols = [c for c in dfs[0].columns if c not in exclude]

        dfs = data_cache[sm]
        ws = exp.get("window_size", WINDOW_SIZE)

        run_experiment(exp, dfs, fcols, short_codes, price_data, window_size=ws)

    # 汇总
    print(f"\n{'='*60}")
    print("实验汇总：")
    completed = []
    for exp in EXPERIMENTS:
        mpath = os.path.join(OUTPUT_DIR, exp["name"], "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            ls = m.get("longshort", {})
            completed.append({
                "实验": exp["name"],
                "IC": f"{m.get('rank_ic_mean', 'nan'):.4f}" if isinstance(m.get('rank_ic_mean'), (int, float)) and not np.isnan(m.get('rank_ic_mean', np.nan)) else "nan",
                "多空年化": f"{ls.get('annual_return', 'nan'):.1f}%",
                "夏普": f"{ls.get('sharpe_ratio', 'nan'):.3f}",
                "回撤": f"{ls.get('max_drawdown', 'nan'):.1f}%",
            })
    if completed:
        summary_df = pd.DataFrame(completed)
        print(summary_df.to_string(index=False))

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed/3600:.1f} 小时")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
