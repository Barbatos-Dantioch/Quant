#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OU 配对·神经网络(MLP)信号实验 - WS + noMktInd + ROLL504
==========================================================

在 OU-P-RPL-LGBEXP-WS-noMktInd-ROLL504-cv5-h5 基础上, **仅把模型由 LightGBM 换成 MLP**,
其余设计(22 特征 noMktInd、滚动 504 日窗、WS 截面收益加权、sign=sign(μ-X)、
top20%/min50、holding=5、5 段、train/val 80/20 + 5 日 gap)完全复用基础脚本
run_ou_pair_lgb_exp_ws.py 的数据加载 / 特征工程 / 选股 / 回测逻辑。

NN 专有处理(与树模型的关键差异):
  缺失值:  train 丢弃含 NaN 的行; val/test 用 train 各列中位数填补
  特征标准化: 全局标准化(每段在 train 上 fit StandardScaler, 应用到 val/test)
  标签标准化: 每段在 train 上对 ΔX 做 z-score(单调仿射, 不影响选股排序, 仅稳训练)

模型 / 训练(具体值):
  MLP: Input(22) → [128 → 64] (each: Linear+BatchNorm+ReLU+Dropout0.2) → Linear(1)
  损失: 加权 MSE(权重 = WS 截面权重); 优化器 Adam(lr=1e-3, weight_decay=1e-5)
  batch=4096, 最多 200 epoch, 早停 patience=20(监控加权 val RMSE)
  多种子: 沿用 ENSEMBLE_SEEDS(默认单种子 42), 各种子预测取均值

环境变量(透传给基础脚本): DROP_GROUPS(默认 mktind)、ROLL_DAYS(默认 504)、
ENSEMBLE_SEEDS、RUN_TAG 等; 本脚本据基础脚本 EXP_NAME 把 LGBEXP 替换为 NN 命名。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

# 线程预算: 本实验与其它实验并行, 限制为机器一半核 (可用 N_THREADS 覆盖)。
# 必须在 import numpy / torch 之前设置 BLAS/OMP 环境变量才生效, 防止底层并行占满核。
_N_THREADS = int(os.environ.get("N_THREADS", str(max(1, (os.cpu_count() or 2) // 2))))
for _ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_ev, str(_N_THREADS))
# numexpr 本机硬上限 64; 设为 min(预算,64) 与其默认上限不冲突, 消除 init 警告
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(min(_N_THREADS, 64)))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir("/root/quant")

# 本实验基线配置(可被外部 env 覆盖): noMktInd(22 特征) + 滚动 504 日窗
os.environ.setdefault("DROP_GROUPS", "mktind")
os.environ.setdefault("ROLL_DAYS", "504")

_PT_DIR = "/root/quant/xgbcode/pair_trading"
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

# 复用基础脚本(LGB 版)的数据加载 / 特征工程 / 选股 + 全部实验常量
import run_ou_pair_lgb_exp_ws as base
from _pair_runner import run_longshort_backtest, save_summary_row

import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── 复用基础脚本常量 ──
FEATURES = list(base.FEATURES)
# 因子消融: 环境变量 DROP_FEATURES (逗号分隔的因子名), 在 base 因子集上再剔除指定因子
#   (特征工程仍照常算全部列, 仅不进入模型输入; mu/X_t 即便被删仍保留供方向/加权用)
_DROP_FEATS = [x.strip() for x in os.environ.get("DROP_FEATURES", "").split(",") if x.strip()]
if _DROP_FEATS:
    _missing = [f for f in _DROP_FEATS if f not in FEATURES]
    assert not _missing, f"DROP_FEATURES 含未知/已不在的因子: {_missing}"
    FEATURES = [f for f in FEATURES if f not in _DROP_FEATS]
ROLLING_SEGMENTS = base.ROLLING_SEGMENTS
TRAIN_START = base.TRAIN_START
TRAIN_FRAC = base.TRAIN_FRAC
TRAIN_VAL_GAP_DAYS = base.TRAIN_VAL_GAP_DAYS
LABEL_HORIZON = base.LABEL_HORIZON
HOLDING_PERIOD = base.HOLDING_PERIOD
COMMISSION = base.COMMISSION
TOP_PCT = base.TOP_PCT
TOP_PCT_MIN_N = base.TOP_PCT_MIN_N
OUTPUT_DIR = base.OUTPUT_DIR
FULL_PAIR_LOG = base.FULL_PAIR_LOG
ENSEMBLE_SEEDS = base.ENSEMBLE_SEEDS
_ROLL_DAYS = base._ROLL_DAYS

# 损失类型: 环境变量 LOSS = "mse"(默认, 加权 MSE 回归) / "rank"(按日 ListNet 截面排序)
LOSS = os.environ.get("LOSS", "mse").strip().lower()
assert LOSS in ("mse", "rank"), f"LOSS 仅支持 mse/rank, 实际 {LOSS!r}"
_LOSS_TAG = "-rank" if LOSS == "rank" else ""

# 开仓方向来源: 环境变量 DIR = "ou"(默认, sign(μ-X) 回归先验) / "model"(sign(预测ΔX))
DIRECTION = os.environ.get("DIR", "ou").strip().lower()
assert DIRECTION in ("ou", "model"), f"DIR 仅支持 ou/model, 实际 {DIRECTION!r}"
assert not (DIRECTION == "model" and LOSS == "rank"), "DIR=model 仅适用于 mse (rank 输出非 ΔX, 无方向语义)"
_DIR_TAG = "-dirModel" if DIRECTION == "model" else ""

# 调参标签: 环境变量 TUNE_TAG(如 h256_128 / dp30 / lr3e-3); 设了则实验名加后缀且归入 nn/tuning/
_TUNE_TAG = os.environ.get("TUNE_TAG", "").strip()
_TUNE_SUFFIX = f"-{_TUNE_TAG}" if _TUNE_TAG else ""
# NN 实验名: 把基础脚本 EXP_NAME 里的 LGBEXP 替换为 NN, 在 -cv5-h5 前插入损失/调参/方向标签
EXP_NAME = base.EXP_NAME.replace("LGBEXP", "NN").replace(
    "-cv5-h5", f"{_LOSS_TAG}{_TUNE_SUFFIX}{_DIR_TAG}-cv5-h5")
# 正式 NN 实验 → 0506_ou_pair/nn/; 调参实验 → 0506_ou_pair/nn/tuning/ (与正式分开)
NN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "nn", "tuning") if _TUNE_TAG else os.path.join(OUTPUT_DIR, "nn")
EXP_DIR = os.path.join(NN_OUTPUT_DIR, EXP_NAME)

# ── NN 超参 (可经环境变量覆盖, 供调参; 默认=基线) ──
HIDDEN = tuple(int(x) for x in os.environ.get("NN_HIDDEN", "128,64").split(",") if x.strip())
DROPOUT = float(os.environ.get("NN_DROPOUT", "0.2"))
WINSOR_Q = (0.1, 99.9)   # 特征缩尾分位 (逐特征, 每段 train 上 fit)
LR = float(os.environ.get("NN_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("NN_WD", "1e-5"))
BATCH = int(os.environ.get("NN_BATCH", "4096"))
MAX_EPOCHS = 200
PATIENCE = 20          # 早停: mse 监控加权 val RMSE↓ / rank 监控日均 rank-IC↑, 连续未改善则停
NO_VAL_EPOCHS = 60     # 无 val 时的固定训练轮数
THREADS = _N_THREADS   # 线程预算 = 机器一半核 (与其它并行实验共享, 见顶部 N_THREADS)
RANK_MIN_PAIRS = 3     # rank 模式下截面 pair 数 < 此值的日子跳过 (无法稳定算 softmax/IC)
RANK_MAX_EPOCHS = 500  # rank 为全量梯度(每 epoch 1 step), 每步廉价, 给足轮数防欠拟合
RANK_PATIENCE = 40     # rank 早停耐心 (相应放大)
_EPS = 1e-12

NN_PARAMS = {
    "model":               "MLP",
    "hidden":              list(HIDDEN),
    "dropout":             DROPOUT,
    "batchnorm":           True,
    "activation":          "relu",
    "loss":                ("weighted_mse" if LOSS == "mse"
                            else "weighted_listnet (per-day softmax-CE on r=sign(mu-X)*dX)"),
    "optimizer":           "adam",
    "lr":                  LR,
    "weight_decay":        WEIGHT_DECAY,
    "batch_size":          BATCH,
    "max_epochs":          MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "winsorize_pct":       list(WINSOR_Q),
    "threads":             THREADS,
    "feature_standardize": "global (per-seg train-fit StandardScaler, after winsorize)",
    "label_standardize":   "z-score (per-seg train-fit)",
    "nan_handling":        "train: drop rows with NaN; val/test: train-median impute",
}

torch.set_num_threads(THREADS)


# =====================================================================
# MLP 模型
# =====================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64), p: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_predict_one_seed(
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
    X_va, y_va, w_va, X_te: np.ndarray, seed: int,
) -> Tuple[np.ndarray, int, float]:
    """单种子: 训练 MLP(加权 MSE + 早停), 返回 (test 预测[标准化 y 空间], best_epoch, best_val_rmse)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)

    model = MLP(X_tr.shape[1], HIDDEN, DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xtr = torch.from_numpy(X_tr); ytr = torch.from_numpy(y_tr); wtr = torch.from_numpy(w_tr)
    n = len(Xtr)
    has_va = X_va is not None and len(X_va) > 0
    if has_va:
        Xva = torch.from_numpy(X_va); yva = torch.from_numpy(y_va); wva = torch.from_numpy(w_va)

    best_rmse = float("inf"); best_state = None; best_ep = -1; bad = 0
    max_ep = MAX_EPOCHS if has_va else NO_VAL_EPOCHS
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:        # BatchNorm 要求 batch>=2
                continue
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = (wtr[idx] * (pred - ytr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        if has_va:
            model.eval()
            with torch.no_grad():
                pv = model(Xva)
                vrmse = float(torch.sqrt((wva * (pv - yva) ** 2).sum() / wva.sum().clamp_min(_EPS)))
            if vrmse < best_rmse - 1e-6:
                best_rmse = vrmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_ep = ep; bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if has_va and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = model(torch.from_numpy(X_te)).numpy()
    return pred_te, (best_ep + 1 if has_va else max_ep), (best_rmse if has_va else float("nan"))


def _mean_daily_rank_ic(s: np.ndarray, r: np.ndarray, grp: np.ndarray) -> float:
    """日均 Spearman(s, r): 各 group(日)内对 s、r 取秩再求 Pearson, 跨日平均."""
    ics = []
    for d in np.unique(grp):
        m = grp == d
        if m.sum() < RANK_MIN_PAIRS:
            continue
        ra = s[m].argsort().argsort().astype(np.float64)
        rb = r[m].argsort().argsort().astype(np.float64)
        if ra.std() < _EPS or rb.std() < _EPS:
            continue
        ics.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def _train_predict_rank_seed(
    X_tr, r_tr, grp_tr, wday_tr, X_va, r_va, grp_va, X_te, seed,
) -> Tuple[np.ndarray, int, float]:
    """单种子: 按日 ListNet 排序训练(目标 r 日内 z-score → softmax 交叉熵, 乘 WS 日权重).
    向量化: 每 epoch 一次全量 forward + 分组(按日)softmax 损失 + 一次 backward/step.
    早停监控 val 日均 rank-IC(越大越好). 返回 (test 打分, best_epoch, best_val_ic)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X_tr.shape[1], HIDDEN, DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── 预计算(一次): 过滤 <RANK_MIN_PAIRS 的日 → 连续 gid + 目标分布 ptar + 行级日权重 ──
    uniq, counts = np.unique(grp_tr, return_counts=True)
    valid = set(uniq[counts >= RANK_MIN_PAIRS].tolist())
    keep = np.fromiter((d in valid for d in grp_tr), dtype=bool, count=len(grp_tr))
    Xk = X_tr[keep]; gk = grp_tr[keep]; rk = r_tr[keep]; wk = wday_tr[keep]
    day_list = np.unique(gk)
    d2g = {d: i for i, d in enumerate(day_list)}
    gid = np.fromiter((d2g[d] for d in gk), dtype=np.int64, count=len(gk))
    G = len(day_list)
    # 目标分布 ptar = softmax(日内 z-score(r)) (与模型无关, 预计算)
    ptar = np.empty(len(rk), dtype=np.float64)
    for d, i in d2g.items():
        m = gid == i
        rr = rk[m]; rz = (rr - rr.mean()) / (rr.std() + _EPS)
        ez = np.exp(rz - rz.max()); ptar[m] = ez / ez.sum()

    Xk_t = torch.from_numpy(Xk)
    gid_t = torch.from_numpy(gid)
    ptar_t = torch.from_numpy(ptar.astype(np.float32))
    wrow_t = torch.from_numpy(wk.astype(np.float32))

    def _grouped_loss(s):
        # 分组(按日) log-softmax: 减去组内 max(detach 稳定) → exp → 组内求和
        gmax = torch.full((G,), -1e30).scatter_reduce(0, gid_t, s, reduce="amax",
                                                       include_self=True).detach()
        s_shift = s - gmax[gid_t]
        Z = torch.zeros(G).index_add(0, gid_t, torch.exp(s_shift))
        logsm = s_shift - torch.log(Z[gid_t] + _EPS)
        # 加权 ListNet 交叉熵, 跨日取均值
        return -(ptar_t * logsm * wrow_t).sum() / G

    has_va = X_va is not None and len(X_va) > 0
    if has_va:
        Xva_t = torch.from_numpy(X_va)

    best_ic = -float("inf"); best_state = None; best_ep = -1; bad = 0
    max_ep = RANK_MAX_EPOCHS if has_va else NO_VAL_EPOCHS
    for ep in range(max_ep):
        model.train()
        opt.zero_grad()
        loss = _grouped_loss(model(Xk_t))      # 一次全量 forward
        loss.backward(); opt.step()
        if has_va:
            model.eval()
            with torch.no_grad():
                sv = model(Xva_t).numpy()
            ic = _mean_daily_rank_ic(sv, r_va, grp_va)
            if ic > best_ic + 1e-5:
                best_ic = ic
                best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
                best_ep = ep; bad = 0
            else:
                bad += 1
                if bad >= RANK_PATIENCE:
                    break

    if has_va and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s_te = model(torch.from_numpy(X_te)).numpy()
    return s_te, (best_ep + 1 if has_va else max_ep), (best_ic if has_va else float("nan"))


def build_holdings_by_score(df_legal_pred: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """rank 模式选股: 按打分 pred_delta_X 降序取 top-k(非 |·|), 方向 sign(μ-X_t).
    n_top 公式与 base.build_holdings 一致, 仅排序键不同."""
    print("\n按打分(rank)选 top, 生成 holdings ...")
    long_rows: List[Tuple] = []; short_rows: List[Tuple] = []; daily_stats: List[Dict] = []
    for d, sub in df_legal_pred.groupby("date"):
        sub = sub[~sub["pred_delta_X"].isna()]
        n_legal = len(sub)
        if n_legal == 0:
            daily_stats.append({"date": d, "n_legal": 0, "n_top": 0}); continue
        n_top = min(n_legal, max(TOP_PCT_MIN_N, int(math.ceil(n_legal * TOP_PCT))))
        sub_top = sub.sort_values("pred_delta_X", ascending=False).head(n_top)
        sign_pred = np.sign(sub_top["mu"] - sub_top["X_t"]).values
        i_arr = sub_top["stock_i"].values; j_arr = sub_top["stock_j"].values
        for k in range(len(sub_top)):
            s = sign_pred[k]
            if s > 0:
                long_rows.append((d, i_arr[k])); short_rows.append((d, j_arr[k]))
            elif s < 0:
                long_rows.append((d, j_arr[k])); short_rows.append((d, i_arr[k]))
        daily_stats.append({"date": d, "n_legal": n_legal, "n_top": len(sub_top)})
    long_df = pd.DataFrame(long_rows, columns=["date", "stock_code"])
    short_df = pd.DataFrame(short_rows, columns=["date", "stock_code"])
    long_h = long_df.groupby("date")["stock_code"].apply(list)
    short_h = short_df.groupby("date")["stock_code"].apply(list)
    all_dates = sorted(set(long_h.index) | set(short_h.index))
    long_h = long_h.reindex(all_dates, fill_value=[])
    short_h = short_h.reindex(all_dates, fill_value=[])
    return long_h, short_h, daily_stats


def build_holdings_model_dir(df_legal_pred: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """DIR=model 选股: 选股仍按 |pred ΔX| 排序取 top-k, 但方向用 sign(预测ΔX)(非 sign(μ-X)).
    pred>0 → 价差预期上升 → 多 i 空 j; pred<0 → 多 j 空 i."""
    print("\n按 |pred ΔX| 选 top, 方向=sign(pred ΔX), 生成 holdings ...")
    long_rows: List[Tuple] = []; short_rows: List[Tuple] = []; daily_stats: List[Dict] = []
    for d, sub in df_legal_pred.groupby("date"):
        sub = sub[~sub["pred_delta_X"].isna()]
        n_legal = len(sub)
        if n_legal == 0:
            daily_stats.append({"date": d, "n_legal": 0, "n_top": 0}); continue
        n_top = min(n_legal, max(TOP_PCT_MIN_N, int(math.ceil(n_legal * TOP_PCT))))
        sub = sub.assign(abs_pred=sub["pred_delta_X"].abs())
        sub_top = sub.sort_values("abs_pred", ascending=False).head(n_top)
        sign_pred = np.sign(sub_top["pred_delta_X"]).values        # 方向由模型预测符号决定
        i_arr = sub_top["stock_i"].values; j_arr = sub_top["stock_j"].values
        for k in range(len(sub_top)):
            s = sign_pred[k]
            if s > 0:
                long_rows.append((d, i_arr[k])); short_rows.append((d, j_arr[k]))
            elif s < 0:
                long_rows.append((d, j_arr[k])); short_rows.append((d, i_arr[k]))
        daily_stats.append({"date": d, "n_legal": n_legal, "n_top": len(sub_top)})
    long_df = pd.DataFrame(long_rows, columns=["date", "stock_code"])
    short_df = pd.DataFrame(short_rows, columns=["date", "stock_code"])
    long_h = long_df.groupby("date")["stock_code"].apply(list)
    short_h = short_df.groupby("date")["stock_code"].apply(list)
    all_dates = sorted(set(long_h.index) | set(short_h.index))
    long_h = long_h.reindex(all_dates, fill_value=[])
    short_h = short_h.reindex(all_dates, fill_value=[])
    return long_h, short_h, daily_stats


# =====================================================================
# 5 段 rolling 训练 + 预测 (MLP 版; 段循环逻辑对齐 base.run_xgb_rolling)
# =====================================================================

def run_nn_rolling(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    print("\n5 段 rolling MLP 训练 + 预测 ...")
    # df 由 main 列裁剪后传入且不再被复用, 原地加列即可 (省一份整表拷贝)
    df["pred_delta_X"] = np.nan
    train_logs: List[Dict] = []

    # 布尔索引本身返回副本且后续不写入这两个变量, 无需显式 .copy()
    df_legal = df[df["is_legal"]]
    df_train_pool = df_legal[~df_legal["delta_X"].isna()]
    print(f"  is_legal + 标签可用样本: {len(df_train_pool):,}")

    pool_dates = np.array(sorted(df_train_pool["date"].unique()))
    pool_min_date = pool_dates[0]
    if _ROLL_DAYS == 0 and pd.Timestamp(TRAIN_START) < pool_min_date:
        raise ValueError(
            f"TRAIN_START={TRAIN_START} 早于训练池可用最早日 {pd.Timestamp(pool_min_date).date()}")

    for seg in ROLLING_SEGMENTS:
        seg_name, train_lo, train_hi, test_lo, test_hi = seg
        train_hi = pd.Timestamp(train_hi)
        test_lo = pd.Timestamp(test_lo); test_hi = pd.Timestamp(test_hi)
        if _ROLL_DAYS > 0:
            elig = pool_dates[pool_dates <= train_hi]
            train_lo = pd.Timestamp(elig[-_ROLL_DAYS] if len(elig) >= _ROLL_DAYS else elig[0])
        else:
            train_lo = pd.Timestamp(train_lo)

        m_train = (df_train_pool["date"] >= train_lo) & (df_train_pool["date"] <= train_hi)
        train_seg = df_train_pool[m_train].sort_values("date").reset_index(drop=True)
        if len(train_seg) == 0:
            print(f"  [{seg_name}] 训练样本 0, 跳过"); continue

        # 段内 train/val: 前 80% + 5 日 gap + 后 20%
        unique_dates = sorted(train_seg["date"].unique())
        n_dates = len(unique_dates)
        n_train_dates = int(np.floor(n_dates * TRAIN_FRAC))
        train_end_date = unique_dates[n_train_dates - 1]
        gap_pos = min(n_train_dates + TRAIN_VAL_GAP_DAYS, n_dates)
        if gap_pos >= n_dates:
            m_tr = train_seg["date"] <= train_end_date
            m_va = pd.Series([False] * len(train_seg))
        else:
            val_start_date = unique_dates[gap_pos]
            m_tr = train_seg["date"] <= train_end_date
            m_va = train_seg["date"] >= val_start_date
        n_tr = int(m_tr.sum()); n_va = int(m_va.sum())

        m_test = (df_legal["date"] >= test_lo) & (df_legal["date"] <= test_hi)
        test_seg = df_legal[m_test].copy()
        n_test = len(test_seg)

        print(f"\n  [{seg_name}] train: {train_lo.date()} ~ {train_hi.date()}, "
              f"test: {test_lo.date()} ~ {test_hi.date()}")
        print(f"    train: {n_tr:,} / val: {n_va:,}; test: {n_test:,}")
        if n_tr < 100 or n_test == 0:
            print(f"  [{seg_name}] 样本不足, 跳过"); continue

        # ── 原始特征 / 标签 / WS 截面权重 ──
        X_tr_raw = train_seg.loc[m_tr, FEATURES].astype(np.float64).values
        y_tr_raw = train_seg.loc[m_tr, "delta_X"].astype(np.float64).values
        X_va_raw = train_seg.loc[m_va, FEATURES].astype(np.float64).values if n_va > 0 else None
        y_va_raw = train_seg.loc[m_va, "delta_X"].astype(np.float64).values if n_va > 0 else None
        X_te_raw = test_seg[FEATURES].astype(np.float64).values

        def _calc_section_weight(mask):
            sub = train_seg.loc[mask]
            r_pair = np.sign(sub["mu"] - sub["X_t"]) * sub["delta_X"]
            date_r = r_pair.groupby(sub["date"]).mean()
            date_w = 2.0 - date_r.rank(pct=True, method="average")
            return sub["date"].map(date_w).astype(np.float64).values
        w_tr_raw = _calc_section_weight(m_tr)
        w_va_raw = _calc_section_weight(m_va) if n_va > 0 else None

        # ── 缺失值: train 丢弃含 NaN 行(标签已非 NaN, 仅查特征) ──
        tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
        X_tr_c = X_tr_raw[tr_ok]; y_tr_c = y_tr_raw[tr_ok]; w_tr_c = w_tr_raw[tr_ok]
        n_drop = int((~tr_ok).sum())

        # ── 仅在 train 上 fit: 中位数填补 → 缩尾 → 标准化 ──
        # 缩尾边界 = train 各列 0.1%/99.9% 分位; 标准化器在缩尾后的 train 上 fit(均值/方差更稳)
        imputer = SimpleImputer(strategy="median").fit(X_tr_c)
        lo_q = np.percentile(X_tr_c, WINSOR_Q[0], axis=0)
        hi_q = np.percentile(X_tr_c, WINSOR_Q[1], axis=0)

        def _prep_raw(X):                       # 填补 + 缩尾 (标准化前)
            return np.clip(imputer.transform(X), lo_q, hi_q)

        X_tr_w = np.clip(X_tr_c, lo_q, hi_q)
        scaler = StandardScaler().fit(X_tr_w)
        X_tr_in = scaler.transform(X_tr_w).astype(np.float32)
        X_va_in = scaler.transform(_prep_raw(X_va_raw)).astype(np.float32) if n_va > 0 else None
        X_te_in = scaler.transform(_prep_raw(X_te_raw)).astype(np.float32)

        # ── 标签标准化(train 拟合, 单调仿射) ──
        y_mean = float(y_tr_c.mean()); y_std = float(y_tr_c.std() + _EPS)
        y_tr_in = ((y_tr_c - y_mean) / y_std).astype(np.float32)
        y_va_in = ((y_va_raw - y_mean) / y_std).astype(np.float32) if n_va > 0 else None
        w_tr_in = w_tr_c.astype(np.float32)
        w_va_in = w_va_raw.astype(np.float32) if n_va > 0 else None

        print(f"    train 去 NaN: 丢 {n_drop:,} 行 → 用 {len(X_tr_c):,}; "
              f"截面权重均值 {w_tr_in.mean():.3f}")

        # ── rank 模式: 排序目标 r=sign(μ-X)·ΔX + 日分组(供按日 ListNet/IC) ──
        if LOSS == "rank":
            _tr_sub = train_seg.loc[m_tr]
            r_tr_full = (np.sign(_tr_sub["mu"] - _tr_sub["X_t"]) * _tr_sub["delta_X"]).to_numpy(np.float64)
            grp_tr_full = _tr_sub["date"].values.astype("datetime64[ns]").astype(np.int64)
            r_tr_c = r_tr_full[tr_ok]; grp_tr_c = grp_tr_full[tr_ok]
            if n_va > 0:
                _va_sub = train_seg.loc[m_va]
                r_va_a = (np.sign(_va_sub["mu"] - _va_sub["X_t"]) * _va_sub["delta_X"]).to_numpy(np.float64)
                grp_va_a = _va_sub["date"].values.astype("datetime64[ns]").astype(np.int64)
            else:
                r_va_a = grp_va_a = None

        # ── 多种子集成: 预测累加取均值 (mse: ΔX 空间; rank: 打分空间) ──
        t0 = time.time()
        n_seeds = len(ENSEMBLE_SEEDS)
        pred_sum = np.zeros(n_test, dtype=np.float64)
        for sd in ENSEMBLE_SEEDS:
            if LOSS == "rank":
                s_te, best_ep, best_metric = _train_predict_rank_seed(
                    X_tr_in, r_tr_c, grp_tr_c, w_tr_in, X_va_in, r_va_a, grp_va_a, X_te_in, sd)
                pred_sum += s_te
                metric_str = f", val_rankIC={best_metric:.4f}" if n_va > 0 else ""
                log_metric = {"best_val_rank_ic": float(best_metric) if n_va > 0 else np.nan}
            else:
                pred_std, best_ep, best_metric = _train_predict_one_seed(
                    X_tr_in, y_tr_in, w_tr_in, X_va_in, y_va_in, w_va_in, X_te_in, sd)
                pred_sum += pred_std * y_std + y_mean       # 反标准化回 ΔX 空间
                metric_str = f", val_rmse(std)={best_metric:.5f}" if n_va > 0 else ""
                log_metric = {"best_val_rmse_std": float(best_metric) if n_va > 0 else np.nan}
            print(f"    [seed {sd}] best_epoch={best_ep}" + (metric_str if n_va > 0 else ""))
            train_logs.append({
                "seg": seg_name, "seed": sd,
                "train_range": f"{train_lo.date()}~{train_hi.date()}",
                "test_range":  f"{test_lo.date()}~{test_hi.date()}",
                "n_train": int(len(X_tr_c)), "n_train_dropped": n_drop,
                "n_val": n_va, "n_test": n_test,
                "best_epoch": int(best_ep),
                **log_metric,
            })
        df.loc[test_seg.index, "pred_delta_X"] = pred_sum / n_seeds
        print(f"    {n_seeds} 种子集成完成, 耗时 {time.time()-t0:.1f}s")

    return df, train_logs


# =====================================================================
# 特征工程缓存 (特征不依赖 DROP_GROUPS, 各实验共享一份; 省 ~233s/次)
# =====================================================================

# 缓存版本: 特征计算逻辑 (base.add_derived_features) 变更时手动 +1, 避免读到旧缓存
_FEAT_CACHE_VERSION = "v1"


def _load_or_build_features(px: Dict) -> pd.DataFrame:
    """读取/构建特征 df. 缓存 key 编码数据范围+train_window+版本(不含 DROP_GROUPS).

    缓存命中则跳过 load_all_pair_log + add_derived_features; px 仍需(回测用), 由 main 传入.
    环境变量 NO_FEAT_CACHE=1 关闭缓存。
    """
    cache_dir = os.path.join(OUTPUT_DIR, ".cache")
    # 缓存 key 须含候选池标识: 不同 PAIR_POOL 的 pair_log 不同, 特征也不同, 不可共用缓存
    # (基线池 _PAIR_POOL="" 时 key 不变, 向后兼容既有缓存)
    _pool = getattr(base, "_PAIR_POOL", "")
    _pool_part = f"_{_pool}" if _pool else ""
    tag = f"{base.DATA_START}_{base.DATA_END}_tw252{_pool_part}_{_FEAT_CACHE_VERSION}".replace("-", "")
    cache_path = os.path.join(cache_dir, f"feat_df_{tag}.parquet")
    use_cache = os.environ.get("NO_FEAT_CACHE", "").strip() == ""

    if use_cache and os.path.exists(cache_path):
        print(f"读取特征缓存: {cache_path}")
        return pd.read_parquet(cache_path)

    pl_full = base.load_all_pair_log()
    df = base.add_derived_features(pl_full, px, train_window=252)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"特征已缓存: {cache_path}")
    return df


# =====================================================================
# main (结构对齐 base.main, 仅模型与参数记录不同)
# =====================================================================

def main():
    print(f"\n{'='*70}")
    print(f"OU 配对·MLP 信号  [{EXP_NAME}]  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}  | 特征 {len(FEATURES)} | 种子 {ENSEMBLE_SEEDS} | ROLL_DAYS {_ROLL_DAYS}")
    print(f"  输出: {EXP_DIR}")
    print(f"{'='*70}\n")

    metrics_path = os.path.join(EXP_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[{EXP_NAME}] 已有 metrics.json, 跳过"); return
    os.makedirs(EXP_DIR, exist_ok=True)

    t_start = time.time()
    px = base.load_price_wides()                 # 回测需用, 始终加载
    df = _load_or_build_features(px)             # 命中缓存则跳过特征工程
    # 列裁剪: 只保留 建模 + 选股 + 回测 所需列, 降内存 (丢弃 mkt/ind 未用列及中间索引列)
    _need = list(dict.fromkeys(
        FEATURES + ["date", "stock_i", "stock_j", "is_legal", "delta_X", "mu", "X_t"]))
    df = df[[c for c in _need if c in df.columns]]
    df, train_logs = run_nn_rolling(df)

    df_legal_pred = df[df["is_legal"] & ~df["pred_delta_X"].isna()].copy()
    # 保存合法 pair 的预测值, 供分层回测(按 |pred ΔX| 分 5 组)后处理使用
    _pred_cols = [c for c in ["date", "stock_i", "stock_j", "signal", "mu", "X_t", "pred_delta_X"]
                  if c in df_legal_pred.columns]
    df_legal_pred[_pred_cols].to_parquet(os.path.join(EXP_DIR, "pred_pairs.parquet"), index=False)
    # rank: 按打分降序选(方向 sign(μ-X)); mse+DIR=model: |pred| 选 + 方向 sign(pred);
    # mse+DIR=ou(默认): |pred| 选 + 方向 sign(μ-X)(复用 base)
    if LOSS == "rank":
        long_h, short_h, daily_stats = build_holdings_by_score(df_legal_pred)
    elif DIRECTION == "model":
        long_h, short_h, daily_stats = build_holdings_model_dir(df_legal_pred)
    else:
        long_h, short_h, daily_stats = base.build_holdings(df_legal_pred)
    print(f"  生成 {len(long_h)} 个截面的 long_holdings")

    print(f"\n启动多空回测 (等权 h={HOLDING_PERIOD}) ...")
    bt = run_longshort_backtest(
        long_holdings=long_h, short_holdings=short_h,
        open_prices=px["open_wide"], close_prices=px["close_wide"],
        status_data=px["status_wide"],
        output_dir=EXP_DIR,
        commission_rate=COMMISSION, holding_period=HOLDING_PERIOD,
    )
    stats = bt["statistics"]

    def _add_calmar(s):
        ann = s.get("annual_return", 0.0); mdd = s.get("max_drawdown", 0.0)
        s["calmar_ratio"] = round(ann / abs(mdd), 3) if mdd != 0 else 0.0
        return s
    for k in ("head", "tail", "tail_raw", "longshort", "benchmark",
              "head_excess", "tail_excess", "ls_excess"):
        if k in stats:
            stats[k] = _add_calmar(stats[k])

    stats_df = pd.DataFrame(daily_stats)
    metrics = {
        "experiment":         EXP_NAME,
        "pair_log_source":    FULL_PAIR_LOG,
        "train_start":        TRAIN_START,
        "roll_days":          _ROLL_DAYS,
        "n_features":         len(FEATURES),
        "features":           FEATURES,
        "holding_period":     HOLDING_PERIOD,
        "label_horizon":      LABEL_HORIZON,
        "commission_rate":    COMMISSION,
        "top_pct":            TOP_PCT,
        "top_pct_min_n":      TOP_PCT_MIN_N,
        "n_signal_days":      int((stats_df["n_top"] > 0).sum()),
        "avg_n_top_per_day":  float(stats_df["n_top"].mean()),
        "nn_params":          NN_PARAMS,
        "ensemble_seeds":     ENSEMBLE_SEEDS,
        "train_logs":         train_logs,
        "long":      stats["head"],
        "short":     stats["tail"],
        "longshort": stats["longshort"],
        "benchmark": stats["benchmark"],
        "long_excess": stats["head_excess"],
        "ls_excess":   stats["ls_excess"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[{EXP_NAME}] metrics.json 已写入")

    if train_logs:
        pd.DataFrame(train_logs).to_csv(os.path.join(EXP_DIR, "train_log.csv"), index=False)

    fieldnames = [
        "exp", "train_window",
        "ls_annual", "ls_sharpe", "ls_calmar", "ls_mdd",
        "long_annual", "long_sharpe", "long_calmar", "long_excess_annual",
        "short_annual", "short_sharpe",
        "avg_n_signal_stocks", "avg_n_long", "avg_n_short",
        "avg_n_legal_pairs", "avg_n_top20_pairs",
    ]
    save_summary_row(os.path.join(NN_OUTPUT_DIR, "summary.csv"), {
        "exp": EXP_NAME, "train_window": "",
        "ls_annual":  stats["longshort"]["annual_return"],
        "ls_sharpe":  stats["longshort"]["sharpe_ratio"],
        "ls_calmar":  stats["longshort"].get("calmar_ratio", 0.0),
        "ls_mdd":     stats["longshort"]["max_drawdown"],
        "long_annual":        stats["head"]["annual_return"],
        "long_sharpe":        stats["head"]["sharpe_ratio"],
        "long_calmar":        stats["head"].get("calmar_ratio", 0.0),
        "long_excess_annual": stats["head_excess"]["annual_return"],
        "short_annual":       stats["tail"]["annual_return"],
        "short_sharpe":       stats["tail"]["sharpe_ratio"],
    }, fieldnames)
    print(f"[{EXP_NAME}] summary.csv 已追加")
    print(f"\n[{EXP_NAME}] 总耗时 {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
