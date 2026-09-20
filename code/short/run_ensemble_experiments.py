#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成模型多空回测实验脚本

实验组：
1. 三模型融合（1 回归 + 2 分类）: E1-A1, E1-A2, E1-A3
2. 粗排 + 精排: E2-B1, E2-B2
3. DoubleEnsemble: E3-C1, E3-C2

每组实验两个版本：
- V1: Data/short/panel.pkl 上训练，头尾组均从融券池选
- V2: Data/all/panel.pkl 上训练，头组从全市场选，尾组用 V1 的尾组，去重后回测

复用 func/: experiment_engine, model_backtest_framework_longshort
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import ctypes
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

_libc = ctypes.CDLL("libc.so.6")

def _force_gc():
    """强制回收 Python 垃圾并归还空闲内存给操作系统。"""
    gc.collect()
    _libc.malloc_trim(0)

# ─── 路径设置 ───
WORK_DIR = "/root/quant"
if WORK_DIR not in sys.path:
    sys.path.append(WORK_DIR)
XGBCODE_DIR = os.path.join(WORK_DIR, "xgbcode")
if XGBCODE_DIR not in sys.path:
    sys.path.append(XGBCODE_DIR)
FUNC_DIR = os.path.join(XGBCODE_DIR, "func")
if FUNC_DIR not in sys.path:
    sys.path.append(FUNC_DIR)
os.chdir(WORK_DIR)

import experiment_engine as eng
from experiment_engine import (
    RANDOM_SEED, params, prep_xy, calculate_rank_ic,
    neutralize_label_mv, half_year,
)

np.random.seed(RANDOM_SEED)

import matplotlib.font_manager as _fm
_cjk_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Microsoft YaHei']
_available = {f.name for f in _fm.fontManager.ttflist}
_font = next((f for f in _cjk_fonts if f in _available), None)
if _font:
    plt.rcParams['font.sans-serif'] = [_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════
SHORT_PANEL_PATH = os.path.join(WORK_DIR, "Data", "short", "panel.pkl")
ALL_PANEL_PATH = os.path.join(WORK_DIR, "Data", "all", "panel.pkl")
PANEL_TRADE_PATH = os.path.join(WORK_DIR, "Data", "all", "panel_trade.pkl")
OUTPUT_ROOT = os.path.join(WORK_DIR, "output", "ensemble_longshort")

NON_FEATURE_COLS = {
    "index", "stock_code", "trade_date", "date",
    "c_pct_5", "market_value", "is_margin_buy",
}

ENSEMBLE_SEEDS = [33, 42, 101]
N_GROUPS = 10
WINDOW_SIZE = 4
TRAIN_RATIO = 0.8
EARLY_STOP_ROUNDS = 30
MIN_IMPROVE = 0.001
NUM_BOOST_ROUND = 500
REBALANCE_DAYS = 5
COMMISSION_RATE = 0.0007
START_DATE = "2020-01-01"
END_DATE = "2026-03-01"

MAX_PARALLEL_WINDOWS = 4
_CPU_COUNT = os.cpu_count() or 224

# nthread 根据实际并行数动态计算，不写死在 params 中
def _nthread_for(n_parallel: int) -> int:
    return max(1, _CPU_COUNT // n_parallel)

XGB_REG_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 5, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.3,
    "gamma": 5.0, "reg_alpha": 1.0, "reg_lambda": 1.0,
    "learning_rate": 0.05, "tree_method": "hist",
}

XGB_CLS_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "max_depth": 5, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.3,
    "gamma": 5.0, "reg_alpha": 1.0, "reg_lambda": 1.0,
    "learning_rate": 0.05, "tree_method": "hist",
}

LGB_CLS_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "num_leaves": 31, "min_child_samples": 20,
    "subsample": 0.8, "colsample_bytree": 0.3,
    "reg_alpha": 1.0, "reg_lambda": 1.0,
    "learning_rate": 0.05,
    "verbosity": -1,
}

# 运行时 nthread，由 rolling_train_predict 设置，训练函数读取
_active_nthread = _nthread_for(MAX_PARALLEL_WINDOWS)


# ═══════════════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # 已完成（有缓存会跳过训练）
    {"name": "E1-A1", "group": "fusion", "top_k": 0.10, "alpha": 0.5, "cls_type": "xgb", "cls_neutralize": False},
    # 实验组 1: 三模型融合
    {"name": "E1-A1n", "group": "fusion", "top_k": 0.10, "alpha": 0.5, "cls_type": "xgb", "cls_neutralize": True},
    {"name": "E1-A2", "group": "fusion", "top_k": 0.20, "alpha": 0.5, "cls_type": "xgb"},
    {"name": "E1-A3", "group": "fusion", "top_k": 0.10, "alpha": 0.5, "cls_type": "lgb"},
    # 实验组 2: 粗排+精排
    {"name": "E2-B1", "group": "coarse_fine", "hard_ratio": 0.20},
    {"name": "E2-B2", "group": "coarse_fine", "hard_ratio": 0.30},
    # 实验组 3: DoubleEnsemble
    {"name": "E3-C1", "group": "double_ensemble", "n_iters": 3, "decay": 0.5, "feat_ratio": 0.8},
    {"name": "E3-C2", "group": "double_ensemble", "n_iters": 5, "decay": 0.5, "feat_ratio": 0.8},
]


# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════

def _load_panel(path: str, tag: str) -> Tuple[List[str], List[pd.DataFrame]]:
    """
    加载 panel → 清洗 → 识别特征列 → 半年切窗 → 释放原始 panel。
    返回 (feature_cols, dfs)。panel 在切窗后立即释放以节省内存。
    """
    print(f"  加载 {tag}: {path} ...")
    panel = pd.read_pickle(path)
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.dropna(inplace=True)
    panel.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    for col in feature_cols:
        if panel[col].dtype != np.float32:
            panel[col] = panel[col].astype(np.float32)

    n_stocks = panel["stock_code"].nunique()
    n_rows = len(panel)
    dfs = half_year(panel, START_DATE, END_DATE)

    # 只保留训练/预测必需的列，丢弃 index/trade_date/is_margin_buy 等
    keep_cols = ["date", "stock_code", "c_pct_5", "market_value"] + feature_cols
    dfs = [df[keep_cols].reset_index(drop=True) for df in dfs]

    # 释放原始 panel，dfs 已经是独立副本
    del panel
    gc.collect()

    print(f"  {tag} 完成: {n_rows:,} 行, {len(feature_cols)} 特征, "
          f"{n_stocks} 只股票, {len(dfs)} 半年区间 (panel 已释放)")
    return feature_cols, dfs


def setup_engine(feature_cols: List[str], dfs: List[pd.DataFrame]) -> None:
    """设置 experiment_engine 全局状态。不设置 eng.panel 以节省内存。"""
    eng.panel = None  # 不常驻，neutralize_label_mv 从 df 自身取 market_value
    eng.dfs = dfs
    eng.feature_cols = feature_cols
    params.FEATURE_COLS = feature_cols


# ═══════════════════════════════════════════════════════════════════════
# 公共工具
# ═══════════════════════════════════════════════════════════════════════

def _extract_features(df: pd.DataFrame, fcols: List[str], copy: bool = True) -> np.ndarray:
    """提取特征矩阵。copy=False 时无 NaN 则取 view（仅用于不需释放 df 的场景）。"""
    raw = df[fcols].values
    if raw.dtype == np.float32 and not np.isnan(raw).any():
        return raw.copy() if copy else raw
    x = raw.astype(np.float32, copy=True)
    np.nan_to_num(x, copy=False, nan=0.0)
    return x


def _neutralize_label_array(df: pd.DataFrame) -> np.ndarray:
    """
    返回中性化标签的 numpy 数组。
    直接在数组层面计算，避免 neutralize_label_mv 的全量 DataFrame 拷贝。
    """
    y = df["c_pct_5"].values.astype(np.float64, copy=True)
    mv = df["market_value"].values.astype(np.float64)
    for idx in df.groupby("date", sort=False).indices.values():
        try:
            mv_grp = np.log(mv[idx] + 1e-10)
            X = np.column_stack([np.ones(len(idx), dtype=np.float64), mv_grp])
            yy = y[idx]
            beta = np.linalg.lstsq(X, yy, rcond=None)[0]
            resid = yy - (X @ beta)
            std = resid.std()
            if std > 1e-8:
                resid = (resid - resid.mean()) / std
            y[idx] = resid
        except Exception:
            continue
    return y.astype(np.float32)


def _prepare_training_data(
    tr: pd.DataFrame, va: pd.DataFrame, fcols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    一次性提取特征矩阵和中性化标签，供多 seed / 多模型复用。
    返回 (x_tr, y_tr, x_va, y_va)，均为 numpy 数组。
    """
    x_tr = _extract_features(tr, fcols)
    y_tr = _neutralize_label_array(tr)
    x_va = _extract_features(va, fcols)
    y_va = _neutralize_label_array(va)
    return x_tr, y_tr, x_va, y_va


def _train_xgb_reg_seeds(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_va: np.ndarray, y_va: np.ndarray,
    extra_params: Optional[Dict] = None,
    sample_weights: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> List[xgb.Booster]:
    """
    直接用数组训练 3 种子 XGBoost 回归，DMatrix 只建一次。
    比 eng.train_silent × 3 节省 ~80% 的 Python 层内存开销。
    """
    dtr = xgb.DMatrix(x_tr, label=y_tr, weight=sample_weights,
                      feature_names=feature_names)
    dva = xgb.DMatrix(x_va, label=y_va, feature_names=feature_names)

    def ric(yp, d):
        ic = calculate_rank_ic(pd.Series(yp), pd.Series(d.get_label()))
        return "rank_ic", float(ic if not np.isnan(ic) else -1.0)

    models = []
    for seed in ENSEMBLE_SEEDS:
        mp = dict(XGB_REG_PARAMS)
        mp["nthread"] = _active_nthread
        if extra_params:
            mp.update(extra_params)
        mp["seed"] = seed
        model = xgb.train(
            mp, dtr, NUM_BOOST_ROUND,
            evals=[(dtr, "train"), (dva, "eval")],
            feval=ric, verbose_eval=False,
            callbacks=[_RankICEarlyStop(EARLY_STOP_ROUNDS, MIN_IMPROVE)],
        )
        models.append(model)

    del dtr, dva
    gc.collect()
    return models


def _predict_xgb_from_array(
    models: list, x: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """用预提取的特征数组预测，避免重复 prep_xy。"""
    dmat = xgb.DMatrix(x, feature_names=feature_names)
    preds = [m.predict(dmat) for m in models]
    del dmat
    gc.collect()
    return np.mean(np.vstack(preds), axis=0).astype(np.float32)


def _build_cls_label(
    df: pd.DataFrame, top_k: float, is_top: bool, neutralize: bool = False,
) -> np.ndarray:
    """
    构造分类标签：横截面 top/bottom k 为正类。
    neutralize=False: 用原始 c_pct_5 的排名（默认）
    neutralize=True:  先做市值中性化再排名
    """
    if neutralize:
        values = _neutralize_label_array(df)
    else:
        values = df["c_pct_5"].values.astype(np.float32)
    ser = pd.Series(values)
    dser = pd.Series(df["date"].values)
    rp = ser.groupby(dser, sort=False).rank(method="average", pct=True)
    if is_top:
        return (rp >= (1.0 - top_k)).astype(np.float32).values
    else:
        return (rp <= top_k).astype(np.float32).values


def _zscore_by_date(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    ser = pd.Series(values)
    dser = pd.Series(dates)
    mean = ser.groupby(dser, sort=False).transform("mean")
    std = ser.groupby(dser, sort=False).transform("std").replace(0, np.nan)
    return ((ser - mean) / std).fillna(0.0).to_numpy(dtype=np.float32)


def _rank_pct(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    ser = pd.Series(values)
    dser = pd.Series(dates)
    return ser.groupby(dser, sort=False).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _predict_xgb(models: list, df: pd.DataFrame, fcols: List[str]) -> np.ndarray:
    x, _ = prep_xy(df, fcols=fcols, lcol="c_pct_5")
    dmat = xgb.DMatrix(x)
    preds = [m.predict(dmat) for m in models]
    del dmat, x
    gc.collect()
    return np.mean(np.vstack(preds), axis=0).astype(np.float32)


def _predict_lgb(models: list, df: pd.DataFrame, fcols: List[str]) -> np.ndarray:
    x = _extract_features(df, fcols)
    preds = [m.predict(x) for m in models]
    del x
    gc.collect()
    return np.mean(np.vstack(preds), axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# 集成训练函数
# ═══════════════════════════════════════════════════════════════════════

def _reg_cache_path(cfg: dict) -> Optional[str]:
    """构造回归模型预测缓存路径，返回 None 表示不缓存。"""
    cache_dir = cfg.get("_cache_dir")
    panel_tag = cfg.get("_panel_tag")
    w_idx = cfg.get("_window_idx")
    if not cache_dir or not panel_tag or w_idx is None:
        return None
    return os.path.join(cache_dir, f"reg_{panel_tag}_w{w_idx}.npy")


def _coarse_cache_paths(cfg: dict) -> Optional[Tuple[str, str, str]]:
    """构造粗排模型预测缓存路径 (tr, va, te)。"""
    cache_dir = cfg.get("_cache_dir")
    panel_tag = cfg.get("_panel_tag")
    w_idx = cfg.get("_window_idx")
    if not cache_dir or not panel_tag or w_idx is None:
        return None
    base = os.path.join(cache_dir, f"coarse_{panel_tag}_w{w_idx}")
    return f"{base}_tr.npy", f"{base}_va.npy", f"{base}_te.npy"


def train_fusion(
    tr: pd.DataFrame, va: pd.DataFrame, test_raw: pd.DataFrame,
    fcols: List[str], cfg: dict,
) -> np.ndarray:
    """
    三模型融合：1 回归 + 2 分类（top/bottom），zscore 融合。
    回归模型按窗口缓存，同组后续实验直接读取跳过训练。
    """
    top_k = cfg["top_k"]
    alpha = cfg["alpha"]
    cls_type = cfg["cls_type"]

    if cfg.get("_pre_sorted"):
        tr_s, va_s = tr, va
    else:
        tr_s = tr.sort_values(["date", "stock_code"]).reset_index(drop=True)
        va_s = va.sort_values(["date", "stock_code"]).reset_index(drop=True)

    x_te = _extract_features(test_raw, fcols, copy=False)  # test_raw 不释放，可取 view
    dmat_te = xgb.DMatrix(x_te)

    # 回归模型：检查缓存
    cache_path = _reg_cache_path(cfg)
    if cache_path and os.path.exists(cache_path):
        reg_pred = np.load(cache_path)
    else:
        x_tr = _extract_features(tr_s, fcols)
        x_va = _extract_features(va_s, fcols)
        y_tr_reg = _neutralize_label_array(tr_s)
        y_va_reg = _neutralize_label_array(va_s)
        reg_models = _train_xgb_reg_seeds(x_tr, y_tr_reg, x_va, y_va_reg)
        reg_pred = _predict_xgb_from_array(reg_models, x_te)
        del reg_models, y_tr_reg, y_va_reg, x_tr, x_va; gc.collect()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, reg_pred)

    # 分类阶段：重新提取特征（回归阶段的 x_tr/x_va 已释放以节省内存）
    cls_neu = cfg.get("cls_neutralize", False)
    y_tr_top = _build_cls_label(tr_s, top_k, is_top=True, neutralize=cls_neu)
    y_va_top = _build_cls_label(va_s, top_k, is_top=True, neutralize=cls_neu)
    y_tr_bot = _build_cls_label(tr_s, top_k, is_top=False, neutralize=cls_neu)
    y_va_bot = _build_cls_label(va_s, top_k, is_top=False, neutralize=cls_neu)

    CLS_BOOST_ROUND = cfg.get("cls_boost_round", 200)
    CLS_EARLY_STOP = cfg.get("cls_early_stop", 20)
    n_cls_seeds = cfg.get("cls_seeds", 1)
    cls_seeds_list = ENSEMBLE_SEEDS[:n_cls_seeds]

    x_tr = _extract_features(tr_s, fcols)
    x_va = _extract_features(va_s, fcols)

    cls_params_override = cfg.get("cls_params_override", {})

    if cls_type == "xgb":
        pos_w_top = float((y_tr_top == 0).sum() / max((y_tr_top == 1).sum(), 1))
        pos_w_bot = float((y_tr_bot == 0).sum() / max((y_tr_bot == 1).sum(), 1))

        # top 分类
        dtr_top = xgb.DMatrix(x_tr, label=y_tr_top)
        dva_top = xgb.DMatrix(x_va, label=y_va_top)
        cls_top_models = []
        for seed in cls_seeds_list:
            mp = dict(XGB_CLS_PARAMS); mp.update(cls_params_override)
            mp["nthread"] = _active_nthread
            mp["seed"] = seed; mp["scale_pos_weight"] = pos_w_top
            cls_top_models.append(xgb.train(mp, dtr_top, CLS_BOOST_ROUND,
                evals=[(dva_top, "eval")], early_stopping_rounds=CLS_EARLY_STOP, verbose_eval=False))
        cls_top_pred = np.mean([m.predict(dmat_te) for m in cls_top_models], axis=0).astype(np.float32)
        del cls_top_models, dtr_top, dva_top; gc.collect()

        # bottom 分类
        dtr_bot = xgb.DMatrix(x_tr, label=y_tr_bot)
        dva_bot = xgb.DMatrix(x_va, label=y_va_bot)
        cls_bot_models = []
        for seed in cls_seeds_list:
            mp2 = dict(XGB_CLS_PARAMS); mp2.update(cls_params_override)
            mp2["nthread"] = _active_nthread
            mp2["seed"] = seed; mp2["scale_pos_weight"] = pos_w_bot
            cls_bot_models.append(xgb.train(mp2, dtr_bot, CLS_BOOST_ROUND,
                evals=[(dva_bot, "eval")], early_stopping_rounds=CLS_EARLY_STOP, verbose_eval=False))
        cls_bot_pred = np.mean([m.predict(dmat_te) for m in cls_bot_models], axis=0).astype(np.float32)
        del cls_bot_models, dtr_bot, dva_bot; gc.collect()

    else:  # lgb
        pos_w_top = float((y_tr_top == 0).sum() / max((y_tr_top == 1).sum(), 1))
        pos_w_bot = float((y_tr_bot == 0).sum() / max((y_tr_bot == 1).sum(), 1))

        cls_top_models = []
        for seed in cls_seeds_list:
            lp = dict(LGB_CLS_PARAMS); lp["n_jobs"] = _active_nthread; lp["scale_pos_weight"] = pos_w_top
            dtr = lgb.Dataset(x_tr, label=y_tr_top); dva = lgb.Dataset(x_va, label=y_va_top, reference=dtr)
            cls_top_models.append(lgb.train(lp, dtr, CLS_BOOST_ROUND, valid_sets=[dva],
                callbacks=[lgb.early_stopping(CLS_EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]))
        cls_top_pred = np.mean([m.predict(x_te) for m in cls_top_models], axis=0).astype(np.float32)
        del cls_top_models

        cls_bot_models = []
        for seed in cls_seeds_list:
            lp2 = dict(LGB_CLS_PARAMS); lp2["n_jobs"] = _active_nthread; lp2["scale_pos_weight"] = pos_w_bot
            dtr2 = lgb.Dataset(x_tr, label=y_tr_bot); dva2 = lgb.Dataset(x_va, label=y_va_bot, reference=dtr2)
            cls_bot_models.append(lgb.train(lp2, dtr2, CLS_BOOST_ROUND, valid_sets=[dva2],
                callbacks=[lgb.early_stopping(CLS_EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]))
        cls_bot_pred = np.mean([m.predict(x_te) for m in cls_bot_models], axis=0).astype(np.float32)
        del cls_bot_models

    del x_tr, x_va, x_te, dmat_te; gc.collect()

    # zscore 融合（保持原公式）
    dates = test_raw["date"].to_numpy(copy=False)
    reg_z = _zscore_by_date(reg_pred, dates)
    top_z = _zscore_by_date(cls_top_pred, dates)
    bot_z = _zscore_by_date(cls_bot_pred, dates)
    if cfg.get("_return_components", False):
        return np.column_stack([reg_z, top_z, bot_z]).astype(np.float32)
    return ((1 - alpha) * reg_z + (alpha / 2) * top_z - (alpha / 2) * bot_z).astype(np.float32)


def train_coarse_fine(
    tr: pd.DataFrame, va: pd.DataFrame, test_raw: pd.DataFrame,
    fcols: List[str], cfg: dict,
) -> np.ndarray:
    """
    粗排+精排：基模型全样本 → 精排模型在头/尾难样本上训练。
    粗排模型按窗口缓存，同组后续实验直接读取跳过训练。
    """
    ratio = cfg["hard_ratio"]
    if cfg.get("_pre_sorted"):
        tr_s, va_s = tr, va
    else:
        tr_s = tr.sort_values(["date", "stock_code"]).reset_index(drop=True)
        va_s = va.sort_values(["date", "stock_code"]).reset_index(drop=True)

    x_tr, y_tr, x_va, y_va = _prepare_training_data(tr_s, va_s, fcols)
    x_te = _extract_features(test_raw, fcols)

    # 粗排模型：检查缓存
    cpaths = _coarse_cache_paths(cfg)
    if cpaths and all(os.path.exists(p) for p in cpaths):
        tr_pred = np.load(cpaths[0])
        va_pred = np.load(cpaths[1])
        coarse_pred_te = np.load(cpaths[2])
    else:
        coarse_models = _train_xgb_reg_seeds(x_tr, y_tr, x_va, y_va)
        tr_pred = _predict_xgb_from_array(coarse_models, x_tr)
        va_pred = _predict_xgb_from_array(coarse_models, x_va)
        coarse_pred_te = _predict_xgb_from_array(coarse_models, x_te)
        del coarse_models; gc.collect()
        if cpaths:
            os.makedirs(os.path.dirname(cpaths[0]), exist_ok=True)
            np.save(cpaths[0], tr_pred)
            np.save(cpaths[1], va_pred)
            np.save(cpaths[2], coarse_pred_te)

    # 粗排打分 → 筛选头尾难样本
    tr_rp = _rank_pct(tr_pred, tr_s["date"].to_numpy(copy=False))
    head_mask = tr_rp >= (1.0 - ratio)
    tail_mask = tr_rp <= ratio

    va_rp = _rank_pct(va_pred, va_s["date"].to_numpy(copy=False))
    va_head_mask = va_rp >= (1.0 - ratio)
    va_tail_mask = va_rp <= ratio

    # 精排 Head 模型（子集切片，无需重新提取特征）
    head_models = []
    if head_mask.sum() > 500 and va_head_mask.sum() > 100:
        head_models = _train_xgb_reg_seeds(
            x_tr[head_mask], y_tr[head_mask], x_va[va_head_mask], y_va[va_head_mask])

    # 精排 Tail 模型
    tail_models = []
    if tail_mask.sum() > 500 and va_tail_mask.sum() > 100:
        tail_models = _train_xgb_reg_seeds(
            x_tr[tail_mask], y_tr[tail_mask], x_va[va_tail_mask], y_va[va_tail_mask])

    del x_tr, y_tr, x_va, y_va; gc.collect()

    # test 打分（coarse_pred_te 已在前面获取，来自缓存或现场计算）
    test_dates = test_raw["date"].to_numpy(copy=False)
    test_rp = _rank_pct(coarse_pred_te, test_dates)

    factor = test_rp * 2 - 1  # 中间层 [-1, 1]

    if head_models:
        head_idx = np.where(test_rp >= (1.0 - ratio))[0]
        head_fine = _predict_xgb_from_array(head_models, x_te[head_idx])
        head_rp = _rank_pct(head_fine, test_dates[head_idx])
        factor[head_idx] = 1.0 + head_rp

    if tail_models:
        tail_idx = np.where(test_rp <= ratio)[0]
        tail_fine = _predict_xgb_from_array(tail_models, x_te[tail_idx])
        tail_rp = _rank_pct(tail_fine, test_dates[tail_idx])
        factor[tail_idx] = -1.0 - (1.0 - tail_rp)

    del head_models, tail_models, x_te; gc.collect()
    return factor.astype(np.float32)


def train_double_ensemble(
    tr: pd.DataFrame, va: pd.DataFrame, test_raw: pd.DataFrame,
    fcols: List[str], cfg: dict,
) -> np.ndarray:
    """
    DoubleEnsemble：迭代式样本加权 + 特征筛选，K 个基学习器取均值。
    标签只算一次，每轮只对特征子集提取数组。
    """
    n_iters = cfg["n_iters"]
    decay = cfg["decay"]
    feat_ratio = cfg["feat_ratio"]

    if cfg.get("_pre_sorted"):
        tr_s, va_s = tr, va
    else:
        tr_s = tr.sort_values(["date", "stock_code"]).reset_index(drop=True)
        va_s = va.sort_values(["date", "stock_code"]).reset_index(drop=True)

    # 标签只算一次
    y_tr = _neutralize_label_array(tr_s)
    y_va = _neutralize_label_array(va_s)
    current_fcols = list(fcols)
    sample_weights = np.ones(len(tr_s), dtype=np.float32)

    all_models: List[Tuple[List[xgb.Booster], List[str]]] = []

    for it in range(n_iters):
        x_tr_iter = _extract_features(tr_s, current_fcols)
        x_va_iter = _extract_features(va_s, current_fcols)
        iter_models = _train_xgb_reg_seeds(
            x_tr_iter, y_tr, x_va_iter, y_va,
            sample_weights=sample_weights,
            feature_names=current_fcols,
        )
        all_models.append((iter_models, list(current_fcols)))
        del x_va_iter

        # 更新样本权重
        pred_tr = _predict_xgb_from_array(iter_models, x_tr_iter, feature_names=current_fcols)
        loss = np.abs(pred_tr - y_tr)
        loss_norm = loss / (loss.max() + 1e-8)
        sample_weights = sample_weights * (1.0 + decay * loss_norm)
        sample_weights = sample_weights / sample_weights.mean()
        del x_tr_iter, pred_tr

        # 更新特征子集
        importance: Dict[str, float] = {}
        for m in iter_models:
            for feat, score in m.get_score(importance_type="gain").items():
                importance[feat] = importance.get(feat, 0) + score
        if importance:
            sorted_feats = sorted(importance, key=importance.get, reverse=True)
            n_keep = max(10, int(len(sorted_feats) * feat_ratio))
            current_fcols = sorted_feats[:n_keep]
        _force_gc()

    # 集成预测
    preds = []
    for iter_models, used_fcols in all_models:
        x_te = _extract_features(test_raw, used_fcols)
        preds.append(_predict_xgb_from_array(iter_models, x_te, feature_names=used_fcols))
        del x_te

    del all_models; gc.collect()
    return np.mean(np.vstack(preds), axis=0).astype(np.float32)


class _RankICEarlyStop(xgb.callback.TrainingCallback):
    def __init__(self, rounds, min_improve):
        self.rounds = rounds; self.min_improve = min_improve
        self.best = -np.inf; self.wait = 0
    def after_iteration(self, model, epoch, evals_log):
        if "eval" not in evals_log or "rank_ic" not in evals_log["eval"]:
            return False
        val = evals_log["eval"]["rank_ic"][-1]
        if val > self.best + self.min_improve:
            self.best = val; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.rounds: return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# 滚动训练（通用版，接受 train_fn）
# ═══════════════════════════════════════════════════════════════════════

_print_lock = threading.Lock()

# train_fn 签名: (tr, va, test_raw, fcols, cfg) -> np.ndarray (factor values)
TrainFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], dict], np.ndarray]

TRAIN_FN_MAP: Dict[str, TrainFn] = {
    "fusion": train_fusion,
    "coarse_fine": train_coarse_fine,
    "double_ensemble": train_double_ensemble,
}


def _process_one_window(
    window_idx: int, train_dfs: List[pd.DataFrame],
    test_raw: pd.DataFrame, fcols: List[str],
    total_windows: int, train_fn: TrainFn, cfg: dict,
) -> pd.DataFrame:
    with _print_lock:
        print(f"    窗口 {window_idx}/{total_windows} 开始 ...", flush=True)

    dtr = pd.concat(train_dfs, ignore_index=True)
    tr, va = eng.split_tv(dtr, TRAIN_RATIO)
    del dtr, train_dfs
    _force_gc()

    tr = tr.sort_values(["date", "stock_code"]).reset_index(drop=True)
    va = va.sort_values(["date", "stock_code"]).reset_index(drop=True)

    cfg_with_window = dict(cfg)
    cfg_with_window["_window_idx"] = window_idx
    cfg_with_window["_pre_sorted"] = True
    factor_values = train_fn(tr, va, test_raw, fcols, cfg_with_window)
    del tr, va; _force_gc()

    y_neutral = _neutralize_label_array(test_raw)
    dates = test_raw["date"].values
    codes = test_raw["stock_code"].values
    returns = test_raw["c_pct_5"].values

    result = pd.DataFrame({
        "date": dates, "stock_code": codes,
        "return_neutral": y_neutral, "return": returns,
    })
    if factor_values.ndim == 2 and factor_values.shape[1] == 3:
        result["reg_z"] = factor_values[:, 0]
        result["top_z"] = factor_values[:, 1]
        result["bot_z"] = factor_values[:, 2]
        alpha = cfg.get("alpha", 0.5)
        result["factor"] = ((1 - alpha) * factor_values[:, 0]
                            + (alpha / 2) * factor_values[:, 1]
                            - (alpha / 2) * factor_values[:, 2]).astype(np.float32)
    else:
        result["factor"] = factor_values

    with _print_lock:
        print(f"    窗口 {window_idx}/{total_windows} 完成, 样本数 {len(result)}", flush=True)

    del factor_values, y_neutral
    _force_gc()
    return result


def rolling_train_predict(
    dfs: List[pd.DataFrame], fcols: List[str],
    train_fn: TrainFn, cfg: dict,
    max_parallel: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    tasks = []
    tdl: List[pd.DataFrame] = []
    total_windows = len(dfs) - 1
    for i in range(1, len(dfs)):
        tdl.append(dfs[i - 1])
        if len(tdl) > WINDOW_SIZE:
            tdl.pop(0)
        if len(tdl) < WINDOW_SIZE:
            continue
        tasks.append((i, list(tdl), dfs[i], fcols, total_windows, train_fn, cfg))

    if not tasks:
        return None
    n_parallel = max_parallel if max_parallel else MAX_PARALLEL_WINDOWS

    global _active_nthread
    _active_nthread = _nthread_for(n_parallel)
    print(f"    共 {len(tasks)} 窗口, {n_parallel} 路并行, "
          f"每模型 nthread={_active_nthread}", flush=True)

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        factor_frames = list(pool.map(lambda a: _process_one_window(*a), tasks))
    return pd.concat(factor_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# 持仓信号 & 多空回测
# ═══════════════════════════════════════════════════════════════════════

def build_holdings_signals(
    factor_df: pd.DataFrame, n_groups: int,
) -> Tuple[pd.Series, pd.Series]:
    head_pairs, tail_pairs = [], []
    for dt, rd in factor_df.groupby("date", sort=True):
        if len(rd) < n_groups * 2:
            continue
        ranked = rd[["stock_code", "factor"]].copy()
        ranked["group"] = pd.qcut(
            ranked["factor"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        head = ranked.loc[ranked["group"] == n_groups - 1, "stock_code"].dropna().astype(str).drop_duplicates().tolist()
        tail = ranked.loc[ranked["group"] == 0, "stock_code"].dropna().astype(str).drop_duplicates().tolist()
        dt_ts = pd.to_datetime(dt)
        head_pairs.append((dt_ts, head))
        tail_pairs.append((dt_ts, tail))
    if not head_pairs:
        return pd.Series(dtype=object), pd.Series(dtype=object)
    h_idx, h_vals = zip(*head_pairs)
    t_idx, t_vals = zip(*tail_pairs)
    return (
        pd.Series(list(h_vals), index=pd.DatetimeIndex(h_idx), name="head"),
        pd.Series(list(t_vals), index=pd.DatetimeIndex(t_idx), name="tail"),
    )


def remove_overlap(head_signal: pd.Series, tail_signal: pd.Series) -> Tuple[pd.Series, int]:
    """从头组中删去与尾组重合的股票，返回 (清理后头组, 总去重股票数)。"""
    total_removed = 0
    head_clean = head_signal.copy()
    for dt in head_clean.index:
        if dt in tail_signal.index:
            overlap = set(head_clean[dt]) & set(tail_signal[dt])
            if overlap:
                head_clean[dt] = [s for s in head_clean[dt] if s not in overlap]
                total_removed += len(overlap)
    return head_clean, total_removed


def _load_price_data():
    src = pd.read_pickle(PANEL_TRADE_PATH)
    src["date"] = pd.to_datetime(src["date"])
    src["stock_code"] = src["stock_code"].astype(str)
    src.sort_values(["date", "stock_code"], inplace=True)
    src.drop_duplicates(["date", "stock_code"], keep="last", inplace=True)
    op = src.pivot(index="date", columns="stock_code", values="open_price").sort_index().astype(np.float32)
    cp = src.pivot(index="date", columns="stock_code", values="close_price").sort_index().astype(np.float32)
    st = src.pivot(index="date", columns="stock_code", values="status").sort_index()
    del src; gc.collect()
    return op, cp, st


def run_longshort_backtest(
    head_signal: pd.Series, tail_signal: pd.Series,
    open_prices, close_prices, status, output_dir: str,
) -> Optional[dict]:
    from model_backtest_framework_longshort import analyze_longshort_holdings

    if head_signal.empty or tail_signal.empty:
        return None
    results = analyze_longshort_holdings(
        head_holdings_data=head_signal, tail_holdings_data=tail_signal,
        open_prices=open_prices, close_prices=close_prices,
        benchmark_data=None, status_data=status,
        method="periodic_rebalance", holding_period=REBALANCE_DAYS,
        commission_rate=COMMISSION_RATE, verbose=False,
    )
    _save_longshort_figure(results, os.path.join(output_dir, "longshort_backtest.png"))
    _save_yearly_longshort_figure(results, os.path.join(output_dir, "yearly_longshort.png"))
    return results


# ═══════════════════════════════════════════════════════════════════════
# 结果保存
# ═══════════════════════════════════════════════════════════════════════

def _save_longshort_figure(results: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nav_df = results["nav_df"]; stats = results["statistics"]
    fig = plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(nav_df.index, nav_df["head"], lw=2, color="red", label="头组(多头)")
    ax1.plot(nav_df.index, nav_df["tail"], lw=2, color="green", label="尾组(做空)")
    ax1.plot(nav_df.index, nav_df["benchmark"], lw=2, color="gray", alpha=0.7, label="基准")
    ax1.set_title("头组 / 尾组做空 / 基准 净值"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(2, 2, 2)
    he = nav_df["head"] / nav_df["benchmark"]
    ax2.plot(nav_df.index, nav_df["longshort"], lw=2, color="blue", label="多空组合")
    ax2.plot(he.index, he, lw=1.5, color="orange", alpha=0.8, label="头组超额")
    ax2.axhline(y=1, color="gray", ls="--", alpha=0.5)
    ax2.set_title("多空组合 & 头组超额"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3 = plt.subplot(2, 2, 3)
    ls_dd = nav_df["longshort"] / nav_df["longshort"].cummax()
    dd = (ls_dd - 1) * 100
    ax3.fill_between(dd.index, dd, 0, alpha=0.3, color="red"); ax3.plot(dd.index, dd, "r-", lw=1)
    ax3.set_title("多空组合回撤"); ax3.set_ylabel("回撤 (%)"); ax3.grid(True, alpha=0.3)
    ax4 = plt.subplot(2, 2, 4); ax4.axis("off")
    headers = ["指标", "头组超额", "尾组做空超额", "多空组合", "基准"]
    mkeys = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "win_rate"]
    mnames = ["年化收益(%)", "年化波动(%)", "夏普比率", "最大回撤(%)", "胜率(%)"]
    tdata = [[mn, *[f"{stats[k][m]:.2f}" for k in ["head_excess","tail_excess","longshort","benchmark"]]]
             for m, mn in zip(mkeys, mnames)]
    tbl = ax4.table(cellText=tdata, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.3, 1.5)
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close(fig)


def _save_yearly_longshort_figure(results: dict, save_path: str) -> None:
    """每年一张子图，画多空组合净值曲线（每年从 1 开始）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nav_df = results["nav_df"]
    ls_nav = nav_df["longshort"]
    ls_ret = ls_nav.pct_change().dropna()

    years = sorted(set(ls_ret.index.year))
    n = len(years)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, year in enumerate(years):
        ax = axes[0, i]
        mask = ls_ret.index.year == year
        yr_ret = ls_ret[mask]
        if yr_ret.empty:
            ax.set_title(f"{year}"); continue
        yr_nav = (1 + yr_ret).cumprod()
        yr_nav = pd.concat([pd.Series([1.0], index=[yr_ret.index[0]]), yr_nav])
        ax.plot(yr_nav.index, yr_nav.values, lw=2, color=colors[i])
        ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
        final = yr_nav.iloc[-1]
        ann_ret = (final - 1) * 100
        ax.set_title(f"{year}  ({ann_ret:+.1f}%)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("多空组合年度净值", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def compute_rank_ic(factor_df: pd.DataFrame) -> Dict[str, float]:
    ics = []
    for _, rd in factor_df.groupby("date"):
        if len(rd) < 10:
            continue
        ic = calculate_rank_ic(rd["factor"], rd["return_neutral"])
        if not np.isnan(ic):
            ics.append(float(ic))
    if not ics:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan, "rank_ic_ir": np.nan}
    m, s = float(np.mean(ics)), float(np.std(ics))
    return {"rank_ic_mean": round(m, 6), "rank_ic_std": round(s, 6),
            "rank_ic_ir": round(m / s if s > 0 else np.nan, 6)}


def save_experiment_results(
    exp_name: str, version: str, factor_df: pd.DataFrame,
    ls_results: Optional[dict], elapsed: float, cfg: dict,
    overlap_removed: int = 0,
) -> Dict[str, Any]:
    exp_dir = os.path.join(OUTPUT_ROOT, f"{exp_name}_{version}")
    os.makedirs(exp_dir, exist_ok=True)

    factor_df.to_pickle(os.path.join(exp_dir, "factor_df.pkl"))

    ic_metrics = compute_rank_ic(factor_df)
    metrics: Dict[str, Any] = {
        "experiment": exp_name, "version": version,
        **ic_metrics, "elapsed_sec": round(elapsed, 1),
        "overlap_removed": overlap_removed,
    }
    if ls_results:
        stats = ls_results["statistics"]
        for k in ["head", "tail", "tail_raw", "longshort", "benchmark",
                   "head_excess", "tail_excess"]:
            if k in stats:
                metrics[k] = stats[k]

    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    # 分组回测
    eng.backtest(factor_df, n=N_GROUPS, save_path=os.path.join(exp_dir, "backtest.png"))
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print(f"集成模型多空回测实验  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共 {len(EXPERIMENTS)} 组实验 × 2 版本 = {len(EXPERIMENTS)*2} 次训练")
    print("=" * 70)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    cache_dir = os.path.join(OUTPUT_ROOT, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for exp_idx, cfg in enumerate(EXPERIMENTS):
        exp_name = cfg["name"]
        group = cfg["group"]
        train_fn = TRAIN_FN_MAP[group]

        print(f"\n{'='*70}")
        print(f"[{exp_idx+1}/{len(EXPERIMENTS)}] {exp_name} ({group})")
        print(f"{'='*70}")

        # 注入缓存元数据，供训练函数做模型共享
        cfg["_cache_dir"] = cache_dir

        # ── V1: short panel ──
        cfg["_panel_tag"] = "short"
        factor_pkl_v1 = os.path.join(OUTPUT_ROOT, f"{exp_name}_v1", "factor_df.pkl")
        if os.path.exists(factor_pkl_v1):
            print(f"\n  V1: 发现缓存, 跳过训练")
            factor_v1 = pd.read_pickle(factor_pkl_v1)
            t_v1 = 0.0
        else:
            print(f"\n  V1: 加载 short panel ...")
            short_fcols, short_dfs = _load_panel(SHORT_PANEL_PATH, "short")
            setup_engine(short_fcols, short_dfs)
            print(f"  V1: 开始训练 ...")
            t0 = time.time()
            factor_v1 = rolling_train_predict(short_dfs, short_fcols, train_fn, cfg)
            t_v1 = time.time() - t0
            del short_dfs
            eng.dfs = []
            _force_gc()
            if factor_v1 is None or factor_v1.empty:
                print(f"  V1: 训练失败, 跳过"); continue
            print(f"  V1: 训练完成, 耗时 {t_v1:.0f}s, 样本数 {len(factor_v1)}")
            # 训练完立刻保存，避免后续回测崩溃导致训练结果丢失
            v1_save_dir = os.path.join(OUTPUT_ROOT, f"{exp_name}_v1")
            os.makedirs(v1_save_dir, exist_ok=True)
            factor_v1.to_pickle(os.path.join(v1_save_dir, "factor_df.pkl"))

        head_v1, tail_v1 = build_holdings_signals(factor_v1, N_GROUPS)

        # V1 回测：加载价格数据 → 回测 → 释放
        print(f"  V1: 回测 ...")
        v1_dir = os.path.join(OUTPUT_ROOT, f"{exp_name}_v1")
        os.makedirs(v1_dir, exist_ok=True)
        open_prices, close_prices, status = _load_price_data()
        ls_v1 = run_longshort_backtest(head_v1, tail_v1, open_prices, close_prices,
                                       status, v1_dir)
        # 分组回测需要 eng.avg_return，从 factor_v1 自身计算
        avg_ret = (factor_v1.groupby("date")["return"].mean()
                   .reset_index().rename(columns={"return": "avg_return"}))
        eng.avg_return = avg_ret
        m_v1 = save_experiment_results(exp_name, "v1", factor_v1, ls_v1, t_v1, cfg)
        all_results.append(m_v1)
        _print_summary(m_v1)
        del open_prices, close_prices, status
        _force_gc()

        # ── V2: all panel, tail from V1 ──
        if cfg.get("_skip_v2"):
            print(f"\n  V2: 已跳过"); del factor_v1, head_v1, tail_v1; _force_gc(); continue
        cfg["_panel_tag"] = "all"
        factor_pkl_v2 = os.path.join(OUTPUT_ROOT, f"{exp_name}_v2", "factor_df.pkl")
        if os.path.exists(factor_pkl_v2):
            print(f"\n  V2: 发现缓存, 跳过训练")
            factor_v2 = pd.read_pickle(factor_pkl_v2)
            t_v2 = 0.0
        else:
            print(f"\n  V2: 加载 all panel ...")
            all_fcols, all_dfs = _load_panel(ALL_PANEL_PATH, "all")
            setup_engine(all_fcols, all_dfs)
            print(f"  V2: 开始训练 ...")
            t0 = time.time()
            factor_v2 = rolling_train_predict(all_dfs, all_fcols, train_fn, cfg,
                                               max_parallel=3)
            t_v2 = time.time() - t0
            del all_dfs
            eng.dfs = []
            _force_gc()
            if factor_v2 is None or factor_v2.empty:
                print(f"  V2: 训练失败, 跳过")
                del factor_v1, head_v1, tail_v1; _force_gc(); continue
            print(f"  V2: 训练完成, 耗时 {t_v2:.0f}s, 样本数 {len(factor_v2)}")
            v2_save_dir = os.path.join(OUTPUT_ROOT, f"{exp_name}_v2")
            os.makedirs(v2_save_dir, exist_ok=True)
            factor_v2.to_pickle(os.path.join(v2_save_dir, "factor_df.pkl"))

        head_v2, _ = build_holdings_signals(factor_v2, N_GROUPS)
        head_v2_clean, n_removed = remove_overlap(head_v2, tail_v1)
        print(f"  V2: 头尾去重 {n_removed} 只股票")

        # V2 回测
        print(f"  V2: 回测 ...")
        v2_dir = os.path.join(OUTPUT_ROOT, f"{exp_name}_v2")
        os.makedirs(v2_dir, exist_ok=True)
        open_prices, close_prices, status = _load_price_data()
        ls_v2 = run_longshort_backtest(head_v2_clean, tail_v1, open_prices, close_prices,
                                       status, v2_dir)
        avg_ret2 = (factor_v2.groupby("date")["return"].mean()
                    .reset_index().rename(columns={"return": "avg_return"}))
        eng.avg_return = avg_ret2
        m_v2 = save_experiment_results(exp_name, "v2", factor_v2, ls_v2, t_v2, cfg,
                                       overlap_removed=n_removed)
        all_results.append(m_v2)
        _print_summary(m_v2)

        del open_prices, close_prices, status, factor_v1, factor_v2
        del head_v1, tail_v1, head_v2, head_v2_clean
        _force_gc()

    # 汇总
    _save_summary(all_results)
    print(f"\n{'='*70}")
    print(f"全部实验完成  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"{'='*70}")


def _print_summary(m: Dict[str, Any]) -> None:
    ls = m.get("longshort", {})
    print(f"    Rank IC: {m.get('rank_ic_mean', np.nan):.6f}  "
          f"多空年化: {ls.get('annual_return', np.nan):.2f}%  "
          f"夏普: {ls.get('sharpe_ratio', np.nan):.3f}  "
          f"最大回撤: {ls.get('max_drawdown', np.nan):.2f}%")


def _save_summary(all_results: List[Dict[str, Any]]) -> None:
    rows = []
    for m in all_results:
        ls = m.get("longshort", {})
        he = m.get("head_excess", {})
        rows.append({
            "experiment": m.get("experiment", ""),
            "version": m.get("version", ""),
            "rank_ic_mean": m.get("rank_ic_mean", np.nan),
            "rank_ic_ir": m.get("rank_ic_ir", np.nan),
            "ls_annual_return": ls.get("annual_return", np.nan),
            "ls_sharpe": ls.get("sharpe_ratio", np.nan),
            "ls_max_drawdown": ls.get("max_drawdown", np.nan),
            "head_excess_annual": he.get("annual_return", np.nan),
            "overlap_removed": m.get("overlap_removed", 0),
            "elapsed_sec": m.get("elapsed_sec", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_ROOT, "summary.csv"), index=False, encoding="utf-8-sig")
    print("\n汇总表:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"FATAL: {err}")
        traceback.print_exc()
        raise
