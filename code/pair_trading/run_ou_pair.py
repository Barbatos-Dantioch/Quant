#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OU 同行业配对选股实验 (OU-P00 baseline 系列)

设计依据: markdown/2026-04-28_ou_pair_selection_plan.md

核心流程 (每个截面 t):
    1. 行业内枚举所有 pair (i, j),i<j,固定字典序。
    2. 训练窗口 [t-W_train-W_valid, t-W_valid-1] 对每对算 X = log_p_i - log_p_j。
    3. 矩阵化 AR(1) 估计 → b, kappa, mu, sigma, half_life
       矩阵化 ADF p 值
    4. 验证窗口 [t-W_valid, t-1] 内每天算 signal = kappa*(mu - X_t),
       label = r_i_{t+1} - r_j_{t+1},Spearman 相关 → valid_rank_ic
    5. OU 硬过滤: b>0 ∧ 5<=half_life<=30 ∧ ADF_p<0.05
    6. 行业内按 valid_rank_ic 贪心去重 (每只股票最多 1 pair)
    7. 合法 pair 筛选: 预测空头方 ∈ short_pool_t
    8. 全市场所有合法 pair 按 |signal_t| 排序取前 20% (向上取整)
    9. 多空组合: 多头 = top pair 中预测多头方; 空头 = top pair 中预测空头方
    10. 等权配置,日频换仓 (T 日信号 → T+1 日开盘建仓)。

实验:
    OU-P00     训练 252 日 验证 20 日
    OU-P00-120 训练 120 日 验证 20 日
    OU-P00-60  训练  60 日 验证 20 日

输出:
    output/0506_ou_pair/
    ├── summary.csv                  跨实验汇总
    ├── OU-P00/
    │   ├── metrics.json             实验指标
    │   ├── pair_log.parquet         每日合法/top pair 明细
    │   ├── stock_pred.parquet       每日股票级预测 (仅 top 部分)
    │   ├── longshort_backtest.png
    │   ├── head_backtest.png
    │   └── tail_backtest.png
    ├── OU-P00-120/
    └── OU-P00-60/

环境变量:
    SMOKE=1   只跑 OU-P00 的第一个截面,打印诊断后退出 (用于冒烟)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

os.chdir("/root/quant")

# ── 通用骨架 ──
_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
if _FACTOR1_DIR not in sys.path:
    sys.path.insert(0, _FACTOR1_DIR)
from _pair_runner import (
    _batch_adf_pvalues,
    build_tradable_mask,
    load_price_industry_mv,
    compute_short_pool,
    run_longshort_backtest,
    run_longshort_backtest_weighted,
    save_summary_row,
)

# ── 常量 ──
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2026-05-13"      # 数据扩展后, 充分利用所有可用日期 (2026-05-13 数据已就绪)
DATA_START     = "2022-09-01"      # OU-P00 需要 252+20=272 个交易日历史,留 ~50 日缓冲
OUTPUT_DIR     = "output/0506_ou_pair"
COMMISSION     = 0.0007
TOP_PCT        = 0.20
TOP_PCT_MIN_N  = 50      # top 选 pair 数下限: n_top = min(n_legal, max(TOP_PCT_MIN_N, ceil(n_legal*TOP_PCT)))
                         # 0 表示无下限 (旧规则); 50 = 至少选 50 对 (合法 pair 不足 50 时全部入选)
HALF_LIFE_MIN  = 5.0
HALF_LIFE_MAX  = 30.0
ADF_P_MAX      = 0.05
# 调试用: 通过环境变量临时覆盖参数 (SMOKE 时探测漏斗变化)。
# 生产运行 (无该环境变量) 始终使用上面的 baseline 值。
_adf_override = os.environ.get("ADF_P_OVERRIDE")
if _adf_override is not None:
    ADF_P_MAX = float(_adf_override)
_hl_min_override = os.environ.get("HL_MIN_OVERRIDE")
if _hl_min_override is not None:
    HALF_LIFE_MIN = float(_hl_min_override)
_hl_max_override = os.environ.get("HL_MAX_OVERRIDE")
if _hl_max_override is not None:
    HALF_LIFE_MAX = float(_hl_max_override)
VALID_WINDOW   = 20

# ── Barra CNE6 风格因子 (供 dedup_rank="barra_dist" 计算配对风格距离) ──
_BARRA_EXPO_PATH = "/root/quant/Data/all/barra_exposure_cne6_sw21.pkl"
_BARRA_COV_PATH  = "/root/quant/Data/all/barra_cov_cne6_sw21.pkl"
# 20 个风格因子 (剔除 31 行业 + COUNTRY)
BARRA_STYLE_FACTORS = [
    "SIZE", "MIDCAP", "BETA", "MOMENTUM", "RESVOL", "LIQUIDTY", "BTOP", "GROWTH",
    "LEVERAGE", "EARNYILD", "EARNQLTY", "EARNVAR", "PROFIT", "INVSQLTY", "DIVYILD",
    "LTREVRSL", "STREVRSL", "INDMOM", "SEASON", "ANALSENTI",
]
# 回撤元凶子集 (size + 各类动量/反转 + 波动)
BARRA_SUBSET_FACTORS = ["SIZE", "MOMENTUM", "STREVRSL", "LTREVRSL", "INDMOM", "RESVOL"]

# 数值防御
_EPS = 1e-12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 日志 ──
LOG_PATH = os.path.join("output", f"0506_ou_pair_run.log")

class _Logger:
    """同时写到原 stdout/stderr 和日志文件;
    若原 stdout 已经被 nohup 重定向到同一日志,会出现"双写",通过比较
    fileno 的 inode 来识别该情形并跳过 terminal 写入。"""
    def __init__(self, fp, original):
        self.terminal = original
        self.log = open(fp, "a", encoding="utf-8")
        self._skip_terminal = self._stdout_already_to_log(original, fp)

    @staticmethod
    def _stdout_already_to_log(original, log_path):
        try:
            log_stat = os.stat(log_path)
            orig_stat = os.fstat(original.fileno())
            return (log_stat.st_dev == orig_stat.st_dev and
                    log_stat.st_ino == orig_stat.st_ino)
        except (OSError, AttributeError, ValueError):
            return False

    def write(self, msg):
        if not self._skip_terminal:
            self.terminal.write(msg)
        self.log.write(msg); self.log.flush()
    def flush(self):
        if not self._skip_terminal:
            self.terminal.flush()
        self.log.flush()


def _install_loggers():
    """仅在 __main__ 入口调用,避免 import 时副作用接管 stdout。"""
    if os.environ.get("SMOKE") == "1":
        return
    sys.stdout = _Logger(LOG_PATH, sys.stdout)
    sys.stderr = _Logger(LOG_PATH, sys.stderr)


# =====================================================================
# OU AR(1) 估计 (矩阵化)
# =====================================================================

def estimate_ou_ar1(spread: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对 (T, P) 形状的价差矩阵批量做 AR(1) 离散估计 + 样本统计。

    模型: X_{t+1} = a + b * X_t + e
    返回 OU 参数 + 样本统计:
        b           (P,)  AR(1) 斜率
        a           (P,)  截距
        kappa       (P,)  -ln(b),仅 b>0 处有效,其余处 NaN
        mu          (P,)  a / (1-b),仅 |1-b|>EPS 处有效     (OU 推算的均值)
        sigma       (P,)  sqrt(res_var * 2*kappa / (1-b**2)),仅 b∈(0,1) 处有效  (OU 推算的稳态 σ)
        half_life   (P,)  ln(2) / kappa
        res_var     (P,)  AR(1) 残差方差 (无偏估计,自由度 T-1-2)
        mean_X      (P,)  全 T 个观测的样本均值                 (用于 simple 信号)
        std_X       (P,)  全 T 个观测的样本标准差 (ddof=1)      (用于 simple 信号)

    注意:
        - b<=0 处所有 OU 参数置 NaN (不是均值回复过程)。
        - b>=1 处 kappa<=0,half_life 为负或无穷,这些 pair 自然会被后续过滤掉。
        - mean_X / std_X 是纯样本统计,与 OU 模型独立,任何分支都可用。
    """
    T, P = spread.shape
    # 样本均值 / 标准差 (与 b/kappa 估计独立)
    mean_X = spread.mean(axis=0) if T > 0 else np.full(P, np.nan)
    std_X = spread.std(axis=0, ddof=1) if T > 1 else np.full(P, np.nan)

    if T < 4:
        # 自由度不够: AR(1) 至少需要 4 个观测才有 1 个自由度做无偏方差
        nan = np.full(P, np.nan)
        return {"b": nan, "a": nan, "kappa": nan, "mu": nan,
                "sigma": nan, "half_life": nan, "res_var": nan,
                "mean_X": mean_X, "std_X": std_X}

    x = spread[:-1, :]                        # (T-1, P)  X_t
    y = spread[1:, :]                         # (T-1, P)  X_{t+1}
    n = T - 1

    x_mean = x.mean(axis=0)                   # (P,)
    y_mean = y.mean(axis=0)                   # (P,)
    xc = x - x_mean                           # (T-1, P)
    yc = y - y_mean
    var_x = (xc ** 2).sum(axis=0)             # (P,)
    cov = (xc * yc).sum(axis=0)               # (P,)
    safe = var_x > _EPS

    b = np.full(P, np.nan)
    a = np.full(P, np.nan)
    b[safe] = cov[safe] / var_x[safe]
    a[safe] = y_mean[safe] - b[safe] * x_mean[safe]

    # 残差方差 (自由度修正,n-2)
    res = y - (a[None, :] + b[None, :] * x)   # (T-1, P)
    ssr = (res ** 2).sum(axis=0)              # (P,)
    dof = n - 2
    res_var = np.full(P, np.nan)
    if dof > 0:
        res_var[safe] = np.maximum(ssr[safe] / dof, 0.0)

    # OU 参数: 仅 b ∈ (0, 1) 时有意义
    valid_b = safe & (b > _EPS) & (b < 1.0 - _EPS)
    kappa = np.full(P, np.nan)
    mu = np.full(P, np.nan)
    sigma = np.full(P, np.nan)
    half_life = np.full(P, np.nan)

    kappa[valid_b] = -np.log(b[valid_b])
    half_life[valid_b] = math.log(2.0) / kappa[valid_b]
    mu[valid_b] = a[valid_b] / (1.0 - b[valid_b])
    sigma_factor = 2.0 * kappa[valid_b] / (1.0 - b[valid_b] ** 2)
    sigma_factor = np.maximum(sigma_factor, 0.0)
    sigma[valid_b] = np.sqrt(np.maximum(res_var[valid_b], 0.0) * sigma_factor)

    return {"b": b, "a": a, "kappa": kappa, "mu": mu,
            "sigma": sigma, "half_life": half_life, "res_var": res_var,
            "mean_X": mean_X, "std_X": std_X}


# =====================================================================
# 矩阵化 Spearman RankIC (按列对配对)
# =====================================================================

def _rank_per_col(mat: np.ndarray) -> np.ndarray:
    """对 (W, P) 的每一列做秩排序 (1..W, ties 不做 average,按 argsort 稳定序)。

    返回 (W, P) 的秩矩阵, NaN 值保留为 NaN。

    NOTE:
        - W 远小于 P 时 (W~20, P~50K),双 argsort 矩阵化比逐列 Python 循环快 ~15 倍。
        - 不做精确 average-rank: W=20 时小数变量 ties 概率极低,对 Spearman 影响可忽略。
        - NaN 处理: 把 NaN 替换为 +inf 排到末尾,排序后再把那些位置 mask 回 NaN。
    """
    W, P = mat.shape
    nan_mask = np.isnan(mat)
    if not nan_mask.any():
        # fast path: 无 NaN, 双 argsort 矩阵化
        order = np.argsort(mat, axis=0, kind="mergesort")
        ranks = np.argsort(order, axis=0, kind="mergesort").astype(np.float64) + 1.0
        return ranks

    # 有 NaN: 用 +inf 替换 → NaN 自动排到末尾
    work = np.where(nan_mask, np.inf, mat)
    order = np.argsort(work, axis=0, kind="mergesort")
    ranks = np.argsort(order, axis=0, kind="mergesort").astype(np.float64) + 1.0
    ranks[nan_mask] = np.nan
    return ranks


def spearman_per_pair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """对 (W, P) 形状的两个矩阵按列计算 Spearman 相关 (P 个独立的 IC)。

    返回 (P,) 的 IC 数组,有效观测 < 5 时返回 NaN。

    NOTE: 必须先按列对取并集 mask,再分别 rank。否则 x 缺失行的 y 值会被
          rank 进去,与 scipy.spearmanr 不一致。
    """
    P = x.shape[1]
    pair_mask = (~np.isnan(x)) & (~np.isnan(y))      # (W, P) 取交集
    # 共同 mask 后再 rank
    x_masked = np.where(pair_mask, x, np.nan)
    y_masked = np.where(pair_mask, y, np.nan)
    rx = _rank_per_col(x_masked)
    ry = _rank_per_col(y_masked)

    out = np.full(P, np.nan)
    n_eff = pair_mask.sum(axis=0).astype(np.float64)
    valid = n_eff >= 5
    if not valid.any():
        return out

    # 在 mask=False 处把 rank 置 0,这样不会影响 sum/sumsq
    rx0 = np.where(pair_mask, rx, 0.0)
    ry0 = np.where(pair_mask, ry, 0.0)
    sx = rx0.sum(axis=0)
    sy = ry0.sum(axis=0)
    sxy = (rx0 * ry0).sum(axis=0)
    sxx = (rx0 * rx0).sum(axis=0)
    syy = (ry0 * ry0).sum(axis=0)

    n = n_eff
    cov = sxy - sx * sy / n
    var_x = sxx - sx * sx / n
    var_y = syy - sy * sy / n
    den = np.sqrt(np.maximum(var_x * var_y, 0.0))

    safe = valid & (den > _EPS)
    out[safe] = cov[safe] / den[safe]
    return out


# =====================================================================
# NZC (Number of Zero Crossings, 零穿越次数) - 均值回归强度的非参数度量
# =====================================================================

def compute_nzc_per_pair(X_train: np.ndarray) -> np.ndarray:
    """对训练窗 X_train 计算每对的 NZC.

    NZC 定义:
        Y_t = X_t - mean(X_train), 沿时间维度去均值
        NZC = |{ t : sign(Y_t) ≠ sign(Y_{t-1}), 且二者皆非零 }|
        即: 严格穿越均值线的次数 (相邻点符号相反)

    参数
    ----
    X_train : (W, n_pair) 训练窗对数价差

    返回
    ----
    nzc : (n_pair,) int32, 每对的零穿越次数, 范围 [0, W-1]

    NaN 处理:
        若某对训练窗存在 NaN, 该对 NZC 设为 0 (不参与排序时排末尾).
        process_one_section 在调用本函数前已确保 X_train 无 NaN (训练段 ok mask).
    """
    if X_train.size == 0:
        return np.zeros(0, dtype=np.int32)
    Y = X_train - X_train.mean(axis=0, keepdims=True)
    s = np.sign(Y).astype(np.int8)        # (W, n_pair), 三态 {-1, 0, 1}
    # 严格穿越: s[t-1] * s[t] == -1 (即从 +1 跳到 -1 或反之, 排除 0)
    cross = (s[1:, :] * s[:-1, :]) == -1   # (W-1, n_pair) bool
    nzc = cross.sum(axis=0).astype(np.int32)
    return nzc


# =====================================================================
# 训练窗内 K 折 CV (用于评估 pair 的稳定性)
# =====================================================================

def compute_cv_mean_ic(
    X_train_full: np.ndarray,   # (W_train, n_pair) 训练段全部价差
    n_pass: int,                # = X_train_full.shape[1]
    k: int = 5,                 # 折数
    signal_type: str = "zscore",
    label_h: int = 5,           # 验证段 label 用 h 日累积价差变化
) -> np.ndarray:
    """
    训练窗内做 K 折经典 CV (B 方案):
        把 W_train 切成 k 段, 折 k 用第 k 段做验证, 其它 k-1 段拼接做训练。
        每折估 OU 参数 + 算验证段 spearman, 最终取 K 折 IC 均值。

    返回 (n_pair,) 的 cv_mean_ic 数组 (NaN 表示不可用)。

    NOTE:
        - K-1 段拼接训练时, 段间断点处 X_t -> X_{t+1} 会有 ~k 个虚假跳跃,
          对 252 日 / 5 折 / 4 个断点 ≈ 1.6% 的污染, 偏差可控。
        - 验证段最后 label_h 天的 label 用了"段外"的数据 (训练段或下一截面),
          这是同窗口内复用, 非 lookahead bias, 偏差可忽略。
        - signal_type 仅决定信号公式 (kappa_dev / zscore / simple / simple_no_hl)。
    """
    W = X_train_full.shape[0]
    if W < k * 10:                    # 每折至少 10 个观测才有意义
        return np.full(n_pass, np.nan)
    # 切段索引 (5 段, 余数加到最后一段)
    seg_size = W // k
    segs = []
    for i in range(k):
        lo = i * seg_size
        hi = (i + 1) * seg_size if i < k - 1 else W
        segs.append((lo, hi))

    fold_ic_list = []                  # 长度 k, 每个 (n_pair,)
    for vi in range(k):
        v_lo, v_hi = segs[vi]
        # 训练段: 拼接其它 k-1 段
        train_idx = []
        for ti in range(k):
            if ti == vi:
                continue
            t_lo, t_hi = segs[ti]
            train_idx.extend(range(t_lo, t_hi))
        train_idx = np.array(train_idx)
        X_tr = X_train_full[train_idx, :]    # (W_train_k-1, n_pair)
        X_va = X_train_full[v_lo:v_hi, :]    # (seg_size, n_pair)

        # OU 估计 (4 段拼接)
        ou_k = estimate_ou_ar1(X_tr)
        kappa_k = ou_k["kappa"]; mu_k = ou_k["mu"]; sigma_k = ou_k["sigma"]
        mean_X_k = ou_k["mean_X"]; std_X_k = ou_k["std_X"]
        # 短窗 σ (每折训练段 X_tr 末段, 仅 abs_dev_* 系列使用)
        if X_tr.shape[0] >= 30:
            sigma_short_20_k = X_tr[-20:, :].std(axis=0, ddof=1)
            sigma_short_30_k = X_tr[-30:, :].std(axis=0, ddof=1)
        elif X_tr.shape[0] >= 20:
            sigma_short_20_k = X_tr[-20:, :].std(axis=0, ddof=1)
            sigma_short_30_k = sigma_short_20_k
        else:
            sigma_short_20_k = std_X_k
            sigma_short_30_k = std_X_k

        # 验证段信号 (按 signal_type)
        if signal_type == "zscore":
            sigma_safe = np.clip(sigma_k, _EPS, None)
            signal_va = kappa_k[None, :] * (mu_k[None, :] - X_va) / sigma_safe[None, :]
        elif signal_type in ("simple", "simple_no_hl"):
            std_safe = np.clip(std_X_k, _EPS, None)
            signal_va = (mean_X_k[None, :] - X_va) / std_safe[None, :]
        elif signal_type == "abs_dev":
            signal_va = mu_k[None, :] - X_va
        elif signal_type == "abs_dev_sqrt_s20":
            s_safe = np.clip(sigma_short_20_k, _EPS, None)
            signal_va = (mu_k[None, :] - X_va) * np.sqrt(s_safe[None, :])
        elif signal_type == "abs_dev_s20":
            s_safe = np.clip(sigma_short_20_k, _EPS, None)
            signal_va = (mu_k[None, :] - X_va) * s_safe[None, :]
        elif signal_type == "abs_dev_sqrt_s30":
            s_safe = np.clip(sigma_short_30_k, _EPS, None)
            signal_va = (mu_k[None, :] - X_va) * np.sqrt(s_safe[None, :])
        elif signal_type == "abs_dev_s30":
            s_safe = np.clip(sigma_short_30_k, _EPS, None)
            signal_va = (mu_k[None, :] - X_va) * s_safe[None, :]
        else:                              # kappa_dev
            signal_va = kappa_k[None, :] * (mu_k[None, :] - X_va)

        # 验证 label: 5 日累积价差变化 (正向, 与 baseline 验证段标签语义一致)
        # baseline 用 (R_i - R_j) ≈ ΔX = X_{t+1} - X_t (1 日)
        # CV 推广到 h 日: label = X_{t'+h} - X_t', 信号高 (X 低于 mu, 预期 X 上涨) 时 label 应高
        # 验证段 [v_lo, v_hi] 内, 对每个 t' 算 X_train_full[v_lo+t'+label_h] - X_va[t']
        # 段尾 t' 的 t'+label_h 可能越出 W, 那些 label 设 NaN
        Wv = v_hi - v_lo
        label_va = np.full((Wv, n_pass), np.nan, dtype=np.float64)
        for t_off in range(Wv):
            fwd = v_lo + t_off + label_h
            if fwd < W:
                # 正向: 用未来价差减当前价差
                label_va[t_off, :] = X_train_full[fwd, :] - X_va[t_off, :]
            # 段尾越出 W 时,该行保持 NaN

        # spearman per pair (signal_va, label_va), 同 pair 内 sigma 等是常数,
        # 所以 signal_va 的归一化对 IC 不影响 (rank 同序)。
        # 但 mu_k 不是常数,跨折 mu_k 不同会让 signal_va 跨时间值不同。
        # spearman_per_pair 已经支持有 NaN 输入。
        ic_fold = spearman_per_pair(signal_va, label_va)
        fold_ic_list.append(ic_fold)

    # 5 折 IC 均值 (允许个别折是 NaN, 用 nanmean)
    cv_mat = np.array(fold_ic_list)         # (k, n_pair)
    with np.errstate(all="ignore"):
        cv_mean = np.nanmean(cv_mat, axis=0)
    return cv_mean


# =====================================================================
# 单截面处理 (按概念分组, 全市场)
# =====================================================================

def process_one_section_concept(
    section_idx: int,
    cal: np.ndarray,
    log_price_wide: np.ndarray,
    ret_wide: np.ndarray,
    stock_codes: np.ndarray,                   # (S_all,) 全市场股票代码
    short_pool_set: set,
    theme_top5_arr: np.ndarray,                # (T_total, S_all, top_k) int32, -1 表示无
    train_window: int,
    valid_window: int = VALID_WINDOW,
    signal_type: str = "zscore",
    cv_folds: int = 0,
    cv_label_h: int = 5,
    dedup_max_per_stock: int = 1,              # 每只股票在贪心去重后最多保留的 pair 数 (1 = 旧行为)
    min_unique_per_side: int = 50,             # 净额化后多空各至少独立票数
    predict_window: int = 0,                   # >0: 信号日 mu/sigma 用前 N 个交易日的滚动 mean/std
    verbose: bool = False,
) -> Dict:
    """
    概念聚类版的单截面处理 (跨行业, 共享 top K 概念才配对)。

    主流程:
        1. 取当天每股的 top K 概念 (theme_top5_arr[t_idx, s_idx, :])
        2. 倒排索引: theme_id -> stocks 列表
        3. 每个 theme 内部枚举 pair, set 去重 (一对 pair 在多个 theme 内只算 1 次)
        4. 全市场一次性矩阵化算 OU + ADF + (CV)
        5. 全市场贪心去重 (按 cv_mean_ic 或 valid_rank_ic 降序)
        6. 合法 pair 过滤 + top 20% 排序

    返回 dict 与 process_one_section 同 schema, 便于 run_one_experiment 通用处理。
    """
    t0 = time.time()
    t_idx = section_idx
    sig_date = pd.Timestamp(cal[t_idx])

    need = train_window + valid_window
    if t_idx < need:
        return {"signal_date": sig_date, "skip": True,
                "reason": f"insufficient history ({t_idx} < {need})"}

    train_lo = t_idx - need
    train_hi = t_idx - valid_window
    valid_lo = t_idx - valid_window
    valid_hi = t_idx

    train_lp = log_price_wide[train_lo:train_hi, :]
    valid_lp = log_price_wide[valid_lo:valid_hi, :]
    today_lp = log_price_wide[t_idx, :]
    label_ret_slice = ret_wide[valid_lo + 1: valid_hi + 1, :]

    # 当天每股的 top K 概念 (顺位)
    today_top5 = theme_top5_arr[t_idx, :, :]   # (S_all, top_k), -1 表示空

    # 当天可用股票: 训练 + 验证 + 信号日 + label 都不缺失 + 至少有 1 个 theme
    train_ok = (~np.isnan(train_lp)).all(axis=0)
    valid_ok = (~np.isnan(valid_lp)).all(axis=0)
    today_ok = ~np.isnan(today_lp)
    label_ok = (~np.isnan(label_ret_slice)).all(axis=0)
    has_theme = (today_top5 >= 0).any(axis=1)
    valid_stock = train_ok & valid_ok & today_ok & label_ok & has_theme

    s_idx_v = np.where(valid_stock)[0]                # 全市场可用股票的全局索引
    n_v = len(s_idx_v)
    if n_v < 2:
        return {"signal_date": sig_date, "skip": True,
                "reason": f"too few valid stocks ({n_v})",
                "n_pairs_total": 0, "n_pairs_ou_pass": 0,
                "n_pairs_dedup": 0, "n_pairs_legal": 0, "n_pairs_top20": 0,
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(), "stock_pred": pd.DataFrame()}

    # 倒排索引: theme_id -> [stock 全局索引列表]
    # 仅遍历 valid_stock, 每股 top K 个 theme
    theme_to_stocks = {}
    for sg in s_idx_v:
        themes = today_top5[sg]
        for t in themes:
            if t < 0:
                continue
            theme_to_stocks.setdefault(int(t), []).append(int(sg))

    # 倒排枚举 pair (set 去重)
    pair_set = set()
    for stocks in theme_to_stocks.values():
        if len(stocks) < 2:
            continue
        # 同 theme 内 C(n, 2) 个 pair
        slist = sorted(stocks)
        for ii in range(len(slist)):
            si = slist[ii]
            for jj in range(ii + 1, len(slist)):
                sj = slist[jj]
                pair_set.add((si, sj))    # 全局 stock 索引 (i < j)
    if not pair_set:
        return {"signal_date": sig_date, "skip": True,
                "reason": "no shared-theme pair",
                "n_pairs_total": 0, "n_pairs_ou_pass": 0,
                "n_pairs_dedup": 0, "n_pairs_legal": 0, "n_pairs_top20": 0,
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(), "stock_pred": pd.DataFrame()}

    pair_arr = np.array(sorted(pair_set), dtype=np.int64)   # (n_pair, 2) 全局 stock 索引
    i_global = pair_arr[:, 0]
    j_global = pair_arr[:, 1]
    n_pair = len(i_global)
    n_total = n_pair

    # ---- 矩阵化算 OU + ADF ----
    # 用 train_lp/valid_lp 在全局索引切片
    train_v_i = train_lp[:, i_global]    # (W_train, n_pair)
    train_v_j = train_lp[:, j_global]
    X_train = train_v_i - train_v_j      # (W_train, n_pair)
    ou = estimate_ou_ar1(X_train)
    b = ou["b"]; kappa = ou["kappa"]; mu = ou["mu"]
    sigma = ou["sigma"]; half_life = ou["half_life"]
    mean_X = ou["mean_X"]; std_X = ou["std_X"]
    adf_p, _, _ = _batch_adf_pvalues(X_train)

    # OU 过滤 (zscore_nok 与 zscore 共用过滤条件, 仅信号公式去掉 kappa)
    if signal_type in ("kappa_dev", "zscore", "zscore_nok"):
        ou_mask = (
            (b > 0)
            & (half_life >= HALF_LIFE_MIN) & (half_life <= HALF_LIFE_MAX)
            & (adf_p < ADF_P_MAX)
            & ~np.isnan(mu) & ~np.isnan(sigma)
        )
    elif signal_type == "simple":
        ou_mask = (
            (b > 0)
            & (half_life >= HALF_LIFE_MIN) & (half_life <= HALF_LIFE_MAX)
            & (adf_p < ADF_P_MAX)
            & ~np.isnan(mean_X) & (std_X > _EPS)
        )
    elif signal_type == "simple_no_hl":
        ou_mask = (
            (b > 0) & (adf_p < ADF_P_MAX)
            & ~np.isnan(mean_X) & (std_X > _EPS)
        )
    else:
        raise ValueError(f"unknown signal_type: {signal_type}")
    n_pass = int(ou_mask.sum())
    if n_pass == 0:
        return {"signal_date": sig_date, "skip": True,
                "reason": "no OU-passing pair",
                "n_pairs_total": n_total, "n_pairs_ou_pass": 0,
                "n_pairs_dedup": 0, "n_pairs_legal": 0, "n_pairs_top20": 0,
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(), "stock_pred": pd.DataFrame()}
    n_ou_pass = n_pass

    ip = np.where(ou_mask)[0]
    b_p = b[ip]; kappa_p = kappa[ip]; mu_p = mu[ip]
    sigma_p = sigma[ip]; half_life_p = half_life[ip]; adf_p_p = adf_p[ip]
    mean_X_p = mean_X[ip]; std_X_p = std_X[ip]
    i_g = i_global[ip]
    j_g = j_global[ip]

    # 验证段 + signal
    valid_v_i = valid_lp[:, i_g]
    valid_v_j = valid_lp[:, j_g]
    X_valid = valid_v_i - valid_v_j
    if signal_type == "zscore":
        sigma_safe = np.clip(sigma_p, _EPS, None)
        signal_valid = kappa_p[None, :] * (mu_p[None, :] - X_valid) / sigma_safe[None, :]
    elif signal_type == "zscore_nok":
        sigma_safe = np.clip(sigma_p, _EPS, None)
        signal_valid = (mu_p[None, :] - X_valid) / sigma_safe[None, :]
    elif signal_type in ("simple", "simple_no_hl"):
        std_safe = np.clip(std_X_p, _EPS, None)
        signal_valid = (mean_X_p[None, :] - X_valid) / std_safe[None, :]
    else:
        signal_valid = kappa_p[None, :] * (mu_p[None, :] - X_valid)
    label_valid = label_ret_slice[:, i_g] - label_ret_slice[:, j_g]
    valid_ic = spearman_per_pair(signal_valid, label_valid)

    # CV (可选)
    if cv_folds > 0:
        X_train_ip = X_train[:, ip]
        cv_mean_ic = compute_cv_mean_ic(
            X_train_full=X_train_ip, n_pass=n_pass,
            k=cv_folds, signal_type=signal_type, label_h=cv_label_h,
        )
    else:
        cv_mean_ic = np.full(n_pass, np.nan)

    # 信号日 X_t
    today_v_i = today_lp[i_g]
    today_v_j = today_lp[j_g]
    X_today = today_v_i - today_v_j
    # predict_window > 0 时, 用信号日前 N 个交易日的滚动 mean/std 替代 OU 的 mu/sigma
    # 仅影响 signal_today, 验证段 signal_valid / valid_ic / CV 全部保留 OU 训练窗的 mu/sigma
    if predict_window > 0 and t_idx >= predict_window:
        pred_lp = log_price_wide[t_idx - predict_window: t_idx, :]    # (N, S_all)
        X_pred = pred_lp[:, i_g] - pred_lp[:, j_g]                     # (N, n_pass)
        with np.errstate(all="ignore"):
            mu_use = np.nanmean(X_pred, axis=0)
            sigma_use = np.nanstd(X_pred, axis=0, ddof=1)
        mean_use = mu_use
        std_use = sigma_use
    else:
        mu_use = mu_p
        sigma_use = sigma_p
        mean_use = mean_X_p
        std_use = std_X_p

    if signal_type == "zscore":
        sigma_safe = np.clip(sigma_use, _EPS, None)
        signal_today = kappa_p * (mu_use - X_today) / sigma_safe
    elif signal_type == "zscore_nok":
        sigma_safe = np.clip(sigma_use, _EPS, None)
        signal_today = (mu_use - X_today) / sigma_safe
    elif signal_type in ("simple", "simple_no_hl"):
        std_safe = np.clip(std_use, _EPS, None)
        signal_today = (mean_use - X_today) / std_safe
    else:
        signal_today = kappa_p * (mu_use - X_today)

    # 转 stock_code (而不是全局索引)
    i_codes = stock_codes[i_g]
    j_codes = stock_codes[j_g]
    i_short = np.array([c in short_pool_set for c in i_codes])
    j_short = np.array([c in short_pool_set for c in j_codes])

    df_pairs = pd.DataFrame({
        "industry": "_concept_",                    # 占位 (兼容下游 schema)
        "stock_i": i_codes,
        "stock_j": j_codes,
        "is_i_short_pool": i_short,
        "is_j_short_pool": j_short,
        "b": b_p,
        "kappa": kappa_p,
        "mu": mu_p,
        "sigma": sigma_p,
        "half_life": half_life_p,
        "mean_X": mean_X_p,
        "std_X": std_X_p,
        "ADF_p": adf_p_p,
        "valid_rank_ic": valid_ic,
        "cv_mean_ic": cv_mean_ic,
        "signal": signal_today,
    })
    dedup_col = "cv_mean_ic" if cv_folds > 0 else "valid_rank_ic"
    df_pairs = df_pairs[~df_pairs[dedup_col].isna()]
    if df_pairs.empty:
        return {"signal_date": sig_date, "skip": True,
                "reason": "all NaN dedup_col",
                "n_pairs_total": n_total, "n_pairs_ou_pass": n_ou_pass,
                "n_pairs_dedup": 0, "n_pairs_legal": 0, "n_pairs_top20": 0,
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(), "stock_pred": pd.DataFrame()}

    # ---- 全市场贪心去重 (按 dedup_col 降序, 每只票最多 dedup_max_per_stock 个 pair) ----
    df_pairs = _greedy_dedup_with_cap(df_pairs, dedup_col, dedup_max_per_stock)
    if df_pairs.empty:
        return {"signal_date": sig_date, "skip": True,
                "reason": "no pair after dedup",
                "n_pairs_total": n_total, "n_pairs_ou_pass": n_ou_pass,
                "n_pairs_dedup": 0, "n_pairs_legal": 0, "n_pairs_top20": 0,
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(), "stock_pred": pd.DataFrame()}
    pair_dedup = df_pairs
    pair_dedup["date"] = sig_date

    # ---- 合法 pair 过滤 + top 20% (与 baseline 同逻辑) ----
    sig = pair_dedup["signal"].to_numpy()
    short_side_in_pool = np.where(
        sig > 0,
        pair_dedup["is_j_short_pool"].to_numpy(),
        np.where(sig < 0, pair_dedup["is_i_short_pool"].to_numpy(), False),
    )
    pair_dedup["is_legal"] = short_side_in_pool & (sig != 0)
    legal_df = pair_dedup[pair_dedup["is_legal"]].copy()
    n_legal = len(legal_df)

    legal_df["abs_signal"] = legal_df["signal"].abs()
    legal_df = legal_df.sort_values("abs_signal", ascending=False).reset_index(drop=True)
    n_top_pct = int(math.ceil(n_legal * TOP_PCT))
    n_top = min(n_legal, max(TOP_PCT_MIN_N, n_top_pct)) if n_legal > 0 else 0

    # 动态扩 top 兜底: 若净额化后 long/short 独立票数 < min_unique_per_side, 继续加 pair
    while n_top < n_legal:
        cur_top = legal_df.iloc[:n_top]
        n_long_uniq, n_short_uniq = _count_unique_after_netting(cur_top)
        if n_long_uniq >= min_unique_per_side and n_short_uniq >= min_unique_per_side:
            break
        n_top += 1

    legal_df["is_top20"] = False
    if n_top > 0:
        legal_df.loc[:n_top - 1, "is_top20"] = True

    top_df = legal_df[legal_df["is_top20"]].copy()
    n_top_actual = len(top_df)

    if dedup_max_per_stock <= 1:
        # 旧路径 (K=1)
        rows = []
        for r in top_df.itertuples(index=False):
            s = r.signal
            if s > 0:
                long_code, short_code = r.stock_i, r.stock_j
            else:
                long_code, short_code = r.stock_j, r.stock_i
            rows.append({"date": sig_date, "stock_code": long_code,
                         "industry": r.industry, "stock_pred": abs(s) / 2.0,
                         "is_short_pool": (long_code in short_pool_set),
                         "paired_stock": short_code, "side": "long",
                         "pair_valid_rank_ic": r.valid_rank_ic, "pair_signal": s})
            rows.append({"date": sig_date, "stock_code": short_code,
                         "industry": r.industry, "stock_pred": -abs(s) / 2.0,
                         "is_short_pool": (short_code in short_pool_set),
                         "paired_stock": long_code, "side": "short",
                         "pair_valid_rank_ic": r.valid_rank_ic, "pair_signal": s})
        stock_pred_df = pd.DataFrame(rows)
    else:
        # D5 路径 (K>1)
        long_w, short_w = _net_pair_to_holdings(top_df)
        rows = []
        for code, w in long_w.items():
            rows.append({"date": sig_date, "stock_code": code,
                         "weight": w, "side": "long",
                         "is_short_pool": (code in short_pool_set)})
        for code, w in short_w.items():
            rows.append({"date": sig_date, "stock_code": code,
                         "weight": w, "side": "short",
                         "is_short_pool": (code in short_pool_set)})
        stock_pred_df = pd.DataFrame(rows)

    n_signal_stocks = len(stock_pred_df)
    n_long = int((stock_pred_df["side"] == "long").sum())
    n_short = int((stock_pred_df["side"] == "short").sum())

    pair_log_out = pair_dedup.copy()
    pair_log_out["abs_signal"] = pair_log_out["signal"].abs()
    pair_log_out["is_top20"] = False
    if n_top > 0:
        top_keys = set(zip(top_df["stock_i"].tolist(), top_df["stock_j"].tolist()))
        pair_log_out["is_top20"] = [
            (i, j) in top_keys
            for i, j in zip(pair_log_out["stock_i"], pair_log_out["stock_j"])
        ]

    elapsed = time.time() - t0
    if verbose:
        print(f"  [{sig_date.date()}] 候选 {n_total:>7d}, OU 通过 {n_ou_pass:>6d}, "
              f"去重后 {len(pair_dedup):>5d}, 合法 {n_legal:>4d}, top20 {n_top_actual:>4d}, "
              f"多头 {n_long:>4d} 空头 {n_short:>4d}, {elapsed:.1f}s")

    return {
        "signal_date": sig_date,
        "skip": False,
        "n_pairs_total": n_total,
        "n_pairs_ou_pass": n_ou_pass,
        "n_pairs_dedup": len(pair_dedup),
        "n_pairs_legal": n_legal,
        "n_pairs_top20": n_top_actual,
        "n_signal_stocks": n_signal_stocks,
        "n_long": n_long,
        "n_short": n_short,
        "pair_log": pair_log_out,
        "stock_pred": stock_pred_df,
        "elapsed_sec": elapsed,
    }


# =====================================================================
# 多 pair 复用辅助函数 (D5 实验)
# =====================================================================

def _greedy_dedup_with_cap(df_ind: pd.DataFrame, dedup_col, cap: int) -> pd.DataFrame:
    """贪心去重: 按 dedup_col 降序, 每只股票最多入选 cap 个 pair.

    dedup_col 可为单列名(str)或多列名列表(list[str], 主键在前, 用于 tie-break)。
    cap=1 时与原 set 去重逻辑数学等价 (used_count[s]>=1 ⇔ s in used).
    """
    df_sorted = df_ind.sort_values(dedup_col, ascending=False).reset_index(drop=True)
    used_count: Dict[str, int] = {}
    keep_rows = []
    for r in df_sorted.itertuples(index=False):
        if used_count.get(r.stock_i, 0) >= cap or used_count.get(r.stock_j, 0) >= cap:
            continue
        keep_rows.append(r)
        used_count[r.stock_i] = used_count.get(r.stock_i, 0) + 1
        used_count[r.stock_j] = used_count.get(r.stock_j, 0) + 1
    if not keep_rows:
        return df_sorted.iloc[0:0]
    return pd.DataFrame(keep_rows)


def _count_unique_after_netting(top_df: pd.DataFrame) -> Tuple[int, int]:
    """模拟净额化, 返回 (n_long_uniq, n_short_uniq).
    净 long_count[s] = #(s 在 long 侧) - #(s 在 short 侧); >0 即净多, <0 即净空.
    """
    long_cnt: Dict[str, int] = {}
    short_cnt: Dict[str, int] = {}
    for r in top_df.itertuples(index=False):
        s = r.signal
        if s > 0:
            long_cnt[r.stock_i] = long_cnt.get(r.stock_i, 0) + 1
            short_cnt[r.stock_j] = short_cnt.get(r.stock_j, 0) + 1
        elif s < 0:
            long_cnt[r.stock_j] = long_cnt.get(r.stock_j, 0) + 1
            short_cnt[r.stock_i] = short_cnt.get(r.stock_i, 0) + 1
    all_codes = set(long_cnt) | set(short_cnt)
    n_long = sum(1 for c in all_codes if long_cnt.get(c, 0) - short_cnt.get(c, 0) > 0)
    n_short = sum(1 for c in all_codes if short_cnt.get(c, 0) - long_cnt.get(c, 0) > 0)
    return n_long, n_short


def _net_pair_to_holdings(top_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """对 top_df (每行 1 个 pair) 做净额化, 返回组内归一化权重 dict.
        long_w  : {stock_code -> weight, ∑w=1}
        short_w : {stock_code -> weight, ∑w=1}
    若净额化后某侧总数=0, 该侧返回空 dict.
    """
    long_cnt: Dict[str, int] = {}
    short_cnt: Dict[str, int] = {}
    for r in top_df.itertuples(index=False):
        s = r.signal
        if s > 0:
            long_cnt[r.stock_i] = long_cnt.get(r.stock_i, 0) + 1
            short_cnt[r.stock_j] = short_cnt.get(r.stock_j, 0) + 1
        elif s < 0:
            long_cnt[r.stock_j] = long_cnt.get(r.stock_j, 0) + 1
            short_cnt[r.stock_i] = short_cnt.get(r.stock_i, 0) + 1
    all_codes = set(long_cnt) | set(short_cnt)
    raw_long: Dict[str, int] = {}
    raw_short: Dict[str, int] = {}
    for c in all_codes:
        net = long_cnt.get(c, 0) - short_cnt.get(c, 0)
        if net > 0:
            raw_long[c] = net
        elif net < 0:
            raw_short[c] = -net
    sl = sum(raw_long.values())
    ss = sum(raw_short.values())
    long_w = {c: v / sl for c, v in raw_long.items()} if sl > 0 else {}
    short_w = {c: v / ss for c, v in raw_short.items()} if ss > 0 else {}
    return long_w, short_w


# =====================================================================
# GLB-semi 实验辅助函数 (全市场枚举 + 半年更新 pair_set)
# =====================================================================

def _get_semi_annual_anchors(
    backtest_dates: List[pd.Timestamp],
    cal: np.ndarray,
) -> List[Tuple[pd.Timestamp, int]]:
    """返回 [(T0_date, T0_cal_idx), ...] 每个 T0 是该半年区间内 backtest_dates 的首个交易日.
    半年定义: 1 月~6 月 = 上半年, 7 月~12 月 = 下半年.
    """
    cal_ts = pd.DatetimeIndex(cal)
    date_to_idx = {dt: i for i, dt in enumerate(cal_ts)}
    anchors: List[Tuple[pd.Timestamp, int]] = []
    seen_periods: set = set()
    for d in backtest_dates:
        period = (d.year, 1 if d.month <= 6 else 2)
        if period in seen_periods:
            continue
        seen_periods.add(period)
        if d not in date_to_idx:
            continue
        anchors.append((d, date_to_idx[d]))
    return anchors


def select_global_pair_set(
    t_idx_T0: int,
    cal: np.ndarray,
    log_price_wide: np.ndarray,
    ret_wide: np.ndarray,
    stock_codes: np.ndarray,
    train_window: int = 252,
    valid_window: int = VALID_WINDOW,
    signal_type: str = "zscore",
    cv_folds: int = 5,
    cv_label_h: int = 5,
    chunk_size: int = 1_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """在 T0 起点全市场枚举 pair, 分块矩阵化 OU+ADF+CV5+全市场去重, 返回稳定 pair_set.

    返回 DataFrame 字段:
        stock_i, stock_j, i_global, j_global,
        b, kappa, mu, sigma, half_life, mean_X, std_X,
        valid_rank_ic, cv_mean_ic
    """
    t0 = time.time()
    T0_date = pd.Timestamp(cal[t_idx_T0])

    need = train_window + valid_window
    if t_idx_T0 < need:
        raise ValueError(f"insufficient history at T0={T0_date.date()} ({t_idx_T0} < {need})")

    train_lo = t_idx_T0 - need
    train_hi = t_idx_T0 - valid_window
    valid_lo = t_idx_T0 - valid_window
    valid_hi = t_idx_T0

    train_lp = log_price_wide[train_lo:train_hi, :]            # (W_train, S)
    valid_lp = log_price_wide[valid_lo:valid_hi, :]            # (W_valid, S)
    label_ret_slice = ret_wide[valid_lo + 1: valid_hi + 1, :]  # (W_valid, S)

    S = log_price_wide.shape[1]
    train_ok = (~np.isnan(train_lp)).all(axis=0)
    valid_ok = (~np.isnan(valid_lp)).all(axis=0)
    label_ok = (~np.isnan(label_ret_slice)).all(axis=0)
    valid_stock = train_ok & valid_ok & label_ok
    s_idx_v = np.where(valid_stock)[0]
    n_v = len(s_idx_v)
    if verbose:
        print(f"  [select_global_pair_set T0={T0_date.date()}] 全市场可用股票 {n_v}/{S}")
    if n_v < 2:
        return pd.DataFrame()

    # 全市场枚举 (i < j) 在可用股票子集上, 局部索引 + 全局索引并行维护
    i_local, j_local = np.triu_indices(n_v, k=1)
    n_pair_total = len(i_local)
    if verbose:
        print(f"  [select_global_pair_set] 全市场 pair 总数 {n_pair_total:,}, 分块 {chunk_size:,}/块")

    i_global_arr = s_idx_v[i_local]
    j_global_arr = s_idx_v[j_local]

    # 切到 valid_stock 子集 (节省内存; float32)
    train_lp_v = train_lp[:, s_idx_v].astype(np.float32, copy=False)
    valid_lp_v = valid_lp[:, s_idx_v].astype(np.float32, copy=False)
    label_ret_v = label_ret_slice[:, s_idx_v].astype(np.float32, copy=False)

    # 分块过滤
    pass_chunks: List[Dict] = []
    n_ou_pass = 0
    n_chunks = (n_pair_total + chunk_size - 1) // chunk_size
    for ci, chunk_start in enumerate(range(0, n_pair_total, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, n_pair_total)
        i_c = i_local[chunk_start:chunk_end]
        j_c = j_local[chunk_start:chunk_end]

        # estimate_ou_ar1 / _batch_adf_pvalues 内部使用 float64, 这里转一次
        X_train_c = (train_lp_v[:, i_c] - train_lp_v[:, j_c]).astype(np.float64, copy=False)
        ou = estimate_ou_ar1(X_train_c)
        b_arr = ou["b"]; kappa_arr = ou["kappa"]; mu_arr = ou["mu"]
        sigma_arr = ou["sigma"]; half_life_arr = ou["half_life"]
        mean_X_arr = ou["mean_X"]; std_X_arr = ou["std_X"]
        adf_p_arr, _, _ = _batch_adf_pvalues(X_train_c)

        # 硬过滤 (zscore / kappa_dev / zscore_nok 共用条件)
        if signal_type in ("kappa_dev", "zscore", "zscore_nok"):
            ou_mask = (
                (b_arr > 0)
                & (half_life_arr >= HALF_LIFE_MIN) & (half_life_arr <= HALF_LIFE_MAX)
                & (adf_p_arr < ADF_P_MAX)
                & ~np.isnan(mu_arr) & ~np.isnan(sigma_arr)
            )
        elif signal_type == "simple":
            ou_mask = (
                (b_arr > 0)
                & (half_life_arr >= HALF_LIFE_MIN) & (half_life_arr <= HALF_LIFE_MAX)
                & (adf_p_arr < ADF_P_MAX)
                & ~np.isnan(mean_X_arr) & (std_X_arr > _EPS)
            )
        elif signal_type == "simple_no_hl":
            ou_mask = (
                (b_arr > 0) & (adf_p_arr < ADF_P_MAX)
                & ~np.isnan(mean_X_arr) & (std_X_arr > _EPS)
            )
        else:
            raise ValueError(f"unknown signal_type: {signal_type}")

        n_pass_c = int(ou_mask.sum())
        n_ou_pass += n_pass_c
        if verbose:
            print(f"    chunk {ci+1:>2d}/{n_chunks}: {chunk_end-chunk_start:>9,} pair → "
                  f"OU 通过 {n_pass_c:>7,}")
        if n_pass_c == 0:
            del X_train_c
            continue

        ip = np.where(ou_mask)[0]
        pass_chunks.append({
            "i_local": i_c[ip],
            "j_local": j_c[ip],
            "i_global": i_global_arr[chunk_start:chunk_end][ip],
            "j_global": j_global_arr[chunk_start:chunk_end][ip],
            "b": b_arr[ip].astype(np.float32),
            "kappa": kappa_arr[ip].astype(np.float32),
            "mu": mu_arr[ip].astype(np.float32),
            "sigma": sigma_arr[ip].astype(np.float32),
            "half_life": half_life_arr[ip].astype(np.float32),
            "mean_X": mean_X_arr[ip].astype(np.float32),
            "std_X": std_X_arr[ip].astype(np.float32),
            "X_train_passed": X_train_c[:, ip].astype(np.float32),
        })
        del X_train_c
        gc.collect()

    if not pass_chunks:
        if verbose:
            print(f"  [select_global_pair_set] 无 OU 通过 pair, 返回空")
        return pd.DataFrame()

    # 拼接所有通过 pair
    i_local_all = np.concatenate([c["i_local"] for c in pass_chunks])
    j_local_all = np.concatenate([c["j_local"] for c in pass_chunks])
    i_global_all = np.concatenate([c["i_global"] for c in pass_chunks])
    j_global_all = np.concatenate([c["j_global"] for c in pass_chunks])
    b_all = np.concatenate([c["b"] for c in pass_chunks])
    kappa_all = np.concatenate([c["kappa"] for c in pass_chunks])
    mu_all = np.concatenate([c["mu"] for c in pass_chunks])
    sigma_all = np.concatenate([c["sigma"] for c in pass_chunks])
    hl_all = np.concatenate([c["half_life"] for c in pass_chunks])
    mean_X_all = np.concatenate([c["mean_X"] for c in pass_chunks])
    std_X_all = np.concatenate([c["std_X"] for c in pass_chunks])
    X_train_pass = np.concatenate([c["X_train_passed"] for c in pass_chunks], axis=1)
    n_pass = len(i_local_all)
    del pass_chunks
    gc.collect()
    if verbose:
        print(f"  [select_global_pair_set] OU 通过合计 {n_pass:,}, 算 valid_rank_ic + CV5...")

    # valid_rank_ic (一次性, 内存可控)
    X_valid = (valid_lp_v[:, i_local_all] - valid_lp_v[:, j_local_all]).astype(np.float64)
    sigma_safe = np.clip(sigma_all.astype(np.float64), _EPS, None)
    if signal_type == "zscore":
        signal_valid = kappa_all.astype(np.float64)[None, :] * (mu_all.astype(np.float64)[None, :] - X_valid) / sigma_safe[None, :]
    elif signal_type == "zscore_nok":
        signal_valid = (mu_all.astype(np.float64)[None, :] - X_valid) / sigma_safe[None, :]
    elif signal_type in ("simple", "simple_no_hl"):
        std_safe = np.clip(std_X_all.astype(np.float64), _EPS, None)
        signal_valid = (mean_X_all.astype(np.float64)[None, :] - X_valid) / std_safe[None, :]
    else:
        signal_valid = kappa_all.astype(np.float64)[None, :] * (mu_all.astype(np.float64)[None, :] - X_valid)
    label_valid = (label_ret_v[:, i_local_all] - label_ret_v[:, j_local_all]).astype(np.float64)
    valid_rank_ic = spearman_per_pair(signal_valid, label_valid).astype(np.float32)
    del X_valid, signal_valid, label_valid
    gc.collect()

    # cv_mean_ic (分批, 避免 5 折 X_tr 内存爆)
    if cv_folds > 0:
        cv_chunk = 500_000
        cv_parts = []
        for cs in range(0, n_pass, cv_chunk):
            ce = min(cs + cv_chunk, n_pass)
            cv_part = compute_cv_mean_ic(
                X_train_full=X_train_pass[:, cs:ce].astype(np.float64),
                n_pass=ce - cs,
                k=cv_folds,
                signal_type=signal_type,
                label_h=cv_label_h,
            )
            cv_parts.append(cv_part.astype(np.float32))
        cv_mean_ic = np.concatenate(cv_parts)
    else:
        cv_mean_ic = np.full(n_pass, np.nan, dtype=np.float32)
    del X_train_pass
    gc.collect()

    i_codes_all = stock_codes[i_global_all]
    j_codes_all = stock_codes[j_global_all]

    df = pd.DataFrame({
        "stock_i": i_codes_all,
        "stock_j": j_codes_all,
        "i_global": i_global_all.astype(np.int32),
        "j_global": j_global_all.astype(np.int32),
        "b": b_all,
        "kappa": kappa_all,
        "mu": mu_all,
        "sigma": sigma_all,
        "half_life": hl_all,
        "mean_X": mean_X_all,
        "std_X": std_X_all,
        "valid_rank_ic": valid_rank_ic,
        "cv_mean_ic": cv_mean_ic,
    })

    # 全市场贪心去重 (cap=1, 按 cv_mean_ic 降序; cv_folds=0 时退化用 valid_rank_ic)
    dedup_col = "cv_mean_ic" if cv_folds > 0 else "valid_rank_ic"
    df = df[~df[dedup_col].isna()]
    df_dedup = _greedy_dedup_with_cap(df, dedup_col, cap=1).reset_index(drop=True)

    if verbose:
        print(f"  [select_global_pair_set] 全市场去重后 {len(df_dedup):,} pair, "
              f"总耗时 {time.time()-t0:.1f}s")
    return df_dedup


def process_one_section_global_semi(
    section_idx: int,
    cal: np.ndarray,
    log_price_wide: np.ndarray,
    stock_codes: np.ndarray,
    short_pool_set: set,
    P_active: pd.DataFrame,
    signal_type: str = "zscore",
    verbose: bool = False,
) -> Dict:
    """在半年期内, 对每天 t 用冻结的 P_active 计算信号 + short_pool 过滤 + top 20%.

    返回 dict 与 process_one_section / process_one_section_concept 同 schema (K=1 路径).
    """
    t0 = time.time()
    t_idx = section_idx
    sig_date = pd.Timestamp(cal[t_idx])

    if P_active.empty:
        return {"signal_date": sig_date, "skip": True,
                "reason": "empty pair_set",
                "n_pairs_total": 0, "n_pairs_ou_pass": 0,
                "n_pairs_dedup": 0, "n_pairs_legal": 0, "n_pairs_top20": 0,
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(), "stock_pred": pd.DataFrame()}

    i_global = P_active["i_global"].to_numpy()
    j_global = P_active["j_global"].to_numpy()
    kappa_p = P_active["kappa"].to_numpy().astype(np.float64)
    mu_p = P_active["mu"].to_numpy().astype(np.float64)
    sigma_p = P_active["sigma"].to_numpy().astype(np.float64)
    mean_X_p = P_active["mean_X"].to_numpy().astype(np.float64)
    std_X_p = P_active["std_X"].to_numpy().astype(np.float64)
    valid_rank_ic_p = P_active["valid_rank_ic"].to_numpy()
    cv_mean_ic_p = P_active["cv_mean_ic"].to_numpy()
    n_pair = len(P_active)

    today_lp = log_price_wide[t_idx, :]
    X_today = today_lp[i_global].astype(np.float64) - today_lp[j_global].astype(np.float64)
    valid_today = ~np.isnan(X_today)

    if signal_type == "zscore":
        sigma_safe = np.clip(sigma_p, _EPS, None)
        signal_today = kappa_p * (mu_p - X_today) / sigma_safe
    elif signal_type == "zscore_nok":
        sigma_safe = np.clip(sigma_p, _EPS, None)
        signal_today = (mu_p - X_today) / sigma_safe
    elif signal_type in ("simple", "simple_no_hl"):
        std_safe = np.clip(std_X_p, _EPS, None)
        signal_today = (mean_X_p - X_today) / std_safe
    else:
        signal_today = kappa_p * (mu_p - X_today)
    signal_today = np.where(valid_today, signal_today, np.nan)

    i_codes = stock_codes[i_global]
    j_codes = stock_codes[j_global]
    i_short = np.array([c in short_pool_set for c in i_codes])
    j_short = np.array([c in short_pool_set for c in j_codes])

    short_side_in_pool = np.where(
        signal_today > 0,
        j_short,
        np.where(signal_today < 0, i_short, False),
    )
    is_legal = short_side_in_pool & ~np.isnan(signal_today) & (signal_today != 0)

    pair_dedup = pd.DataFrame({
        "industry": "_global_",
        "stock_i": i_codes,
        "stock_j": j_codes,
        "is_i_short_pool": i_short,
        "is_j_short_pool": j_short,
        "b": P_active["b"].to_numpy(),
        "kappa": kappa_p,
        "mu": mu_p,
        "sigma": sigma_p,
        "half_life": P_active["half_life"].to_numpy(),
        "mean_X": mean_X_p,
        "std_X": std_X_p,
        "valid_rank_ic": valid_rank_ic_p,
        "cv_mean_ic": cv_mean_ic_p,
        "signal": signal_today,
        "is_legal": is_legal,
        "date": sig_date,
    })
    legal_df = pair_dedup[pair_dedup["is_legal"]].copy()
    n_legal = len(legal_df)

    legal_df["abs_signal"] = legal_df["signal"].abs()
    legal_df = legal_df.sort_values("abs_signal", ascending=False).reset_index(drop=True)
    n_top_pct = int(math.ceil(n_legal * TOP_PCT))
    n_top = min(n_legal, max(TOP_PCT_MIN_N, n_top_pct)) if n_legal > 0 else 0

    legal_df["is_top20"] = False
    if n_top > 0:
        legal_df.loc[:n_top - 1, "is_top20"] = True

    top_df = legal_df[legal_df["is_top20"]].copy()
    n_top_actual = len(top_df)

    # K=1 schema (与 process_one_section 行业版完全一致)
    rows = []
    for r in top_df.itertuples(index=False):
        s = r.signal
        if s > 0:
            long_code, short_code = r.stock_i, r.stock_j
        else:
            long_code, short_code = r.stock_j, r.stock_i
        rows.append({"date": sig_date, "stock_code": long_code,
                     "industry": r.industry, "stock_pred": abs(s) / 2.0,
                     "is_short_pool": (long_code in short_pool_set),
                     "paired_stock": short_code, "side": "long",
                     "pair_valid_rank_ic": r.valid_rank_ic, "pair_signal": s})
        rows.append({"date": sig_date, "stock_code": short_code,
                     "industry": r.industry, "stock_pred": -abs(s) / 2.0,
                     "is_short_pool": (short_code in short_pool_set),
                     "paired_stock": long_code, "side": "short",
                     "pair_valid_rank_ic": r.valid_rank_ic, "pair_signal": s})
    stock_pred_df = pd.DataFrame(rows)

    n_signal_stocks = len(stock_pred_df)
    if not stock_pred_df.empty:
        n_long = int((stock_pred_df["side"] == "long").sum())
        n_short = int((stock_pred_df["side"] == "short").sum())
    else:
        n_long = n_short = 0

    pair_log_out = pair_dedup.copy()
    pair_log_out["abs_signal"] = pair_log_out["signal"].abs()
    pair_log_out["is_top20"] = False
    if n_top > 0:
        top_keys = set(zip(top_df["stock_i"].tolist(), top_df["stock_j"].tolist()))
        pair_log_out["is_top20"] = [
            (i, j) in top_keys
            for i, j in zip(pair_log_out["stock_i"], pair_log_out["stock_j"])
        ]

    elapsed = time.time() - t0
    if verbose:
        print(f"  [{sig_date.date()}] P {n_pair:,}, 合法 {n_legal:>5d}, top20 {n_top_actual:>4d}, "
              f"多头 {n_long:>4d} 空头 {n_short:>4d}, {elapsed:.2f}s")

    return {
        "signal_date": sig_date,
        "skip": False,
        "n_pairs_total": n_pair,
        "n_pairs_ou_pass": n_pair,
        "n_pairs_dedup": len(pair_dedup),
        "n_pairs_legal": n_legal,
        "n_pairs_top20": n_top_actual,
        "n_signal_stocks": n_signal_stocks,
        "n_long": n_long,
        "n_short": n_short,
        "pair_log": pair_log_out,
        "stock_pred": stock_pred_df,
        "elapsed_sec": elapsed,
    }


# =====================================================================
# 单截面处理
# =====================================================================

def process_one_section(
    section_idx: int,
    cal: np.ndarray,                       # 全局交易日序列 (np.datetime64)
    log_price_wide: np.ndarray,            # (T_total, S_all) log close (np.float32)
    ret_wide: np.ndarray,                  # (T_total, S_all) close-to-close 日收益 (用于 valid label)
    industry_codes: np.ndarray,            # (S_all,) 每只股票的 sw_l1 (object)
    stock_codes: np.ndarray,               # (S_all,) 每只股票代码 (object)
    short_pool_set: set,                   # 当日空头池 (set of stock_code)
    train_window: int,
    valid_window: int = VALID_WINDOW,
    signal_type: str = "kappa_dev",        # baseline: kappa*(mu-X); zscore: kappa*(mu-X)/sigma_OU
    cv_folds: int = 0,                     # >0: 训练窗内做 K 折 CV, 用 cv_mean_ic 替代 valid_rank_ic
    cv_label_h: int = 5,                   # CV 每折验证段的 label 周期 (5 = 5 日调仓)
    corr_filter_method: str = "none",      # 行业内 pair 相似度预筛: none / pearson / spearman / ssd
    corr_filter_pct: float = 1.0,          # 行业内保留 top X% 高相似 pair (1.0=不筛, 0.5=top 50%)
    nzc_filter_pct: float = 1.0,           # 行业内 OU 后 NZC top X% 保留 (1.0=不筛, 0.5=top 50%)
    dedup_max_per_stock: int = 1,          # 每只股票在贪心去重后最多保留的 pair 数 (1 = 旧行为)
    dedup_rank: str = "cv_mean_ic",        # 行业内贪心去重排序键: cv_mean_ic(默认)/nzc(训练窗零穿越次数降序, cv_mean_ic 次级 tie-break)
    min_unique_per_side: int = 50,         # 净额化后多空各至少独立票数 (>1 时动态扩 top 兜底)
    predict_window: int = 0,               # >0: 信号日 mu/sigma 用前 N 个交易日的滚动 mean/std (仅影响 signal_today)
    predict_window_mu_only: bool = False,  # True: predict_window 仅替换 mu(均值), sigma/std 仍用 OU 训练窗值
    barra_expo: np.ndarray = None,         # (T_total, S_all, K) 风格暴露; 配 dedup_rank="barra_dist" 用
    barra_cov: np.ndarray = None,          # (T_total, K, K) 因子协方差; barra_metric="mahal" 时用
    barra_factor_idx: np.ndarray = None,   # 选用的风格因子在 K 维中的列索引 (None=全部)
    barra_metric: str = "euclid",          # 距离度量: euclid(L2) / mahal(协方差加权 ΔxᵀΣΔx)
    verbose: bool = False,
    mv_wide: np.ndarray = None,            # (T_total, S_all) log market value; 配 mv_gap_max 用于配对时市值差距过滤
    mv_gap_max: float = None,              # 市值差距硬过滤上限: 仅保留 |log mv_i - log mv_j| < mv_gap_max 的 pair (None=不过滤)
) -> Dict:
    """
    处理一个截面日 (section_idx 是 cal 数组中的索引,代表当前截面 t)。

    signal_type:
        kappa_dev: signal = kappa * (mu - X_t)               (baseline)
        zscore:    signal = kappa * (mu - X_t) / sigma_OU    (按稳态 sigma 归一化)

    返回 dict:
        signal_date: pd.Timestamp
        pair_log:    DataFrame 每行一个合法 pair (包含 is_top20)
        stock_pred:  DataFrame 每行一只股票 (仅 top20 内)
        n_pairs_total / n_pairs_ou_pass / n_pairs_legal / n_pairs_top20
        n_signal_stocks / n_long / n_short
        elapsed_sec
    """
    if signal_type not in ("kappa_dev", "zscore", "zscore_nok", "simple", "simple_no_hl",
                           "abs_dev", "abs_dev_sqrt_s20", "abs_dev_s20",
                           "abs_dev_sqrt_s30", "abs_dev_s30"):
        raise ValueError(f"unknown signal_type: {signal_type}")
    if corr_filter_method not in ("none", "pearson", "spearman", "ssd"):
        raise ValueError(f"unknown corr_filter_method: {corr_filter_method}")
    if not (0.0 < nzc_filter_pct <= 1.0):
        raise ValueError(f"nzc_filter_pct must be in (0, 1], got {nzc_filter_pct}")
    if dedup_rank not in ("cv_mean_ic", "nzc", "barra_dist"):
        raise ValueError(f"unknown dedup_rank: {dedup_rank}")
    t0 = time.time()
    t_idx = section_idx
    sig_date = pd.Timestamp(cal[t_idx])

    # 时间窗口范围: 训练 [t-W_train-W_valid, t-W_valid-1], 验证 [t-W_valid, t-1]
    # 信号日: t (即 cal[t_idx])
    need = train_window + valid_window
    if t_idx < need:
        return {"signal_date": sig_date, "skip": True,
                "reason": f"insufficient history ({t_idx} < {need})"}

    train_lo = t_idx - need
    train_hi = t_idx - valid_window      # exclusive
    valid_lo = t_idx - valid_window
    valid_hi = t_idx                     # exclusive (信号日不算验证段)
    # 信号是 valid_hi=t 当日的 X_t = log_p[t] - log_p[t]; 不参与 IC 计算

    # 取窗口内 log price 切片
    train_lp = log_price_wide[train_lo:train_hi, :]   # (W_train, S_all)
    valid_lp = log_price_wide[valid_lo:valid_hi, :]   # (W_valid, S_all)
    # 信号 X_t 的标量价格
    today_lp = log_price_wide[t_idx, :]               # (S_all,)

    # 验证段标签: r_i_{t+1} - r_j_{t+1}
    # ret_wide[t] = (close_t - close_{t-1})/close_{t-1};
    # 为了让 signal_{t'} (t' ∈ valid window) 对齐 label_{t'+1},
    # label 取 ret_wide[valid_lo+1 : valid_hi+1] 的两只股票之差。
    label_ret_slice = ret_wide[valid_lo + 1: valid_hi + 1, :]  # (W_valid, S_all)

    # 按行业分组
    pair_records = []
    n_total = 0
    n_ou_pass = 0
    industries = pd.Series(industry_codes).unique()

    for ind in industries:
        if not ind or pd.isna(ind):
            continue
        ind_mask = (industry_codes == ind)
        s_idx = np.where(ind_mask)[0]
        n_s = len(s_idx)
        if n_s < 2:
            continue

        # 行业内: 要求训练 + 验证 + 信号日的所有日期都不缺失
        train_sub = train_lp[:, s_idx]      # (W_train, n_s)
        valid_sub = valid_lp[:, s_idx]      # (W_valid, n_s)
        today_sub = today_lp[s_idx]         # (n_s,)
        label_sub = label_ret_slice[:, s_idx]  # (W_valid, n_s)

        # 各只股票"全窗口无 NaN"的 mask
        train_ok = (~np.isnan(train_sub)).all(axis=0)         # (n_s,)
        valid_ok = (~np.isnan(valid_sub)).all(axis=0)
        today_ok = ~np.isnan(today_sub)
        label_ok = (~np.isnan(label_sub)).all(axis=0)
        ok = train_ok & valid_ok & today_ok & label_ok
        if ok.sum() < 2:
            continue
        keep = np.where(ok)[0]
        s_idx_v = s_idx[keep]
        train_v = train_sub[:, keep]
        valid_v = valid_sub[:, keep]
        today_v = today_sub[keep]
        label_v = label_sub[:, keep]
        n_v = len(keep)

        # 行业内枚举 pair (i<j)
        i_idx_local, j_idx_local = np.triu_indices(n_v, k=1)
        n_pair = len(i_idx_local)
        n_total += n_pair
        if n_pair == 0:
            continue

        # 训练段价差 X = log_p_i - log_p_j (W_train, n_pair)
        X_train = train_v[:, i_idx_local] - train_v[:, j_idx_local]

        # ---- 市值差距硬过滤 (可选): 仅保留 |log mv_i - log mv_j| < mv_gap_max 的 pair ----
        # 在 OU 估计前裁剪 (与 corr_filter 同层), 省 OU/ADF 算力。市值缺失(NaN)的 pair 视为不通过。
        if mv_wide is not None and mv_gap_max is not None:
            mv_ind = mv_wide[t_idx, s_idx_v]                              # (n_v,) 该行业当日 log 市值
            mv_gap = np.abs(mv_ind[i_idx_local] - mv_ind[j_idx_local])    # (n_pair,)
            mv_keep = np.isfinite(mv_gap) & (mv_gap < mv_gap_max)
            i_idx_local = i_idx_local[mv_keep]
            j_idx_local = j_idx_local[mv_keep]
            X_train = X_train[:, mv_keep]
            n_pair = len(i_idx_local)
            if n_pair == 0:
                continue

        # ---- 行业内相似度预筛 (可选, 按比例保留 top X%) ----
        # 在 OU 估计之前裁剪 pair, 节省 OU + ADF 算力。
        # 注意: pair_score 越大 = 越相似, SSD 取负使得方向一致。
        if corr_filter_method != "none" and corr_filter_pct < 1.0:
            if corr_filter_method == "pearson":
                # 行业内 daily return 的 Pearson 相关
                ret_v = np.diff(train_v, axis=0)             # (W-1, n_v)
                rcorr = np.corrcoef(ret_v.T)                 # (n_v, n_v)
                pair_score = rcorr[i_idx_local, j_idx_local] # (n_pair,)
            elif corr_filter_method == "spearman":
                # 日收益做秩, 然后 Pearson on ranks
                ret_v = np.diff(train_v, axis=0)
                rank_v = _rank_per_col(ret_v)                # (W-1, n_v)
                rcorr = np.corrcoef(rank_v.T)
                pair_score = rcorr[i_idx_local, j_idx_local]
            else:                                            # ssd
                # SSD = sum_t (X_t - X_0)^2, 越小越像 → 取负
                Xc = X_train - X_train[0:1, :]               # 归一化为 X_0 = 0
                pair_score = -(Xc ** 2).sum(axis=0)

            # NaN 用 -inf 替代, 保证排到末尾不入选
            pair_score = np.where(np.isnan(pair_score), -np.inf, pair_score)
            # 行业内保留 top pct 比例 (向上取整, 至少留 1 个)
            n_keep = max(1, int(np.ceil(n_pair * corr_filter_pct)))
            keep_idx = np.argsort(pair_score)[-n_keep:]
            i_idx_local = i_idx_local[keep_idx]
            j_idx_local = j_idx_local[keep_idx]
            X_train = X_train[:, keep_idx]
            n_pair = len(i_idx_local)

        # AR(1) 估计 + 样本统计 (estimate_ou_ar1 同时返回 mean_X / std_X)
        ou = estimate_ou_ar1(X_train)
        b = ou["b"]; kappa = ou["kappa"]; mu = ou["mu"]
        sigma = ou["sigma"]; half_life = ou["half_life"]
        mean_X = ou["mean_X"]; std_X = ou["std_X"]
        # 短窗 σ: 训练窗末段 20/30 日 X 的样本标准差,
        # 仅 abs_dev_* 系列信号使用 (signal = (μ-X) · [√]σ_短窗)
        # 训练窗 252 日时不会触发 fallback; 短训练窗 (60/120) 时优雅降级到 std_X
        if X_train.shape[0] >= 30:
            sigma_short_20 = X_train[-20:, :].std(axis=0, ddof=1)
            sigma_short_30 = X_train[-30:, :].std(axis=0, ddof=1)
        elif X_train.shape[0] >= 20:
            sigma_short_20 = X_train[-20:, :].std(axis=0, ddof=1)
            sigma_short_30 = sigma_short_20
        else:
            sigma_short_20 = std_X
            sigma_short_30 = std_X
        # ADF
        adf_p, _, _ = _batch_adf_pvalues(X_train)

        # 硬过滤 (kappa 范围已被 half_life 范围等价表达,无需重复)
        # signal_type 控制是否保留 hl 过滤 + 是否要求 OU mu/sigma 非 NaN
        if signal_type in ("kappa_dev", "zscore", "zscore_nok",
                           "abs_dev", "abs_dev_sqrt_s20", "abs_dev_s20",
                           "abs_dev_sqrt_s30", "abs_dev_s30"):
            # OU 信号 / 偏离量信号: 必须有 mu/sigma 才能算 (sigma 用于过滤质量, abs_dev 不进入信号)
            ou_mask = (
                (b > 0)
                & (half_life >= HALF_LIFE_MIN) & (half_life <= HALF_LIFE_MAX)
                & (adf_p < ADF_P_MAX)
                & ~np.isnan(mu) & ~np.isnan(sigma)
            )
        elif signal_type == "simple":
            # simple 信号: 用样本 mean/std, 不依赖 OU mu/sigma
            # 仍保留 hl 过滤作为质量门槛 (这要求 b 在 (0,1) 区间, half_life 有定义)
            ou_mask = (
                (b > 0)
                & (half_life >= HALF_LIFE_MIN) & (half_life <= HALF_LIFE_MAX)
                & (adf_p < ADF_P_MAX)
                & ~np.isnan(mean_X) & (std_X > _EPS)
            )
        elif signal_type == "simple_no_hl":
            # simple 信号 + 不要求 hl: 只需 b>0 + ADF + 样本统计有效
            ou_mask = (
                (b > 0)
                & (adf_p < ADF_P_MAX)
                & ~np.isnan(mean_X) & (std_X > _EPS)
            )
        else:
            raise ValueError(f"unknown signal_type: {signal_type}")
        n_pass = int(ou_mask.sum())
        if n_pass == 0:
            continue
        n_ou_pass += n_pass

        # ---- NZC 筛选 (可选, 行业内按 NZC 降序保留 top X%) ----
        # 仅对 OU 硬过滤通过的 pair 算 NZC, 然后在该子集内按比例保留高 NZC pair.
        # NZC 越大 = 价差越频繁回到均值 = 均值回归特性越强 (非参数度量).
        # 实现: 算出 nzc 后, 将"未入选 top X%"的 pair 在 ou_mask 中置 False,
        # 这样下游所有按 ip 提取 / 排序的逻辑零修改.
        if nzc_filter_pct < 1.0:
            ip_pre = np.where(ou_mask)[0]
            X_train_passed = X_train[:, ip_pre]              # (W_train, n_pass)
            nzc_passed = compute_nzc_per_pair(X_train_passed)  # (n_pass,)
            n_keep = max(1, int(np.ceil(n_pass * nzc_filter_pct)))
            # argsort 升序, 取末尾 n_keep 个 = NZC 最高的 n_keep 个 (在 ip_pre 内的局部索引)
            keep_local = np.argsort(nzc_passed)[-n_keep:]
            keep_global = ip_pre[keep_local]                  # 还原到 i_idx_local 的索引
            new_mask = np.zeros_like(ou_mask)
            new_mask[keep_global] = True
            ou_mask = new_mask
            n_pass = int(ou_mask.sum())
            if n_pass == 0:
                continue

        # 仅对通过过滤的 pair 计算验证 IC + 信号 (节省算力)
        ip = np.where(ou_mask)[0]
        b_p = b[ip]; kappa_p = kappa[ip]; mu_p = mu[ip]
        sigma_p = sigma[ip]; half_life_p = half_life[ip]; adf_p_p = adf_p[ip]
        mean_X_p = mean_X[ip]; std_X_p = std_X[ip]
        sigma_short_20_p = sigma_short_20[ip]
        sigma_short_30_p = sigma_short_30[ip]
        i_g = i_idx_local[ip]   # 局部索引 (相对于 s_idx_v)
        j_g = j_idx_local[ip]

        # 验证段价差 + signal (按 signal_type 分支)
        X_valid = valid_v[:, i_g] - valid_v[:, j_g]               # (W_valid, n_pass)
        if signal_type == "zscore":
            sigma_safe = np.clip(sigma_p, _EPS, None)
            kdev_valid = kappa_p[None, :] * (mu_p[None, :] - X_valid)
            signal_valid = kdev_valid / sigma_safe[None, :]
        elif signal_type == "zscore_nok":
            # 与 zscore 唯一区别: 去掉 kappa 系数
            sigma_safe = np.clip(sigma_p, _EPS, None)
            signal_valid = (mu_p[None, :] - X_valid) / sigma_safe[None, :]
        elif signal_type in ("simple", "simple_no_hl"):
            std_safe = np.clip(std_X_p, _EPS, None)
            # simple 信号 = (mean_X - X) / std_X, 完全样本统计, 不依赖 OU 模型
            signal_valid = (mean_X_p[None, :] - X_valid) / std_safe[None, :]
        elif signal_type == "abs_dev":
            # 偏离量信号: 直接用 |μ-X|, σ 不参与排序
            signal_valid = mu_p[None, :] - X_valid
        elif signal_type == "abs_dev_sqrt_s20":
            s_safe = np.clip(sigma_short_20_p, _EPS, None)
            signal_valid = (mu_p[None, :] - X_valid) * np.sqrt(s_safe[None, :])
        elif signal_type == "abs_dev_s20":
            s_safe = np.clip(sigma_short_20_p, _EPS, None)
            signal_valid = (mu_p[None, :] - X_valid) * s_safe[None, :]
        elif signal_type == "abs_dev_sqrt_s30":
            s_safe = np.clip(sigma_short_30_p, _EPS, None)
            signal_valid = (mu_p[None, :] - X_valid) * np.sqrt(s_safe[None, :])
        elif signal_type == "abs_dev_s30":
            s_safe = np.clip(sigma_short_30_p, _EPS, None)
            signal_valid = (mu_p[None, :] - X_valid) * s_safe[None, :]
        else:  # kappa_dev (baseline)
            signal_valid = kappa_p[None, :] * (mu_p[None, :] - X_valid)
        # 验证标签 = r_i_{t'+1} - r_j_{t'+1}
        label_valid = label_v[:, i_g] - label_v[:, j_g]            # (W_valid, n_pass)

        # 矩阵化 Spearman
        # NOTE: 同一 pair 内 sigma/std/mu/mean 都是常数, signal_valid 与各种"-X"
        # 同序变换, 故 valid_rank_ic 在所有 signal_type 之间数值完全相同。
        valid_ic = spearman_per_pair(signal_valid, label_valid)

        # ---- 训练窗内 K 折 CV (可选) ----
        # 若 cv_folds > 0, 在训练窗 X_train_ip 内做 K 折 CV, 算 cv_mean_ic
        # 用于替代 valid_ic 做行业内贪心去重 (但 pair_log 仍记录原 valid_ic 供诊断)。
        if cv_folds > 0:
            X_train_ip = X_train[:, ip]    # (W_train, n_pass) 通过过滤 pair 的训练段
            cv_mean_ic = compute_cv_mean_ic(
                X_train_full=X_train_ip,
                n_pass=n_pass,
                k=cv_folds,
                signal_type=signal_type,
                label_h=cv_label_h,
            )
        else:
            cv_mean_ic = np.full(n_pass, np.nan)   # 占位

        # 信号日 X_t
        X_today = today_v[i_g] - today_v[j_g]                     # (n_pass,)
        # predict_window > 0 时, 用信号日前 N 个交易日的滚动 mean/std 替代 OU 的 mu/sigma
        # 仅影响 signal_today, 验证段 signal_valid / valid_ic / CV 全部保留 OU 训练窗的 mu/sigma
        if predict_window > 0 and t_idx >= predict_window:
            pred_lp = log_price_wide[t_idx - predict_window: t_idx, s_idx_v]  # (N, n_v)
            X_pred = pred_lp[:, i_g] - pred_lp[:, j_g]                         # (N, n_pass)
            with np.errstate(all="ignore"):
                mu_use = np.nanmean(X_pred, axis=0)
                sigma_use = np.nanstd(X_pred, axis=0, ddof=1)
            if predict_window_mu_only:
                # 仅用短窗均值替代 mu; sigma/std 仍取 OU 训练窗值 (kappa 本就保留 OU)
                sigma_use = sigma_p
                mean_use = mu_use
                std_use = std_X_p
            else:
                mean_use = mu_use      # simple 路径复用同一个 mean
                std_use = sigma_use    # simple 路径复用同一个 std
        else:
            mu_use = mu_p
            sigma_use = sigma_p
            mean_use = mean_X_p
            std_use = std_X_p

        if signal_type == "zscore":
            sigma_safe = np.clip(sigma_use, _EPS, None)
            signal_today = kappa_p * (mu_use - X_today) / sigma_safe
        elif signal_type == "zscore_nok":
            sigma_safe = np.clip(sigma_use, _EPS, None)
            signal_today = (mu_use - X_today) / sigma_safe
        elif signal_type in ("simple", "simple_no_hl"):
            std_safe = np.clip(std_use, _EPS, None)
            signal_today = (mean_use - X_today) / std_safe
        elif signal_type == "abs_dev":
            # abs_dev_* 系列: 信号日 σ 短窗仍取训练窗末段, 不受 predict_window 影响
            signal_today = mu_use - X_today
        elif signal_type == "abs_dev_sqrt_s20":
            s_safe = np.clip(sigma_short_20_p, _EPS, None)
            signal_today = (mu_use - X_today) * np.sqrt(s_safe)
        elif signal_type == "abs_dev_s20":
            s_safe = np.clip(sigma_short_20_p, _EPS, None)
            signal_today = (mu_use - X_today) * s_safe
        elif signal_type == "abs_dev_sqrt_s30":
            s_safe = np.clip(sigma_short_30_p, _EPS, None)
            signal_today = (mu_use - X_today) * np.sqrt(s_safe)
        elif signal_type == "abs_dev_s30":
            s_safe = np.clip(sigma_short_30_p, _EPS, None)
            signal_today = (mu_use - X_today) * s_safe
        else:  # kappa_dev
            signal_today = kappa_p * (mu_use - X_today)

        # 全行业 pair 候选: 转换回原 (i, j) 的 stock_code
        codes_v = stock_codes[s_idx_v]                            # (n_v,)
        i_codes = codes_v[i_g]                                    # (n_pass,)
        j_codes = codes_v[j_g]
        i_short = np.array([c in short_pool_set for c in i_codes])
        j_short = np.array([c in short_pool_set for c in j_codes])

        df_ind = pd.DataFrame({
            "industry": ind,
            "stock_i": i_codes,
            "stock_j": j_codes,
            "is_i_short_pool": i_short,
            "is_j_short_pool": j_short,
            "b": b_p,
            "kappa": kappa_p,
            "mu": mu_p,
            "sigma": sigma_p,
            "half_life": half_life_p,
            "mean_X": mean_X_p,
            "std_X": std_X_p,
            "ADF_p": adf_p_p,
            "valid_rank_ic": valid_ic,
            "cv_mean_ic": cv_mean_ic,
            "signal": signal_today,
        })
        # 选去重排序基准: dedup_rank 决定行业内贪心去重排序键
        #   cv_mean_ic(默认): cv_folds>0 用 cv_mean_ic, 否则 valid_rank_ic
        #   nzc: 训练窗 NZC 降序(越大回归性越强, 越优先保留), ic_col 次级 tie-break
        ic_col = "cv_mean_ic" if cv_folds > 0 else "valid_rank_ic"
        if dedup_rank == "nzc":
            # X_train[:, ip] 为该行业 OU 通过 pair 的训练段价差, 行序与 df_ind 一致
            df_ind["nzc_dedup"] = compute_nzc_per_pair(X_train[:, ip])
            base_col = "nzc_dedup"
            dedup_col = ["nzc_dedup", ic_col]   # NZC 主键降序, ic_col 次级降序破平局
        elif dedup_rank == "barra_dist":
            # 两腿风格距离: 距离越小=因子越相似=越优先保留。贪心按降序, 故存负距离。
            gi = s_idx_v[i_g]; gj = s_idx_v[j_g]                  # 两腿全局股票索引, 与 df_ind 行序一致
            fidx = barra_factor_idx if barra_factor_idx is not None else slice(None)
            xi = barra_expo[t_idx, gi, :][:, fidx]                # (n_pass, k)
            xj = barra_expo[t_idx, gj, :][:, fidx]
            dx = (xi - xj).astype(np.float64)                     # (n_pass, k)
            bad = ~np.isfinite(dx).all(axis=1)
            if barra_metric == "mahal":
                sig = barra_cov[t_idx][np.ix_(
                    barra_factor_idx, barra_factor_idx)] if barra_factor_idx is not None else barra_cov[t_idx]
                d2 = np.einsum("nk,kl,nl->n", dx, sig, dx)        # Δxᵀ Σ Δx = 配对因子风险
                dist = np.sqrt(np.maximum(d2, 0.0))
                if not np.isfinite(sig).all():
                    bad = np.ones(len(dx), dtype=bool)
            else:
                dist = np.sqrt((dx ** 2).sum(axis=1))
            dist[bad] = np.inf                                    # 缺失暴露 -> 距离设 inf, 最低优先级
            df_ind["barra_dist_neg"] = -dist                      # 负距离: 降序贪心 = 距离升序
            base_col = "barra_dist_neg"
            dedup_col = ["barra_dist_neg", ic_col]                # 风格距离主键, ic_col 次级破平局
        else:
            base_col = ic_col
            dedup_col = ic_col
        df_ind = df_ind[~df_ind[base_col].isna()]
        if df_ind.empty:
            continue

        # ---- 行业内贪心去重: 按 dedup_col 降序, 每只股票最多 dedup_max_per_stock 个 pair ----
        # cap=1 时与原 set 去重逻辑数学等价
        df_ind = _greedy_dedup_with_cap(df_ind, dedup_col, dedup_max_per_stock)
        if df_ind.empty:
            continue

        pair_records.append(df_ind)

    if not pair_records:
        return {"signal_date": sig_date, "skip": True, "reason": "no surviving pairs",
                "n_pairs_total": n_total, "n_pairs_ou_pass": n_ou_pass,
                "n_pairs_legal": 0, "n_pairs_top20": 0,
                "elapsed_sec": time.time() - t0,
                "pair_log": pd.DataFrame(),
                "stock_pred": pd.DataFrame(),
                "n_signal_stocks": 0, "n_long": 0, "n_short": 0}

    pair_dedup = pd.concat(pair_records, ignore_index=True)
    pair_dedup["date"] = sig_date

    # ---- 合法 pair 筛选: 预测空头方 ∈ short_pool ----
    # 预测空头方: signal>0 → j; signal<0 → i; signal==0 直接丢弃
    sig = pair_dedup["signal"].to_numpy()
    short_side_in_pool = np.where(
        sig > 0,
        pair_dedup["is_j_short_pool"].to_numpy(),
        np.where(sig < 0, pair_dedup["is_i_short_pool"].to_numpy(), False),
    )
    pair_dedup["is_legal"] = short_side_in_pool & (sig != 0)
    legal_df = pair_dedup[pair_dedup["is_legal"]].copy()
    n_legal = len(legal_df)

    # ---- 全市场按 |signal| 排序取前 20% ----
    legal_df["abs_signal"] = legal_df["signal"].abs()
    legal_df = legal_df.sort_values("abs_signal", ascending=False).reset_index(drop=True)
    # 初始 n_top = max(50, ceil(n_legal*0.20)), 不超过 n_legal
    n_top_pct = int(math.ceil(n_legal * TOP_PCT))
    n_top = min(n_legal, max(TOP_PCT_MIN_N, n_top_pct)) if n_legal > 0 else 0

    # 动态扩 top 兜底: 若净额化后 long/short 独立票数 < min_unique_per_side, 继续加 pair
    # cap=1 时初始 top 必满足(因 1 pair = 1 长 + 1 短独立票, 50 pair = 50 vs 50), 不进入循环
    while n_top < n_legal:
        cur_top = legal_df.iloc[:n_top]
        n_long_uniq, n_short_uniq = _count_unique_after_netting(cur_top)
        if n_long_uniq >= min_unique_per_side and n_short_uniq >= min_unique_per_side:
            break
        n_top += 1

    legal_df["is_top20"] = False
    if n_top > 0:
        legal_df.loc[:n_top - 1, "is_top20"] = True

    top_df = legal_df[legal_df["is_top20"]].copy()
    n_top_actual = len(top_df)

    # ---- 输出股票级预测 ----
    if dedup_max_per_stock <= 1:
        # 旧路径 (K=1): stock_pred 保留原 schema, 上游用等权回测
        rows = []
        for r in top_df.itertuples(index=False):
            s = r.signal
            if s > 0:
                long_code, short_code = r.stock_i, r.stock_j
            else:
                long_code, short_code = r.stock_j, r.stock_i
            rows.append({"date": sig_date, "stock_code": long_code,
                         "industry": r.industry, "stock_pred": abs(s) / 2.0,
                         "is_short_pool": (long_code in short_pool_set),
                         "paired_stock": short_code, "side": "long",
                         "pair_valid_rank_ic": r.valid_rank_ic,
                         "pair_signal": s})
            rows.append({"date": sig_date, "stock_code": short_code,
                         "industry": r.industry, "stock_pred": -abs(s) / 2.0,
                         "is_short_pool": (short_code in short_pool_set),
                         "paired_stock": long_code, "side": "short",
                         "pair_valid_rank_ic": r.valid_rank_ic,
                         "pair_signal": s})
        stock_pred_df = pd.DataFrame(rows)
    else:
        # D5 路径 (K>1): 净额化后输出权重, 上游走加权回测
        long_w, short_w = _net_pair_to_holdings(top_df)
        rows = []
        for code, w in long_w.items():
            rows.append({"date": sig_date, "stock_code": code,
                         "weight": w, "side": "long",
                         "is_short_pool": (code in short_pool_set)})
        for code, w in short_w.items():
            rows.append({"date": sig_date, "stock_code": code,
                         "weight": w, "side": "short",
                         "is_short_pool": (code in short_pool_set)})
        stock_pred_df = pd.DataFrame(rows)

    n_signal_stocks = len(stock_pred_df)
    n_long = int((stock_pred_df["side"] == "long").sum())
    n_short = int((stock_pred_df["side"] == "short").sum())

    # 完整 pair_log: 包含合法但未入选 top 的 (上游可统计 / 复用)
    pair_log_out = pair_dedup.copy()
    pair_log_out["abs_signal"] = pair_log_out["signal"].abs()
    pair_log_out["is_top20"] = False
    if n_top > 0:
        # 将 legal_df 的 top 标记 join 回去 (按 stock_i, stock_j 唯一)
        top_keys = set(zip(top_df["stock_i"].tolist(), top_df["stock_j"].tolist()))
        pair_log_out["is_top20"] = [
            (i, j) in top_keys
            for i, j in zip(pair_log_out["stock_i"], pair_log_out["stock_j"])
        ]

    elapsed = time.time() - t0
    if verbose:
        print(f"  [{sig_date.date()}] 候选 {n_total:>7d}, OU 通过 {n_ou_pass:>6d}, "
              f"去重后 {len(pair_dedup):>5d}, 合法 {n_legal:>4d}, top20 {n_top_actual:>4d}, "
              f"多头 {n_long:>4d} 空头 {n_short:>4d}, {elapsed:.1f}s")

    return {
        "signal_date": sig_date,
        "skip": False,
        "n_pairs_total": n_total,
        "n_pairs_ou_pass": n_ou_pass,
        "n_pairs_dedup": len(pair_dedup),
        "n_pairs_legal": n_legal,
        "n_pairs_top20": n_top_actual,
        "n_signal_stocks": n_signal_stocks,
        "n_long": n_long,
        "n_short": n_short,
        "pair_log": pair_log_out,
        "stock_pred": stock_pred_df,
        "elapsed_sec": elapsed,
    }


# =====================================================================
# 单实验流程
# =====================================================================

def run_one_experiment(
    exp_name: str,
    train_window: int,
    cal: np.ndarray,
    log_price_wide: np.ndarray,
    ret_wide: np.ndarray,
    stock_codes: np.ndarray,
    industry_codes: np.ndarray,
    short_pool_by_date: Dict[pd.Timestamp, set],
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    status_wide: pd.DataFrame,
    backtest_dates: List[pd.Timestamp],
    signal_type: str = "kappa_dev",
    holding_period: int = 1,
    cv_folds: int = 0,
    cv_label_h: int = 5,
    corr_filter_method: str = "none",
    corr_filter_pct: float = 1.0,
    nzc_filter_pct: float = 1.0,
    pairing_method: str = "industry",
    theme_top5_arr: Optional[np.ndarray] = None,
    dedup_max_per_stock: int = 1,
    min_unique_per_side: int = 50,
    predict_window: int = 0,
    predict_window_mu_only: bool = False,
    dedup_rank: str = "cv_mean_ic",
    barra_expo: np.ndarray = None,
    barra_cov: np.ndarray = None,
    barra_factor_idx: np.ndarray = None,
    barra_metric: str = "euclid",
):
    """
    执行单个 OU 实验。

    signal_type:        见 process_one_section 的同名参数说明。
    holding_period:     1 = 日频换仓 (baseline); >1 = 持有 N 日轮换。
                        信号阶段每日仍生成,持有期内多余信号被回测忽略。
    cv_folds:           0 = 不做 CV (默认); >0 = 训练窗内 K 折 CV, 用 cv_mean_ic
                        替代 valid_rank_ic 做行业内贪心去重。
    cv_label_h:         CV 每折验证段 label 累积周期 (5 = 5 日调仓).
    corr_filter_method: 行业内 pair 相似度预筛: none/pearson/spearman/ssd.
    corr_filter_pct:    行业内保留 top X% 高相似 pair (1.0=不筛, 0.5=top 50%).
    nzc_filter_pct:     行业内 OU 后 NZC top X% 保留 (1.0=不筛, 0.5=top 50%).
                        在 OU 硬过滤之后、CV5 之前; pairing_method="industry" 才生效.
    pairing_method:     "industry" (默认, 同行业枚举) 或 "shared_top_theme" (跨行业, 共享 top K 概念).
    theme_top5_arr:     pairing_method="shared_top_theme" 时必传, (T, S, K) int32 数组.
    dedup_max_per_stock: 1 = 旧行为 (每只票最多 1 pair, 等权回测);
                         >1 = 每只票最多 K pair, 净额化后输出权重, 走加权回测.
    min_unique_per_side: 净额化后多空各至少独立票数 (>1 时动态扩 top 兜底; 1 时不触发).

    输入数据均为各实验间共享的预处理结果 (节省内存/时间)。
    """
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    metrics_path = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"\n[{exp_name}] 已有 metrics.json,跳过")
        return

    # 入参校验放在跳过逻辑之后, 避免"已完成实验缺数据时仍被拦下"
    if pairing_method == "shared_top_theme" and theme_top5_arr is None:
        raise ValueError("pairing_method='shared_top_theme' 需要传 theme_top5_arr")

    print(f"\n{'='*60}")
    print(f"[{exp_name}] train_window={train_window}, valid_window={VALID_WINDOW}, "
          f"signal_type={signal_type}, holding_period={holding_period}, "
          f"cv_folds={cv_folds}, corr_filter={corr_filter_method}/{corr_filter_pct:.2f}, "
          f"nzc_filter_pct={nzc_filter_pct:.2f}, pairing={pairing_method}, "
          f"dedup_max={dedup_max_per_stock}, min_uniq={min_unique_per_side}")
    print(f"{'='*60}")
    t_exp = time.time()

    # 把 cal 转 np.datetime64 数组 + 建立 timestamp -> idx 索引
    cal_ts = pd.DatetimeIndex(cal)
    date_to_idx = {dt: i for i, dt in enumerate(cal_ts)}

    pair_log_chunks = []
    stock_pred_chunks = []
    daily_stats = []   # 每日统计

    # ---- GLB-semi 预处理: 在循环之前生成所有半年起点的稳定 pair_set ----
    pair_sets_global_semi: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    anchor_dates_sorted: Optional[List[pd.Timestamp]] = None
    if pairing_method == "global_semi":
        anchors = _get_semi_annual_anchors(backtest_dates, cal)
        print(f"\n[{exp_name}] GLB-semi 半年起点 {len(anchors)} 个: " +
              ", ".join(str(d.date()) for d, _ in anchors))
        pair_sets_global_semi = {}
        for T0_date, T0_idx in anchors:
            cache_path = os.path.join(exp_dir, f"pair_set_{T0_date.date()}.pkl")
            if os.path.exists(cache_path):
                P = pd.read_pickle(cache_path)
                print(f"  [{T0_date.date()}] 加载缓存 {cache_path} ({len(P):,} pair)")
            else:
                print(f"  [{T0_date.date()}] 生成 P_T0 ...")
                P = select_global_pair_set(
                    t_idx_T0=T0_idx, cal=cal,
                    log_price_wide=log_price_wide,
                    ret_wide=ret_wide,
                    stock_codes=stock_codes,
                    train_window=train_window,
                    valid_window=VALID_WINDOW,
                    signal_type=signal_type,
                    cv_folds=cv_folds,
                    cv_label_h=cv_label_h,
                    verbose=True,
                )
                P.to_pickle(cache_path)
                print(f"  [{T0_date.date()}] 已缓存 {cache_path} ({len(P):,} pair)")
            pair_sets_global_semi[T0_date] = P
        anchor_dates_sorted = sorted(pair_sets_global_semi.keys())

    n_dates = len(backtest_dates)
    for k, sig_date in enumerate(backtest_dates):
        if sig_date not in date_to_idx:
            continue
        t_idx = date_to_idx[sig_date]
        sp_set = short_pool_by_date.get(sig_date, set())

        verbose = (k < 5) or (k % 10 == 0) or (k == n_dates - 1)
        if pairing_method == "global_semi":
            # 找 sig_date 落在哪个半年起点 (>= 最大的 T0 ≤ sig_date)
            active_T0 = None
            for T0 in anchor_dates_sorted:
                if T0 <= sig_date:
                    active_T0 = T0
                else:
                    break
            if active_T0 is None:
                continue
            out = process_one_section_global_semi(
                section_idx=t_idx, cal=cal,
                log_price_wide=log_price_wide,
                stock_codes=stock_codes,
                short_pool_set=sp_set,
                P_active=pair_sets_global_semi[active_T0],
                signal_type=signal_type,
                verbose=verbose,
            )
        elif pairing_method == "shared_top_theme":
            out = process_one_section_concept(
                section_idx=t_idx, cal=cal,
                log_price_wide=log_price_wide, ret_wide=ret_wide,
                stock_codes=stock_codes, short_pool_set=sp_set,
                theme_top5_arr=theme_top5_arr,
                train_window=train_window, valid_window=VALID_WINDOW,
                signal_type=signal_type, cv_folds=cv_folds,
                cv_label_h=cv_label_h,
                dedup_max_per_stock=dedup_max_per_stock,
                min_unique_per_side=min_unique_per_side,
                predict_window=predict_window,
                verbose=verbose,
            )
        else:
            out = process_one_section(
                section_idx=t_idx,
                cal=cal,
                log_price_wide=log_price_wide,
                ret_wide=ret_wide,
                industry_codes=industry_codes,
                stock_codes=stock_codes,
                short_pool_set=sp_set,
                train_window=train_window,
                valid_window=VALID_WINDOW,
                signal_type=signal_type,
                cv_folds=cv_folds,
                cv_label_h=cv_label_h,
                corr_filter_method=corr_filter_method,
                corr_filter_pct=corr_filter_pct,
                nzc_filter_pct=nzc_filter_pct,
                dedup_max_per_stock=dedup_max_per_stock,
                min_unique_per_side=min_unique_per_side,
                predict_window=predict_window,
                predict_window_mu_only=predict_window_mu_only,
                dedup_rank=dedup_rank,
                barra_expo=barra_expo,
                barra_cov=barra_cov,
                barra_factor_idx=barra_factor_idx,
                barra_metric=barra_metric,
                verbose=verbose,
            )
        if out.get("skip"):
            daily_stats.append({
                "date": sig_date,
                "n_pairs_total": out.get("n_pairs_total", 0),
                "n_pairs_ou_pass": out.get("n_pairs_ou_pass", 0),
                "n_pairs_dedup": out.get("n_pairs_dedup", 0),
                "n_pairs_legal": out.get("n_pairs_legal", 0),
                "n_pairs_top20": out.get("n_pairs_top20", 0),
                "n_signal_stocks": out.get("n_signal_stocks", 0),
                "n_long": out.get("n_long", 0),
                "n_short": out.get("n_short", 0),
                "elapsed_sec": out.get("elapsed_sec", 0.0),
            })
            continue

        pair_log_chunks.append(out["pair_log"])
        stock_pred_chunks.append(out["stock_pred"])
        daily_stats.append({
            "date": sig_date,
            "n_pairs_total": out["n_pairs_total"],
            "n_pairs_ou_pass": out["n_pairs_ou_pass"],
            "n_pairs_dedup": out["n_pairs_dedup"],
            "n_pairs_legal": out["n_pairs_legal"],
            "n_pairs_top20": out["n_pairs_top20"],
            "n_signal_stocks": out["n_signal_stocks"],
            "n_long": out["n_long"],
            "n_short": out["n_short"],
            "elapsed_sec": out["elapsed_sec"],
        })

    if not stock_pred_chunks:
        print(f"[{exp_name}] 无任何有效信号,放弃回测")
        return

    pair_log_full = pd.concat(pair_log_chunks, ignore_index=True)
    stock_pred_full = pd.concat(stock_pred_chunks, ignore_index=True)
    stats_df = pd.DataFrame(daily_stats)

    # 写中间产物
    pair_log_path = os.path.join(exp_dir, "pair_log.parquet")
    stock_pred_path = os.path.join(exp_dir, "stock_pred.parquet")
    daily_stats_path = os.path.join(exp_dir, "daily_stats.parquet")
    pair_log_full.to_parquet(pair_log_path, index=False)
    stock_pred_full.to_parquet(stock_pred_path, index=False)
    stats_df.to_parquet(daily_stats_path, index=False)
    print(f"\n[{exp_name}] 信号阶段完成, "
          f"耗时 {time.time()-t_exp:.0f}s, "
          f"pair_log {len(pair_log_full)} 行, stock_pred {len(stock_pred_full)} 行")

    # ---- 多空回测 ----
    if dedup_max_per_stock <= 1:
        # K=1 旧路径: 等权回测
        long_holdings = (
            stock_pred_full[stock_pred_full["side"] == "long"]
            .groupby("date")["stock_code"].apply(list)
        )
        short_holdings = (
            stock_pred_full[stock_pred_full["side"] == "short"]
            .groupby("date")["stock_code"].apply(list)
        )
        all_dates_with_signal = sorted(set(long_holdings.index) | set(short_holdings.index))
        long_holdings = long_holdings.reindex(all_dates_with_signal, fill_value=[])
        short_holdings = short_holdings.reindex(all_dates_with_signal, fill_value=[])

        print(f"[{exp_name}] 启动多空回测 (等权)...")
        bt_results = run_longshort_backtest(
            long_holdings=long_holdings,
            short_holdings=short_holdings,
            open_prices=open_wide,
            close_prices=close_wide,
            status_data=status_wide,
            output_dir=exp_dir,
            commission_rate=COMMISSION,
            holding_period=holding_period,
        )
    else:
        # K>1 加权路径: 净额化权重 → run_longshort_backtest_weighted
        long_w = (
            stock_pred_full[stock_pred_full["side"] == "long"]
            .groupby("date")
            .apply(lambda g: dict(zip(g["stock_code"], g["weight"])))
        )
        short_w = (
            stock_pred_full[stock_pred_full["side"] == "short"]
            .groupby("date")
            .apply(lambda g: dict(zip(g["stock_code"], g["weight"])))
        )
        all_dates_with_signal = sorted(set(long_w.index) | set(short_w.index))
        long_w = long_w.reindex(all_dates_with_signal, fill_value={})
        short_w = short_w.reindex(all_dates_with_signal, fill_value={})

        print(f"[{exp_name}] 启动多空回测 (按净敞口加权)...")
        bt_results = run_longshort_backtest_weighted(
            long_holdings_w=long_w,
            short_holdings_w=short_w,
            open_prices=open_wide,
            close_prices=close_wide,
            status_data=status_wide,
            output_dir=exp_dir,
            commission_rate=COMMISSION,
            holding_period=holding_period,
        )
    bt_stats = bt_results["statistics"]

    # ---- 写 metrics.json ----
    valid_mask_stats = stats_df["n_signal_stocks"] > 0
    if valid_mask_stats.any():
        valid_stats_df = stats_df[valid_mask_stats]
        ns = valid_stats_df["n_signal_stocks"]
        n_long_arr = valid_stats_df["n_long"]
        n_short_arr = valid_stats_df["n_short"]
        n_legal_arr = valid_stats_df["n_pairs_legal"]
        n_top_arr = valid_stats_df["n_pairs_top20"]
        valid_stock_stats = {
            "avg_n_signal_stocks_per_day": float(ns.mean()),
            "median_n_signal_stocks_per_day": float(ns.median()),
            "min_n_signal_stocks_per_day": int(ns.min()),
            "max_n_signal_stocks_per_day": int(ns.max()),
        }
        n_long_avg = float(n_long_arr.mean())
        n_short_avg = float(n_short_arr.mean())
        n_legal_avg = float(n_legal_arr.mean())
        n_top_avg = float(n_top_arr.mean())
    else:
        valid_stock_stats = {"avg_n_signal_stocks_per_day": 0,
                             "median_n_signal_stocks_per_day": 0,
                             "min_n_signal_stocks_per_day": 0,
                             "max_n_signal_stocks_per_day": 0}
        n_long_avg = n_short_avg = n_legal_avg = n_top_avg = 0.0

    # 给所有 stats 字典补 calmar_ratio = annual_return / |max_drawdown|
    # (annual_return 和 max_drawdown 都是百分比, 不需要再换算)
    def _add_calmar(s):
        ann = s.get("annual_return", 0.0)
        mdd = s.get("max_drawdown", 0.0)
        s["calmar_ratio"] = round(ann / abs(mdd), 3) if mdd != 0 else 0.0
        return s
    for _key in ("head", "tail", "tail_raw", "longshort", "benchmark",
                 "head_excess", "tail_excess", "ls_excess"):
        if _key in bt_stats:
            bt_stats[_key] = _add_calmar(bt_stats[_key])

    metrics = {
        "experiment": exp_name,
        "train_window": train_window,
        "valid_window": VALID_WINDOW,
        "signal_type": signal_type,
        "holding_period": holding_period,
        "cv_folds": cv_folds,
        "cv_label_h": cv_label_h if cv_folds > 0 else None,
        "corr_filter_method": corr_filter_method,
        "corr_filter_pct": corr_filter_pct if corr_filter_method != "none" else None,
        "nzc_filter_pct": nzc_filter_pct if nzc_filter_pct < 1.0 else None,
        "pairing_method": pairing_method,
        "dedup_max_per_stock": dedup_max_per_stock,
        "min_unique_per_side": min_unique_per_side if dedup_max_per_stock > 1 else None,
        "predict_window": predict_window if predict_window > 0 else None,
        "backtest_range": f"{BACKTEST_START} ~ {BACKTEST_END}",
        "commission_rate": COMMISSION,
        "top_pct": TOP_PCT,
        "top_pct_min_n": TOP_PCT_MIN_N,
        # 多空指标 (主)
        "long":         bt_stats["head"],
        "short":        bt_stats["tail"],
        "longshort":    bt_stats["longshort"],
        "benchmark":    bt_stats["benchmark"],
        "long_excess":  bt_stats["head_excess"],
        "ls_excess":    bt_stats["ls_excess"],
        # 信号侧统计
        "valid_stock_stats": valid_stock_stats,
        "n_long_per_day_avg":         n_long_avg,
        "n_short_per_day_avg":        n_short_avg,
        "n_pairs_legal_per_day_avg":  n_legal_avg,
        "n_pairs_top20_per_day_avg":  n_top_avg,
        "n_signal_days":              int(valid_mask_stats.sum()),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{exp_name}] metrics.json 已写入")

    # ---- 写 summary.csv 一行 ----
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    fieldnames = [
        "exp", "train_window",
        "ls_annual", "ls_sharpe", "ls_calmar", "ls_mdd",
        "long_annual", "long_sharpe", "long_calmar", "long_excess_annual",
        "short_annual", "short_sharpe",
        "avg_n_signal_stocks", "avg_n_long", "avg_n_short",
        "avg_n_legal_pairs", "avg_n_top20_pairs",
    ]
    save_summary_row(summary_path, {
        "exp": exp_name,
        "train_window": train_window,
        "ls_annual":  bt_stats["longshort"]["annual_return"],
        "ls_sharpe":  bt_stats["longshort"]["sharpe_ratio"],
        "ls_calmar":  bt_stats["longshort"].get("calmar_ratio", 0.0),
        "ls_mdd":     bt_stats["longshort"]["max_drawdown"],
        "long_annual":         bt_stats["head"]["annual_return"],
        "long_sharpe":         bt_stats["head"]["sharpe_ratio"],
        "long_calmar":         bt_stats["head"].get("calmar_ratio", 0.0),
        "long_excess_annual":  bt_stats["head_excess"]["annual_return"],
        "short_annual":        bt_stats["tail"]["annual_return"],
        "short_sharpe":        bt_stats["tail"]["sharpe_ratio"],
        "avg_n_signal_stocks": round(valid_stock_stats["avg_n_signal_stocks_per_day"], 1),
        "avg_n_long":          round(n_long_avg, 1),
        "avg_n_short":         round(n_short_avg, 1),
        "avg_n_legal_pairs":   round(n_legal_avg, 1),
        "avg_n_top20_pairs":   round(n_top_avg, 1),
    }, fieldnames)
    print(f"[{exp_name}] summary.csv 已写入")
    print(f"[{exp_name}] 总耗时 {time.time()-t_exp:.0f}s")


# =====================================================================
# 数据预处理
# =====================================================================

def prepare_data():
    """
    加载并构造所有实验共用的数据结构。

    返回:
        cal: np.ndarray (T_total,) 全局交易日 (np.datetime64[ns])
        stock_codes: (S_all,)  字符串数组,固定列顺序
        industry_codes: (S_all,) 字符串数组,与 stock_codes 一一对应
        log_price_wide: (T_total, S_all) np.float32  log close
        ret_wide:       (T_total, S_all) np.float64  close-to-close 日收益
        short_pool_by_date: dict[pd.Timestamp -> set(stock_code)]
        open_wide / close_wide / status_wide: pd.DataFrame 宽表 (供 longshort 回测)
        backtest_dates: List[pd.Timestamp]
    """
    print("=" * 60)
    print(f"加载数据 [{DATA_START} ~ {BACKTEST_END}]")
    print("=" * 60)

    # 长表
    panel = load_price_industry_mv(DATA_START, BACKTEST_END)

    # 短表 → 宽表 (close / open / status / log_close / ret)
    print("构造宽表 (close/open/status/log_close/ret)...")
    t0 = time.time()
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
    cal_index = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    stock_index = panel["stock_code"].drop_duplicates().sort_values().reset_index(drop=True)
    cal = cal_index.to_numpy(dtype="datetime64[ns]")
    stock_codes = stock_index.to_numpy(dtype=object)

    # 利用 pivot 构造宽表
    close_wide = panel.pivot(index="date", columns="stock_code", values="close_price").reindex(
        index=cal_index, columns=stock_codes)
    open_wide = panel.pivot(index="date", columns="stock_code", values="open_price").reindex(
        index=cal_index, columns=stock_codes)
    status_wide = panel.pivot(index="date", columns="stock_code", values="status").reindex(
        index=cal_index, columns=stock_codes)
    print(f"  宽表 shape: close={close_wide.shape}, "
          f"耗时 {time.time()-t0:.0f}s")

    close_arr = close_wide.to_numpy(dtype=np.float64)
    log_price_wide = np.where(close_arr > 0, np.log(close_arr), np.nan).astype(np.float32)
    # 日收益 close-to-close
    ret_wide = np.full_like(close_arr, np.nan)
    ret_wide[1:, :] = (close_arr[1:, :] - close_arr[:-1, :]) / close_arr[:-1, :]

    # 市值宽表 (log market value): 配对时按 |log mv_i - log mv_j| 过滤市值差距用;
    # 列序与 close/log_price 一致 (reindex 到同 cal_index, stock_codes)。
    mv_wide_df = panel.pivot(index="date", columns="stock_code", values="market_value").reindex(
        index=cal_index, columns=stock_codes)
    mv_arr = mv_wide_df.to_numpy(dtype=np.float64)
    log_mv_wide = np.where(mv_arr > 0, np.log(mv_arr), np.nan).astype(np.float32)

    # 行业映射: 每只股票取最后一次出现的非空 sw_l1 (允许一只股票全程一个行业)
    print("构造行业映射 (stock_code -> sw_industry_l1_code)...")
    t0 = time.time()
    ind_series = panel.groupby("stock_code")["sw_industry_l1_code"].last()
    industry_codes = ind_series.reindex(stock_codes).to_numpy(dtype=object)
    n_with_ind = (industry_codes != None).sum()  # noqa: E711
    n_total = len(industry_codes)
    print(f"  {n_with_ind}/{n_total} 只股票有行业, "
          f"{pd.Series(industry_codes).nunique()} 个行业, "
          f"耗时 {time.time()-t0:.0f}s")

    # 空头池
    sp_df = compute_short_pool(panel)
    sp_df = sp_df[sp_df["is_short_pool"]]
    short_pool_by_date = {
        ts: set(grp["stock_code"].tolist())
        for ts, grp in sp_df.groupby("date")
    }

    # 回测日期集合 (在 cal 中过滤到 BACKTEST_START~END)
    bt_lo = pd.Timestamp(BACKTEST_START)
    bt_hi = pd.Timestamp(BACKTEST_END)
    backtest_dates = [
        ts for ts in cal_index.tolist()
        if bt_lo <= ts <= bt_hi
    ]
    print(f"回测交易日数: {len(backtest_dates)}")

    return {
        "cal": cal,
        "stock_codes": stock_codes,
        "industry_codes": industry_codes,
        "log_price_wide": log_price_wide,
        "log_mv_wide": log_mv_wide,
        "ret_wide": ret_wide,
        "short_pool_by_date": short_pool_by_date,
        "open_wide": open_wide,
        "close_wide": close_wide,
        "status_wide": status_wide,
        "backtest_dates": backtest_dates,
    }


def load_barra_arrays(stock_codes: np.ndarray, cal: np.ndarray, need_cov: bool):
    """加载 Barra CNE6 风格因子暴露 (+可选协方差), 构造与 (cal, stock_codes) 对齐的数组。

    返回:
        expo: (T, S, K) float32  K=20 风格因子, 缺失为 NaN
        cov:  (T, K, K) float64 或 None (need_cov=True 时返回)
        name_to_idx: dict 风格因子名 -> K 维列索引
    用于 dedup_rank="barra_dist": 配对距离 = 两腿风格暴露向量之差的 (欧氏 / 协方差加权) 范数。
    """
    K = BARRA_STYLE_FACTORS
    name_to_idx = {f: i for i, f in enumerate(K)}
    cal_idx = pd.DatetimeIndex(cal)
    d2i = {d: i for i, d in enumerate(cal_idx)}
    c2i = {c: i for i, c in enumerate(stock_codes)}
    T, S = len(cal_idx), len(stock_codes)

    print(f"加载 Barra 风格暴露: {_BARRA_EXPO_PATH}")
    t0 = time.time()
    exp = pd.read_pickle(_BARRA_EXPO_PATH)
    exp["date"] = pd.to_datetime(exp["date"])
    exp["stock_code"] = exp["stock_code"].astype(str).str.zfill(6)
    exp["di"] = exp["date"].map(d2i)
    exp["ci"] = exp["stock_code"].map(c2i)
    exp = exp.dropna(subset=["di", "ci"])
    di = exp["di"].astype(int).to_numpy(); ci = exp["ci"].astype(int).to_numpy()
    expo = np.full((T, S, len(K)), np.nan, dtype=np.float32)
    for k, f in enumerate(K):
        expo[di, ci, k] = exp[f].to_numpy(dtype=np.float32)
    print(f"  暴露数组 {expo.shape}, 耗时 {time.time()-t0:.0f}s")

    cov = None
    if need_cov:
        print(f"加载 Barra 协方差: {_BARRA_COV_PATH}")
        t0 = time.time()
        cv = pd.read_pickle(_BARRA_COV_PATH)
        cv["date"] = pd.to_datetime(cv["date"])
        cv = cv[cv["factor_name"].isin(K)].copy()
        cv["di"] = cv["date"].map(d2i)
        cv = cv.dropna(subset=["di"])
        cv["ri"] = cv["factor_name"].map(name_to_idx)
        di_c = cv["di"].astype(int).to_numpy(); ri_c = cv["ri"].astype(int).to_numpy()
        cov = np.full((T, len(K), len(K)), np.nan, dtype=np.float64)
        for cj, f in enumerate(K):
            cov[di_c, ri_c, cj] = cv[f].to_numpy(dtype=np.float64)
        print(f"  协方差数组 {cov.shape}, 耗时 {time.time()-t0:.0f}s")

    return expo, cov, name_to_idx


# =====================================================================
# 实验列表 + main
# =====================================================================

EXPERIMENTS = [
    # baseline (signal = kappa * (mu - X_t), 日频换仓)
    {"name": "OU-P00",     "train_window": 252, "signal_type": "kappa_dev", "holding_period": 1},
    {"name": "OU-P00-120", "train_window": 120, "signal_type": "kappa_dev", "holding_period": 1},
    {"name": "OU-P00-60",  "train_window":  60, "signal_type": "kappa_dev", "holding_period": 1},
    # zscore 对照 (signal = kappa * (mu - X_t) / sigma_OU, 日频换仓)
    {"name": "OU-P00-zs",     "train_window": 252, "signal_type": "zscore", "holding_period": 1},
    {"name": "OU-P00-120-zs", "train_window": 120, "signal_type": "zscore", "holding_period": 1},
    {"name": "OU-P00-60-zs",  "train_window":  60, "signal_type": "zscore", "holding_period": 1},
    # 5 日轮换持仓对照 (信号每日生成,持仓每 5 日轮换)
    {"name": "OU-P00-h5",        "train_window": 252, "signal_type": "kappa_dev", "holding_period": 5},
    {"name": "OU-P00-120-h5",    "train_window": 120, "signal_type": "kappa_dev", "holding_period": 5},
    {"name": "OU-P00-zs-h5",     "train_window": 252, "signal_type": "zscore",    "holding_period": 5},
    {"name": "OU-P00-120-zs-h5", "train_window": 120, "signal_type": "zscore",    "holding_period": 5},
    # zscore 去 kappa 对照 (signal = (mu - X_t) / sigma_OU, 与 zs-h5 唯一区别: 跨 pair 不再被 kappa 加权)
    {"name": "OU-P00-zs-noK-h5",     "train_window": 252, "signal_type": "zscore_nok", "holding_period": 5},
    {"name": "OU-P00-120-zs-noK-h5", "train_window": 120, "signal_type": "zscore_nok", "holding_period": 5},
    # simple 对照 (signal = (mean_X - X) / std_X, 完全不用 OU kappa/mu/sigma)
    # simple:       保留 hl ∈ [5, 30] 过滤
    # simple_no_hl: 不保留 hl 过滤 (只靠 b > 0 + ADF)
    {"name": "OU-P00-simple-h5",      "train_window": 252, "signal_type": "simple",       "holding_period": 5},
    {"name": "OU-P00-simple-noHL-h5", "train_window": 252, "signal_type": "simple_no_hl", "holding_period": 5},
    # 5 折 CV 对照 (训练窗内 K 折 CV, cv_mean_ic 替代 valid_rank_ic 选 pair)
    # 与 OU-P00-zs-h5 严格对偶, 只换"行业内贪心去重"的排序基准 (zscore 信号 + h5)
    {"name": "OU-P00-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5},
    # 训练窗 120 日对照 (在 OU-P00-zs-cv5-h5 基础上, 仅 train_window 改为 120)
    {"name": "OU-P00-120-zs-cv5-h5", "train_window": 120, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5},
    # 相似度预筛对照 (行业内 top X% 高相似度 pair, 其余设定与 zs-cv5-h5 一致)
    # 自 2026-05-08 起新实验默认 cv_folds=5
    # 3 种相似度 (Pearson / Spearman / SSD) × 2 种比例 (50% / 30%) = 6 个实验
    {"name": "OU-P-CRP50-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "corr_filter_method": "pearson",  "corr_filter_pct": 0.50},
    {"name": "OU-P-CRP30-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "corr_filter_method": "pearson",  "corr_filter_pct": 0.30},
    {"name": "OU-P-CRS50-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "corr_filter_method": "spearman", "corr_filter_pct": 0.50},
    {"name": "OU-P-CRS30-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "corr_filter_method": "spearman", "corr_filter_pct": 0.30},
    {"name": "OU-P-CRD50-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "corr_filter_method": "ssd",      "corr_filter_pct": 0.50},
    {"name": "OU-P-CRD30-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "corr_filter_method": "ssd",      "corr_filter_pct": 0.30},
    # 概念聚类对照 (跨行业, 共享 top 5 unmarket_norm_score 概念) + cv5
    # 行业内贪心去重 → 改为 全市场贪心去重 (在 process_one_section_concept 内自动)
    {"name": "OU-P-CC5-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "pairing_method": "shared_top_theme"},
    # NZC 零穿越筛选对照 (行业内, OU 后按 NZC 降序保留 top X%, 其余设定与 zs-cv5-h5 一致)
    # NZC = 训练窗 252 日 (X - mean(X)) 的零穿越次数, 越大越频繁回归均值
    {"name": "OU-P-NZC50-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "nzc_filter_pct": 0.50},
    {"name": "OU-P-NZC30-zs-cv5-h5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "nzc_filter_pct": 0.30},
    # D5 对照 (每只股票最多 5 pair, 净额化后按净敞口加权回测)
    # 在 OU-P00-zs-cv5-h5 / OU-P00-120-zs-cv5-h5 基础上, 仅放宽 dedup_max_per_stock=5
    {"name": "OU-P00-zs-cv5-h5-D5", "train_window": 252, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "dedup_max_per_stock": 5},
    {"name": "OU-P00-120-zs-cv5-h5-D5", "train_window": 120, "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5, "dedup_max_per_stock": 5},
    # 全市场 + 半年更新 pair_set 对照 (GLB-semi)
    # 与 OU-P00-zs-cv5-h5 区别: 全市场枚举 (非行业内) + pair_set 半年重选 (非每日)
    {"name": "OU-P-GLB-semi-zs-cv5-h5", "train_window": 252,
     "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5,
     "pairing_method": "global_semi"},
    # 滚动 30 日 mean/std 对照 (signal_today = kappa_252 × (mean_30 - X_t) / std_30)
    # 与 OU-P00-zs-cv5-h5 区别: 信号日 mu/sigma 用前 30 个交易日的滚动 mean/std 替代 OU 训练窗 mu/sigma_OU
    # 仅影响 signal_today, 验证段 / CV / OU 过滤 / 贪心去重 / pair 池子完全一致
    {"name": "OU-P00-zs-cv5-m30-h5", "train_window": 252,
     "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5,
     "predict_window": 30},
    # 前10日均值(仅mu)对照 (signal_today = kappa_252 × (mean_10 - X_t) / sigma_OU)
    # 与 OU-P00-zs-cv5-h5 区别: 信号日 mu 用前10日滚动均值替代 OU mu; sigma/kappa 仍用 OU 训练窗值。
    # 与 m30 区别: m30 同时换 mu 和 sigma, 本实验只换 mu。仅影响 signal_today(top选择+方向)。
    {"name": "OU-P00-zs-cv5-mu10-h5", "train_window": 252,
     "signal_type": "zscore", "holding_period": 5,
     "cv_folds": 5, "cv_label_h": 5,
     "predict_window": 10, "predict_window_mu_only": True},
    # Barra 风格距离贪心去重对照 (在 OU-P00-zs-cv5-h5 基础上, 仅把行业内贪心去重排序键
    # 由 cv_mean_ic 改为"两腿 Barra 风格暴露距离升序", cv_mean_ic 破平局; 初筛/候选池/top20 不变)
    # 4 变体: 全20风格 / 回撤子集 × 欧氏 / 协方差加权(ΔxᵀΣΔx=配对因子风险)
    {"name": "OU-P-BARRADIST-all-eu-cv5-h5", "train_window": 252, "signal_type": "zscore",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5,
     "dedup_rank": "barra_dist", "barra_factors": BARRA_STYLE_FACTORS, "barra_metric": "euclid"},
    {"name": "OU-P-BARRADIST-all-mh-cv5-h5", "train_window": 252, "signal_type": "zscore",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5,
     "dedup_rank": "barra_dist", "barra_factors": BARRA_STYLE_FACTORS, "barra_metric": "mahal"},
    {"name": "OU-P-BARRADIST-sub-eu-cv5-h5", "train_window": 252, "signal_type": "zscore",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5,
     "dedup_rank": "barra_dist", "barra_factors": BARRA_SUBSET_FACTORS, "barra_metric": "euclid"},
    {"name": "OU-P-BARRADIST-sub-mh-cv5-h5", "train_window": 252, "signal_type": "zscore",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5,
     "dedup_rank": "barra_dist", "barra_factors": BARRA_SUBSET_FACTORS, "barra_metric": "mahal"},
    # ─── DEV 系列 (σ 不放分母, 直接用 |μ-X| 或 |μ-X|·σ_短窗 排序) ───
    # 设计动机: 实证发现"σ 越准, 当分母效果越差"(短窗 σ 引入 B 类陷阱),
    # 且 σ 自身 mean-reverting + σ 下降伴随 X 朝 μ 移动, 说明高 σ pair 才是回归潜力大的.
    # 故反向使用 σ: 把它当成正向加权因子, 而非标准化的分母.
    # 公式:
    #   abs_dev:          signal = μ-X                       (σ 完全不参与)
    #   abs_dev_sqrt_sN:  signal = (μ-X) · √σ_N              (短窗 σ 弱加权)
    #   abs_dev_sN:       signal = (μ-X) · σ_N               (短窗 σ 等权加权)
    # σ 短窗用训练窗 252 日的末 20/30 日, 不重新拉数据.
    {"name": "OU-P-DEV-cv5-h5",      "train_window": 252, "signal_type": "abs_dev",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5},
    {"name": "OU-P-DEV-SQ20-cv5-h5", "train_window": 252, "signal_type": "abs_dev_sqrt_s20",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5},
    {"name": "OU-P-DEV-S20-cv5-h5",  "train_window": 252, "signal_type": "abs_dev_s20",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5},
    {"name": "OU-P-DEV-SQ30-cv5-h5", "train_window": 252, "signal_type": "abs_dev_sqrt_s30",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5},
    {"name": "OU-P-DEV-S30-cv5-h5",  "train_window": 252, "signal_type": "abs_dev_s30",
     "holding_period": 5, "cv_folds": 5, "cv_label_h": 5},
]


def main():
    smoke = (os.environ.get("SMOKE") == "1")
    print(f"\n{'='*60}")
    print(f"OU 配对实验  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}")
    print(f"  日志: {LOG_PATH}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  SMOKE 模式: {'是' if smoke else '否'}")
    print(f"  生效参数: ADF_P_MAX={ADF_P_MAX}, "
          f"half_life∈[{HALF_LIFE_MIN},{HALF_LIFE_MAX}], "
          f"TOP_PCT={TOP_PCT}, VALID_WINDOW={VALID_WINDOW}")
    print(f"{'='*60}")

    data = prepare_data()

    # 按需加载 theme 数据 (仅当有 pairing_method=="shared_top_theme" 实验时)
    theme_top5_arr = None
    need_theme = any(
        e.get("pairing_method") == "shared_top_theme"
        and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "metrics.json"))
        for e in EXPERIMENTS
    )
    if need_theme and not smoke:
        from _pair_runner import load_theme_top5_array
        theme_top5_arr = load_theme_top5_array(
            stock_codes=data["stock_codes"],
            dates=data["cal"],
            top_k=5,
            score_col="unmarket_norm_score",
        )

    if smoke:
        # 冒烟: 只跑 OU-P00 的前 SMOKE_N 个截面 (默认 10),打印漏斗诊断后退出
        smoke_n = int(os.environ.get("SMOKE_N", "10"))
        smoke_signal = os.environ.get("SMOKE_SIGNAL_TYPE", "kappa_dev")
        print(f"\n{'='*60}\nSMOKE: OU-P00 前 {smoke_n} 个截面冒烟 (signal_type={smoke_signal})\n{'='*60}")
        cal = data["cal"]
        cal_ts = pd.DatetimeIndex(cal)
        date_to_idx = {dt: i for i, dt in enumerate(cal_ts)}
        sect_dates = data["backtest_dates"][:smoke_n]
        rows = []
        for sig_date in sect_dates:
            if sig_date not in date_to_idx:
                continue
            t_idx = date_to_idx[sig_date]
            sp_set = data["short_pool_by_date"].get(sig_date, set())
            out = process_one_section(
                section_idx=t_idx,
                cal=cal,
                log_price_wide=data["log_price_wide"],
                ret_wide=data["ret_wide"],
                industry_codes=data["industry_codes"],
                stock_codes=data["stock_codes"],
                short_pool_set=sp_set,
                train_window=252,
                valid_window=VALID_WINDOW,
                signal_type=smoke_signal,
                verbose=True,
            )
            rows.append({
                "date": pd.Timestamp(sig_date).date(),
                "n_total":   out.get("n_pairs_total", 0),
                "n_ou_pass": out.get("n_pairs_ou_pass", 0),
                "n_dedup":   out.get("n_pairs_dedup", 0),
                "n_legal":   out.get("n_pairs_legal", 0),
                "n_top20":   out.get("n_pairs_top20", 0),
                "n_long":    out.get("n_long", 0),
                "n_short":   out.get("n_short", 0),
                "elapsed":   round(out.get("elapsed_sec", 0.0), 1),
            })
        print()
        print("=" * 90)
        print("漏斗逐日明细 (前 {} 截面):".format(len(rows)))
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print("=" * 90)
        print("汇总统计:")
        for col in ["n_total", "n_ou_pass", "n_dedup", "n_legal",
                    "n_top20", "n_long", "n_short", "elapsed"]:
            s = df[col]
            print(f"  {col:>10s}: 平均 {s.mean():>9.1f}  "
                  f"中位 {s.median():>9.1f}  "
                  f"范围 [{s.min()}, {s.max()}]")
        # 保存 CSV 方便后续看趋势
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        smoke_csv = os.path.join(OUTPUT_DIR, "smoke_funnel.csv")
        df.to_csv(smoke_csv, index=False)
        print(f"\n已保存: {smoke_csv}")
        print("=" * 90)
        return

    # 正式运行 (EXP_ONLY 环境变量可指定仅跑某些实验, 逗号分隔; 避免误跑其它未完成实验)
    exp_only = set(x.strip() for x in os.environ.get("EXP_ONLY", "").split(",") if x.strip())

    # 按需加载 Barra 风格因子 (仅当待跑实验里有 dedup_rank=="barra_dist")
    barra_expo = barra_cov = barra_name_idx = None
    _barra_exps = [e for e in EXPERIMENTS
                   if e.get("dedup_rank") == "barra_dist"
                   and (not exp_only or e["name"] in exp_only)
                   and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "metrics.json"))]
    if _barra_exps:
        need_cov = any(e.get("barra_metric", "euclid") == "mahal" for e in _barra_exps)
        barra_expo, barra_cov, barra_name_idx = load_barra_arrays(
            data["stock_codes"], data["cal"], need_cov=need_cov)

    for exp in EXPERIMENTS:
        if exp_only and exp["name"] not in exp_only:
            continue
        _is_barra = exp.get("dedup_rank") == "barra_dist"
        _bidx = None
        if _is_barra:
            _facs = exp.get("barra_factors", BARRA_STYLE_FACTORS)
            _bidx = np.array([barra_name_idx[f] for f in _facs], dtype=int)
        run_one_experiment(
            exp_name=exp["name"],
            train_window=exp["train_window"],
            cal=data["cal"],
            log_price_wide=data["log_price_wide"],
            ret_wide=data["ret_wide"],
            stock_codes=data["stock_codes"],
            industry_codes=data["industry_codes"],
            short_pool_by_date=data["short_pool_by_date"],
            open_wide=data["open_wide"],
            close_wide=data["close_wide"],
            status_wide=data["status_wide"],
            backtest_dates=data["backtest_dates"],
            signal_type=exp.get("signal_type", "kappa_dev"),
            holding_period=exp.get("holding_period", 1),
            cv_folds=exp.get("cv_folds", 0),
            cv_label_h=exp.get("cv_label_h", 5),
            corr_filter_method=exp.get("corr_filter_method", "none"),
            corr_filter_pct=exp.get("corr_filter_pct", 1.0),
            nzc_filter_pct=exp.get("nzc_filter_pct", 1.0),
            pairing_method=exp.get("pairing_method", "industry"),
            theme_top5_arr=theme_top5_arr,
            dedup_max_per_stock=exp.get("dedup_max_per_stock", 1),
            min_unique_per_side=exp.get("min_unique_per_side", 50),
            predict_window=exp.get("predict_window", 0),
            predict_window_mu_only=exp.get("predict_window_mu_only", False),
            dedup_rank=exp.get("dedup_rank", "cv_mean_ic"),
            barra_expo=barra_expo if _is_barra else None,
            barra_cov=barra_cov if _is_barra else None,
            barra_factor_idx=_bidx,
            barra_metric=exp.get("barra_metric", "euclid"),
        )
        gc.collect()

    print(f"\n{'='*60}\n所有实验完成\n{'='*60}")


if __name__ == "__main__":
    _install_loggers()
    main()
