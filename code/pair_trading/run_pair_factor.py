#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pair Trading 因子实验 (baseline, n=20)

因子逻辑:
    对每只股票 s 在截面日 t:
    1. 取 s 的 [t-20, t-1] 特质收益率序列 ret_s  (20天)
    2. 取所有股票 j 的 [t-40, t-21] 特质收益率序列 ret_j (20天)
    3. corr(ret_s, ret_j) → 只保留正相关部分，softmax 归一化为权重
    4. 用权重对各股票 j 在 [t-20, t-1] 的累计收益进行加权 → s 在 t 的因子值

回测:
    - 日频调仓，十分位分组
    - 调仓时过滤涨跌停、停牌
    - 对头组进行持仓回测，计算超额收益
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

# ── 引入 func/model_backtest_framework 以对齐 autopilot 的头组回测口径 ──
_FUNC_DIR = "/root/quant/xgbcode/func"
if _FUNC_DIR not in sys.path:
    sys.path.insert(0, _FUNC_DIR)
from model_backtest_framework import analyze_model_holdings

# ── 通用骨架 (与 run_ou_pair.py 共用,见 _pair_runner.py) ──
_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
if _FACTOR1_DIR not in sys.path:
    sys.path.insert(0, _FACTOR1_DIR)
from _pair_runner import (
    _tau_to_pvalue, _batch_adf_pvalues,
    build_tradable_mask,
    group_backtest, head_group_backtest,
    calc_stats,
    plot_group_backtest, plot_head_backtest, plot_yearly_head,
)

# ── 路径 ──
os.chdir("/root/quant")

# ── 常量 ──
N = 20                      # 回看窗口
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2025-12-31"
N_GROUPS = 10
COMMISSION_RATE = 0.0007    # 单边手续费
LIMIT_THRESHOLD = 0.095     # 涨跌停判断阈值（日收益率 >= 9.5%）

DATA_DIR = "Data/all"
OUTPUT_DIR = "output/0416_pair_factor"
SPRET_PATH = os.path.join(DATA_DIR, "specific_ret_cne6_sw21.pkl")
SPRET_NON_ST_PATH = os.path.join(DATA_DIR, "s_ret_cne6_non_st.pkl")
PRICE_PATH = os.path.join(DATA_DIR, "panel_trade.pkl")
PRICE_NON_ST_PATH = os.path.join(DATA_DIR, "price_non_st.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 日志 
LOG_PATH = os.path.join("output", "0416_pair_factor_run.log")

class _Logger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "a", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

sys.stdout = _Logger(LOG_PATH)
sys.stderr = _Logger(LOG_PATH)


# =====================================================================
# 因子计算（矩阵化，高性能）
# =====================================================================

def build_pair_factor(
    spret_df: pd.DataFrame,
    n: int = 20,
    calc_every: int = 1,
    weight_window: str = "recent",
    corr_topk: float = 1.0,
    lag: int = 0,
) -> pd.DataFrame:
    """
    矩阵化计算 pair trading 因子。

    参数:
        spret_df:      包含 [stock_code, date, spret]
        n:             回看窗口长度
        calc_every:    每隔多少天计算一次因子（1=日频, 5=周频）
        weight_window: 加权收益的窗口
                       "recent"  — j 在 [t-n, t-1] 的累计收益 (BL-01)
                       "post5"   — j 在 [t-n, t-n+5] 的累计收益 (BL-02)
                       "lag"     — j 在 [t-lag, t-lag+calc_every] 的累计收益
        corr_topk:     只保留正相关中排名前 topk 比例的配对 (1.0=全部正相关)
        lag:           j 相对 s 的领先天数 (0=无偏移)
                       lag>0 时: rj = mat[t-2n-lag : t-n-lag], 加权窗口也相应前移

    输出: DataFrame [date, stock_code, factor]
    """
    topk_desc = f"top{int(corr_topk*100)}%" if corr_topk < 1.0 else "全部"
    lag_desc = f"lag={lag}" if lag > 0 else "no-lag"
    print(f"构建 pair trading 因子 (weight={weight_window}, every={calc_every}d, "
          f"corr={topk_desc}, {lag_desc}) ...")
    t0 = time.time()

    pivot = spret_df.pivot(index="date", columns="stock_code", values="spret")
    pivot = pivot.sort_index()
    all_dates = pivot.index.values
    all_stocks = pivot.columns.values
    mat = pivot.values.astype(np.float64)  # (T, S)

    T, S = mat.shape
    print(f"  矩阵大小: {T} 天 × {S} 只股票")

    log_ret = np.log1p(mat / 100.0)
    cum_log = np.nancumsum(log_ret, axis=0)  # (T, S)

    backtest_start_dt = pd.Timestamp(BACKTEST_START)
    bt_end_dt = pd.Timestamp(BACKTEST_END)
    min_idx = 2 * n + lag

    bt_mask = all_dates >= backtest_start_dt
    if not bt_mask.any():
        raise ValueError(f"数据中没有 >= {BACKTEST_START} 的日期")

    results = []
    n_calc = 0
    calc_count = 0

    for t_idx in range(min_idx, T):
        dt = all_dates[t_idx]
        if dt < backtest_start_dt or dt > bt_end_dt:
            continue

        # 周频: 只在每 calc_every 个截面日计算
        calc_count += 1
        if (calc_count - 1) % calc_every != 0:
            continue

        rs = mat[t_idx - n: t_idx, :]
        rj_start = t_idx - 2 * n - lag
        rj_end = t_idx - n - lag
        rj = mat[rj_start: rj_end, :]

        rs_valid = np.sum(~np.isnan(rs), axis=0) >= n // 2
        rj_valid = np.sum(~np.isnan(rj), axis=0) >= n // 2

        rs_clean = np.where(np.isnan(rs), 0.0, rs)
        rj_clean = np.where(np.isnan(rj), 0.0, rj)

        # ── 矩阵化 Pearson 相关性 ──
        rs_count = np.sum(~np.isnan(mat[t_idx - n: t_idx, :]), axis=0).clip(min=1)
        rs_mean = np.nansum(mat[t_idx - n: t_idx, :], axis=0) / rs_count
        rs_c = rs_clean - rs_mean[np.newaxis, :]
        rs_c = np.where(np.isnan(mat[t_idx - n: t_idx, :]), 0.0, rs_c)

        rj_count = np.sum(~np.isnan(mat[rj_start: rj_end, :]), axis=0).clip(min=1)
        rj_mean = np.nansum(mat[rj_start: rj_end, :], axis=0) / rj_count
        rj_c = rj_clean - rj_mean[np.newaxis, :]
        rj_c = np.where(np.isnan(mat[rj_start: rj_end, :]), 0.0, rj_c)

        rs_norm = np.sqrt(np.sum(rs_c ** 2, axis=0)).clip(min=1e-10)
        rj_norm = np.sqrt(np.sum(rj_c ** 2, axis=0)).clip(min=1e-10)

        dot_prod = rs_c.T @ rj_c
        corr_mat = dot_prod / (rs_norm[:, np.newaxis] * rj_norm[np.newaxis, :])

        np.fill_diagonal(corr_mat, 0.0)
        corr_mat[:, ~rj_valid] = 0.0
        corr_mat = np.maximum(corr_mat, 0.0)

        # 只保留每行正相关中排名前 corr_topk 的配对
        if corr_topk < 1.0:
            n_pos_per_row = (corr_mat > 0).sum(axis=1)  # (S,)
            k_per_row = np.maximum(1, (n_pos_per_row * corr_topk).astype(int))
            # 对每行按降序排序，找到第 k 大的值作为阈值
            sorted_desc = np.sort(corr_mat, axis=1)[:, ::-1]  # (S, S) 降序
            thresholds = sorted_desc[np.arange(S), k_per_row - 1]  # (S,)
            corr_mat[corr_mat < thresholds[:, np.newaxis]] = 0.0

        # ── 加权目标收益 ──
        if weight_window == "lag":
            # j 在 [t-lag, t-lag+calc_every) 的累计收益
            w_start = t_idx - lag
            w_end = min(t_idx - lag + calc_every, T) - 1
            if w_start - 1 >= 0:
                cum_ret_j = np.exp(cum_log[w_end, :] - cum_log[w_start - 1, :]) - 1
            else:
                cum_ret_j = np.exp(cum_log[w_end, :]) - 1
        elif weight_window == "post5":
            # j 在 [t-n, t-n+5) 的累计收益
            w_start = t_idx - n
            w_end = min(t_idx - n + 5, T) - 1
            if w_start - 1 >= 0:
                cum_ret_j = np.exp(cum_log[w_end, :] - cum_log[w_start - 1, :]) - 1
            else:
                cum_ret_j = np.exp(cum_log[w_end, :]) - 1
        else:
            # j 在 [t-n, t-1] 的累计收益
            if t_idx - n - 1 >= 0:
                cum_ret_j = np.exp(cum_log[t_idx - 1, :] - cum_log[t_idx - n - 1, :]) - 1
            else:
                cum_ret_j = np.exp(cum_log[t_idx - 1, :]) - 1

        cum_ret_j = np.where(np.isnan(cum_ret_j), 0.0, cum_ret_j)

        # ── softmax 归一化 + 加权 ──
        row_max = np.max(corr_mat, axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 0.0)
        exp_corr = np.exp(corr_mat - row_max)
        exp_corr = np.where(corr_mat > 0, exp_corr, 0.0)
        row_sum = exp_corr.sum(axis=1, keepdims=True).clip(min=1e-10)
        weights = exp_corr / row_sum

        factor_vals = weights @ cum_ret_j
        factor_vals[~rs_valid] = np.nan

        valid_mask = ~np.isnan(factor_vals)
        if valid_mask.any():
            day_df = pd.DataFrame({
                "date": dt,
                "stock_code": all_stocks[valid_mask],
                "factor": factor_vals[valid_mask],
            })
            results.append(day_df)

        n_calc += 1
        if n_calc % 50 == 0:
            elapsed = time.time() - t0
            print(f"  已计算 {n_calc} 个截面, 耗时 {elapsed:.0f}s")

    if not results:
        raise RuntimeError("因子计算结果为空")

    factor_df = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  因子计算完成: {len(factor_df)} 条, "
          f"{factor_df['date'].nunique()} 个截面, "
          f"{factor_df['stock_code'].nunique()} 只股票, "
          f"耗时 {elapsed:.0f}s")

    return factor_df


# =====================================================================
# Granger 因果因子 (GL-01)
# =====================================================================

def _spearman_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """对 a(W,Sa) 和 b(W,Sb) 计算 Spearman 秩相关矩阵 (Sa, Sb)。
    要求 a, b 均无 NaN (由调用方保证通过预筛)。
    """
    from scipy.stats import rankdata
    a_rank = rankdata(a, axis=0).astype(np.float64)  # 平均秩, 精确处理 ties
    b_rank = rankdata(b, axis=0).astype(np.float64)
    a_c = a_rank - a_rank.mean(axis=0)
    b_c = b_rank - b_rank.mean(axis=0)
    a_norm = np.sqrt((a_c ** 2).sum(axis=0)).clip(min=1e-10)
    b_norm = np.sqrt((b_c ** 2).sum(axis=0)).clip(min=1e-10)
    return (a_c.T @ b_c) / (a_norm[:, np.newaxis] * b_norm[np.newaxis, :])


def _batch_granger_pvals_residual(
    y: np.ndarray,        # (W,) 因变量 ret_s[τ], 无 NaN
    y_lag1: np.ndarray,   # (W,) 自滞后 ret_s[τ-1], 无 NaN
    X_cand: np.ndarray,   # (W, K) 候选 K 只 j 的 ret_j[τ-lag], 无 NaN
) -> np.ndarray:
    """
    残差法批量计算 K 个候选的 Granger F 检验 p 值。
    
    受限模型: y = α₀ + β₀ y_lag1
    无限制模型: y = α₁ + β₁ y_lag1 + γ x_j
    
    使用残差化技巧:
        1. 把 y 和 X_cand 都对 [1, y_lag1] 做残差 (投影到正交补空间)
        2. 偏相关 ρ = (ỹ·x̃_j) / (‖ỹ‖·‖x̃_j‖)
        3. F = ρ² · (W-3) / (1-ρ²), p = 1 - F.cdf(F_val, 1, W-3)
    
    返回 shape=(K,) 的 p 值数组。
    """
    from scipy.stats import f as f_dist

    W = len(y)
    K = X_cand.shape[1]
    df1, df2 = 1, W - 3
    if df2 <= 0:
        return np.ones(K)

    # 防御: y_lag1 退化(全相同/全0)时受限模型奇异
    if y_lag1.std() < 1e-10:
        return np.ones(K)

    # 构造 [1, y_lag1] 的残差矩阵 M_r = I - X_r(X_r'X_r)^(-1)X_r'
    X_r = np.column_stack([np.ones(W), y_lag1])   # (W, 2)
    try:
        XtX_inv = np.linalg.inv(X_r.T @ X_r)      # (2, 2)
    except np.linalg.LinAlgError:
        return np.ones(K)
    P_coef = XtX_inv @ X_r.T                       # (2, W)

    # y 残差
    beta_y = P_coef @ y                            # (2,)
    y_tilde = y - X_r @ beta_y                     # (W,)
    ssr_r = float(y_tilde @ y_tilde)

    # X_cand 每列残差 (批量)
    beta_X = P_coef @ X_cand                       # (2, K)
    X_tilde = X_cand - X_r @ beta_X                # (W, K)

    # 偏相关
    yx = y_tilde @ X_tilde                         # (K,)
    xx = (X_tilde ** 2).sum(axis=0)                # (K,)

    denom = ssr_r * xx
    rho_sq = np.where(denom > 1e-12, yx ** 2 / denom.clip(min=1e-12), 0.0)
    rho_sq = np.clip(rho_sq, 0.0, 1.0 - 1e-10)     # 避免 1/0

    F_vals = rho_sq * df2 / (1.0 - rho_sq)
    p_vals = 1.0 - f_dist.cdf(F_vals, df1, df2)

    return p_vals


def build_granger_factor(
    spret_df: pd.DataFrame,
    n: int = 60,
    lags=(1, 3, 5, 8, 10),
    topk1: int = 100,
    granger_p: float = 0.05,
    topk2: int = 5,
    topk_list=None,
) -> pd.DataFrame:
    """
    Granger 因果因子 (GL-01)。

    流程:
        Step0: 预筛窗口内无 NaN 的有效股票
        Step1: 对每个 lag 用 Spearman 秩相关初筛 top-K₁ 候选 j
        Step2: 残差法批量 Granger F 检验, 保留 p < granger_p
        Step3: 同一 j 在多 lag 下显著则保留最小 p 值的 lag
        Step4: 按 p 值升序取 top-K₂, 做 OLS 多元回归
        Step5: 用拟合方程预测 ret_s[t] 作为因子值

    参数 topk_list: 若提供 (如 (1,2,3)), 则同时输出多个 topk 下的预测结果,
        返回宽表含列 factor_top{k}。此时 topk2 被忽略, 筛选门槛对每个 k 独立
        (即 len(sig_pairs) >= k 才产生对应 factor_top{k})。
        若为 None, 则保持原行为 (返回单列 factor, 门槛=topk2)。
    """
    if topk_list is None:
        topk_list_eff = None
        print(f"构建 Granger 因果因子 (n={n}, lags={list(lags)}, "
              f"topk1={topk1}, p<{granger_p}, topk2={topk2}) ...")
    else:
        topk_list_eff = tuple(sorted(int(k) for k in topk_list))
        print(f"构建 Granger 因果因子 (宽表) (n={n}, lags={list(lags)}, "
              f"topk1={topk1}, p<{granger_p}, topk_list={list(topk_list_eff)}) ...")
    t0 = time.time()

    pivot = spret_df.pivot(index="date", columns="stock_code", values="spret")
    pivot = pivot.sort_index()
    all_dates = pivot.index.values
    all_stocks = pivot.columns.values
    mat = pivot.values.astype(np.float64)  # (T, S)
    T, S = mat.shape
    print(f"  矩阵大小: {T} 天 × {S} 只股票")

    lags = tuple(lags)
    max_lag = max(lags)
    min_idx = n + max_lag + 1  # 需要 ret_s[τ-1] 和 ret_j[τ-lag]

    backtest_start_dt = pd.Timestamp(BACKTEST_START)
    bt_end_dt = pd.Timestamp(BACKTEST_END)

    results = []
    n_calc = 0

    for t_idx in range(min_idx, T):
        dt = all_dates[t_idx]
        if dt < backtest_start_dt or dt > bt_end_dt:
            continue

        # ── Step 0: 预筛有效股票 (窗口 [t-max_lag-n-1, t-1] 内严格无 NaN 且非退化) ──
        check_start = t_idx - n - max_lag - 1
        check_window = mat[check_start: t_idx, :]                          # (71, S)
        no_nan = ~np.isnan(check_window).any(axis=0)                       # (S,)
        # 非退化: 窗口内方差 > 1e-8 (排除全 0 或全相同的停牌段)
        with np.errstate(invalid='ignore'):
            non_degenerate = np.nanstd(check_window, axis=0) > 1e-6        # (S,)
        valid_mask_s = no_nan & non_degenerate
        valid_idx = np.where(valid_mask_s)[0]                              # (S_v,)
        S_v = len(valid_idx)

        if S_v < topk1 + topk2 + 10:  # 候选股票过少
            continue

        mat_v = mat[:, valid_idx]                                          # (T, S_v)

        # 共享量: y, y_lag1 (对所有 s 和 lag 不变)
        y_mat = mat_v[t_idx - n: t_idx, :]                                 # (n, S_v)
        y_lag1_mat = mat_v[t_idx - n - 1: t_idx - 1, :]                    # (n, S_v)

        # ── Step 1: 每个 lag 做 Spearman 初筛 ──
        k1 = min(topk1, S_v - 1)
        # per_lag_topk[lg]: (S_v, k1) 存 j 在 valid_idx 内的局部索引
        per_lag_topk = {}
        rs = y_mat                                                          # (n, S_v)
        for lg in lags:
            rj = mat_v[t_idx - n - lg: t_idx - lg, :]                      # (n, S_v)
            corr_mat = _spearman_corr_matrix(rs, rj)                       # (S_v, S_v)
            np.fill_diagonal(corr_mat, 0.0)                                # 排除自身
            # 每行降序 top-k1
            top_idx = np.argpartition(-corr_mat, k1, axis=1)[:, :k1]       # (S_v, k1)
            per_lag_topk[lg] = top_idx

        # ── Step 2~5: 逐只 s 计算 ──
        if topk_list_eff is None:
            factor_vals_v = np.full(S_v, np.nan)
            min_k_required = topk2
        else:
            # 宽表: 每个 k 一列
            factor_vals_v_multi = {k: np.full(S_v, np.nan) for k in topk_list_eff}
            min_k_required = min(topk_list_eff)  # 至少 len(sig_pairs) >= min_k 才需要进入后续流程

        for s_local in range(S_v):
            y = y_mat[:, s_local]
            y_lag1 = y_lag1_mat[:, s_local]

            # 防御: 若 y 退化 (全相同), 跳过
            if y.std() < 1e-10:
                continue

            # 为该 s 收集所有 lag 下的 Granger 显著候选
            sig_pairs = {}  # j_local -> (p_value, lag)

            for lg in lags:
                cand_locals = per_lag_topk[lg][s_local]                    # (k1,)
                cand_locals = cand_locals[cand_locals != s_local]          # 排除自身
                if len(cand_locals) == 0:
                    continue

                X_cand = mat_v[t_idx - n - lg: t_idx - lg, cand_locals]    # (n, k1-?)
                pvals = _batch_granger_pvals_residual(y, y_lag1, X_cand)

                for p, j_local in zip(pvals, cand_locals):
                    if p < granger_p:
                        prev = sig_pairs.get(j_local)
                        if prev is None or p < prev[0]:
                            sig_pairs[j_local] = (p, lg)

            if len(sig_pairs) < min_k_required:
                continue  # 显著配对不足, 因子为 NaN

            # ── Step 3: 按 p 升序排序 (保留全部, 后续按需截断) ──
            sorted_pairs = sorted(sig_pairs.items(), key=lambda x: x[1][0])

            if topk_list_eff is None:
                # ── 单列: 按 topk2 截断后做 OLS ──
                used_pairs = sorted_pairs[:topk2]
                X_cols = [np.ones(n), y_lag1]
                for j_local, (_, lg) in used_pairs:
                    X_cols.append(mat_v[t_idx - n - lg: t_idx - lg, j_local])
                X = np.column_stack(X_cols)
                try:
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                x_pred_list = [1.0, mat_v[t_idx - 1, s_local]]
                for j_local, (_, lg) in used_pairs:
                    x_pred_list.append(mat_v[t_idx - lg, j_local])
                x_pred = np.array(x_pred_list)
                factor_vals_v[s_local] = float(x_pred @ beta)
            else:
                # ── 宽表: 对每个 k 独立做 OLS 和预测 ──
                n_sig = len(sorted_pairs)
                base_cols = [np.ones(n), y_lag1]
                base_pred = [1.0, mat_v[t_idx - 1, s_local]]
                # 逐步累加 j 列, 避免重复构造
                X_cols_k = list(base_cols)
                pred_list_k = list(base_pred)
                cur_len = 0
                for k in topk_list_eff:
                    if n_sig < k:
                        # 该 s 显著配对数不足 k, factor_top{k} = NaN
                        continue
                    # 补齐到 k 个 j 列
                    while cur_len < k:
                        j_local, (_, lg) = sorted_pairs[cur_len]
                        X_cols_k.append(mat_v[t_idx - n - lg: t_idx - lg, j_local])
                        pred_list_k.append(mat_v[t_idx - lg, j_local])
                        cur_len += 1
                    X = np.column_stack(X_cols_k)
                    try:
                        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    x_pred = np.array(pred_list_k)
                    factor_vals_v_multi[k][s_local] = float(x_pred @ beta)

        # 收集结果 (映射回全局 stock_code)
        if topk_list_eff is None:
            valid_fac_mask = ~np.isnan(factor_vals_v)
            if valid_fac_mask.any():
                global_idx = valid_idx[valid_fac_mask]
                day_df = pd.DataFrame({
                    "date": dt,
                    "stock_code": all_stocks[global_idx],
                    "factor": factor_vals_v[valid_fac_mask],
                })
                results.append(day_df)
            n_valid_fac = int(valid_fac_mask.sum()) if valid_fac_mask.any() else 0
        else:
            # 宽表: 只要任一 k 有值就保留该行
            any_valid_mask = np.zeros(S_v, dtype=bool)
            for k in topk_list_eff:
                any_valid_mask |= ~np.isnan(factor_vals_v_multi[k])
            if any_valid_mask.any():
                global_idx = valid_idx[any_valid_mask]
                data_dict = {
                    "date": dt,
                    "stock_code": all_stocks[global_idx],
                }
                for k in topk_list_eff:
                    data_dict[f"factor_top{k}"] = factor_vals_v_multi[k][any_valid_mask]
                day_df = pd.DataFrame(data_dict)
                results.append(day_df)
            n_valid_fac = int(any_valid_mask.sum()) if any_valid_mask.any() else 0

        n_calc += 1
        if n_calc % 10 == 0:
            elapsed = time.time() - t0
            avg_t = elapsed / n_calc
            print(f"  [{n_calc}] {pd.Timestamp(dt).date()}, "
                  f"S_v={S_v}, 有效={n_valid_fac}, "
                  f"耗时 {elapsed:.0f}s (avg {avg_t:.1f}s/截面)")

    if not results:
        raise RuntimeError("因子计算结果为空")

    factor_df = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  Granger 因子计算完成: {len(factor_df)} 条, "
          f"{factor_df['date'].nunique()} 个截面, "
          f"{factor_df['stock_code'].nunique()} 只股票, "
          f"耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return factor_df


# =====================================================================
# 相关系数因子 (GL-06/07): 去掉 Granger 检验, 直接按 Spearman 相关选 1 对 (j*, lag*)
# 一次计算产出两列宽表: factor_pos (只正相关) 和 factor_abs (|相关|最大)
# =====================================================================

def build_correlation_factor(
    spret_df: pd.DataFrame,
    n: int = 60,
    lags=(1, 3, 5, 8, 10),
) -> pd.DataFrame:
    """
    相关系数 lead-lag 因子 (GL-06/07)。

    流程:
        Step0: 预筛窗口内无 NaN 且非退化的有效股票 (与 Granger 一致)
        Step1: 对每个 lag 计算 Spearman(ret_s[τ∈n], ret_j[τ-lag∈n]) 全矩阵
        Step2: 对每只 s:
            - pos 版本: 从 (M-1)*|lags| 对中选 Spearman 最大且 > 0 的 1 对 (j*, lag*)
            - abs 版本: 选 |Spearman| 最大的 1 对 (j*, lag*)
            (两个版本可能挑到不同的 j/lag, 各自独立做 OLS)
        Step3: 对 2 个候选分别做 OLS:
            ret_s[τ] = α + β₀·ret_s[τ-1] + γ·ret_j[τ-lag*] + ε
        Step4: 因子值 = α + β₀·ret_s[t-1] + γ·ret_j[t-lag*]
            (使用 t 日已知的 ret_s[t-1] 和 ret_j[t-lag*])

    返回宽表, 列: date, stock_code, factor_pos, factor_abs
    """
    print(f"构建相关系数因子 (GL-06/07 共享宽表) (n={n}, lags={list(lags)}) ...")
    t0 = time.time()

    pivot = spret_df.pivot(index="date", columns="stock_code", values="spret")
    pivot = pivot.sort_index()
    all_dates = pivot.index.values
    all_stocks = pivot.columns.values
    mat = pivot.values.astype(np.float64)
    T, S = mat.shape
    print(f"  矩阵大小: {T} 天 × {S} 只股票")

    lags = tuple(lags)
    L = len(lags)
    max_lag = max(lags)
    min_idx = n + max_lag + 1  # 需要 ret_s[τ-1] 和 ret_j[τ-lag]

    backtest_start_dt = pd.Timestamp(BACKTEST_START)
    bt_end_dt = pd.Timestamp(BACKTEST_END)

    results = []
    n_calc = 0

    for t_idx in range(min_idx, T):
        dt = all_dates[t_idx]
        if dt < backtest_start_dt or dt > bt_end_dt:
            continue

        # ── Step 0: 预筛有效股票 ──
        check_start = t_idx - n - max_lag - 1
        check_window = mat[check_start: t_idx, :]
        no_nan = ~np.isnan(check_window).any(axis=0)
        with np.errstate(invalid='ignore'):
            non_degenerate = np.nanstd(check_window, axis=0) > 1e-6
        valid_mask_s = no_nan & non_degenerate
        valid_idx = np.where(valid_mask_s)[0]
        S_v = len(valid_idx)

        if S_v < 10:
            continue

        mat_v = mat[:, valid_idx]

        # ── Step 1: 每个 lag 的 Spearman 全矩阵 (堆成 (L, S_v, S_v)) ──
        y_mat = mat_v[t_idx - n: t_idx, :]                     # (n, S_v)
        corr_stack = np.full((L, S_v, S_v), -np.inf, dtype=np.float64)
        for li, lg in enumerate(lags):
            rj = mat_v[t_idx - n - lg: t_idx - lg, :]          # (n, S_v)
            corr_mat = _spearman_corr_matrix(y_mat, rj)        # (S_v, S_v)
            np.fill_diagonal(corr_mat, -np.inf)                # 排除自身
            corr_stack[li] = corr_mat

        # (L, S_v, S_v) -> 对每只 s, 从 L*(S_v-1) 对中找极值
        # 重排: (S_v, L*S_v)
        # 每只 s 的候选矩阵: corr_stack[:, s, :].reshape(-1) = L*S_v
        # 其中 j_local = k % S_v, lag_idx = k // S_v (k 是索引)

        factor_pos = np.full(S_v, np.nan)
        factor_abs = np.full(S_v, np.nan)

        # ── Step 2~4: 逐只 s 处理 (OLS 部分无法向量化, 但 Spearman 已一次性算完) ──
        for s_local in range(S_v):
            y = y_mat[:, s_local]
            y_lag1 = mat_v[t_idx - n - 1: t_idx - 1, s_local]

            if y.std() < 1e-10 or y_lag1.std() < 1e-10:
                continue

            # (L, S_v) -> flatten
            row_corrs = corr_stack[:, s_local, :]              # (L, S_v)

            # ── pos 版本: 最大正相关 ──
            pos_max = row_corrs.max()
            pos_valid = pos_max > 0  # 必须正相关
            if pos_valid:
                flat_idx = int(np.argmax(row_corrs))
                lg_pos = lags[flat_idx // S_v]
                j_pos = flat_idx % S_v
            else:
                lg_pos = j_pos = None

            # ── abs 版本: |相关| 最大 ──
            # 屏蔽 -inf (对角/非法) 的地方的 |.| 运算
            abs_mat = np.abs(row_corrs)
            # -inf 取 abs 会变 +inf, 需重新屏蔽
            mask_finite = np.isfinite(row_corrs)
            abs_mat[~mask_finite] = -np.inf
            abs_max = abs_mat.max()
            abs_valid = abs_max > 0 and np.isfinite(abs_max)
            if abs_valid:
                flat_idx_a = int(np.argmax(abs_mat))
                lg_abs = lags[flat_idx_a // S_v]
                j_abs = flat_idx_a % S_v
            else:
                lg_abs = j_abs = None

            # 2 个版本的 OLS (若 pos 和 abs 挑到同一对, OLS 结果一致, 不再去重)
            for tag, lg, j_local, out_arr in [
                ("pos", lg_pos, j_pos, factor_pos),
                ("abs", lg_abs, j_abs, factor_abs),
            ]:
                if lg is None:
                    continue
                x_j = mat_v[t_idx - n - lg: t_idx - lg, j_local]  # (n,)
                X = np.column_stack([np.ones(n), y_lag1, x_j])
                try:
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                # 因子值 = α + β₀·ret_s[t-1] + γ·ret_j[t-lg]
                x_pred = np.array([1.0, mat_v[t_idx - 1, s_local],
                                   mat_v[t_idx - lg, j_local]])
                out_arr[s_local] = float(x_pred @ beta)

        # 收集结果
        any_valid = ~np.isnan(factor_pos) | ~np.isnan(factor_abs)
        if any_valid.any():
            global_idx = valid_idx[any_valid]
            day_df = pd.DataFrame({
                "date": dt,
                "stock_code": all_stocks[global_idx],
                "factor_pos": factor_pos[any_valid],
                "factor_abs": factor_abs[any_valid],
            })
            results.append(day_df)

        n_calc += 1
        if n_calc % 10 == 0:
            elapsed = time.time() - t0
            n_pos = int((~np.isnan(factor_pos)).sum())
            n_abs = int((~np.isnan(factor_abs)).sum())
            print(f"  [{n_calc}] {pd.Timestamp(dt).date()}, "
                  f"S_v={S_v}, 有效(pos/abs)={n_pos}/{n_abs}, "
                  f"耗时 {elapsed:.0f}s (avg {elapsed/n_calc:.1f}s/截面)")

    if not results:
        raise RuntimeError("相关系数因子计算结果为空")

    factor_df = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0
    n_pos_total = int(factor_df["factor_pos"].notna().sum())
    n_abs_total = int(factor_df["factor_abs"].notna().sum())
    print(f"  相关系数因子计算完成: {len(factor_df)} 条 (pos={n_pos_total}, abs={n_abs_total}), "
          f"{factor_df['date'].nunique()} 个截面, "
          f"{factor_df['stock_code'].nunique()} 只股票, "
          f"耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return factor_df


# =====================================================================
# 协整因子 (CO-01 ~ CO-09)
# =====================================================================
# _tau_to_pvalue / _DF_TAU_C / _DF_P_LEVEL 已搬至 _pair_runner.py


def _batch_coint_regression(
    y_mat: np.ndarray,    # (n, S_v) log_p_s[t-n:t], 每列一只股票
    x_mat: np.ndarray,    # (n, K)   log_p_j[t-n-lag:t-lag], K 为候选数
    s_to_cands: list,     # 长度 S_v,s_to_cands[s]=candidate j 的局部索引数组 (k1,)
):
    """
    对每个候选 (s, j, lag) 批量做协整 OLS: y = α + β·x + ε,返回 α, β, spread。

    输入:
        y_mat: 每只 s 的 log_p_s 时间序列 (n, S_v)
        x_mat: 所有股票的 log_p_j 时间序列 (n, S_v)  [同 lag 下]
        s_to_cands: 每只 s 对应的候选 j 局部索引列表 [(k1,)] × S_v

    返回:
        alpha: (S_v, k1) 截距
        beta:  (S_v, k1) 斜率
        spread: (n, S_v, k1) 残差序列
        mu:    (S_v, k1) spread 均值(实际为0,因OLS残差)
        sigma: (S_v, k1) spread 标准差

    实现要点:
        - 利用 x_j 的均值/方差只和 j 有关的特性,先对 S_v 只 x 序列批量算 sum_x, sum_xx
        - α = ȳ_s - β·x̄_j, β = (Σ(x-x̄)(y-ȳ)) / Σ(x-x̄)²
        - spread_τ = y_s,τ - α - β·x_j,τ
    """
    n, S_v = y_mat.shape
    k1 = s_to_cands[0].shape[0] if len(s_to_cands) > 0 else 0

    # 每只股票的统计量 (S_v,)
    x_mean = x_mat.mean(axis=0)                             # (S_v,)
    x_centered = x_mat - x_mean                             # (n, S_v)
    x_var = (x_centered ** 2).sum(axis=0).clip(min=1e-12)   # (S_v,)

    y_mean = y_mat.mean(axis=0)                             # (S_v,)
    y_centered = y_mat - y_mean                             # (n, S_v)

    alpha = np.zeros((S_v, k1), dtype=np.float64)
    beta = np.zeros((S_v, k1), dtype=np.float64)
    spread = np.zeros((n, S_v, k1), dtype=np.float64)
    sigma = np.zeros((S_v, k1), dtype=np.float64)

    for s_local in range(S_v):
        cands = s_to_cands[s_local]                         # (k1,)
        yc = y_centered[:, s_local]                         # (n,)
        ym = y_mean[s_local]

        xc_k = x_centered[:, cands]                         # (n, k1)
        x_var_k = x_var[cands]                              # (k1,)
        x_mean_k = x_mean[cands]                            # (k1,)

        cov_xy = (xc_k * yc[:, None]).sum(axis=0)           # (k1,)
        b = cov_xy / x_var_k                                # (k1,)
        a = ym - b * x_mean_k                               # (k1,)

        # spread = y - α - β·x = yc - β·xc  (因为 y = yc + ym, α+β·x̄ = ym)
        sp = yc[:, None] - b[None, :] * xc_k                # (n, k1)
        sd = sp.std(axis=0, ddof=0).clip(min=1e-10)         # (k1,)

        alpha[s_local] = a
        beta[s_local] = b
        spread[:, s_local, :] = sp
        sigma[s_local] = sd

    mu = np.zeros_like(alpha)  # OLS 残差均值为 0,保留接口
    return alpha, beta, spread, mu, sigma


# _batch_adf_pvalues 已搬至 _pair_runner.py


def build_cointegration_factor(
    log_panel: pd.DataFrame,
    n: int = 60,
    lags=(1, 3, 5, 8, 10),
    calc_every: int = 5,
    topk1: int = 100,
    coint_p: float = 0.05,
    half_life_max: Optional[float] = None,
    factor_mode: str = "predict",   # "predict" / "zscore" / "mixed" / "predict_decay" / "predict_prob"
    industry_map: Optional[pd.Series] = None,
    save_leader: bool = False,
    use_log: bool = True,
    pairing_mode: str = "adf",      # "adf" (默认,沿用) / "holdout" (CO-14 新增)
    valid_window: int = 20,         # holdout 模式下的验证段长度
    leader_mv_pct: Optional[float] = None,  # holdout 模式:leader 池=同行业市值前该比例
    mv_panel: Optional[pd.DataFrame] = None,  # [date, stock_code, market_value]
) -> pd.DataFrame:
    """
    协整 lead-lag 因子。

    输入:
        log_panel: [date, stock_code, log_price]
                   - use_log=True:  log_price = log(close) 或 cumsum(spret/100)  (默认)
                   - use_log=False: log_price 列承载原始 close_price (语义是"价格尺度协整")
                                    函数内部名字仍叫 log_price, 但所有 OLS/ADF/因子
                                    公式在"原始价格尺度"上直接成立, 无需额外改动。
        n:         协整窗口长度 (pairing_mode="holdout" 时为训练段长度)
        lags:      候选领先期
        calc_every:每隔多少个截面日计算一次 (5=周频)
        topk1:     粗筛 Spearman(log_price) 每只 s 的正相关 top-k1
                   (Spearman 对单调变换不敏感, 取 log 与否几乎不改变粗筛结果)
        coint_p:   ADF 显著性阈值
        half_life_max: 半衰期上限(None=不过滤, 仅 pairing_mode="adf" 生效)
        factor_mode:
            "predict": α + β·log_p_j[t] + μ - log_p_s[t]  (方案 C,默认)
            "zscore":  -(log_p_s[t] - α - β·log_p_j[t-lag] - μ)/σ  (方案 A)
            "mixed":   -z + β·(log_p_j[t] - log_p_j[t-lag])  (方案 B)
            "predict_decay": (1-φ*) * predict  (CO-18, 仅 holdout 分支,
                             φ*=训练段 AR(1) 系数 = 1 + ρ_ADF; 1-φ* = -ρ_ADF = 回归速率;
                             1-φ* 夹到 [1e-4, 1.0])
            "predict_prob":  Φ(|z|·√((1-φ)/(1+φ))) * predict  (CO-20, 仅 holdout 分支,
                             Φ=标准正态 CDF, z=(log_p_s[t]-α-β·log_p_j[t-lag]-μ)/σ,
                             φ=1+ρ_ADF, 仅当 φ∈(-1,0) 时生效, 其余情形 prob=0.5;
                             predict 提供正负号, Φ(·) 仅作振幅调制)
        industry_map: Series(index=stock_code, value=行业代码) 若传入, 粗筛仅在同行业内进行
        save_leader: True 时在 day_df 额外记录 leader_code (选中 j*) 与 best_lag 两列
        use_log:     False 时按价格尺度做协整 (CO-12 对照), 仅影响打印与语义标识

        pairing_mode:
            "adf":     原路径 (CO-11), 训练段 n 天, ADF 筛 p<coint_p, 取 p 最小
            "holdout": CO-14, 训练段 n 天 + 验证段 valid_window 天
                      训练段 OLS -> ADF 筛 p<coint_p -> 验证段外推 -> 选 valid_RMSE/σ_train 最小
        valid_window: holdout 模式验证段长度 (默认 20)
        leader_mv_pct: holdout 模式下, 只有同行业 market_value 前该比例的股票才能当 leader;
                      s 本身必须在后 (1-该比例) 才生成因子. None=不过滤. 需配合 mv_panel 使用.
        mv_panel: [date, stock_code, market_value] 长表. leader_mv_pct 非 None 时必须提供.

    输出: DataFrame [date, stock_code, factor] (save_leader=True 时多两列 leader_code, best_lag)
    """
    if pairing_mode not in ("adf", "holdout"):
        raise ValueError(f"pairing_mode 必须是 'adf' 或 'holdout', 得到: {pairing_mode}")
    if pairing_mode == "holdout" and leader_mv_pct is not None and mv_panel is None:
        raise ValueError("pairing_mode='holdout' 且 leader_mv_pct 非 None 时必须提供 mv_panel")

    print(f"构建协整因子 (n={n}, lags={list(lags)}, topk1={topk1}, "
          f"p<{coint_p}, every={calc_every}d, half_life<={half_life_max}, "
          f"mode={factor_mode}, same_industry={industry_map is not None}, "
          f"save_leader={save_leader}, use_log={use_log}, "
          f"pairing_mode={pairing_mode}"
          + (f", valid_window={valid_window}, leader_mv_pct={leader_mv_pct}"
             if pairing_mode == "holdout" else "")
          + ") ...")
    t0 = time.time()

    pivot = log_panel.pivot(index="date", columns="stock_code", values="log_price")
    pivot = pivot.sort_index()
    all_dates = pivot.index.values
    all_stocks = pivot.columns.values
    mat = pivot.values.astype(np.float64)                 # (T, S) log 价格
    T, S = mat.shape
    print(f"  矩阵大小: {T} 天 × {S} 只股票")

    # ── 行业掩码(静态): ind_codes[i] = 股票 all_stocks[i] 的行业代码整数 id ──
    # 未覆盖的股票或缺失值一律分配独一无二的 id (保证不会与其他股票同行业)
    if industry_map is not None:
        aligned = industry_map.reindex(all_stocks)
        # 把字符串行业代码编码成整数; NaN 编码为唯一负值
        cat = aligned.astype("category")
        codes = cat.cat.codes.values.astype(np.int64)     # (S,), NaN -> -1
        # 给 NaN 每一个独立负编号(避免所有 NaN 被当同行业)
        nan_mask = codes < 0
        if nan_mask.any():
            nan_ids = -1 - np.arange(int(nan_mask.sum()), dtype=np.int64)
            codes[nan_mask] = nan_ids
        ind_codes_full = codes                            # (S,)
        n_cov = int((~nan_mask).sum())
        n_ind = int(cat.cat.categories.size)
        print(f"  同行业过滤: 覆盖 {n_cov}/{S} 只股票, {n_ind} 个行业")
    else:
        ind_codes_full = None

    # ── holdout 模式: 加载市值面板, 与 log 价格面板时间/股票对齐 ──
    mv_mat: Optional[np.ndarray] = None
    if pairing_mode == "holdout" and leader_mv_pct is not None:
        assert mv_panel is not None
        mv_pivot = (mv_panel.pivot(index="date", columns="stock_code", values="market_value")
                    .reindex(index=pivot.index, columns=pivot.columns))
        mv_mat = mv_pivot.values.astype(np.float64)
        n_mv_valid = int(np.isfinite(mv_mat).sum())
        print(f"  市值面板对齐完成: 有效值 {n_mv_valid} / {mv_mat.size} "
              f"({n_mv_valid / mv_mat.size * 100:.1f}%)")

    lags = tuple(lags)
    max_lag = max(lags)
    # adf 模式: 最早需要 mat_v[t_idx - n + 1 - max_lag], 要求 t_idx >= n + max_lag - 1
    # holdout 模式: 训练段 [t_idx - n - valid_window + 1, t_idx - valid_window] + 验证段 [t_idx - valid_window + 1, t_idx]
    #   训练段 x 最早 t_idx - n - valid_window + 1 - max_lag, 要求 t_idx >= n + valid_window + max_lag - 1
    if pairing_mode == "adf":
        min_idx = n + max_lag - 1
    else:
        min_idx = n + valid_window + max_lag - 1

    backtest_start_dt = pd.Timestamp(BACKTEST_START)
    bt_end_dt = pd.Timestamp(BACKTEST_END)

    results = []
    n_calc = 0
    calc_count = 0

    for t_idx in range(min_idx, T):
        dt = all_dates[t_idx]
        if dt < backtest_start_dt or dt > bt_end_dt:
            continue

        # 周频: 只在每 calc_every 个 bt 截面日算
        calc_count += 1
        if (calc_count - 1) % calc_every != 0:
            continue

        # ── Step 0: 预筛 (训练+验证+max_lag 窗口内无 NaN 且非退化) ──
        # adf 模式: [t_idx-n+1-max_lag, t_idx]
        # holdout 模式: [t_idx-n-valid_window+1-max_lag, t_idx] (覆盖训练段+验证段+lag shift)
        if pairing_mode == "adf":
            check_start = t_idx - n + 1 - max_lag
        else:
            check_start = t_idx - n - valid_window + 1 - max_lag
        check_window = mat[check_start: t_idx + 1, :]
        no_nan = ~np.isnan(check_window).any(axis=0)
        with np.errstate(invalid='ignore'):
            non_degenerate = np.nanstd(check_window, axis=0) > 1e-6
        valid_mask_s = no_nan & non_degenerate
        valid_idx = np.where(valid_mask_s)[0]
        S_v = len(valid_idx)

        if S_v < topk1 + 10:
            continue

        mat_v = mat[:, valid_idx]                         # (T, S_v)

        # 行业掩码: same_ind_mask[s, j] = True 当 s 和 j 同行业 (对角=False 排除自身)
        if ind_codes_full is not None:
            ind_v = ind_codes_full[valid_idx]
            same_ind_mask = (ind_v[:, None] == ind_v[None, :])  # (S_v, S_v)
            np.fill_diagonal(same_ind_mask, False)
        else:
            same_ind_mask = None

        # ── pairing_mode 分支: holdout 走独立逻辑, 完成后直接跳到下一截面 ──
        if pairing_mode == "holdout":
            factor_vals_v = np.full(S_v, np.nan)
            if save_leader:
                j_stars_v = np.full(S_v, -1, dtype=np.int64)
                lag_stars_v = np.full(S_v, -1, dtype=np.int64)

            # ── H1: 基于 T 日市值确定 leader 池 / 非 leader 池 (仅同行业内比较) ──
            # leader_mv_pct=0.3 表示同行业市值前 30% 为 leader 池, 后 70% 生成因子
            if leader_mv_pct is not None and mv_mat is not None:
                mv_today_v = mv_mat[t_idx, valid_idx]      # (S_v,)
                is_leader_v = np.zeros(S_v, dtype=bool)
                is_nonleader_v = np.zeros(S_v, dtype=bool)
                # 按行业分组排名: 每只股票在同行业 mv 值的 rank / 行业内股票数
                # 为效率用矩阵法: 同行业掩码(含对角) × mv 比较
                if same_ind_mask is None:
                    raise ValueError("holdout + leader_mv_pct 目前仅在 same_industry=True 下支持")
                same_ind_self_v = same_ind_mask.copy()
                np.fill_diagonal(same_ind_self_v, True)      # 算行业内排名时包含自身
                ind_sizes_v = same_ind_self_v.sum(axis=1)    # 每个 s 所在行业内股票数
                # 对于 s: 行业内 mv 严格大于 s.mv 的股票数 >= ceil((1-leader_mv_pct)*ind_size) 则 s 不是 leader
                # 用 mv 广播:  same_ind_self_v & (mv > mv[s])
                mv_mat_brd = mv_today_v[None, :]              # (1, S_v)
                mv_self_brd = mv_today_v[:, None]             # (S_v, 1)
                # mv_today_v 中有 NaN (行业数据漏) 的股票: 直接剔除出 pool
                mv_valid = np.isfinite(mv_today_v)            # (S_v,)
                rank_matrix = same_ind_self_v & (mv_mat_brd > mv_self_brd)
                rank_higher = rank_matrix.sum(axis=1).astype(np.float64)  # 行业内比 s 大的数量
                # s 在行业内 mv 排名 = rank_higher + 1 (1 为最大); 归一化: rank_higher / ind_size
                # leader: rank_higher / ind_size < leader_mv_pct (前 leader_mv_pct 比例)
                with np.errstate(invalid='ignore', divide='ignore'):
                    rank_pct = np.where(ind_sizes_v > 0, rank_higher / np.maximum(ind_sizes_v, 1), np.nan)
                is_leader_v = mv_valid & (rank_pct < leader_mv_pct)
                is_nonleader_v = mv_valid & (~is_leader_v)
            else:
                # 无市值过滤: 所有股票既可当 leader 也可当非 leader
                is_leader_v = np.ones(S_v, dtype=bool)
                is_nonleader_v = np.ones(S_v, dtype=bool)

            # ── H2: Spearman 粗筛 (训练段 n 天) 每个 lag 选 top-k1 候选 leader ──
            # 训练段: [t_idx - n - valid_window + 1, t_idx - valid_window]
            y_train = mat_v[t_idx - n - valid_window + 1: t_idx - valid_window + 1, :]  # (n, S_v)
            k1 = min(topk1, S_v - 1)
            per_lag_topk_h = {}
            for lg in lags:
                rj = mat_v[t_idx - n - valid_window + 1 - lg: t_idx - valid_window + 1 - lg, :]  # (n, S_v)
                corr_mat = _spearman_corr_matrix(y_train, rj)
                np.fill_diagonal(corr_mat, -np.inf)
                corr_mat = np.where(corr_mat > 0, corr_mat, -np.inf)
                if same_ind_mask is not None:
                    corr_mat = np.where(same_ind_mask, corr_mat, -np.inf)
                # 只有 leader 池可当 j: 非 leader 列置 -inf
                corr_mat = np.where(is_leader_v[None, :], corr_mat, -np.inf)
                top_idx = np.argpartition(-corr_mat, k1, axis=1)[:, :k1]
                top_corr = np.take_along_axis(corr_mat, top_idx, axis=1)
                per_lag_topk_h[lg] = (top_idx, np.isfinite(top_corr))

            # ── H3: 训练段 OLS + ADF 筛选 + 验证段 RMSE 归一化 ──
            # 对每个 lag: 训练段 OLS → 训练段 ADF (p_train<coint_p) → 验证段 spread → 归一化 RMSE
            lag_results_h = {}
            for lg in lags:
                cand, cand_valid = per_lag_topk_h[lg]
                x_train_lg = mat_v[t_idx - n - valid_window + 1 - lg:
                                   t_idx - valid_window + 1 - lg, :]   # (n, S_v)
                s_to_cands = [cand[i] for i in range(S_v)]

                # 训练段 OLS
                alpha, beta, spread_train, mu, sigma = _batch_coint_regression(
                    y_train, x_train_lg, s_to_cands
                )   # spread_train: (n, S_v, k1), sigma/alpha/beta/mu: (S_v, k1)

                # 训练段 ADF
                spread_flat = spread_train.reshape(n, S_v * k1)
                pvals_flat, rho_flat, _ = _batch_adf_pvalues(spread_flat)
                pvals_train = pvals_flat.reshape(S_v, k1)
                rho_train = rho_flat.reshape(S_v, k1)   # AR(1) 系数, 供 CO-18/CO-20 使用

                # 验证段 spread: 用训练段 (alpha, beta), 外推到 [t_idx - valid_window + 1, t_idx]
                y_valid = mat_v[t_idx - valid_window + 1: t_idx + 1, :]              # (v, S_v)
                x_valid_lg = mat_v[t_idx - valid_window + 1 - lg: t_idx + 1 - lg, :] # (v, S_v)
                # 对每只 s, 取其 k1 个候选 j 的 x_valid 列: x_valid_lg[:, cand[s]]
                # 向量化: 用 fancy index, cand shape (S_v, k1), x_valid_lg.T[cand] -> (S_v, k1, v)
                # 改用每只 s 的 spread_valid[:, s, k] = y_valid[:, s] - alpha[s,k] - beta[s,k]*x_valid_lg[:, cand[s,k]]
                # 数据规模: S_v≈几千, k1=100, valid_window=20 → S_v*k1*valid_window 约 4e6,直接向量化可行
                # x_valid_cand[:, s, k] = x_valid_lg[:, cand[s, k]]
                x_valid_cand = x_valid_lg[:, cand]                                    # (v, S_v, k1)
                # y_valid[:, :, None] broadcast 到 (v, S_v, k1)
                spread_valid = y_valid[:, :, None] - alpha[None, :, :] - beta[None, :, :] * x_valid_cand - mu[None, :, :]

                # 验证段 RMSE = sqrt(mean(spread_valid**2))
                rmse_valid = np.sqrt(np.mean(spread_valid ** 2, axis=0))              # (S_v, k1)
                # 归一化: score = rmse_valid / sigma_train (sigma 越小 spread 越紧, 直接比 rmse 对 σ 小的不公)
                with np.errstate(invalid='ignore', divide='ignore'):
                    norm_rmse = rmse_valid / np.maximum(sigma, 1e-10)                 # (S_v, k1)

                lag_results_h[lg] = {
                    "alpha": alpha, "beta": beta, "mu": mu, "sigma": sigma,
                    "pvals_train": pvals_train,
                    "rho_train": rho_train,
                    "norm_rmse": norm_rmse,
                    "cand": cand,
                    "cand_valid": cand_valid,
                }

            # ── H4: 每只 s 选 (lg*, j*): 只有非 leader s 才生成因子;
            #   从训练段 ADF p_train<coint_p 的候选中, 选 norm_rmse 最小者 ──
            for s_local in range(S_v):
                if not is_nonleader_v[s_local]:
                    continue

                best_score = np.inf
                best = None

                for lg in lags:
                    info = lag_results_h[lg]
                    p_row = info["pvals_train"][s_local]
                    nr_row = info["norm_rmse"][s_local]
                    cand_row = info["cand"][s_local]
                    valid_row = info["cand_valid"][s_local]

                    valid_cand = (
                        valid_row & (p_row < coint_p) & (cand_row != s_local)
                        & np.isfinite(nr_row)
                    )
                    if not valid_cand.any():
                        continue

                    idx = np.where(valid_cand)[0]
                    k_best = idx[np.argmin(nr_row[idx])]
                    score_k = nr_row[k_best]
                    if score_k < best_score:
                        best_score = score_k
                        best = (lg, k_best, s_local)

                if best is None:
                    continue

                lg_b, k_b, s_b = best
                info = lag_results_h[lg_b]
                j_local = int(info["cand"][s_b, k_b])
                if save_leader:
                    j_stars_v[s_b] = j_local
                    lag_stars_v[s_b] = lg_b
                alpha_b = float(info["alpha"][s_b, k_b])
                beta_b = float(info["beta"][s_b, k_b])
                sigma_b = float(info["sigma"][s_b, k_b])
                mu_b = float(info["mu"][s_b, k_b])
                rho_b = float(info["rho_train"][s_b, k_b])
                log_p_s_t = float(mat_v[t_idx, s_b])
                log_p_j_t = float(mat_v[t_idx, j_local])
                log_p_j_t_lag = float(mat_v[t_idx - lg_b, j_local])

                # 因子计算: α/β/σ/ρ 均来自训练段
                if factor_mode == "predict":
                    factor_vals_v[s_b] = alpha_b + beta_b * log_p_j_t + mu_b - log_p_s_t
                elif factor_mode == "zscore":
                    spread_now = log_p_s_t - alpha_b - beta_b * log_p_j_t_lag
                    factor_vals_v[s_b] = -(spread_now - mu_b) / max(sigma_b, 1e-10)
                elif factor_mode == "mixed":
                    spread_now = log_p_s_t - alpha_b - beta_b * log_p_j_t_lag
                    z = (spread_now - mu_b) / max(sigma_b, 1e-10)
                    lead = beta_b * (log_p_j_t - log_p_j_t_lag)
                    factor_vals_v[s_b] = -z + lead
                elif factor_mode == "predict_decay":
                    # CO-18: (1-φ*) * predict, φ=AR(1)系数
                    # ADF 方程 Δs=c+ρ·s[τ-1]+γ·Δs[τ-1] 中的 ρ = φ - 1,
                    # 所以 AR(1) 系数 φ = 1 + rho_b, 回归速率 1-φ = -rho_b
                    # 稳定 AR(1): φ∈(-1,1) → rho_b∈(-2,0) → 1-φ∈(0,2)
                    # φ≥0 (rho_b≥-1): 慢回归/无回归, 夹到 [1e-4, 1.0]; φ<0: 超快回归, 夹到 1.0
                    predict_val = alpha_b + beta_b * log_p_j_t + mu_b - log_p_s_t
                    one_minus_phi = -rho_b
                    one_minus_phi = min(max(one_minus_phi, 1e-4), 1.0)
                    factor_vals_v[s_b] = one_minus_phi * predict_val
                elif factor_mode == "predict_prob":
                    # CO-20: P(|s[τ+1]| < |s[τ]|) * predict, φ=AR(1)系数=1+rho_ADF
                    # 推导: spread AR(1) s[τ+1]=φ·s[τ]+u, σ_u=σ√(1-φ²)
                    #   P(|s+| < |s|) = P(-|s|-φ·s < u < |s|-φ·s)
                    #               = Φ((1-φ)|s|/σ_u) + Φ((1+φ)|s|/σ_u) - 1
                    #               = Φ(|z|·√((1-φ)/(1+φ))) + Φ(|z|·√((1+φ)/(1-φ))) - 1
                    # 对 φ∈(-1,1) 全有效; φ→±1 退化为 0.5, φ→0 接近 2Φ(|z|)-1
                    predict_val = alpha_b + beta_b * log_p_j_t + mu_b - log_p_s_t
                    spread_now = log_p_s_t - alpha_b - beta_b * log_p_j_t_lag
                    z_abs = abs((spread_now - mu_b) / max(sigma_b, 1e-10))
                    phi = 1.0 + rho_b
                    if -0.9999 < phi < 0.9999:
                        # 夹紧防止 1-φ 或 1+φ → 0 时 ratio 溢出
                        one_minus = max(1.0 - phi, 1e-6)
                        one_plus = max(1.0 + phi, 1e-6)
                        ratio1 = one_minus / one_plus      # (1-φ)/(1+φ)
                        ratio2 = one_plus / one_minus      # (1+φ)/(1-φ)
                        prob = float(
                            norm.cdf(z_abs * np.sqrt(ratio1))
                            + norm.cdf(z_abs * np.sqrt(ratio2))
                            - 1.0
                        )
                    else:
                        prob = 0.5   # 随机游走 (φ=±1) 时回归概率退化为 0.5
                    factor_vals_v[s_b] = prob * predict_val
                else:
                    raise ValueError(f"未知 factor_mode: {factor_mode}")

            valid_fac_mask = ~np.isnan(factor_vals_v)
            if valid_fac_mask.any():
                global_idx = valid_idx[valid_fac_mask]
                day_df = pd.DataFrame({
                    "date": dt,
                    "stock_code": all_stocks[global_idx],
                    "factor": factor_vals_v[valid_fac_mask],
                })
                if save_leader:
                    leader_local = j_stars_v[valid_fac_mask]
                    leader_global_idx = valid_idx[leader_local]
                    day_df["leader_code"] = all_stocks[leader_global_idx]
                    day_df["best_lag"] = lag_stars_v[valid_fac_mask]
                results.append(day_df)

            n_calc += 1
            if n_calc % 5 == 0:
                elapsed = time.time() - t0
                avg_t = elapsed / n_calc
                print(f"  [{n_calc}] {pd.Timestamp(dt).date()}, "
                      f"S_v={S_v}, leader池={int(is_leader_v.sum())}, "
                      f"有效={int(valid_fac_mask.sum())}, "
                      f"耗时 {elapsed:.0f}s (avg {avg_t:.1f}s/截面)")
            continue

        # ── 以下为 pairing_mode="adf" 原路径 ──

        # y_mat: log_p_s[t_idx-n+1 : t_idx+1], 包含 T 日 close (n, S_v)
        y_mat = mat_v[t_idx - n + 1: t_idx + 1, :]

        # ── Step 1: 每个 lag 独立 Spearman 粗筛 ──
        k1 = min(topk1, S_v - 1)
        per_lag_topk = {}
        rs = y_mat                                         # (n, S_v) log_p_s
        for lg in lags:
            rj = mat_v[t_idx - n + 1 - lg: t_idx + 1 - lg, :]  # (n, S_v) log_p_j[τ-lg]
            corr_mat = _spearman_corr_matrix(rs, rj)      # (S_v, S_v)
            # 自身 + 非正相关一律设为 -inf, 确保不被 top-k1 选中
            np.fill_diagonal(corr_mat, -np.inf)
            corr_mat = np.where(corr_mat > 0, corr_mat, -np.inf)
            # 同行业过滤: 非同行业置 -inf
            if same_ind_mask is not None:
                corr_mat = np.where(same_ind_mask, corr_mat, -np.inf)
            # 每行降序 top-k1 的局部索引
            top_idx = np.argpartition(-corr_mat, k1, axis=1)[:, :k1]
            # 标记该候选是否真正正相关 (-inf 即无效, 后续 Step 4 过滤)
            top_corr = np.take_along_axis(corr_mat, top_idx, axis=1)  # (S_v, k1)
            per_lag_topk[lg] = (top_idx, np.isfinite(top_corr))

        # ── Step 2+3: 对每个 lag 批量协整 + ADF ──
        lag_results = {}  # lg -> dict of arrays (S_v, k1)

        for lg in lags:
            cand, cand_valid = per_lag_topk[lg]           # (S_v, k1), (S_v, k1)
            x_mat_lg = mat_v[t_idx - n + 1 - lg: t_idx + 1 - lg, :]  # (n, S_v) log_p_j[τ-lg]
            s_to_cands = [cand[i] for i in range(S_v)]

            alpha, beta, spread, mu, sigma = _batch_coint_regression(
                y_mat, x_mat_lg, s_to_cands
            )
            # spread shape (n, S_v, k1) → reshape (n, S_v*k1) 批量 ADF
            spread_flat = spread.reshape(n, S_v * k1)
            pvals_flat, rho_flat, _ = _batch_adf_pvalues(spread_flat)
            pvals = pvals_flat.reshape(S_v, k1)
            rho = rho_flat.reshape(S_v, k1)

            # 半衰期: -log(2) / log(1 + ρ)  (ρ<0 时才有意义)
            with np.errstate(invalid='ignore', divide='ignore'):
                safe_rho = np.where(rho < -1e-8, rho, -1e-8)     # ρ>=0 → 不回归,给大值
                half_life = -np.log(2.0) / np.log1p(safe_rho)    # (S_v, k1)
            half_life = np.where(rho < -1e-8, half_life, np.inf)

            lag_results[lg] = {
                "alpha": alpha, "beta": beta, "mu": mu, "sigma": sigma,
                "pvals": pvals, "half_life": half_life,
                "cand": cand,                                 # (S_v, k1)
                "cand_valid": cand_valid,                     # (S_v, k1) bool, 假候选为 False
            }

        # ── Step 4+5: 每只 s 选最优 (j, lag),构造因子 ──
        factor_vals_v = np.full(S_v, np.nan)
        # save_leader 时记录每只 s 的 j*(局部索引)与 lag* (-1 = 无有效配对)
        if save_leader:
            j_stars_v = np.full(S_v, -1, dtype=np.int64)
            lag_stars_v = np.full(S_v, -1, dtype=np.int64)

        for s_local in range(S_v):
            best_p = np.inf
            best = None       # (lg, j_local, alpha, beta, sigma)

            for lg in lags:
                info = lag_results[lg]
                p_row = info["pvals"][s_local]                # (k1,)
                hl_row = info["half_life"][s_local]           # (k1,)
                cand_row = info["cand"][s_local]              # (k1,)
                valid_row = info["cand_valid"][s_local]       # (k1,) 排除假候选

                valid_cand = valid_row & (p_row < coint_p)
                if half_life_max is not None:
                    valid_cand = valid_cand & (hl_row <= half_life_max) & (hl_row > 0)

                # 排除自身
                valid_cand = valid_cand & (cand_row != s_local)
                if not valid_cand.any():
                    continue

                idx = np.where(valid_cand)[0]
                p_valid = p_row[idx]
                k_best = idx[np.argmin(p_valid)]
                p_k = p_row[k_best]
                if p_k < best_p:
                    best_p = p_k
                    best = (lg, k_best, s_local)

            if best is None:
                continue

            lg_b, k_b, s_b = best
            info = lag_results[lg_b]
            j_local = int(info["cand"][s_b, k_b])
            if save_leader:
                j_stars_v[s_b] = j_local
                lag_stars_v[s_b] = lg_b
            alpha_b = float(info["alpha"][s_b, k_b])
            beta_b = float(info["beta"][s_b, k_b])
            sigma_b = float(info["sigma"][s_b, k_b])
            mu_b = float(info["mu"][s_b, k_b])                # =0
            # log 价格当前点: 使用 T 日 close (mat_v[t_idx])
            log_p_s_t = float(mat_v[t_idx, s_b])
            log_p_j_t = float(mat_v[t_idx, j_local])
            log_p_j_t_lag = float(mat_v[t_idx - lg_b, j_local])

            if factor_mode == "predict":
                # α + β·log_p_j[t] + μ - log_p_s[t]
                factor_vals_v[s_b] = alpha_b + beta_b * log_p_j_t + mu_b - log_p_s_t
            elif factor_mode == "zscore":
                spread_now = log_p_s_t - alpha_b - beta_b * log_p_j_t_lag
                factor_vals_v[s_b] = -(spread_now - mu_b) / max(sigma_b, 1e-10)
            elif factor_mode == "mixed":
                spread_now = log_p_s_t - alpha_b - beta_b * log_p_j_t_lag
                z = (spread_now - mu_b) / max(sigma_b, 1e-10)
                lead = beta_b * (log_p_j_t - log_p_j_t_lag)
                factor_vals_v[s_b] = -z + lead
            else:
                raise ValueError(f"未知 factor_mode: {factor_mode}")

        valid_fac_mask = ~np.isnan(factor_vals_v)
        if valid_fac_mask.any():
            global_idx = valid_idx[valid_fac_mask]
            day_df = pd.DataFrame({
                "date": dt,
                "stock_code": all_stocks[global_idx],
                "factor": factor_vals_v[valid_fac_mask],
            })
            if save_leader:
                # j_stars_v[s_local] 是"valid_idx 之内"的局部索引, 转全局 stock_code:
                # leader_global_idx = valid_idx[j_stars_v[valid_fac_mask]]
                leader_local = j_stars_v[valid_fac_mask]
                leader_global_idx = valid_idx[leader_local]
                day_df["leader_code"] = all_stocks[leader_global_idx]
                day_df["best_lag"] = lag_stars_v[valid_fac_mask]
            results.append(day_df)

        n_calc += 1
        if n_calc % 5 == 0:
            elapsed = time.time() - t0
            avg_t = elapsed / n_calc
            print(f"  [{n_calc}] {pd.Timestamp(dt).date()}, "
                  f"S_v={S_v}, 有效={int(valid_fac_mask.sum())}, "
                  f"耗时 {elapsed:.0f}s (avg {avg_t:.1f}s/截面)")

    if not results:
        raise RuntimeError("协整因子计算结果为空")

    factor_df = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  协整因子计算完成: {len(factor_df)} 条, "
          f"{factor_df['date'].nunique()} 个截面, "
          f"{factor_df['stock_code'].nunique()} 只股票, "
          f"耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return factor_df


# =====================================================================
# CO-16: spret 直接 OLS 预测因子 (holdout 验证 + 行业龙头驱动)
# =====================================================================

def build_predictive_factor(
    spret_panel: pd.DataFrame,
    n: int = 60,
    lags=(1,),
    calc_every: int = 1,
    topk1: int = 100,
    industry_map: Optional[pd.Series] = None,
    save_leader: bool = False,
    valid_window: int = 20,
    leader_mv_pct: Optional[float] = None,
    mv_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    spret (特质日收益率) 的 OLS 预测因子。跳出协整框架, 直接对 I(0) 序列做单变量回归:

        spret_s[τ] = α + β·spret_j[τ-lag] + ε[τ],  τ ∈ 训练段 [t-n-valid_window+1, t-valid_window]

    然后在验证段 [t-valid_window+1, t] 用训练段 (α, β) 外推计算 RMSE, 选归一化 RMSE 最小的 j*:
        score(j) = RMSE(ε_valid) / σ_train(ε),  j* = argmin score(j)

    T 日因子 (预测 T 日 s 的特质收益, 用 T-lag 日 leader 的 spret):
        factor_s[T] = α* + β*·spret_j*[T-lag]

    输入:
        spret_panel: [date, stock_code, spret]  (特质日收益率, 单位 %; 直接使用不累加)
        n:           训练段长度
        lags:        候选领先期 (CO-16 固定 {1})
        calc_every:  每隔多少截面日计算一次
        topk1:       粗筛 Spearman(spret) 每只 s 的正相关 top-K1
        industry_map: 同 CO-14
        save_leader: 同 CO-14
        valid_window:holdout 验证段长度
        leader_mv_pct: 同 CO-14
        mv_panel:    同 CO-14, [date, stock_code, market_value]

    输出: [date, stock_code, factor] (save_leader=True 多 leader_code / best_lag)

    与 build_cointegration_factor 相比的关键差异:
        - 输入是 spret (I(0)), 不是 log 价格 (I(1))
        - 不做 ADF 检验 (spret 平稳, ADF 几乎必过, 无区分能力)
        - 因子是"预测收益率"而非"标准化偏离度", 语义上等价于"α* + β*·spret_j*[T-1]"
    """
    if leader_mv_pct is not None and mv_panel is None:
        raise ValueError("leader_mv_pct 非 None 时必须提供 mv_panel")

    print(f"构建 spret 预测因子 (n={n}, lags={list(lags)}, topk1={topk1}, "
          f"every={calc_every}d, valid_window={valid_window}, "
          f"leader_mv_pct={leader_mv_pct}, "
          f"same_industry={industry_map is not None}, "
          f"save_leader={save_leader}) ...")
    t0 = time.time()

    pivot = spret_panel.pivot(index="date", columns="stock_code", values="spret")
    pivot = pivot.sort_index()
    all_dates = pivot.index.values
    all_stocks = pivot.columns.values
    mat = pivot.values.astype(np.float64)   # (T, S) spret
    T, S = mat.shape
    print(f"  矩阵大小: {T} 天 × {S} 只股票")

    # 行业掩码 (静态)
    if industry_map is not None:
        aligned = industry_map.reindex(all_stocks)
        cat = aligned.astype("category")
        codes = cat.cat.codes.values.astype(np.int64)
        nan_mask = codes < 0
        if nan_mask.any():
            nan_ids = -1 - np.arange(int(nan_mask.sum()), dtype=np.int64)
            codes[nan_mask] = nan_ids
        ind_codes_full = codes
        n_cov = int((~nan_mask).sum())
        n_ind = int(cat.cat.categories.size)
        print(f"  同行业过滤: 覆盖 {n_cov}/{S} 只股票, {n_ind} 个行业")
    else:
        ind_codes_full = None

    # 市值面板对齐
    mv_mat: Optional[np.ndarray] = None
    if leader_mv_pct is not None:
        assert mv_panel is not None
        mv_pivot = (mv_panel.pivot(index="date", columns="stock_code", values="market_value")
                    .reindex(index=pivot.index, columns=pivot.columns))
        mv_mat = mv_pivot.values.astype(np.float64)
        n_mv_valid = int(np.isfinite(mv_mat).sum())
        print(f"  市值面板对齐完成: 有效值 {n_mv_valid} / {mv_mat.size} "
              f"({n_mv_valid / mv_mat.size * 100:.1f}%)")

    lags = tuple(lags)
    max_lag = max(lags)
    # 训练段 [t-n-valid_window+1, t-valid_window] 的 x 最早为 t-n-valid_window+1-max_lag
    min_idx = n + valid_window + max_lag - 1

    backtest_start_dt = pd.Timestamp(BACKTEST_START)
    bt_end_dt = pd.Timestamp(BACKTEST_END)

    results = []
    n_calc = 0
    calc_count = 0

    for t_idx in range(min_idx, T):
        dt = all_dates[t_idx]
        if dt < backtest_start_dt or dt > bt_end_dt:
            continue

        calc_count += 1
        if (calc_count - 1) % calc_every != 0:
            continue

        # ── 预筛: 训练+验证+max_lag 窗口内 spret 无 NaN 且非退化 ──
        check_start = t_idx - n - valid_window + 1 - max_lag
        check_window = mat[check_start: t_idx + 1, :]
        no_nan = ~np.isnan(check_window).any(axis=0)
        with np.errstate(invalid='ignore'):
            non_degenerate = np.nanstd(check_window, axis=0) > 1e-6
        valid_mask_s = no_nan & non_degenerate
        valid_idx = np.where(valid_mask_s)[0]
        S_v = len(valid_idx)

        if S_v < topk1 + 10:
            continue

        mat_v = mat[:, valid_idx]

        # 行业掩码
        if ind_codes_full is not None:
            ind_v = ind_codes_full[valid_idx]
            same_ind_mask = (ind_v[:, None] == ind_v[None, :])
            np.fill_diagonal(same_ind_mask, False)
        else:
            same_ind_mask = None

        factor_vals_v = np.full(S_v, np.nan)
        if save_leader:
            j_stars_v = np.full(S_v, -1, dtype=np.int64)
            lag_stars_v = np.full(S_v, -1, dtype=np.int64)

        # ── H1: leader / 非 leader 划分 (与 CO-14 同构) ──
        if leader_mv_pct is not None and mv_mat is not None:
            mv_today_v = mv_mat[t_idx, valid_idx]
            if same_ind_mask is None:
                raise ValueError("leader_mv_pct 目前仅在 same_industry=True 下支持")
            same_ind_self_v = same_ind_mask.copy()
            np.fill_diagonal(same_ind_self_v, True)
            ind_sizes_v = same_ind_self_v.sum(axis=1)
            mv_mat_brd = mv_today_v[None, :]
            mv_self_brd = mv_today_v[:, None]
            mv_valid = np.isfinite(mv_today_v)
            rank_matrix = same_ind_self_v & (mv_mat_brd > mv_self_brd)
            rank_higher = rank_matrix.sum(axis=1).astype(np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                rank_pct = np.where(ind_sizes_v > 0, rank_higher / np.maximum(ind_sizes_v, 1), np.nan)
            is_leader_v = mv_valid & (rank_pct < leader_mv_pct)
            is_nonleader_v = mv_valid & (~is_leader_v)
        else:
            is_leader_v = np.ones(S_v, dtype=bool)
            is_nonleader_v = np.ones(S_v, dtype=bool)

        # ── H2: Spearman(spret) 粗筛训练段, leader 池内选 top-K1 候选 ──
        # 训练段: [t_idx - n - valid_window + 1, t_idx - valid_window]  (n 天)
        y_train = mat_v[t_idx - n - valid_window + 1: t_idx - valid_window + 1, :]   # (n, S_v)
        k1 = min(topk1, S_v - 1)
        per_lag_topk_h = {}
        for lg in lags:
            rj = mat_v[t_idx - n - valid_window + 1 - lg: t_idx - valid_window + 1 - lg, :]  # (n, S_v)
            corr_mat = _spearman_corr_matrix(y_train, rj)
            np.fill_diagonal(corr_mat, -np.inf)
            # 方案 B: 不约束 β 符号, 正负相关都保留 → 不再过滤正相关
            # 仅保留有限值 (排除 NaN/±inf 污染)
            corr_mat = np.where(np.isfinite(corr_mat), corr_mat, -np.inf)
            if same_ind_mask is not None:
                corr_mat = np.where(same_ind_mask, corr_mat, -np.inf)
            corr_mat = np.where(is_leader_v[None, :], corr_mat, -np.inf)
            # 按相关性"绝对值"降序取 top-K1 (保留强负相关), 同时保持 -inf 的无效候选在最后
            abs_corr = np.where(corr_mat > -np.inf, np.abs(corr_mat), -np.inf)
            top_idx = np.argpartition(-abs_corr, k1, axis=1)[:, :k1]
            top_corr_abs = np.take_along_axis(abs_corr, top_idx, axis=1)
            per_lag_topk_h[lg] = (top_idx, np.isfinite(top_corr_abs))

        # ── H3: 训练段 OLS + 验证段 RMSE 归一化 ──
        lag_results_h = {}
        for lg in lags:
            cand, cand_valid = per_lag_topk_h[lg]
            x_train_lg = mat_v[t_idx - n - valid_window + 1 - lg:
                               t_idx - valid_window + 1 - lg, :]
            s_to_cands = [cand[i] for i in range(S_v)]

            # 训练段 OLS (复用 _batch_coint_regression, 虽然这里不是"协整", 但 OLS 公式完全相同)
            alpha, beta, spread_train, mu, sigma = _batch_coint_regression(
                y_train, x_train_lg, s_to_cands
            )

            # 验证段 spread = y_valid - α - β·x_valid - μ
            y_valid = mat_v[t_idx - valid_window + 1: t_idx + 1, :]
            x_valid_lg = mat_v[t_idx - valid_window + 1 - lg: t_idx + 1 - lg, :]
            x_valid_cand = x_valid_lg[:, cand]                              # (v, S_v, k1)
            spread_valid = y_valid[:, :, None] - alpha[None, :, :] - beta[None, :, :] * x_valid_cand - mu[None, :, :]

            rmse_valid = np.sqrt(np.mean(spread_valid ** 2, axis=0))        # (S_v, k1)
            with np.errstate(invalid='ignore', divide='ignore'):
                norm_rmse = rmse_valid / np.maximum(sigma, 1e-10)

            lag_results_h[lg] = {
                "alpha": alpha, "beta": beta,
                "norm_rmse": norm_rmse,
                "cand": cand,
                "cand_valid": cand_valid,
            }

        # ── H4: 每只非 leader 股票选 (lg*, j*), 输出因子 = α* + β*·spret_j*[T-lg*] ──
        for s_local in range(S_v):
            if not is_nonleader_v[s_local]:
                continue

            best_score = np.inf
            best = None

            for lg in lags:
                info = lag_results_h[lg]
                nr_row = info["norm_rmse"][s_local]
                cand_row = info["cand"][s_local]
                valid_row = info["cand_valid"][s_local]

                valid_cand = (
                    valid_row & (cand_row != s_local) & np.isfinite(nr_row)
                )
                if not valid_cand.any():
                    continue

                idx = np.where(valid_cand)[0]
                k_best = idx[np.argmin(nr_row[idx])]
                score_k = nr_row[k_best]
                if score_k < best_score:
                    best_score = score_k
                    best = (lg, k_best, s_local)

            if best is None:
                continue

            lg_b, k_b, s_b = best
            info = lag_results_h[lg_b]
            j_local = int(info["cand"][s_b, k_b])
            if save_leader:
                j_stars_v[s_b] = j_local
                lag_stars_v[s_b] = lg_b
            alpha_b = float(info["alpha"][s_b, k_b])
            beta_b = float(info["beta"][s_b, k_b])
            spret_j_t_lag = float(mat_v[t_idx - lg_b, j_local])

            # T 日因子 = α* + β*·spret_j*[T-lg*] (预测 T 日 s 的特质收益率, 单位 %)
            factor_vals_v[s_b] = alpha_b + beta_b * spret_j_t_lag

        valid_fac_mask = ~np.isnan(factor_vals_v)
        if valid_fac_mask.any():
            global_idx = valid_idx[valid_fac_mask]
            day_df = pd.DataFrame({
                "date": dt,
                "stock_code": all_stocks[global_idx],
                "factor": factor_vals_v[valid_fac_mask],
            })
            if save_leader:
                leader_local = j_stars_v[valid_fac_mask]
                leader_global_idx = valid_idx[leader_local]
                day_df["leader_code"] = all_stocks[leader_global_idx]
                day_df["best_lag"] = lag_stars_v[valid_fac_mask]
            results.append(day_df)

        n_calc += 1
        if n_calc % 5 == 0:
            elapsed = time.time() - t0
            avg_t = elapsed / n_calc
            print(f"  [{n_calc}] {pd.Timestamp(dt).date()}, "
                  f"S_v={S_v}, leader池={int(is_leader_v.sum())}, "
                  f"有效={int(valid_fac_mask.sum())}, "
                  f"耗时 {elapsed:.0f}s (avg {avg_t:.1f}s/截面)")

    if not results:
        raise RuntimeError("spret 预测因子计算结果为空")

    factor_df = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  spret 预测因子计算完成: {len(factor_df)} 条, "
          f"{factor_df['date'].nunique()} 个截面, "
          f"{factor_df['stock_code'].nunique()} 只股票, "
          f"耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return factor_df


# =====================================================================
# 通用骨架已抽至 _pair_runner.py:
#   - build_tradable_mask
#   - group_backtest / head_group_backtest
#   - calc_stats
#   - plot_group_backtest / plot_head_backtest / plot_yearly_head
# =====================================================================


# =====================================================================
# 主函数
# =====================================================================

def run_one_experiment(
    exp_name: str,
    tradable_df: pd.DataFrame,
    spret: Optional[pd.DataFrame],
    factor_cfg: Dict,
    backtest_cfg: Dict,
    log_panel: Optional[pd.DataFrame] = None,
    shared_wide_df: Optional[pd.DataFrame] = None,
    factor_col: Optional[str] = None,
    industry_map: Optional[pd.Series] = None,
    mv_panel: Optional[pd.DataFrame] = None,
):
    """运行单个实验：因子计算 + 回测 + 保存。

    若提供 shared_wide_df + factor_col, 则从共享宽表取对应列作为因子
    (不再调用因子计算函数)。适用于 GL-03/04/05 共享同一张 Granger 宽表的场景。
    """
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    metrics_path = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"\n[{exp_name}] 已有 metrics.json，跳过")
        return

    os.makedirs(exp_dir, exist_ok=True)
    t0 = time.time()
    fwd_days = backtest_cfg.get("fwd_days", 1)
    factor_type = factor_cfg.get("factor_type", "pair")

    print(f"\n{'─'*50}")
    print(f"[{exp_name}] 开始  {time.strftime('%H:%M:%S')}")
    if shared_wide_df is not None:
        print(f"  因子: 共享 Granger 宽表 (factor_col={factor_col})")
    elif factor_type == "granger":
        print(f"  因子: Granger (n={factor_cfg.get('n', 60)}, "
              f"lags={list(factor_cfg.get('lags', (1,3,5,8,10)))}, "
              f"topk1={factor_cfg.get('topk1', 100)}, "
              f"topk2={factor_cfg.get('topk2', 5)})")
    elif factor_type == "cointegration":
        print(f"  因子: Cointegration (n={factor_cfg.get('n', 60)}, "
              f"lags={list(factor_cfg.get('lags', (1,3,5,8,10)))}, "
              f"topk1={factor_cfg.get('topk1', 100)}, "
              f"p<{factor_cfg.get('coint_p', 0.05)}, "
              f"half_life_max={factor_cfg.get('half_life_max')}, "
              f"mode={factor_cfg.get('factor_mode', 'predict')}, "
              f"use_log={factor_cfg.get('use_log', True)})")
    elif factor_type == "predictive":
        print(f"  因子: Predictive (spret OLS 预测, n={factor_cfg.get('n', 60)}, "
              f"lags={list(factor_cfg.get('lags', (1,)))}, "
              f"topk1={factor_cfg.get('topk1', 100)}, "
              f"valid_window={factor_cfg.get('valid_window', 20)}, "
              f"leader_mv_pct={factor_cfg.get('leader_mv_pct')})")
    elif factor_type == "correlation":
        print(f"  因子: Correlation (n={factor_cfg.get('n', 60)}, "
              f"lags={list(factor_cfg.get('lags', (1,3,5,8,10)))}, "
              f"共享宽表 {factor_col})")
    else:
        topk = factor_cfg.get("corr_topk", 1.0)
        topk_str = f"top{int(topk*100)}%" if topk < 1.0 else "全部"
        lag_val = factor_cfg.get("lag", 0)
        lag_str = f", lag={lag_val}" if lag_val > 0 else ""
        print(f"  因子: weight={factor_cfg.get('weight_window','recent')}, "
              f"calc_every={factor_cfg.get('calc_every',1)}d, corr={topk_str}{lag_str}")
    print(f"  回测: fwd_days={fwd_days}")

    # 因子：先尝试加载实验目录缓存
    cached_path = os.path.join(exp_dir, "factor_df.pkl")
    if os.path.exists(cached_path):
        print(f"  加载已有因子: {cached_path}")
        factor_df = pd.read_pickle(cached_path)
        factor_df["date"] = pd.to_datetime(factor_df["date"])
    elif shared_wide_df is not None:
        # 从共享宽表取列
        if factor_col is None or factor_col not in shared_wide_df.columns:
            raise ValueError(f"[{exp_name}] factor_col={factor_col} 不存在于共享宽表, "
                             f"可用列: {list(shared_wide_df.columns)}")
        sub = shared_wide_df[["date", "stock_code", factor_col]].copy()
        sub = sub.rename(columns={factor_col: "factor"})
        # 去除该列为 NaN 的行 (门槛不满足时)
        sub = sub[sub["factor"].notna()].reset_index(drop=True)
        factor_df = sub
        factor_df.to_pickle(cached_path)
        print(f"  从共享宽表取列 {factor_col}: {len(factor_df)} 条, "
              f"{factor_df['date'].nunique()} 个截面, "
              f"{factor_df['stock_code'].nunique()} 只股票")
    else:
        if factor_type == "granger":
            granger_kwargs = {k: v for k, v in factor_cfg.items() if k != "factor_type"}
            factor_df = build_granger_factor(spret, **granger_kwargs)
        elif factor_type == "cointegration":
            coint_kwargs = {k: v for k, v in factor_cfg.items()
                            if k not in ("factor_type", "same_industry")}
            if factor_cfg.get("same_industry", False):
                if industry_map is None:
                    raise RuntimeError(f"[{exp_name}] same_industry=True 但未提供 industry_map")
                coint_kwargs["industry_map"] = industry_map
            # CO-14 holdout + 市值过滤: 注入 mv_panel
            if (factor_cfg.get("pairing_mode") == "holdout"
                    and factor_cfg.get("leader_mv_pct") is not None):
                if mv_panel is None:
                    raise RuntimeError(
                        f"[{exp_name}] pairing_mode='holdout' + leader_mv_pct 但未提供 mv_panel"
                    )
                coint_kwargs["mv_panel"] = mv_panel
            factor_df = build_cointegration_factor(log_panel, **coint_kwargs)
        elif factor_type == "predictive":
            # CO-16: spret 直接 OLS 预测因子 (跳出协整, 使用 build_predictive_factor)
            pred_kwargs = {k: v for k, v in factor_cfg.items()
                           if k not in ("factor_type", "same_industry")}
            if factor_cfg.get("same_industry", False):
                if industry_map is None:
                    raise RuntimeError(f"[{exp_name}] same_industry=True 但未提供 industry_map")
                pred_kwargs["industry_map"] = industry_map
            if factor_cfg.get("leader_mv_pct") is not None:
                if mv_panel is None:
                    raise RuntimeError(
                        f"[{exp_name}] factor_type='predictive' + leader_mv_pct 但未提供 mv_panel"
                    )
                pred_kwargs["mv_panel"] = mv_panel
            if spret is None:
                raise RuntimeError(f"[{exp_name}] factor_type='predictive' 但未提供 spret")
            factor_df = build_predictive_factor(spret, **pred_kwargs)
        else:
            factor_df = build_pair_factor(spret, n=N, **factor_cfg)
        factor_df.to_pickle(cached_path)

    # 分层回测
    bt_result = group_backtest(factor_df, tradable_df,
                               n_groups=N_GROUPS, fwd_days=fwd_days)

    # 头组持仓回测
    nav_df = head_group_backtest(factor_df, tradable_df,
                                 n_groups=N_GROUPS,
                                 commission_rate=COMMISSION_RATE,
                                 fwd_days=fwd_days)

    # 统计
    stats = {
        "head": calc_stats(nav_df["head_nav"], "头组", fwd_days=fwd_days),
        "benchmark": calc_stats(nav_df["benchmark_nav"], "基准", fwd_days=fwd_days),
        "excess": calc_stats(nav_df["excess_nav"], "超额", fwd_days=fwd_days),
    }

    # metrics
    metrics = {
        "experiment": exp_name,
        "factor_type": factor_type,
        "n": factor_cfg.get("n", N),
        "fwd_days": fwd_days,
        "factor_cfg": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in factor_cfg.items()},
        "backtest_range": f"{BACKTEST_START} ~ {BACKTEST_END}",
        "rank_ic_mean": round(bt_result["ic_mean"], 6),
        "rank_ic_std": round(bt_result["ic_std"], 6),
        "rank_ic_ir": round(bt_result["ic_ir"], 6),
        "n_dates": len(bt_result["dates"]),
        "head_excess_return_annualized": round(bt_result["head_ann_excess"], 6),
        "head_excess_volatility_annualized": round(bt_result["head_ann_excess_vol"], 6),
        "head_excess_ir": round(bt_result["head_excess_ir"], 6),
        "head_excess_max_drawdown": round(bt_result["head_excess_max_dd"], 6),
        "top_group_excess_return": round(bt_result["head_excess_total"], 6),
        "head": stats["head"],
        "benchmark": stats["benchmark"],
        "head_excess": stats["excess"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # 图表
    plot_group_backtest(bt_result, os.path.join(exp_dir, "backtest.png"))
    plot_head_backtest(nav_df, os.path.join(exp_dir, "head_backtest.png"), stats)
    plot_yearly_head(nav_df, os.path.join(exp_dir, "yearly_head.png"))

    elapsed = time.time() - t0
    print(f"  [{exp_name}] 完成, {elapsed:.0f}s")
    print(f"  IC={bt_result['ic_mean']:.4f}, ICIR={bt_result['ic_ir']:.4f}")
    print(f"  头组超额: 年化={stats['excess']['annual_return']:.1f}%, "
          f"夏普={stats['excess']['sharpe_ratio']:.3f}, "
          f"回撤={stats['excess']['max_drawdown']:.1f}%")

    del factor_df, nav_df
    gc.collect()


# ── 实验列表 ──
# data="default" 用原始数据, data="non_st" 用剔除ST后的数据
EXPERIMENTS = [
    # ── 原始数据 ──
    {"name": "BL-01", "factor_cfg": {"calc_every": 1, "weight_window": "recent"}, "backtest_cfg": {"fwd_days": 1}, "data": "default"},
    {"name": "BL-02", "factor_cfg": {"calc_every": 5, "weight_window": "post5"}, "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    {"name": "BL-03", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.2}, "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    {"name": "BL-04", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.4}, "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    {"name": "BL-05", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.6}, "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    {"name": "BL-06", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.8}, "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    # ── 剔除ST ──
    {"name": "BL-07", "factor_cfg": {"calc_every": 1, "weight_window": "recent"}, "backtest_cfg": {"fwd_days": 1}, "data": "non_st"},
    {"name": "BL-08", "factor_cfg": {"calc_every": 5, "weight_window": "post5"}, "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    {"name": "BL-09", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.2}, "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    {"name": "BL-10", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.4}, "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    {"name": "BL-11", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.6}, "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    {"name": "BL-12", "factor_cfg": {"calc_every": 5, "weight_window": "post5", "corr_topk": 0.8}, "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    # ── lag 实验 (特质收益率) ──
    {"name": "BL-13", "factor_cfg": {"calc_every": 1, "weight_window": "lag", "lag": 1},  "backtest_cfg": {"fwd_days": 1}, "data": "default"},
    {"name": "BL-14", "factor_cfg": {"calc_every": 2, "weight_window": "lag", "lag": 2},  "backtest_cfg": {"fwd_days": 2}, "data": "default"},
    {"name": "BL-15", "factor_cfg": {"calc_every": 3, "weight_window": "lag", "lag": 3},  "backtest_cfg": {"fwd_days": 3}, "data": "default"},
    {"name": "BL-16", "factor_cfg": {"calc_every": 5, "weight_window": "lag", "lag": 8},  "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    {"name": "BL-17", "factor_cfg": {"calc_every": 5, "weight_window": "lag", "lag": 10}, "backtest_cfg": {"fwd_days": 5}, "data": "default"},
    # ── lag 实验 (价格收益率, non_st) ──
    {"name": "BL-18", "factor_cfg": {"calc_every": 1, "weight_window": "lag", "lag": 1},  "backtest_cfg": {"fwd_days": 1}, "data": "non_st"},
    {"name": "BL-19", "factor_cfg": {"calc_every": 2, "weight_window": "lag", "lag": 2},  "backtest_cfg": {"fwd_days": 2}, "data": "non_st"},
    {"name": "BL-20", "factor_cfg": {"calc_every": 3, "weight_window": "lag", "lag": 3},  "backtest_cfg": {"fwd_days": 3}, "data": "non_st"},
    {"name": "BL-21", "factor_cfg": {"calc_every": 5, "weight_window": "lag", "lag": 8},  "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    {"name": "BL-22", "factor_cfg": {"calc_every": 5, "weight_window": "lag", "lag": 10}, "backtest_cfg": {"fwd_days": 5}, "data": "non_st"},
    # ── Granger 因果因子 (日频, 回测=panel_trade 全市场) ──
    # GL-01: 因子=非ST特质收益率
    {"name": "GL-01",
     "factor_cfg": {"factor_type": "granger", "n": 60,
                    "lags": (1, 3, 5, 8, 10),
                    "topk1": 100, "granger_p": 0.05, "topk2": 5},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret"},
    # GL-02: 因子=非ST价格日收益率
    {"name": "GL-02",
     "factor_cfg": {"factor_type": "granger", "n": 60,
                    "lags": (1, 3, 5, 8, 10),
                    "topk1": 100, "granger_p": 0.05, "topk2": 5},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # ── GL-03/04/05: 改变 OLS 中来自其他股票的自变量数量 (1,2,3) ──
    # 三组共享同一张 Granger 宽表, 门槛同步收窄 (len(sig_pairs) >= k)
    # 数据源 = non_st_spret (对应 GL-01 系列)
    {"name": "GL-03",
     "factor_cfg": {"factor_type": "granger", "n": 60,
                    "lags": (1, 3, 5, 8, 10),
                    "topk1": 100, "granger_p": 0.05, "topk2": 1},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret",
     "shared_cache": "granger_non_st_spret_topk123", "shared_topk": 1},
    {"name": "GL-04",
     "factor_cfg": {"factor_type": "granger", "n": 60,
                    "lags": (1, 3, 5, 8, 10),
                    "topk1": 100, "granger_p": 0.05, "topk2": 2},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret",
     "shared_cache": "granger_non_st_spret_topk123", "shared_topk": 2},
    {"name": "GL-05",
     "factor_cfg": {"factor_type": "granger", "n": 60,
                    "lags": (1, 3, 5, 8, 10),
                    "topk1": 100, "granger_p": 0.05, "topk2": 3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret",
     "shared_cache": "granger_non_st_spret_topk123", "shared_topk": 3},
    # ── GL-06/07: 去 Granger, 直接用 Spearman 选 1 对 (j*, lag*) ──
    # 两组共享同一张宽表 (Spearman 矩阵只算一次)
    # GL-06 (pos): 只允许正相关的 j
    # GL-07 (abs): 允许 |相关| 最大, 正负皆可
    {"name": "GL-06",
     "factor_cfg": {"factor_type": "correlation", "n": 60,
                    "lags": (1, 3, 5, 8, 10)},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret",
     "shared_cache": "correlation_non_st_spret", "shared_col": "factor_pos"},
    {"name": "GL-07",
     "factor_cfg": {"factor_type": "correlation", "n": 60,
                    "lags": (1, 3, 5, 8, 10)},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret",
     "shared_cache": "correlation_non_st_spret", "shared_col": "factor_abs"},
    # ── 协整 lead-lag 因子 (CO-01 ~ CO-09) ──
    # 基线: 日频 zscore; CO-04 周频作长持仓对照; CO-08/09 对比 predict/mixed 模式
    # CO-01: 日频 zscore 基线
    {"name": "CO-01",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-02: +短半衰期过滤 (<=5 天)
    {"name": "CO-02",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": 5.0,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-03: 粗筛放宽到 top-200
    {"name": "CO-03",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 200, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-04: 周频持仓对照 (lag/period 均放大)
    {"name": "CO-04",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (5, 10, 15, 20), "calc_every": 5,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 5}, "data": "non_st_price"},
    # CO-05: 严格 p 值 < 0.01
    {"name": "CO-05",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.01, "half_life_max": None,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-06: lag 候选更密集 {1,2,3,5,8,10,15}
    {"name": "CO-06",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 2, 3, 5, 8, 10, 15), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-07: 数据源对照 (cumsum(spret/100) 伪对数价格)
    {"name": "CO-07",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret_cumsum"},
    # CO-08: 因子模式对照 (predict)
    {"name": "CO-08",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "predict"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-09: 因子模式对照 (mixed = -z + 领先动量)
    {"name": "CO-09",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "mixed"},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-10: 同行业内配对 (申万一级行业), 其余同 CO-01
    {"name": "CO-10",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore",
                    "same_industry": True},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-11: 在 CO-10 基础上固定 lag=1; 同时保存 leader_code 以便后续分析领涨股分布
    {"name": "CO-11",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore",
                    "same_industry": True,
                    "save_leader": True},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-12: 在 CO-10 基础上改用"原始价格 (不取 log)" 做协整, 用于验证 log 的必要性
    {"name": "CO-12",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1, 3, 5, 8, 10), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore",
                    "same_industry": True,
                    "use_log": False},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-13: 在 CO-11 基础上将协整检验窗口由 60 扩到 120, 其它参数完全相同
    #        检验"更长窗口能否缓解样本内选择偏差"
    {"name": "CO-13",
     "factor_cfg": {"factor_type": "cointegration", "n": 120,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore",
                    "same_industry": True,
                    "save_leader": True},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-14: 行业龙头驱动 + holdout 验证
    #   leader 池=同行业市值前 30% (日动态, 每日重算)
    #   非 leader 股票 s 从 leader 池中选 j:
    #     训练段 60 天 OLS -> ADF p<0.05 过滤 -> 验证段 20 天 -> 选归一化 RMSE 最小 j*
    #   因子 α/β/σ 均来自训练段, 用于检验"out-of-sample 稳定性"能否改善 CO-11 的样本内过拟合
    {"name": "CO-14",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore",
                    "same_industry": True,
                    "save_leader": True,
                    "use_log": True,
                    "pairing_mode": "holdout",
                    "valid_window": 20,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-15: 在 CO-14 基础上把训练段从 60 天扩到 80 天, 验证段从 20 天扩到 40 天
    #        lookback 总长 60+20=80 -> 80+40=120, 其他参数完全相同
    #        检验"更长训练/更严格验证能否进一步缓解过拟合"
    {"name": "CO-15",
     "factor_cfg": {"factor_type": "cointegration", "n": 80,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "zscore",
                    "same_industry": True,
                    "save_leader": True,
                    "use_log": True,
                    "pairing_mode": "holdout",
                    "valid_window": 40,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-16: 跳出协整框架, 直接用 spret 做 OLS 预测 + holdout 验证 + 行业龙头驱动
    #   训练段: spret_s[τ] = α + β·spret_j[τ-1] + ε (60 天)
    #   验证段: 20 天, 选归一化 RMSE 最小的 j*
    #   T 日因子 = α* + β*·spret_j*[T-1]  (即预测 T 日 s 的特质收益率)
    #   方案 B: 不约束 β 符号, 粗筛按 |Spearman| 降序; 无 ADF 检验 (spret 本身平稳)
    {"name": "CO-16",
     "factor_cfg": {"factor_type": "predictive", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100,
                    "same_industry": True,
                    "save_leader": True,
                    "valid_window": 20,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_spret"},
    # CO-17: 在 CO-14 基础上 factor_mode 由 zscore 改为 predict
    #   因子 = α* + β*·log_p_j[T] + μ - log_p_s[T]
    #   利用 leader T 日价格直接预测 s 的 T+1 对数收益 (弥补 zscore 模式不用 log_p_j[T] 的缺陷)
    {"name": "CO-17",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "predict",
                    "same_industry": True,
                    "save_leader": True,
                    "use_log": True,
                    "pairing_mode": "holdout",
                    "valid_window": 20,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-18: 在 CO-17 基础上乘以回归速率 (1-ρ*), ρ*=训练段 AR(1) 系数
    #   因子 = (1-ρ*) * predict
    #   ρ*>=0 时 (1-ρ*) 夹到 [1e-4,1.0], 避免非回归情形干扰
    {"name": "CO-18",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "predict_decay",
                    "same_industry": True,
                    "save_leader": True,
                    "use_log": True,
                    "pairing_mode": "holdout",
                    "valid_window": 20,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-19: 在 CO-14 基础上 factor_mode 改为 mixed (均值回归 + 领先动量)
    #   因子 = -z + β*·(log_p_j[T] - log_p_j[T-1])
    {"name": "CO-19",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "mixed",
                    "same_industry": True,
                    "save_leader": True,
                    "use_log": True,
                    "pairing_mode": "holdout",
                    "valid_window": 20,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
    # CO-20: 回归概率 × 预测收益
    #   因子 = Φ(|z|·√((1-ρ)/(1+ρ))) * predict
    #   Φ=标准正态 CDF; z=(log_p_s[T]-α-β·log_p_j[T-1]-μ)/σ (训练段 α/β/σ/ρ);
    #   predict=α+β·log_p_j[T]+μ-log_p_s[T] 提供正负号, Φ(·) 仅作振幅调制 (>=0.5)
    {"name": "CO-20",
     "factor_cfg": {"factor_type": "cointegration", "n": 60,
                    "lags": (1,), "calc_every": 1,
                    "topk1": 100, "coint_p": 0.05, "half_life_max": None,
                    "factor_mode": "predict_prob",
                    "same_industry": True,
                    "save_leader": True,
                    "use_log": True,
                    "pairing_mode": "holdout",
                    "valid_window": 20,
                    "leader_mv_pct": 0.3},
     "backtest_cfg": {"fwd_days": 1}, "data": "non_st_price"},
]


def _resolve_paths(data_tag: str):
    """根据 data_tag 返回 (price_path, spret_path)。
    - default:             回测=panel_trade, 因子=specific_ret_cne6_sw21
    - non_st:              回测=price_non_st, 因子=从 price_non_st 算日收益率
    - non_st_spret:        回测=panel_trade, 因子=s_ret_cne6_non_st
    - non_st_price:        回测=panel_trade, 因子=从 price_non_st 算日收益率
    - non_st_spret_cumsum: 回测=panel_trade, 因子=cumsum(spret/100) 伪对数价格 (CO-09)
    """
    if data_tag == "non_st":
        return PRICE_NON_ST_PATH, None
    if data_tag == "non_st_spret":
        return PRICE_PATH, SPRET_NON_ST_PATH
    if data_tag == "non_st_price":
        return PRICE_PATH, None  # 因子从 PRICE_NON_ST_PATH 计算, 在 main 中处理
    if data_tag == "non_st_spret_cumsum":
        return PRICE_PATH, SPRET_NON_ST_PATH  # spret 在 main 中 cumsum 成伪对数价格
    return PRICE_PATH, SPRET_PATH


def _price_to_ret(price_df: pd.DataFrame) -> pd.DataFrame:
    """从价格数据计算 close-to-close 日收益率，输出格式同特质收益率。"""
    df = price_df[["date", "stock_code", "close_price"]].copy()
    df = df.sort_values(["stock_code", "date"])
    df["spret"] = df.groupby("stock_code")["close_price"].pct_change() * 100  # 百分比，与 spret 单位一致
    df = df.dropna(subset=["spret"])[["date", "stock_code", "spret"]].reset_index(drop=True)
    return df


def _build_log_panel(source: str, use_log: bool = True) -> pd.DataFrame:
    """
    构造协整因子所需的面板: [date, stock_code, log_price]。
    列名恒为 log_price (下游统一), 其语义由 use_log 控制:
        source = "non_st_price":
            - use_log=True (默认): log(close_price) from price_non_st.pkl
            - use_log=False:       close_price from price_non_st.pkl (CO-12 对照)
        source = "non_st_spret_cumsum":
            cumsum(spret/100) from s_ret_cne6_non_st.pkl (伪对数价格, 固定为 log 版本)
            use_log=False 对此 source 无定义, 直接报错。
    """
    if source == "non_st_price":
        if use_log:
            print(f"构造协整面板: log(close) from {PRICE_NON_ST_PATH}")
        else:
            print(f"构造协整面板: close (不取 log) from {PRICE_NON_ST_PATH}")
        df = pd.read_pickle(PRICE_NON_ST_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "stock_code", "close_price"]].copy()
        df = df[df["close_price"] > 0].dropna(subset=["close_price"])
        if use_log:
            df["log_price"] = np.log(df["close_price"].astype(np.float64))
        else:
            df["log_price"] = df["close_price"].astype(np.float64)
        return df[["date", "stock_code", "log_price"]].sort_values(["stock_code", "date"]).reset_index(drop=True)

    if source == "non_st_spret_cumsum":
        if not use_log:
            raise ValueError("source=non_st_spret_cumsum 不支持 use_log=False (cumsum 本身就是伪对数价格)")
        print(f"构造协整面板: cumsum(spret/100) from {SPRET_NON_ST_PATH}")
        df = pd.read_pickle(SPRET_NON_ST_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "stock_code", "spret"]].copy()
        df = df.dropna(subset=["spret"]).sort_values(["stock_code", "date"])
        # spret 是百分点,除以 100 得到近似对数收益,累加成伪对数价格
        df["log_price"] = (df["spret"].astype(np.float64) / 100.0)
        df["log_price"] = df.groupby("stock_code")["log_price"].cumsum()
        return df[["date", "stock_code", "log_price"]].reset_index(drop=True)

    raise ValueError(f"不支持的 source: {source}")


def main():
    t_total = time.time()
    # 白名单: 通过环境变量 EXP_FILTER (逗号分隔) 指定只跑哪些实验
    # 例: EXP_FILTER="GL-03,GL-04,GL-05" python3 run_pair_factor.py
    filter_raw = os.environ.get("EXP_FILTER", "").strip()
    if filter_raw:
        white = {s.strip() for s in filter_raw.split(",") if s.strip()}
        exps_run = [e for e in EXPERIMENTS if e["name"] in white]
        print(f"\n>> EXP_FILTER 生效: 只跑 {sorted(e['name'] for e in exps_run)} "
              f"(共 {len(exps_run)}/{len(EXPERIMENTS)} 个)")
    else:
        exps_run = list(EXPERIMENTS)

    print(f"\n{'='*60}")
    print(f"Pair Trading 因子实验  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  n={N}, 回测={BACKTEST_START}~{BACKTEST_END}")
    print(f"  实验数: {len(exps_run)}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # 按 data tag 分组运行，同一数据源只加载一次
    data_tags = sorted(set(exp.get("data", "default") for exp in exps_run))
    tradable_cache = {}
    spret_cache = {}
    log_panel_cache = {}  # CO 因子专用: tag -> DataFrame [date, stock_code, log_price]
    industry_map_cache = {}  # CO same_industry 专用: tag -> Series(index=stock_code, value=sw_l1_code)
    mv_panel_cache = {}  # CO-14 holdout 专用: tag -> DataFrame [date, stock_code, market_value]

    for tag in data_tags:
        tag_exps = [e for e in exps_run if e.get("data", "default") == tag]
        price_path, spret_path = _resolve_paths(tag)

        tag_label = {
            "default": "原始数据",
            "non_st": "剔除ST (因子+回测均用price_non_st)",
            "non_st_spret": "因子=非ST特质收益率, 回测=panel_trade",
            "non_st_price": "因子=非ST价格日收益率, 回测=panel_trade",
            "non_st_spret_cumsum": "因子=cumsum(非ST特质收益率), 回测=panel_trade",
        }.get(tag, tag)
        print(f"\n── 数据源 [{tag}]: {tag_label} ──")

        # 该 tag 下是否有协整实验 / 其他因子实验
        has_coint = any(e["factor_cfg"].get("factor_type") == "cointegration" for e in tag_exps)
        has_non_coint = any(e["factor_cfg"].get("factor_type", "pair") != "cointegration" for e in tag_exps)

        # 加载价格 & 可交易掩码 (所有实验都需要回测价格)
        if tag not in tradable_cache:
            print(f"加载价格数据: {price_path} ...")
            pt = pd.read_pickle(price_path)
            pt["date"] = pd.to_datetime(pt["date"])
            print(f"  {len(pt)} 行")
            print("构建可交易掩码 ...")
            tradable_df = build_tradable_mask(pt)
            n_tradable = tradable_df["tradable"].sum()
            print(f"  可交易: {n_tradable}/{len(tradable_df)} "
                  f"({n_tradable/len(tradable_df)*100:.1f}%)")
            tradable_cache[tag] = tradable_df
            del pt
            gc.collect()

        # 按需加载特质收益率 (非协整因子需要)
        if has_non_coint and tag not in spret_cache:
            need_spret = any(
                e["factor_cfg"].get("factor_type", "pair") != "cointegration"
                and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "factor_df.pkl"))
                and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "metrics.json"))
                for e in tag_exps
            )
            if need_spret:
                if spret_path is not None and tag != "non_st_spret_cumsum":
                    print(f"加载特质收益率: {spret_path} ...")
                    spret = pd.read_pickle(spret_path)
                    spret["date"] = pd.to_datetime(spret["date"])
                elif tag == "non_st_price":
                    print(f"从 {PRICE_NON_ST_PATH} 计算日收益率 ...")
                    pnst = pd.read_pickle(PRICE_NON_ST_PATH)
                    pnst["date"] = pd.to_datetime(pnst["date"])
                    spret = _price_to_ret(pnst)
                    del pnst
                    gc.collect()
                else:
                    # 默认 non_st: 因子和回测共享 price_non_st.pkl
                    print(f"从价格数据计算日收益率 ...")
                    spret = _price_to_ret(tradable_cache[tag])
                print(f"  {len(spret)} 行, {spret['stock_code'].nunique()} 只股票")
                spret_cache[tag] = spret
            else:
                spret_cache[tag] = None

        # 按需加载 log_panel (协整因子需要); 按 (tag, use_log) 切分, 同一 tag 可并存 log/price 两套
        if has_coint:
            coint_exps = [e for e in tag_exps if e["factor_cfg"].get("factor_type") == "cointegration"]
            use_log_variants = sorted({
                bool(e["factor_cfg"].get("use_log", True)) for e in coint_exps
            }, reverse=True)  # True 优先, 便于日志顺序
            for ul in use_log_variants:
                cache_key = (tag, ul)
                if cache_key in log_panel_cache:
                    continue
                need_panel = any(
                    bool(e["factor_cfg"].get("use_log", True)) == ul
                    and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "factor_df.pkl"))
                    and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "metrics.json"))
                    for e in coint_exps
                )
                if need_panel:
                    panel = _build_log_panel(tag, use_log=ul)
                    log_panel_cache[cache_key] = panel
                    print(f"  协整面板[use_log={ul}]: {len(panel)} 行, "
                          f"{panel['stock_code'].nunique()} 只股票")
                else:
                    log_panel_cache[cache_key] = None

        # 按需加载 industry_map (CO/predictive 同行业过滤需要): stock_code -> sw_l1_code
        # 仅在有 same_industry=True 的 CO 或 predictive 实验且尚未完成时加载
        if tag not in industry_map_cache:
            need_ind_map = any(
                e["factor_cfg"].get("factor_type") in ("cointegration", "predictive")
                and e["factor_cfg"].get("same_industry", False)
                and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "factor_df.pkl"))
                and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "metrics.json"))
                for e in tag_exps
            )
            if need_ind_map:
                # 行业数据随 tag 选择源: non_st_spret_cumsum 用 s_ret_cne6_non_st, 其他都用 price_non_st
                ind_src = SPRET_NON_ST_PATH if tag == "non_st_spret_cumsum" else PRICE_NON_ST_PATH
                print(f"加载行业映射 (stock_code -> sw_industry_l1_code) from {ind_src} ...")
                ind_df = pd.read_pickle(ind_src)
                # 每只股票行业代码在全时段唯一, 直接 first 即可
                ind_map = (ind_df.groupby("stock_code")["sw_industry_l1_code"]
                           .first())
                print(f"  行业映射: {len(ind_map)} 只股票, {ind_map.nunique()} 个行业")
                industry_map_cache[tag] = ind_map
                del ind_df
                gc.collect()
            else:
                industry_map_cache[tag] = None

        # 按需加载 market_value 面板 (CO-14 holdout + leader_mv_pct 过滤 / CO-16 predictive 均需要)
        if tag not in mv_panel_cache:
            need_mv = any(
                (
                    (e["factor_cfg"].get("factor_type") == "cointegration"
                     and e["factor_cfg"].get("pairing_mode") == "holdout"
                     and e["factor_cfg"].get("leader_mv_pct") is not None)
                    or
                    (e["factor_cfg"].get("factor_type") == "predictive"
                     and e["factor_cfg"].get("leader_mv_pct") is not None)
                )
                and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "factor_df.pkl"))
                and not os.path.exists(os.path.join(OUTPUT_DIR, e["name"], "metrics.json"))
                for e in tag_exps
            )
            if need_mv:
                # market_value 仅存在于 price_non_st.pkl 中 (spret 表没有市值)
                print(f"加载市值面板 (stock_code, date, market_value) from {PRICE_NON_ST_PATH} ...")
                mv_df = pd.read_pickle(PRICE_NON_ST_PATH)
                mv_df["date"] = pd.to_datetime(mv_df["date"])
                mv_df = mv_df[["date", "stock_code", "market_value"]].copy()
                # 过滤 mv<=0 (退市/异常样本) 以免影响行业内百分位排名
                mv_df = mv_df[mv_df["market_value"] > 0]
                print(f"  市值面板: {len(mv_df)} 条, "
                      f"{mv_df['stock_code'].nunique()} 只股票, "
                      f"{mv_df['date'].nunique()} 个截面")
                mv_panel_cache[tag] = mv_df
                del mv_df
                gc.collect()
            else:
                mv_panel_cache[tag] = None

        # 按 shared_cache 分组, 预生成 Granger 宽表 (GL-03/04/05 共享)
        # shared_wide_cache[name] -> DataFrame (含 factor_top{k} 列)
        shared_wide_cache = {}
        shared_groups = {}
        for e in tag_exps:
            sc_name = e.get("shared_cache")
            if sc_name:
                shared_groups.setdefault(sc_name, []).append(e)

        for sc_name, members in shared_groups.items():
            # 所有成员都已完成则无需生成
            all_done = all(
                os.path.exists(os.path.join(OUTPUT_DIR, m["name"], "metrics.json"))
                or os.path.exists(os.path.join(OUTPUT_DIR, m["name"], "factor_df.pkl"))
                for m in members
            )
            if all_done:
                shared_wide_cache[sc_name] = None
                continue

            wide_dir = os.path.join(OUTPUT_DIR, f"_shared_{sc_name}")
            os.makedirs(wide_dir, exist_ok=True)
            wide_path = os.path.join(wide_dir, "factor_wide.pkl")

            if os.path.exists(wide_path):
                print(f"\n[shared_cache={sc_name}] 加载已有宽表: {wide_path}")
                wide_df = pd.read_pickle(wide_path)
                wide_df["date"] = pd.to_datetime(wide_df["date"])
            else:
                # 以第一个成员的 factor_cfg 为基础生成宽表
                base_cfg = dict(members[0]["factor_cfg"])
                factor_type_shared = base_cfg.get("factor_type", "granger")
                base_cfg = {k: v for k, v in base_cfg.items() if k != "factor_type"}

                if factor_type_shared == "granger":
                    # Granger: 宽表多列 factor_top{k}, 成员用 shared_topk 指定 k
                    topk_list_union = sorted({
                        int(m.get("shared_topk", 0)) for m in members
                    })
                    topk_list_union = [k for k in topk_list_union if k > 0]
                    if not topk_list_union:
                        raise ValueError(f"shared_cache={sc_name} 的成员未声明 shared_topk")
                    base_cfg.pop("topk2", None)
                    base_cfg["topk_list"] = tuple(topk_list_union)
                    print(f"\n[shared_cache={sc_name}] 生成 Granger 宽表, "
                          f"topk_list={topk_list_union}")
                    wide_df = build_granger_factor(spret_cache.get(tag), **base_cfg)
                elif factor_type_shared == "correlation":
                    # Correlation: 宽表固定 2 列 factor_pos/factor_abs, 成员用 shared_col 指定列
                    print(f"\n[shared_cache={sc_name}] 生成相关系数宽表 (factor_pos/factor_abs)")
                    wide_df = build_correlation_factor(spret_cache.get(tag), **base_cfg)
                else:
                    raise ValueError(f"shared_cache={sc_name} 不支持的因子类型: {factor_type_shared}")

                wide_df.to_pickle(wide_path)
                print(f"  宽表已保存: {wide_path}, 行数={len(wide_df)}")

            shared_wide_cache[sc_name] = wide_df

        # 运行实验
        for exp in tag_exps:
            sc_name = exp.get("shared_cache")
            sc_df = shared_wide_cache.get(sc_name) if sc_name else None
            # shared_col 优先 (correlation 等); 否则 shared_topk 兼容 GL-03/04/05
            if sc_name and exp.get("shared_col"):
                sc_col = exp["shared_col"]
            elif sc_name and exp.get("shared_topk"):
                sc_col = f"factor_top{int(exp['shared_topk'])}"
            else:
                sc_col = None
            # CO 实验按 use_log 选取对应面板; 非 CO 实验拿不到面板也无所谓
            ul_key = bool(exp["factor_cfg"].get("use_log", True))
            panel_for_exp = log_panel_cache.get((tag, ul_key))
            run_one_experiment(
                exp_name=exp["name"],
                tradable_df=tradable_cache[tag],
                spret=spret_cache.get(tag),
                log_panel=panel_for_exp,
                factor_cfg=exp["factor_cfg"],
                backtest_cfg=exp["backtest_cfg"],
                shared_wide_df=sc_df,
                factor_col=sc_col,
                industry_map=industry_map_cache.get(tag),
                mv_panel=mv_panel_cache.get(tag),
            )

    # 汇总
    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print("实验汇总:")
    for exp in exps_run:
        mpath = os.path.join(OUTPUT_DIR, exp["name"], "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            ex = m.get("head_excess", {})
            data_tag = exp.get("data", "default")
            tag_label = f"[{data_tag}]" if data_tag != "default" else ""
            print(f"  {exp['name']}{tag_label}: IC={m.get('rank_ic_mean','nan'):.4f}, "
                  f"超额年化={ex.get('annual_return','nan')}%, "
                  f"夏普={ex.get('sharpe_ratio','nan')}, "
                  f"回撤={ex.get('max_drawdown','nan')}%")
    print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
