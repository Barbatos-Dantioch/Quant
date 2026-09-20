#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pair-trading 实验通用骨架

本模块从 run_pair_factor.py 抽出可复用的工具函数:
  ─ 搬运（一字不动）─
    _DF_TAU_C / _DF_P_LEVEL / _tau_to_pvalue / _batch_adf_pvalues
    build_tradable_mask
    group_backtest / head_group_backtest
    calc_stats
    plot_group_backtest / plot_head_backtest / plot_yearly_head

  ─ 新增（OU 配对实验专用）─
    load_price_industry_mv  : 读取价格 + 行业 + 市值长表
    compute_short_pool      : 每日空头池（市值前 50% ∩ 波动率前 40% ∩ tradable）
    run_longshort_backtest  : 多空组合回测 + 绘图，封装 analyze_longshort_holdings
    save_summary_row        : 追加写入 summary.csv

run_pair_factor.py 与 run_ou_pair.py 都从本模块 import。
"""
from __future__ import annotations

import csv
import gc
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

# ── 引入 func/ 目录 ──
_FUNC_DIR = "/root/quant/xgbcode/func"
if _FUNC_DIR not in sys.path:
    sys.path.insert(0, _FUNC_DIR)
from model_backtest_framework import analyze_model_holdings
from model_backtest_framework_longshort import analyze_longshort_holdings


# =====================================================================
# ADF 矩阵化（搬自 run_pair_factor.py 第 741-951 行）
# =====================================================================

# Dickey-Fuller 检验临界值表 (MacKinnon 1996, 带常数项, 样本量接近 60)
# 对应显著性 [0.01, 0.025, 0.05, 0.10]
_DF_TAU_C = np.array([-3.56, -3.22, -2.92, -2.60])
_DF_P_LEVEL = np.array([0.01, 0.025, 0.05, 0.10])


def _tau_to_pvalue(tau_stat: np.ndarray) -> np.ndarray:
    """
    ADF t 统计量转 p 值 (单边左尾近似,使用 Dickey-Fuller 临界值表线性插值)。

    原理:
        - tau 越负 → 越可能拒绝原假设(存在单位根) → p 越小
        - tau >= -1.5 近似 p ≈ 1.0 (无法拒绝)
        - tau <= -4.0 近似 p ≈ 0.001
        - 区间内按 (tau, p) 表线性插值
    """
    p_out = np.full_like(tau_stat, 0.5, dtype=np.float64)

    # 极端区间近似
    p_out[tau_stat >= -1.5] = np.clip(0.5 + (tau_stat[tau_stat >= -1.5] + 1.5) * 0.1, 0.1, 1.0)
    p_out[tau_stat <= -4.0] = 0.001

    # 中间区间 [-4.0, -1.5] 用表插值
    mid = (tau_stat > -4.0) & (tau_stat < -1.5)
    if mid.any():
        tau_mid = tau_stat[mid]
        # 插值节点: tau 从小到大, p 从小到大 (tau 越大 p 越大)
        tau_nodes = np.array([-4.0, -3.56, -3.22, -2.92, -2.60, -1.50])
        p_nodes = np.array([0.001, 0.01, 0.025, 0.05, 0.10, 0.50])
        p_out[mid] = np.interp(tau_mid, tau_nodes, p_nodes)

    return p_out


def _batch_adf_pvalues(
    spread: np.ndarray,        # (n, M)
) -> tuple:
    """
    对 M 个 spread 序列批量做带常数项 + 1 阶滞后差分的 ADF 检验。

    模型: Δs[τ] = c + ρ·s[τ-1] + γ·Δs[τ-1] + η[τ]
    样本量: n-2 (丢失 2 个观测: 一阶差分 + 再滞后一阶差分)

    返回:
        pvals: (M,) ADF p 值
        rho:   (M,) ρ 的估计值 (用于半衰期)
        tau:   (M,) t 统计量 (诊断用)
    """
    n, M = spread.shape
    if n < 10:
        return np.ones(M), np.zeros(M), np.zeros(M)

    # Δs[τ]     = s[τ]  - s[τ-1],   τ = 2 ... n-1  → 长度 n-2
    # s[τ-1]                                      → 长度 n-2
    # Δs[τ-1]   = s[τ-1]- s[τ-2]                   → 长度 n-2
    dy = spread[2:, :] - spread[1:-1, :]          # (n-2, M)
    lag_y = spread[1:-1, :]                        # (n-2, M)
    dy_lag = spread[1:-1, :] - spread[:-2, :]     # (n-2, M)

    W = n - 2
    # 设计矩阵每列: [1, lag_y, dy_lag]   → 3 列
    # OLS 对每个 spread 独立: (3, 3) × M, 矩阵化实现

    # 构造 X.T @ X 的 6 个唯一分量 (对称矩阵)
    one_sum = float(W)                                                  # Σ1
    lag_sum = lag_y.sum(axis=0)                                          # (M,)
    dylag_sum = dy_lag.sum(axis=0)                                       # (M,)
    lag_sq = (lag_y ** 2).sum(axis=0)                                    # (M,)
    dylag_sq = (dy_lag ** 2).sum(axis=0)                                 # (M,)
    lag_dylag = (lag_y * dy_lag).sum(axis=0)                             # (M,)

    # X.T @ dy
    Xty_0 = dy.sum(axis=0)                                               # (M,) : Σdy
    Xty_1 = (lag_y * dy).sum(axis=0)                                     # (M,) : Σ lag_y·dy
    Xty_2 = (dy_lag * dy).sum(axis=0)                                    # (M,) : Σ dy_lag·dy

    # 求解 3x3 线性方程 A·β = b,批量 M 个 (3x3 对称)
    # A = [[W,        lag_sum,  dylag_sum],
    #      [lag_sum,  lag_sq,   lag_dylag],
    #      [dylag_sum,lag_dylag,dylag_sq ]]
    # 行列式和伴随矩阵 (3x3 通用公式)
    a11 = np.full(M, one_sum)
    a12 = lag_sum;    a13 = dylag_sum
    a22 = lag_sq;     a23 = lag_dylag; a33 = dylag_sq
    # A.T = A,使用对称性

    # 行列式 det(A)
    det = (a11 * (a22 * a33 - a23 * a23)
           - a12 * (a12 * a33 - a23 * a13)
           + a13 * (a12 * a23 - a22 * a13))

    # 防御: 奇异矩阵直接给 p=1
    safe = np.abs(det) > 1e-10
    pvals = np.ones(M, dtype=np.float64)
    rho_est = np.zeros(M, dtype=np.float64)
    tau_est = np.zeros(M, dtype=np.float64)
    if not safe.any():
        return pvals, rho_est, tau_est

    # 逆矩阵 (3x3),只需要第二列 (对应 lag_y 的系数即 ρ)
    # A^(-1)[i,j] = cofactor(A)[j,i] / det
    # ρ = A^(-1)[1,:] · Xty = (c21·Xty0 + c22·Xty1 + c23·Xty2) / det
    # c21 = -(a12·a33 - a23·a13),  c22 = (a11·a33 - a13·a13),  c23 = -(a11·a23 - a13·a12)
    c21 = -(a12 * a33 - a23 * a13)
    c22 =  (a11 * a33 - a13 * a13)
    c23 = -(a11 * a23 - a13 * a12)

    # 逆矩阵对角线第 2 项(用于 ρ 的方差):A^(-1)[1,1] = c22 / det
    Ainv_11 = np.where(safe, c22 / np.where(safe, det, 1.0), np.nan)

    rho_numer = c21 * Xty_0 + c22 * Xty_1 + c23 * Xty_2
    rho = np.where(safe, rho_numer / np.where(safe, det, 1.0), np.nan)

    # 还要算常数项和 γ 以计算残差
    c11 =  (a22 * a33 - a23 * a23)
    c12 = -(a12 * a33 - a23 * a13)
    c13 =  (a12 * a23 - a22 * a13)
    c31 = c13   # 对称
    c32 = c23
    c33 =  (a11 * a22 - a12 * a12)

    c_numer = c11 * Xty_0 + c12 * Xty_1 + c13 * Xty_2
    g_numer = c31 * Xty_0 + c32 * Xty_1 + c33 * Xty_2
    c_const = np.where(safe, c_numer / np.where(safe, det, 1.0), 0.0)
    gamma = np.where(safe, g_numer / np.where(safe, det, 1.0), 0.0)

    # 残差平方和 (避免构造 W×M 的 pred 矩阵):
    #   SSR = Σ dy² - β^T · (X^T dy) = Σdy² - (c·Σdy + ρ·Σ(lag_y·dy) + γ·Σ(dy_lag·dy))
    yty = (dy ** 2).sum(axis=0)                                          # (M,) Σdy²
    ssr = yty - (c_const * Xty_0 + rho * Xty_1 + gamma * Xty_2)
    ssr = np.maximum(ssr, 0.0)                                           # 防御数值误差
    dof = W - 3
    if dof <= 0:
        return pvals, rho_est, tau_est
    sigma2 = ssr / dof                                                   # (M,)

    var_rho = np.where(safe, sigma2 * Ainv_11, np.inf)
    se_rho = np.sqrt(np.maximum(var_rho, 1e-24))
    tau = np.where(safe, rho / se_rho, 0.0)                              # (M,)

    pvals = _tau_to_pvalue(tau)
    rho_est = np.where(safe, rho, 0.0)
    tau_est = np.where(safe, tau, 0.0)
    return pvals, rho_est, tau_est


# =====================================================================
# 可交易掩码（搬自 run_pair_factor.py 第 1802-1814 行）
# =====================================================================

def build_tradable_mask(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建可交易掩码 DataFrame (date, stock_code, tradable, status, open_price, close_price)。
    status==0 即为正常可交易（已包含涨跌停、停牌等过滤）。
    """
    df = price_df.copy()
    df["tradable"] = (
        (df["status"] == 0)
        & (df["open_price"] > 0)
        & df["open_price"].notna()
        & df["close_price"].notna()
    )
    return df[["date", "stock_code", "tradable", "status", "open_price", "close_price"]].copy()


# =====================================================================
# 分层回测（搬自 run_pair_factor.py 第 1821-1915 行）
# =====================================================================

def group_backtest(
    factor_df: pd.DataFrame,
    tradable_df: pd.DataFrame,
    n_groups: int = 10,
    fwd_days: int = 1,
) -> Dict:
    """
    分层回测。
    fwd_days=1: 日频，用下一日收益; fwd_days=5: 周频，用未来5日收益。
    输出各组相对基准的累计超额净值。
    """
    freq_desc = "日频" if fwd_days == 1 else f"{fwd_days}日"
    print(f"分层回测（{freq_desc}调仓）...")
    t0 = time.time()

    price = tradable_df.sort_values(["stock_code", "date"]).copy()
    price["fwd_close"] = price.groupby("stock_code")["close_price"].shift(-fwd_days)
    price["fwd_tradable"] = price.groupby("stock_code")["tradable"].shift(-1)
    price["fwd_ret"] = (price["fwd_close"] - price["close_price"]) / price["close_price"]

    merged = factor_df.merge(
        price[["date", "stock_code", "tradable", "fwd_ret", "fwd_tradable"]],
        on=["date", "stock_code"],
        how="inner",
    )
    merged = merged[merged["tradable"] & merged["fwd_tradable"].fillna(False)].copy()
    merged = merged.dropna(subset=["factor", "fwd_ret"])

    # 逐日分组
    group_returns = {g: [] for g in range(n_groups)}
    avg_returns = []
    ic_list = []
    dates_list = []

    for dt, grp in merged.groupby("date"):
        if len(grp) < n_groups * 2:
            continue
        grp = grp.copy()
        grp["group"] = pd.qcut(
            grp["factor"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        dates_list.append(dt)
        avg_returns.append(grp["fwd_ret"].mean())

        for g in range(n_groups):
            g_ret = grp.loc[grp["group"] == g, "fwd_ret"].mean()
            group_returns[g].append(g_ret if not np.isnan(g_ret) else 0.0)

        ic, _ = spearmanr(grp["factor"], grp["fwd_ret"])
        if not np.isnan(ic):
            ic_list.append(ic)

    # 各组超额收益 = 组收益 - 全截面等权收益, 然后累乘得到超额净值
    avg_arr = np.array(avg_returns)
    excess_nav = {}
    for g in range(n_groups):
        ret_arr = np.array(group_returns[g])
        excess_ret = ret_arr - avg_arr
        excess_nav[g] = np.cumprod(1 + excess_ret)

    elapsed = time.time() - t0
    ic_mean = np.mean(ic_list) if ic_list else np.nan
    ic_std = np.std(ic_list) if ic_list else np.nan
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan

    # 头组超额统计
    head_excess_ret = np.array(group_returns[n_groups - 1]) - avg_arr
    head_excess_nav_series = np.cumprod(1 + head_excess_ret)
    ann_factor = 250.0 / fwd_days
    head_ann_excess = float(np.mean(head_excess_ret)) * ann_factor
    head_ann_excess_vol = float(np.std(head_excess_ret)) * np.sqrt(ann_factor)
    head_excess_ir = head_ann_excess / head_ann_excess_vol if head_ann_excess_vol > 0 else 0.0
    running_peak = np.maximum.accumulate(head_excess_nav_series)
    head_excess_max_dd = float(np.min(head_excess_nav_series / running_peak - 1))

    print(f"  分层回测完成: {len(dates_list)} 个截面, "
          f"IC={ic_mean:.4f}, ICIR={ic_ir:.4f}, 耗时 {elapsed:.0f}s")
    print(f"  头组超额: 年化={head_ann_excess*100:.2f}%, "
          f"IR={head_excess_ir:.3f}, 回撤={head_excess_max_dd*100:.2f}%")

    return {
        "excess_nav": excess_nav,
        "dates": dates_list,
        "ic_list": ic_list,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "n_groups": n_groups,
        "head_ann_excess": head_ann_excess,
        "head_ann_excess_vol": head_ann_excess_vol,
        "head_excess_ir": head_excess_ir,
        "head_excess_max_dd": head_excess_max_dd,
        "head_excess_total": float(head_excess_nav_series[-1]),
    }


# =====================================================================
# 头组持仓回测（搬自 run_pair_factor.py 第 1922-2032 行）
# =====================================================================

def head_group_backtest(
    factor_df: pd.DataFrame,
    tradable_df: pd.DataFrame,
    n_groups: int = 10,
    commission_rate: float = 0.0007,
    fwd_days: int = 1,
) -> pd.DataFrame:
    """
    头组持仓回测 (对齐 func/model_backtest_framework.analyze_model_holdings):
    - T 日选股 → T+1 日开盘建仓, 每 fwd_days 个交易日换仓
    - 持仓按 "隔夜(旧仓) + 日内(新仓)" 分段记账, 等价于 open(T+1) → open(T+1+fwd_days)
    - fwd_days=1: 日频; fwd_days=5: 周频
    - 基准: func 默认 close-to-close 全样本等权 (benchmark_data=None)

    返回字段:
        head_nav, benchmark_nav, excess_nav, head_ret, bm_ret, n_holdings
    """
    freq_desc = "日频" if fwd_days == 1 else f"{fwd_days}日"
    print(f"头组持仓回测（{freq_desc}调仓, 对齐 func）...")
    t0 = time.time()

    # ── Step 1: 构造 open_prices / close_prices / status 宽表 ──
    price = tradable_df.sort_values(["stock_code", "date"]).copy()
    open_wide = price.pivot(index="date", columns="stock_code", values="open_price").sort_index()
    close_wide = price.pivot(index="date", columns="stock_code", values="close_price").sort_index()
    status_wide = price.pivot(index="date", columns="stock_code", values="status").sort_index()
    tradable_wide = price.pivot(index="date", columns="stock_code", values="tradable").sort_index().fillna(False)

    # ── Step 2: 构造头组信号 (T 日信号 → T+1 日建仓) ──
    # 使用与原实现一致的逻辑: 当日 tradable 过滤 + qcut 分十组 + 取最后一组
    signal_rows = []
    all_dates = sorted(pd.to_datetime(factor_df["date"].unique()))
    for dt in all_dates:
        if dt not in tradable_wide.index:
            continue
        tradable_today = tradable_wide.loc[dt]
        tradable_codes = set(tradable_today[tradable_today].index)

        day_factor = factor_df[factor_df["date"] == dt]
        day_factor = day_factor[day_factor["stock_code"].isin(tradable_codes)]
        if len(day_factor) < n_groups * 2:
            continue

        group = pd.qcut(
            day_factor["factor"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        head_codes = (
            day_factor.loc[group == n_groups - 1, "stock_code"]
            .dropna().astype(str).drop_duplicates().tolist()
        )
        if head_codes:
            signal_rows.append((dt, head_codes))

    if not signal_rows:
        print("  头组信号为空, 返回空 nav_df")
        return pd.DataFrame(columns=["head_nav", "benchmark_nav", "head_ret", "bm_ret",
                                      "n_holdings", "excess_nav"]).rename_axis("date")

    signal_idx, signal_vals = zip(*signal_rows)
    holdings_series = pd.Series(list(signal_vals), index=pd.DatetimeIndex(signal_idx),
                                 name="head_holdings")

    # ── Step 3: 调用 func 的 analyze_model_holdings ──
    method = "daily_rebalance" if fwd_days == 1 else "periodic_rebalance"
    results = analyze_model_holdings(
        holdings_data=holdings_series,
        open_prices=open_wide,
        close_prices=close_wide,
        benchmark_data=None,          # A 方案: 用 func 默认 close-to-close 全样本等权
        status_data=status_wide,      # 仅买入日要求 status == 0
        method=method,
        holding_period=fwd_days,
        commission_rate=commission_rate,
        verbose=False,
    )

    # ── Step 4: 转成现有返回格式 (保持字段名不变, 上游画图/stats 不用改) ──
    nav = results["nav"]
    bm_nav = results["benchmark_nav"]
    excess_nav = results["excess_nav"]
    holdings_schedule = results.get("holdings_schedule", {}) or {}

    nav_df = pd.DataFrame({
        "head_nav": nav,
        "benchmark_nav": bm_nav,
        "excess_nav": excess_nav,
    })
    nav_df["head_ret"] = nav_df["head_nav"].pct_change().fillna(0.0)
    nav_df["bm_ret"] = nav_df["benchmark_nav"].pct_change().fillna(0.0)

    # n_holdings: 每日追踪当前持仓大小
    # analyze_model_holdings 的 holdings_schedule 仅在"实际调仓日"有 key,
    # 非调仓日沿用前一次的 holdings.
    sched_sorted = sorted(holdings_schedule.keys())
    counts = []
    cur_n = 0
    j = 0
    for dt in nav_df.index:
        while j < len(sched_sorted) and sched_sorted[j] <= dt:
            cur_n = len(holdings_schedule[sched_sorted[j]])
            j += 1
        counts.append(cur_n)
    nav_df["n_holdings"] = counts
    nav_df.index.name = "date"

    elapsed = time.time() - t0
    print(f"  头组回测完成: {len(nav_df)} 个交易日, "
          f"调仓 {len(holdings_schedule)} 次, 耗时 {elapsed:.0f}s")

    return nav_df


# =====================================================================
# 净值统计（搬自 run_pair_factor.py 第 2039-2062 行）
# =====================================================================

def calc_stats(nav_series: pd.Series, name: str, fwd_days: int = 1) -> Dict:
    """从净值序列计算常用统计指标。

    nav_series 每个点代表 fwd_days 个交易日的复合净值,
    因此年化需按 fwd_days 调整 (否则周频/双周频会严重高估年化)。
    """
    ret = nav_series.pct_change().dropna()
    total_ret = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
    yrs = len(nav_series) * fwd_days / 250.0
    ann_ret = ((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    ann_vol = ret.std() * np.sqrt(250.0 / fwd_days) * 100
    sharpe = (ann_ret - 3) / ann_vol if ann_vol > 0 else 0
    dd = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min() * 100
    win_rate = (ret > 0).sum() / len(ret) * 100 if len(ret) > 0 else 0

    return {
        "name": name,
        "total_return": round(total_ret, 2),
        "annual_return": round(ann_ret, 2),
        "annual_volatility": round(ann_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(dd, 2),
        "win_rate": round(win_rate, 1),
    }


# =====================================================================
# 绘图（搬自 run_pair_factor.py 第 2069-2179 行）
# =====================================================================

def plot_group_backtest(bt_result: Dict, save_path: str):
    """分组超额净值图，与 BL-01 backtest.png 格式一致。"""
    excess_nav = bt_result["excess_nav"]
    dates = bt_result["dates"]
    n_groups = bt_result["n_groups"]

    plot_dates = pd.to_datetime(dates)
    # 前插一个起始点 (净值=1)
    start_date = plot_dates[0] - pd.Timedelta(days=1)
    plot_dates = pd.DatetimeIndex([start_date]).append(plot_dates)

    plt.figure(figsize=(12, 6))
    for g in range(n_groups):
        nav_arr = np.concatenate([[1.0], excess_nav[g]])
        plt.plot(plot_dates, nav_arr, label=f"Group {g}", alpha=0.8)
    plt.title("Group Backtest Excess Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Excess Return")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close("all")
    print(f"  已保存: {save_path}")


# =====================================================================
# Pair 级分组回测 (按 abs(signal) 分组, 绝对累计净值)
# =====================================================================

def pair_group_backtest(
    pair_log: pd.DataFrame,
    tradable_df: pd.DataFrame,
    n_groups: int = 5,
    fwd_days: int = 5,
    legal_only: bool = True,
    score_col: Optional[str] = None,
) -> Dict:
    """对 pair_log 按打分列分组, 计算每组的"sign(signal)*(ret_i - ret_j)"绝对累计净值。

    参数
    ----
    pair_log : 需要包含 [date, stock_i, stock_j, signal, is_legal] (+ score_col 若指定)
    tradable_df : 长表 [date, stock_code, close_price, tradable], 由 build_tradable_mask 产出
    n_groups : 分组数 (按打分列排序后分位)
    fwd_days : 持有期 (close_{t+fwd}/close_t - 1); 同时也是采样步长 (无重叠)
    legal_only : 仅保留 is_legal=True 的 pair
    score_col : 分组打分列名; None(默认)=按 |signal| 分组(向后兼容);
                指定列名(如 abs_pred)则按该列分组, 方向仍用 sign(signal)

    采样口径 (与 holding_period=fwd_days 对齐, 无重叠):
        - 取 pair_log 中所有 unique 日期排序, 每 fwd_days 个取一个 (第 0, fwd, 2*fwd, ...)
        - 每个采样日: 截面分组 -> 计算每组的 5 日 pair 收益均值 -> 累乘
        - 结果即"每 fwd_days 日刷新一次持仓"的纯净累计净值, 不重叠不重叠

    截面口径:
        - 仅保留 tradable_t & tradable_{t+1} 的 i、j
        - 截面合法 pair 数 < n_groups*2 时跳过该截面
        - 每对收益 = sign(signal) * (ret_i - ret_j), ret = close_{t+fwd}/close_t - 1
        - 组收益 = 组内 pair 收益等权均值
        - 累计净值 = cumprod(1 + 组收益), 绝对值, 不减基准

    返回 dict:
        abs_nav / dates / n_groups / ic_mean / ic_ir / n_pairs_per_group_avg / fwd_days
    """
    print(f"Pair 分组回测 (n_groups={n_groups}, fwd_days={fwd_days}, "
          f"legal_only={legal_only}, 采样步长={fwd_days}d 无重叠)...")
    t0 = time.time()

    pl = pair_log
    if legal_only and "is_legal" in pl.columns:
        pl = pl[pl["is_legal"] == True]  # noqa: E712
    _keep = ["date", "stock_i", "stock_j", "signal"]
    if score_col is not None and score_col not in _keep:
        _keep.append(score_col)
    pl = pl[_keep].copy()
    pl["date"] = pd.to_datetime(pl["date"]).dt.normalize()
    pl["stock_i"] = pl["stock_i"].astype(str).str.zfill(6)
    pl["stock_j"] = pl["stock_j"].astype(str).str.zfill(6)
    pl = pl.dropna(subset=["signal"])
    pl = pl[pl["signal"] != 0.0].reset_index(drop=True)

    # 每 fwd_days 个截面取一个 (无重叠采样, 与 holding_period=fwd_days 对齐)
    all_dates_sorted = np.sort(pl["date"].unique())
    sampled_dates = all_dates_sorted[::fwd_days]
    sampled_set = set(pd.to_datetime(sampled_dates))
    pl = pl[pl["date"].isin(sampled_set)].reset_index(drop=True)
    print(f"  截面采样: 全部 {len(all_dates_sorted)} 截面 -> 每 {fwd_days}d 取 1, "
          f"保留 {len(sampled_dates)} 个采样截面")

    # 准备 fwd 收益与可交易标志 (基于完整 tradable_df, 不受采样影响)
    px = tradable_df.sort_values(["stock_code", "date"]).copy()
    px["fwd_close"] = px.groupby("stock_code")["close_price"].shift(-fwd_days)
    px["fwd_tradable"] = px.groupby("stock_code")["tradable"].shift(-1)
    px["fwd_ret"] = (px["fwd_close"] - px["close_price"]) / px["close_price"]
    px = px[["date", "stock_code", "tradable", "fwd_tradable", "fwd_ret"]]

    pl = pl.merge(
        px.rename(columns={"stock_code": "stock_i", "tradable": "i_trad",
                            "fwd_tradable": "i_fwd_trad", "fwd_ret": "i_ret"}),
        on=["date", "stock_i"], how="left",
    )
    pl = pl.merge(
        px.rename(columns={"stock_code": "stock_j", "tradable": "j_trad",
                            "fwd_tradable": "j_fwd_trad", "fwd_ret": "j_ret"}),
        on=["date", "stock_j"], how="left",
    )
    mask = (pl["i_trad"].fillna(False) & pl["j_trad"].fillna(False)
            & pl["i_fwd_trad"].fillna(False) & pl["j_fwd_trad"].fillna(False))
    pl = pl[mask].dropna(subset=["i_ret", "j_ret"]).copy()

    pl["pair_ret"] = np.sign(pl["signal"]) * (pl["i_ret"] - pl["j_ret"])
    # 分组打分: 默认按 |signal| (向后兼容); 指定 score_col 时按该列 (如 |pred ΔX|)
    pl["_score"] = np.abs(pl["signal"]) if score_col is None else pl[score_col]

    group_returns = {g: [] for g in range(n_groups)}
    dates_list = []
    ic_list = []
    pair_counts = []

    for dt, grp in pl.groupby("date"):
        if len(grp) < n_groups * 2:
            continue
        grp = grp.copy()
        grp["group"] = pd.qcut(
            grp["_score"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        if grp["group"].nunique() < n_groups:
            continue
        dates_list.append(dt)
        pair_counts.append(len(grp))

        for g in range(n_groups):
            g_ret = grp.loc[grp["group"] == g, "pair_ret"].mean()
            group_returns[g].append(float(g_ret))

        ic, _ = spearmanr(grp["_score"], grp["pair_ret"])
        if not np.isnan(ic):
            ic_list.append(ic)

    # 绝对累计净值 (不减基准)
    abs_nav = {g: np.cumprod(1.0 + np.array(group_returns[g])) for g in range(n_groups)}

    elapsed = time.time() - t0
    ic_mean = float(np.mean(ic_list)) if ic_list else float("nan")
    ic_std = float(np.std(ic_list)) if ic_list else float("nan")
    ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")
    n_pairs_avg = float(np.mean(pair_counts)) if pair_counts else 0.0

    print(f"  Pair 分组完成: {len(dates_list)} 个截面, "
          f"截面均 {n_pairs_avg:.0f} pair, IC={ic_mean:.4f}, ICIR={ic_ir:.4f}, "
          f"耗时 {elapsed:.0f}s")
    for g in range(n_groups):
        ret_arr = np.array(group_returns[g])
        ann = float(np.mean(ret_arr)) * (250.0 / fwd_days)
        vol = float(np.std(ret_arr)) * np.sqrt(250.0 / fwd_days)
        sharpe = ann / vol if vol > 0 else 0.0
        print(f"    Group {g}: 终值={abs_nav[g][-1]:.3f}, 年化={ann*100:+.2f}%, "
              f"夏普={sharpe:.3f}")

    return {
        "abs_nav": abs_nav,
        "dates": dates_list,
        "n_groups": n_groups,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "n_pairs_per_group_avg": n_pairs_avg,
        "fwd_days": fwd_days,
    }


def plot_pair_group_backtest(bt_result: Dict, save_path: str, score_name: str = "|signal|"):
    """Pair 分组绝对累计净值图 (按 score_name 分组, 不减基准)."""
    abs_nav = bt_result["abs_nav"]
    dates = bt_result["dates"]
    n_groups = bt_result["n_groups"]
    fwd = bt_result.get("fwd_days", 5)
    ic_mean = bt_result.get("ic_mean", float("nan"))
    ic_ir = bt_result.get("ic_ir", float("nan"))
    n_pairs_avg = bt_result.get("n_pairs_per_group_avg", 0.0)

    plot_dates = pd.to_datetime(dates)
    start_date = plot_dates[0] - pd.Timedelta(days=1)
    plot_dates = pd.DatetimeIndex([start_date]).append(plot_dates)

    plt.figure(figsize=(12, 6))
    for g in range(n_groups):
        nav_arr = np.concatenate([[1.0], abs_nav[g]])
        plt.plot(plot_dates, nav_arr, label=f"Group {g} ({score_name} {'最弱' if g==0 else '最强' if g==n_groups-1 else f'q{g}'})",
                 alpha=0.85, lw=1.5)
    plt.title(f"Pair Group Backtest (按{score_name}分{n_groups}组, fwd={fwd}d, "
              f"截面均{n_pairs_avg:.0f}对, IC={ic_mean:.3f}, ICIR={ic_ir:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative NAV (绝对, 不减基准)")
    plt.axhline(1.0, color="gray", ls="--", alpha=0.5)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close("all")
    print(f"  已保存: {save_path}")


def plot_head_backtest(nav_df: pd.DataFrame, save_path: str, stats: Dict):
    """头组持仓回测图。"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 头组 & 基准 净值
    ax = axes[0, 0]
    ax.plot(nav_df.index, nav_df["head_nav"], lw=2, color="red", label="头组")
    ax.plot(nav_df.index, nav_df["benchmark_nav"], lw=1.5, color="gray", alpha=0.7, label="基准")
    ax.set_title("头组 / 基准 净值", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 超额净值
    ax2 = axes[0, 1]
    ax2.plot(nav_df.index, nav_df["excess_nav"], lw=2, color="orange")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.set_title("头组超额净值", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # 3. 超额回撤
    ax3 = axes[1, 0]
    ex_nav = nav_df["excess_nav"]
    dd = (ex_nav - ex_nav.cummax()) / ex_nav.cummax() * 100
    ax3.fill_between(dd.index, dd, 0, alpha=0.3, color="red")
    ax3.plot(dd.index, dd, "r-", lw=1)
    ax3.set_title("超额回撤", fontsize=13)
    ax3.set_ylabel("回撤 (%)")
    ax3.grid(True, alpha=0.3)

    # 4. 统计表
    ax4 = axes[1, 1]
    ax4.axis("off")
    headers = ["指标", "头组", "基准", "超额"]
    metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "win_rate"]
    labels = ["年化收益(%)", "年化波动(%)", "夏普比率", "最大回撤(%)", "胜率(%)"]
    tdata = [
        [lab, f"{stats['head'][m]:.2f}", f"{stats['benchmark'][m]:.2f}", f"{stats['excess'][m]:.2f}"]
        for m, lab in zip(metrics, labels)
    ]
    tbl = ax4.table(cellText=tdata, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  已保存: {save_path}")


def plot_yearly_head(nav_df: pd.DataFrame, save_path: str):
    """每年一张子图，画头组超额净值（每年从 1.0 起始）。"""
    ex_nav = nav_df["excess_nav"]
    ex_ret = ex_nav.pct_change().dropna()

    years = sorted(set(ex_ret.index.year))
    n = len(years)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, year in enumerate(years):
        ax = axes[0, i]
        mask = ex_ret.index.year == year
        yr_ret = ex_ret[mask]
        if yr_ret.empty:
            ax.set_title(f"{year}")
            continue
        yr_nav = (1 + yr_ret).cumprod()
        yr_nav = pd.concat([pd.Series([1.0], index=[yr_ret.index[0]]), yr_nav])
        ax.plot(yr_nav.index, yr_nav.values, lw=2, color=colors[i])
        ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
        final = yr_nav.iloc[-1]
        ann_ret = (final - 1) * 100
        ax.set_title(f"{year}  ({ann_ret:+.1f}%)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("头组超额年度净值", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  已保存: {save_path}")


def plot_yearly_longshort(nav_df: pd.DataFrame, save_path: str,
                           col: str = "longshort", suptitle: str = "多空组合年度净值"):
    """每年一张子图,画多空组合净值(每年从 1.0 起始)。

    参数:
        nav_df: analyze_longshort_holdings 返回的 nav_df, 含列 col (默认 "longshort")
        save_path: 输出图片路径
        col: nav_df 中要画的列名 (默认 "longshort"; 也可传 "head"/"tail")
        suptitle: 整体标题
    """
    if col not in nav_df.columns:
        print(f"  [WARN] nav_df 缺少列 {col}, 跳过 yearly 图")
        return
    nav = nav_df[col]
    ret = nav.pct_change().dropna()

    years = sorted(set(ret.index.year))
    n = len(years)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, year in enumerate(years):
        ax = axes[0, i]
        mask = ret.index.year == year
        yr_ret = ret[mask]
        if yr_ret.empty:
            ax.set_title(f"{year}")
            continue
        yr_nav = (1 + yr_ret).cumprod()
        # 前插 1.0 起点
        yr_nav = pd.concat([pd.Series([1.0], index=[yr_ret.index[0]]), yr_nav])
        ax.plot(yr_nav.index, yr_nav.values, lw=2, color=colors[i])
        ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
        final = yr_nav.iloc[-1]
        ann_ret = (final - 1) * 100
        ax.set_title(f"{year}  ({ann_ret:+.1f}%)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  已保存: {save_path}")


# =====================================================================
# 新增：OU 配对实验专用工具
# =====================================================================

DATA_DIR = "Data/all"
PRICE_NON_ST_PATH = os.path.join(DATA_DIR, "price_non_st.pkl")
PANEL_TRADE_PATH = os.path.join(DATA_DIR, "panel_trade.pkl")
THEME_LONG_PATH = os.path.join(DATA_DIR, "theme_long.parquet")


def load_price_industry_mv(start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 Data/all/price_non_st.pkl 读取日频面板,过滤到 [start_date, end_date]。

    返回长表字段:
        date, stock_code, close_price, open_price, status,
        market_value, sw_industry_l1_code

    NOTE: market_value <=0 视为异常,过滤掉。行业代码缺失也过滤掉。
    """
    print(f"加载价格/行业/市值面板: {PRICE_NON_ST_PATH}")
    t0 = time.time()
    df = pd.read_pickle(PRICE_NON_ST_PATH)
    df["date"] = pd.to_datetime(df["date"])
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    df = df[(df["date"] >= sd) & (df["date"] <= ed)].copy()
    df = df[df["market_value"] > 0]
    df = df.dropna(subset=["sw_industry_l1_code"])
    keep = ["date", "stock_code", "close_price", "open_price",
            "status", "market_value", "sw_industry_l1_code"]
    df = df[keep].sort_values(["stock_code", "date"]).reset_index(drop=True)
    elapsed = time.time() - t0
    print(f"  面板: {len(df)} 行, {df['stock_code'].nunique()} 只股票, "
          f"{df['date'].nunique()} 个截面, 耗时 {elapsed:.0f}s")
    return df


def compute_short_pool(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日空头池 = 市值前 50% ∩ 波动率综合排名前 40% ∩ tradable。

    波动率综合分:
        vol_score = 1/3 * rank_pct(vol20) + 1/3 * rank_pct(vol60) + 1/3 * rank_pct(vol120)
    其中 vol_w = 过去 w 日的日收益率标准差; rank_pct 为日内全市场升序百分位 ∈ (0, 1]。

    返回 DataFrame [date, stock_code, is_short_pool],
    is_short_pool=True 表示当日属于空头池。
    """
    print("计算每日空头池 (市值前 50% ∩ vol_score 前 40% ∩ tradable)...")
    t0 = time.time()

    df = price_panel[["date", "stock_code", "close_price",
                      "status", "open_price", "market_value"]].copy()
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # 日收益率 (close-to-close, group 内 shift)
    df["ret"] = df.groupby("stock_code")["close_price"].pct_change()

    # 滚动波动率 (按股票内 rolling, 最少 80% 观测)
    grp = df.groupby("stock_code", group_keys=False)["ret"]
    df["vol20"] = grp.transform(lambda s: s.rolling(20, min_periods=16).std())
    df["vol60"] = grp.transform(lambda s: s.rolling(60, min_periods=48).std())
    df["vol120"] = grp.transform(lambda s: s.rolling(120, min_periods=96).std())

    # tradable
    df["tradable"] = (
        (df["status"] == 0)
        & (df["open_price"] > 0)
        & df["open_price"].notna()
        & df["close_price"].notna()
    )

    # 截面排名: 注意 rank_pct 仅在该截面非 NaN 的股票内排
    df["mv_rank_pct"] = df.groupby("date")["market_value"].rank(pct=True, method="first")
    # 三个波动率分位 (升序; 越高表示波动越大)
    df["vol20_rank"] = df.groupby("date")["vol20"].rank(pct=True, method="first")
    df["vol60_rank"] = df.groupby("date")["vol60"].rank(pct=True, method="first")
    df["vol120_rank"] = df.groupby("date")["vol120"].rank(pct=True, method="first")
    df["vol_score"] = (df["vol20_rank"] + df["vol60_rank"] + df["vol120_rank"]) / 3.0
    # vol_score 本身在截面内再排一次(升序; >=0.6 表示波动率综合分位于前 40%)
    df["vol_score_rank"] = df.groupby("date")["vol_score"].rank(pct=True, method="first")

    df["is_short_pool"] = (
        (df["mv_rank_pct"] >= 0.5)
        & (df["vol_score_rank"] >= 0.6)
        & df["tradable"].fillna(False)
    )

    out = df[["date", "stock_code", "is_short_pool"]].copy()
    elapsed = time.time() - t0
    n_avg = out.groupby("date")["is_short_pool"].sum().mean()
    print(f"  空头池构造完成: 平均每日 {n_avg:.0f} 只, 耗时 {elapsed:.0f}s")
    return out


def load_theme_top5_array(stock_codes: np.ndarray, dates: np.ndarray,
                           top_k: int = 5,
                           score_col: str = "unmarket_norm_score"
                           ) -> np.ndarray:
    """加载概念-个股关联表, 预处理为 (n_dates, n_stocks, top_k) 的 int 数组。

    每股每天按 score_col 降序取 top_k 个 theme_id; 不足填 -1。

    参数:
        stock_codes: (n_stocks,) 全市场股票代码 (与 prepare_data 输出一致)
        dates:       (n_dates,)  全市场交易日 (np.datetime64[ns])
        top_k:       每股保留 top K 概念 (默认 5)
        score_col:   排序字段 (默认 unmarket_norm_score)

    返回:
        top_k_arr:   (n_dates, n_stocks, top_k) int32, -1 表示无该位置
                     行索引 = dates 的位置, 列索引 = stock_codes 的位置

    NOTE:
        - 内存约 (1500 × 5181 × 5 × 4 字节) ≈ 156 MB int32
        - 加载 + 预处理总耗时约 30-60s (含 pivot)
        - 仅在 pairing_method == "shared_top_theme" 时调用
    """
    print(f"加载 theme_long: {THEME_LONG_PATH}")
    t0 = time.time()
    df = pd.read_parquet(THEME_LONG_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    print(f"  原始 {len(df):,} 行, 加载 {time.time()-t0:.0f}s")

    # 过滤到所需 date 和 stock 范围 (节省内存)
    t1 = time.time()
    dates_set = set(pd.DatetimeIndex(dates).tolist())
    stocks_set = set(stock_codes.tolist())
    df = df[df["date"].isin(dates_set) & df["stock_code"].isin(stocks_set)]
    print(f"  过滤到所需 date×stock 后: {len(df):,} 行, "
          f"过滤耗时 {time.time()-t1:.0f}s")

    # 每股每天按 score_col 降序取 top_k
    t1 = time.time()
    df = df.sort_values(["date", "stock_code", score_col], ascending=[True, True, False])
    top_df = df.groupby(["date", "stock_code"]).head(top_k).reset_index(drop=True)
    # 给每行附 within-group rank (0..top_k-1)
    top_df["rk"] = top_df.groupby(["date", "stock_code"]).cumcount()
    print(f"  排序+取 top {top_k} 完成, {len(top_df):,} 行, 耗时 {time.time()-t1:.0f}s")

    # 构造 (n_dates, n_stocks, top_k) 矩阵
    t1 = time.time()
    n_dates = len(dates)
    n_stocks = len(stock_codes)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(pd.DatetimeIndex(dates))}
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    top_k_arr = np.full((n_dates, n_stocks, top_k), -1, dtype=np.int32)

    # 矩阵化填充: 用 to_numpy 一次性拿出 (date, stock, theme, rk)
    d_arr = top_df["date"].to_numpy()
    s_arr = top_df["stock_code"].to_numpy()
    t_arr = top_df["theme_id"].to_numpy()
    r_arr = top_df["rk"].to_numpy()
    # 转索引
    d_idx = np.array([date_to_idx.get(pd.Timestamp(x), -1) for x in d_arr])
    s_idx = np.array([code_to_idx.get(x, -1) for x in s_arr])
    valid = (d_idx >= 0) & (s_idx >= 0)
    d_idx = d_idx[valid]; s_idx = s_idx[valid]
    t_arr = t_arr[valid]; r_arr = r_arr[valid]
    top_k_arr[d_idx, s_idx, r_arr] = t_arr
    print(f"  矩阵化填充完成: top_k_arr shape={top_k_arr.shape}, "
          f"内存 {top_k_arr.nbytes/1024/1024:.0f} MB, 耗时 {time.time()-t1:.0f}s")

    # 抽查
    has_theme = (top_k_arr >= 0).any(axis=2)        # (n_dates, n_stocks)
    cov = has_theme.sum(axis=1).mean()
    avg_themes = (top_k_arr >= 0).sum(axis=2)[has_theme].mean()
    print(f"  覆盖率: 平均每日 {cov:.0f} 股有 theme, 平均 top {avg_themes:.1f} 个/股")
    print(f"  load_theme_top5_array 总耗时 {time.time()-t0:.0f}s")
    return top_k_arr


def run_longshort_backtest(
    long_holdings: pd.Series,   # index=date, values=list[stock_code]
    short_holdings: pd.Series,  # index=date, values=list[stock_code]
    open_prices: pd.DataFrame,  # 宽表
    close_prices: pd.DataFrame, # 宽表
    status_data: pd.DataFrame,  # 宽表
    output_dir: str,
    commission_rate: float = 0.0007,
    holding_period: int = 1,    # 1 = 日频; >1 = 每 holding_period 日轮换持仓
) -> Dict:
    """
    多空组合回测 + 绘图。

    holding_period=1: 日频调仓 (T 日信号 → T+1 日开盘建仓,每日换仓)
    holding_period=N: 每 N 个交易日轮换一次持仓 (调用 analyze_longshort_holdings
                      的 periodic_rebalance 模式;持有 N 日内信号被忽略)

    输出 longshort_backtest.png / head_backtest.png / tail_backtest.png 三张图。

    返回 dict 包含 statistics、nav_df 等,直接给上游写 metrics.json 用。
    """
    os.makedirs(output_dir, exist_ok=True)

    method = "daily_rebalance" if holding_period == 1 else "periodic_rebalance"
    desc = "日频调仓" if holding_period == 1 else f"持有 {holding_period} 日轮换"
    print(f"  调用 analyze_longshort_holdings ({desc})...")
    t0 = time.time()
    results = analyze_longshort_holdings(
        head_holdings_data=long_holdings,
        tail_holdings_data=short_holdings,
        open_prices=open_prices,
        close_prices=close_prices,
        benchmark_data=None,
        status_data=status_data,
        method=method,
        holding_period=holding_period,
        commission_rate=commission_rate,
        verbose=False,
    )
    elapsed = time.time() - t0
    nav_df = results["nav_df"]
    stats = results["statistics"]
    print(f"  多空回测完成: {len(nav_df)} 个交易日, 耗时 {elapsed:.0f}s")

    # ── 绘图 1: longshort_backtest.png ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 左上: 三条净值
    ax = axes[0, 0]
    ax.plot(nav_df.index, nav_df["longshort"], lw=2, color="blue", label="多空")
    ax.plot(nav_df.index, nav_df["head"], lw=1.5, color="red", alpha=0.8, label="多头")
    ax.plot(nav_df.index, nav_df["tail"], lw=1.5, color="green", alpha=0.8, label="空头")
    ax.plot(nav_df.index, nav_df["benchmark"], lw=1, color="gray", alpha=0.7, label="基准")
    ax.set_title("多空 / 多头 / 空头 / 基准 净值", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)

    # 右上: 多空超额
    ax2 = axes[0, 1]
    ls_excess = nav_df["longshort"] / nav_df["benchmark"]
    ax2.plot(ls_excess.index, ls_excess.values, lw=2, color="orange")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.set_title("多空超额净值", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # 左下: 多空回撤
    ax3 = axes[1, 0]
    ls_nav = nav_df["longshort"]
    dd = (ls_nav - ls_nav.cummax()) / ls_nav.cummax() * 100
    ax3.fill_between(dd.index, dd, 0, alpha=0.3, color="red")
    ax3.plot(dd.index, dd, "r-", lw=1)
    ax3.set_title("多空回撤", fontsize=13); ax3.set_ylabel("回撤 (%)")
    ax3.grid(True, alpha=0.3)

    # 右下: 统计表
    ax4 = axes[1, 1]; ax4.axis("off")
    headers = ["指标", "多头", "空头", "多空", "基准"]
    ms = ["annual_return", "annual_volatility", "sharpe_ratio",
          "max_drawdown", "win_rate"]
    mns = ["年化收益(%)", "年化波动(%)", "夏普比率", "最大回撤(%)", "胜率(%)"]
    tdata = [
        [mn,
         f"{stats['head'][m]:.2f}",
         f"{stats['tail'][m]:.2f}",
         f"{stats['longshort'][m]:.2f}",
         f"{stats['benchmark'][m]:.2f}"]
        for m, mn in zip(ms, mns)
    ]
    tbl = ax4.table(cellText=tdata, colLabels=headers,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.3, 1.5)
    plt.tight_layout()
    longshort_path = os.path.join(output_dir, "longshort_backtest.png")
    plt.savefig(longshort_path, dpi=120); plt.close(fig)
    print(f"  已保存: {longshort_path}")

    # ── 绘图 2: head_backtest.png (多头独立曲线 + 超额) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    ax.plot(nav_df.index, nav_df["head"], lw=2, color="red", label="多头")
    ax.plot(nav_df.index, nav_df["benchmark"], lw=1.2, color="gray", alpha=0.7, label="基准")
    ax.set_title(f"多头净值 (年化 {stats['head']['annual_return']:.1f}%, "
                 f"夏普 {stats['head']['sharpe_ratio']:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    head_ex = nav_df["head"] / nav_df["benchmark"]
    ax2.plot(head_ex.index, head_ex.values, lw=2, color="orange")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.set_title(f"多头超额 (年化 {stats['head_excess']['annual_return']:.1f}%, "
                  f"夏普 {stats['head_excess']['sharpe_ratio']:.2f})")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    head_path = os.path.join(output_dir, "head_backtest.png")
    plt.savefig(head_path, dpi=120); plt.close(fig)
    print(f"  已保存: {head_path}")

    # ── 绘图 3: tail_backtest.png (空头独立曲线) ──
    # nav_df["tail"] = analyze_longshort_holdings 中的 t_nav,即"做空收益累计净值",
    # 也就是 -tail_raw_ret 累计;曲线向上代表做空赚钱。
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    ax.plot(nav_df.index, nav_df["tail"], lw=2, color="green", label="空头(做空收益)")
    ax.plot(nav_df.index, nav_df["tail_raw"], lw=1.2, color="gray", alpha=0.7,
            label="空头标的实际涨跌")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_title(f"空头组合 (做空年化 {stats['tail']['annual_return']:.1f}%, "
                 f"夏普 {stats['tail']['sharpe_ratio']:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    tail_dd = (nav_df["tail"] - nav_df["tail"].cummax()) / nav_df["tail"].cummax() * 100
    ax2.fill_between(tail_dd.index, tail_dd, 0, alpha=0.3, color="red")
    ax2.plot(tail_dd.index, tail_dd, "r-", lw=1)
    ax2.set_title("空头回撤")
    ax2.set_ylabel("回撤 (%)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    tail_path = os.path.join(output_dir, "tail_backtest.png")
    plt.savefig(tail_path, dpi=120); plt.close(fig)
    print(f"  已保存: {tail_path}")

    # ── 绘图 4: yearly_longshort.png (按年切分多空净值, 与 W-01 同款) ──
    yearly_path = os.path.join(output_dir, "yearly_longshort.png")
    plot_yearly_longshort(nav_df, yearly_path, col="longshort",
                          suptitle="多空组合年度净值")

    return results


def run_longshort_backtest_weighted(
    long_holdings_w: pd.Series,    # index=date, values=dict[stock_code -> weight], 组内 ∑w=1
    short_holdings_w: pd.Series,   # index=date, values=dict[stock_code -> weight], 组内 ∑w=1
    open_prices: pd.DataFrame,     # 宽表
    close_prices: pd.DataFrame,    # 宽表
    status_data: pd.DataFrame,     # 宽表
    output_dir: str,
    commission_rate: float = 0.0007,
    holding_period: int = 1,
) -> Dict:
    """
    加权多空回测 (与 run_longshort_backtest 等价语义, 但每只票按给定权重持仓)。

    与 analyze_longshort_holdings 的区别:
        - 持仓: 等权 -> 给定权重 dict (组内 ∑w=1)
        - 隔夜/日内收益: 简单平均 -> 加权平均
        - 换手成本: 按"集合差异比例"-> 按"逐票权重变化绝对值之和 / 2"

    入参与原 run_longshort_backtest 严格对偶; 输出三张图 + nav_df + statistics。

    实现复用思路:
        - 调仓计划 (date -> dict): 复用相同的"日频/周期"切片逻辑
        - 持仓股票被 status/价格过滤后, 剩余票按"原权重 / 剩余 ∑w"重新归一化, 保持组内 ∑w=1
        - 换手成本 = 0.5 * Σ_code |w_new - w_old| * 2 * commission_rate
          (把多头组内 ∑w=1 看作一个标准化资金, |Δw|/2 即"被换出/换入"的资金比例)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 标准化输入 (Series of dict) ──
    def _to_weight_series(data):
        s = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        return s.apply(lambda x: x if isinstance(x, dict) else {})

    head_s = _to_weight_series(long_holdings_w)
    tail_s = _to_weight_series(short_holdings_w)

    common_dates = sorted(
        set(head_s.index) & set(tail_s.index)
        & set(open_prices.index) & set(close_prices.index)
    )
    head_s = head_s.loc[common_dates]
    tail_s = tail_s.loc[common_dates]
    open_p = open_prices.loc[common_dates]
    close_p = close_prices.loc[common_dates]
    print(f"  数据对齐: {len(common_dates)} 个交易日 (加权回测)")

    # ── 生成调仓计划: {交易日 -> dict[code, weight]}, T 信号 → T+1 持仓 ──
    def _build_weight_schedule(weight_series, dates, op, cp, period, status_df):
        sched = {}
        method_periodic = (period > 1)
        for i in range(len(dates) - 1):
            sig_d, trd_d = dates[i], dates[i + 1]
            if method_periodic and i % period != 0:
                continue
            wdict = weight_series.loc[sig_d]
            if not wdict:
                continue
            op_row = op.loc[trd_d]
            cp_row = cp.loc[trd_d]
            st_row = (status_df.loc[trd_d]
                      if status_df is not None and trd_d in status_df.index
                      else None)
            valid = {}
            for stk, w in wdict.items():
                if stk not in op_row.index or stk not in cp_row.index:
                    continue
                if not (pd.notna(op_row[stk]) and pd.notna(cp_row[stk]) and op_row[stk] > 0):
                    continue
                if st_row is not None and st_row.get(stk, np.nan) != 0:
                    continue
                valid[stk] = float(w)
            if not valid:
                continue
            # 按剩余 ∑w 重新归一化, 保持组内 ∑w = 1
            tot = sum(valid.values())
            if tot <= 0:
                continue
            sched[trd_d] = {k: v / tot for k, v in valid.items()}
        return sched

    head_sch = _build_weight_schedule(head_s, common_dates, open_p, close_p,
                                       holding_period, status_data)
    tail_sch = _build_weight_schedule(tail_s, common_dates, open_p, close_p,
                                       holding_period, status_data)
    print(f"  头组 {len(head_sch)} 个调仓日, 尾组 {len(tail_sch)} 个调仓日")

    def _weighted_return(weight_dict, price_from, price_to):
        """加权平均收益: ∑ w * (to-from)/from; 缺失/无效价的票按原权重视为 0 收益."""
        if not weight_dict:
            return 0.0
        r = 0.0
        for stk, w in weight_dict.items():
            if stk in price_from.index and stk in price_to.index:
                pf, pt = price_from[stk], price_to[stk]
                if pd.notna(pf) and pd.notna(pt) and pf > 0:
                    r += w * (pt - pf) / pf
        return r

    def _turnover_cost_w(old_dict, new_dict, rate):
        """加权换手成本 = 0.5 * Σ_code |w_new - w_old| * 2 * rate."""
        if not old_dict and not new_dict:
            return 0.0
        codes = set(old_dict) | set(new_dict)
        delta = sum(abs(new_dict.get(c, 0.0) - old_dict.get(c, 0.0)) for c in codes)
        return 0.5 * delta * 2 * rate

    # ── 回测主循环 (与 analyze_longshort_holdings 同骨架) ──
    h_nav = t_nav = t_raw_nav = ls_nav = bm_nav = 1.0
    h_curr, t_curr = {}, {}
    records = []

    for i, date in enumerate(common_dates):
        if i == 0:
            records.append({"date": date, "head": 1.0, "tail": 1.0,
                            "tail_raw": 1.0, "longshort": 1.0, "benchmark": 1.0})
            continue
        yday = common_dates[i - 1]
        prev_close_row = close_p.loc[yday]
        curr_open_row = open_p.loc[date]
        curr_close_row = close_p.loc[date]

        # ── 头组 ──
        hp = head_sch.get(date)
        h_rebal = hp is not None and hp != h_curr
        h_overnight_h = h_curr
        h_intraday_h = hp if hp else h_curr
        h_overnight = _weighted_return(h_overnight_h, prev_close_row, curr_open_row)
        h_cost = _turnover_cost_w(h_curr, hp, commission_rate) if h_rebal and hp else 0.0
        if h_rebal and hp:
            h_curr = hp
        h_intraday = _weighted_return(h_intraday_h, curr_open_row, curr_close_row)
        h_ret = h_overnight + h_intraday - h_cost

        # ── 尾组 ──
        tp = tail_sch.get(date)
        t_rebal = tp is not None and tp != t_curr
        t_overnight_h = t_curr
        t_intraday_h = tp if tp else t_curr
        t_overnight = _weighted_return(t_overnight_h, prev_close_row, curr_open_row)
        t_cost = _turnover_cost_w(t_curr, tp, commission_rate) if t_rebal and tp else 0.0
        if t_rebal and tp:
            t_curr = tp
        t_intraday = _weighted_return(t_intraday_h, curr_open_row, curr_close_row)
        t_raw = t_overnight + t_intraday
        short_ret = -t_raw - t_cost
        ls_ret = h_ret + short_ret

        # 基准 = close-to-close 全样本等权 (与 analyze_longshort_holdings 一致)
        bm_ret = 0.0
        try:
            mask = ((prev_close_row > 0) & pd.notna(prev_close_row)
                    & pd.notna(curr_close_row))
            if mask.any():
                bm_ret = ((curr_close_row[mask] - prev_close_row[mask])
                          / prev_close_row[mask]).mean()
        except Exception:
            pass

        h_nav *= (1 + h_ret)
        t_nav *= (1 + short_ret)
        t_raw_nav *= (1 + t_raw)
        ls_nav *= (1 + ls_ret)
        bm_nav *= (1 + bm_ret)

        records.append({"date": date, "head": h_nav, "tail": t_nav,
                        "tail_raw": t_raw_nav, "longshort": ls_nav, "benchmark": bm_nav})

    nav_df = pd.DataFrame(records).set_index("date")

    # ── 统计指标 (沿用 calc_stats, fwd_days=1 因为 nav_df 是逐日的) ──
    stats = {}
    stats["head"] = calc_stats(nav_df["head"], "头组(多头)")
    stats["tail"] = calc_stats(nav_df["tail"], "尾组(做空)")
    stats["tail_raw"] = calc_stats(nav_df["tail_raw"], "尾组(实际)")
    stats["longshort"] = calc_stats(nav_df["longshort"], "多空组合")
    stats["benchmark"] = calc_stats(nav_df["benchmark"], "基准")
    stats["head_excess"] = calc_stats(nav_df["head"] / nav_df["benchmark"], "头组超额")
    stats["tail_excess"] = calc_stats(nav_df["tail"] / nav_df["benchmark"], "尾组做空超额")
    stats["ls_excess"] = calc_stats(nav_df["longshort"] / nav_df["benchmark"], "多空超额")

    # ── 绘图 (复用 run_longshort_backtest 的三张图骨架) ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax = axes[0, 0]
    ax.plot(nav_df.index, nav_df["longshort"], lw=2, color="blue", label="多空")
    ax.plot(nav_df.index, nav_df["head"], lw=1.5, color="red", alpha=0.8, label="多头")
    ax.plot(nav_df.index, nav_df["tail"], lw=1.5, color="green", alpha=0.8, label="空头")
    ax.plot(nav_df.index, nav_df["benchmark"], lw=1, color="gray", alpha=0.7, label="基准")
    ax.set_title("多空 / 多头 / 空头 / 基准 净值 (加权)", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ls_excess = nav_df["longshort"] / nav_df["benchmark"]
    ax2.plot(ls_excess.index, ls_excess.values, lw=2, color="orange")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.set_title("多空超额净值", fontsize=13); ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ls_nav = nav_df["longshort"]
    dd = (ls_nav - ls_nav.cummax()) / ls_nav.cummax() * 100
    ax3.fill_between(dd.index, dd, 0, alpha=0.3, color="red")
    ax3.plot(dd.index, dd, "r-", lw=1)
    ax3.set_title("多空回撤", fontsize=13); ax3.set_ylabel("回撤 (%)")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]; ax4.axis("off")
    headers = ["指标", "多头", "空头", "多空", "基准"]
    ms = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "win_rate"]
    mns = ["年化收益(%)", "年化波动(%)", "夏普比率", "最大回撤(%)", "胜率(%)"]
    tdata = [
        [mn, f"{stats['head'][m]:.2f}", f"{stats['tail'][m]:.2f}",
         f"{stats['longshort'][m]:.2f}", f"{stats['benchmark'][m]:.2f}"]
        for m, mn in zip(ms, mns)
    ]
    tbl = ax4.table(cellText=tdata, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.3, 1.5)
    plt.tight_layout()
    longshort_path = os.path.join(output_dir, "longshort_backtest.png")
    plt.savefig(longshort_path, dpi=120); plt.close(fig)
    print(f"  已保存: {longshort_path}")

    # head_backtest.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    ax.plot(nav_df.index, nav_df["head"], lw=2, color="red", label="多头(加权)")
    ax.plot(nav_df.index, nav_df["benchmark"], lw=1.2, color="gray", alpha=0.7, label="基准")
    ax.set_title(f"多头净值 (年化 {stats['head']['annual_return']:.1f}%, "
                 f"夏普 {stats['head']['sharpe_ratio']:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    head_ex = nav_df["head"] / nav_df["benchmark"]
    ax2.plot(head_ex.index, head_ex.values, lw=2, color="orange")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.set_title(f"多头超额 (年化 {stats['head_excess']['annual_return']:.1f}%, "
                  f"夏普 {stats['head_excess']['sharpe_ratio']:.2f})")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    head_path = os.path.join(output_dir, "head_backtest.png")
    plt.savefig(head_path, dpi=120); plt.close(fig)
    print(f"  已保存: {head_path}")

    # tail_backtest.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    ax.plot(nav_df.index, nav_df["tail"], lw=2, color="green", label="空头(做空收益)")
    ax.plot(nav_df.index, nav_df["tail_raw"], lw=1.2, color="gray", alpha=0.7,
            label="空头标的实际涨跌")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_title(f"空头组合 (做空年化 {stats['tail']['annual_return']:.1f}%, "
                 f"夏普 {stats['tail']['sharpe_ratio']:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    tail_dd = (nav_df["tail"] - nav_df["tail"].cummax()) / nav_df["tail"].cummax() * 100
    ax2.fill_between(tail_dd.index, tail_dd, 0, alpha=0.3, color="red")
    ax2.plot(tail_dd.index, tail_dd, "r-", lw=1)
    ax2.set_title("空头回撤"); ax2.set_ylabel("回撤 (%)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    tail_path = os.path.join(output_dir, "tail_backtest.png")
    plt.savefig(tail_path, dpi=120); plt.close(fig)
    print(f"  已保存: {tail_path}")

    # yearly_longshort.png
    yearly_path = os.path.join(output_dir, "yearly_longshort.png")
    plot_yearly_longshort(nav_df, yearly_path, col="longshort",
                           suptitle="多空组合年度净值 (加权)")

    return {"nav_df": nav_df, "statistics": stats,
            "head_schedule": head_sch, "tail_schedule": tail_sch}


def save_summary_row(summary_csv_path: str, row: Dict, fieldnames: List[str]):
    """以 0409_tail_pool/summary.csv 风格追加写入一行;首次写入自动创建表头。

    NOTE: 字段顺序必须显式传入 fieldnames,以保证多次写入列顺序一致。
    """
    file_exists = os.path.exists(summary_csv_path)
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    # 仅取字段集中的列,缺失值用空字符串
    safe_row = {k: row.get(k, "") for k in fieldnames}
    with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(safe_row)
