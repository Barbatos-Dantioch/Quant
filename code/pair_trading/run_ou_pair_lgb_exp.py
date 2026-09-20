#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OU 配对·LightGBM 信号实验 - expanding 训练窗 (OU-P-RPL-LGBEXP-cv5-h5)
====================================================================

与 OU-P-RPL-LGB-cv5-h5 完全一致, 仅训练窗从 rolling (固定 6 个月) 改为
expanding (训练起点固定 2023-07-01, 训练终点随段增长).
其它: 同样 28 特征, 5 段, train/val 80/20 + 5 日 gap,
基线 pair 池, sign=sign(μ-X_t), magnitude=|预测 ΔX|, holding=5.

5 段 expanding 训练 (训练起点固定 2023-07-01):
  段 1: train 2023-07-01 ~ 2023-12-26 → test 2024-01-02 ~ 2024-06-30
  段 2: train 2023-07-01 ~ 2024-06-25 → test 2024-07-01 ~ 2024-12-31
  段 3: train 2023-07-01 ~ 2024-12-25 → test 2025-01-02 ~ 2025-06-30
  段 4: train 2023-07-01 ~ 2025-06-25 → test 2025-07-01 ~ 2025-12-31
  段 5: train 2023-07-01 ~ 2025-12-25 → test 2026-01-02 ~ 2026-05-13

段内 train/val: 前 80% / 5 日 gap / 后 20%, 全部样本.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir("/root/quant")

_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
if _FACTOR1_DIR not in sys.path:
    sys.path.insert(0, _FACTOR1_DIR)
from _pair_runner import (
    load_price_industry_mv,
    run_longshort_backtest,
    save_summary_row,
)

import lightgbm as lgb

# ── 常量 ──
EXP_NAME = "OU-P-RPL-LGBEXP-cv5-h5"
BASELINE = "OU-P00-zs-cv5-h5"
EXTRA_TRAIN = "_train_extra_2023H2"
OUTPUT_DIR = "output/0506_ou_pair"
EXP_DIR = os.path.join(OUTPUT_DIR, EXP_NAME)

COMMISSION = 0.0007
TOP_PCT = 0.20
TOP_PCT_MIN_N = 50
HOLDING_PERIOD = 5
LABEL_HORIZON = 5      # ΔX_{t→t+5}, 与 holding_period 一致
DATA_START = "2021-01-04"   # 与 extend 一致, 保证训练窗历史充足
DATA_END = "2026-05-13"
_EPS = 1e-12

# 5 段 expanding 训练划分 (训练起点固定 2023-07-01, 训练终点随段增长)
ROLLING_SEGMENTS = [
    # (段名, train_start, train_end_inclusive, test_start, test_end_inclusive)
    ("seg1", "2023-07-01", "2023-12-26", "2024-01-02", "2024-06-30"),
    ("seg2", "2023-07-01", "2024-06-25", "2024-07-01", "2024-12-31"),
    ("seg3", "2023-07-01", "2024-12-25", "2025-01-02", "2025-06-30"),
    ("seg4", "2023-07-01", "2025-06-25", "2025-07-01", "2025-12-31"),
    ("seg5", "2023-07-01", "2025-12-25", "2026-01-02", "2026-05-13"),
]

# 段内 train/val: 5 日 gap
TRAIN_VAL_GAP_DAYS = 5
TRAIN_FRAC = 0.80         # train 占训练段前 80%

# LightGBM 配置 (与 XGB 对齐: max_depth=5 ≈ num_leaves=31)
LGB_PARAMS = {
    "objective":          "regression",
    "metric":             "rmse",
    "max_depth":          5,
    "num_leaves":         31,        # ≈ XGB max_depth=5 (2^5)
    "learning_rate":      0.05,
    "bagging_fraction":   0.8,       # 对应 XGB subsample
    "bagging_freq":       1,
    "feature_fraction":   0.8,       # 对应 XGB colsample_bytree
    "min_child_samples":  20,        # LGB 习惯 20+ (XGB 用 10)
    "reg_lambda":         1.0,
    "random_state":       42,
    "verbose":            -1,
}
LGB_N_ROUNDS = 500
LGB_EARLY_STOPPING = 30

# 28 个特征
FEATURE_BASE = [   # 16 个 pair_log 直接取的
    "mu", "kappa", "sigma", "half_life", "b", "ADF_p", "mean_X", "std_X",
    "valid_rank_ic", "cv_mean_ic", "signal",
    "X_t", "abs_dev", "sigma_20", "sigma_30", "z_OU",
]
FEATURE_DERIVED = [  # 6 个时序衍生
    "sigma_20_quantile", "X_t_quantile", "NZC_60",
    "ret_i_20", "ret_j_20", "corr_ij_60",
]
FEATURE_MARKET = [   # 3 个市场截面
    "mkt_ret_5", "mkt_vol_20", "mkt_avg_sigma_20",
]
FEATURE_INDUSTRY = [ # 3 个行业截面
    "ind_ret_5", "ind_vol_20", "ind_avg_signal_zs",
]
FEATURES = FEATURE_BASE + FEATURE_DERIVED + FEATURE_MARKET + FEATURE_INDUSTRY
assert len(FEATURES) == 28, f"特征数应为 28, 实际 {len(FEATURES)}"


# =====================================================================
# 1) 加载所有数据 + 合并 pair_log (基线 + 扩展段)
# =====================================================================

def load_all_pair_log() -> pd.DataFrame:
    """合并基线 OU-P00-zs-cv5-h5 + 扩展段 2023H2 的 pair_log."""
    print("加载基线 pair_log + 扩展段 ...")
    base_path = os.path.join(OUTPUT_DIR, BASELINE, "pair_log.parquet")
    extra_path = os.path.join(OUTPUT_DIR, EXTRA_TRAIN, "pair_log.parquet")

    pl_base = pd.read_parquet(base_path)
    print(f"  基线: {len(pl_base):,} 行, "
          f"{pl_base['date'].min()} ~ {pl_base['date'].max()}")

    if not os.path.exists(extra_path):
        raise FileNotFoundError(f"扩展段未生成: {extra_path}. 先跑 extend_pair_log_2023H2.py")
    pl_extra = pd.read_parquet(extra_path)
    print(f"  扩展段: {len(pl_extra):,} 行, "
          f"{pl_extra['date'].min()} ~ {pl_extra['date'].max()}")

    # 列对齐: 扩展段 schema 应与基线一致 (来自同一个 run_ou_pair.process_one_section)
    common_cols = [c for c in pl_base.columns if c in pl_extra.columns]
    pl_base = pl_base[common_cols]
    pl_extra = pl_extra[common_cols]
    pl_full = pd.concat([pl_extra, pl_base], ignore_index=True)
    pl_full["date"] = pd.to_datetime(pl_full["date"])
    pl_full["stock_i"] = pl_full["stock_i"].astype(str).str.zfill(6)
    pl_full["stock_j"] = pl_full["stock_j"].astype(str).str.zfill(6)
    pl_full = pl_full.sort_values(["date","stock_i","stock_j"]).reset_index(drop=True)
    print(f"  合并后: {len(pl_full):,} 行, "
          f"{pl_full['date'].min().date()} ~ {pl_full['date'].max().date()}, "
          f"{pl_full['date'].nunique()} 截面")
    return pl_full


def load_price_wides() -> Dict:
    """加载价格宽表 + 行业映射."""
    print("\n加载价格表 ...")
    panel = load_price_industry_mv(DATA_START, DATA_END)
    panel = panel.sort_values(["date","stock_code"]).reset_index(drop=True)
    cal_index = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    stock_index = panel["stock_code"].drop_duplicates().sort_values().reset_index(drop=True)
    stock_codes = stock_index.to_numpy(dtype=object)

    close_wide = panel.pivot(index="date", columns="stock_code", values="close_price").reindex(
        index=cal_index, columns=stock_codes)
    open_wide = panel.pivot(index="date", columns="stock_code", values="open_price").reindex(
        index=cal_index, columns=stock_codes)
    status_wide = panel.pivot(index="date", columns="stock_code", values="status").reindex(
        index=cal_index, columns=stock_codes)

    close_arr = close_wide.to_numpy(dtype=np.float64)
    log_arr = np.where(close_arr > 0, np.log(close_arr), np.nan)
    # 日收益
    ret_arr = np.full_like(close_arr, np.nan)
    ret_arr[1:, :] = (close_arr[1:, :] - close_arr[:-1, :]) / close_arr[:-1, :]

    # 行业映射: stock_code -> sw_l1_code
    industry_map = panel.groupby("stock_code")["sw_industry_l1_code"].last()

    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    date_to_idx = {d: i for i, d in enumerate(cal_index)}

    print(f"  log_close shape: {log_arr.shape}")
    return {
        "log_arr": log_arr, "ret_arr": ret_arr,
        "open_wide": open_wide, "close_wide": close_wide, "status_wide": status_wide,
        "code_to_idx": code_to_idx, "date_to_idx": date_to_idx,
        "stock_codes": stock_codes, "industry_map": industry_map,
    }


# =====================================================================
# 2) 特征工程: 在 pair_log 上添加 12 个衍生特征
# =====================================================================

def add_derived_features(pl: pd.DataFrame, px: Dict, train_window: int = 252) -> pd.DataFrame:
    """对 pair_log 添加 12 个衍生特征 (6 时序 + 3 市场 + 3 行业).
    + 标签 ΔX_{t→t+5}.

    pl 已含 16 个基础特征 + 9 个其它列 (stock_i, stock_j, date, is_legal 等).
    """
    print("\n特征工程: 添加 12 个衍生 + 标签 ...")
    t0 = time.time()
    log_arr = px["log_arr"]
    ret_arr = px["ret_arr"]
    code_to_idx = px["code_to_idx"]
    date_to_idx = px["date_to_idx"]
    industry_map = px["industry_map"]
    T_total = log_arr.shape[0]

    # 对齐 pair_log 到价格表索引
    pl = pl[pl["stock_i"].isin(code_to_idx) & pl["stock_j"].isin(code_to_idx)].copy()
    pl["i_idx"] = pl["stock_i"].map(code_to_idx).astype(np.int32)
    pl["j_idx"] = pl["stock_j"].map(code_to_idx).astype(np.int32)
    pl["t_idx"] = pl["date"].map(date_to_idx)
    pl = pl.dropna(subset=["t_idx"])
    pl["t_idx"] = pl["t_idx"].astype(np.int32)

    # z_OU = |μ-X|/σ_OU (基础特征 16 之一)
    if "z_OU" not in pl.columns:
        pl["z_OU"] = (pl["mu"] - pl["X_t"]).abs() / pl["sigma"].clip(lower=_EPS) if "X_t" in pl.columns else np.nan

    # 预计算市场截面特征 (按 date 算, 全市场聚合)
    print("  市场截面特征 (mkt_ret_5, mkt_vol_20) ...")
    # 全市场等权日收益 = ret_arr 跨股票均值
    mkt_daily_ret = np.nanmean(ret_arr, axis=1)   # (T,)
    # mkt_ret_5: 过去 5 日累积 log return; 用 log(1+ret) 累加
    # 简化: 用 sum of daily ret 近似 (小变化下接近)
    mkt_ret_5 = np.full(T_total, np.nan)
    mkt_vol_20 = np.full(T_total, np.nan)
    for t in range(20, T_total):
        mkt_ret_5[t] = np.nansum(mkt_daily_ret[t-5:t])
        mkt_vol_20[t] = np.nanstd(mkt_daily_ret[t-20:t], ddof=1)

    # 行业级日收益: ind_daily_ret[ind][t] = 行业内股票均值
    print("  行业截面特征 (ind_ret_5, ind_vol_20) ...")
    industries = pd.Series(industry_map.values).dropna().unique()
    industries = [ind for ind in industries if ind is not None and not pd.isna(ind)]
    ind_to_cols = {ind: [code_to_idx[c] for c in industry_map[industry_map==ind].index
                         if c in code_to_idx] for ind in industries}
    ind_daily_ret = {}
    for ind, cols in ind_to_cols.items():
        if len(cols) == 0:
            continue
        cols_arr = np.array(cols, dtype=np.int32)
        ind_daily_ret[ind] = np.nanmean(ret_arr[:, cols_arr], axis=1)  # (T,)
    # ind_ret_5, ind_vol_20: (T, n_ind)
    ind_ret_5 = {}
    ind_vol_20 = {}
    for ind, drs in ind_daily_ret.items():
        r5 = np.full(T_total, np.nan)
        v20 = np.full(T_total, np.nan)
        for t in range(20, T_total):
            r5[t] = np.nansum(drs[t-5:t])
            v20[t] = np.nanstd(drs[t-20:t], ddof=1)
        ind_ret_5[ind] = r5
        ind_vol_20[ind] = v20

    # 主循环: 按截面分组, 算 pair 级时序衍生 + 截面聚合
    print("  pair 级特征 (sigma_20_quantile, X_t_quantile, NZC_60, ret_i/j_20, corr_ij_60) ...")
    rows = []
    n_section = pl["t_idx"].nunique()
    print(f"    截面总数: {n_section}")
    for k, (t_idx, sub) in enumerate(pl.groupby("t_idx", sort=True)):
        if t_idx < train_window:
            continue  # 没足够历史
        if t_idx + LABEL_HORIZON >= T_total:
            continue  # 没未来标签 (但允许预测时 fwd_X=NaN, 见后)
        i_arr = sub["i_idx"].values
        j_arr = sub["j_idx"].values
        n_pair = len(sub)

        # 训练窗 X (含信号日, 共 train_window+1 = 253 日; 但实际 train_window 取靠前 252 日)
        # NOTE: 基线 pair_log 不含 sigma_20/sigma_30 字段 (DEV 实验后才加), 故这里重新算
        train_lp_i = log_arr[t_idx-train_window:t_idx, i_arr]   # (W, n_pair)
        train_lp_j = log_arr[t_idx-train_window:t_idx, j_arr]
        X_train = train_lp_i - train_lp_j                        # (W, n_pair)
        # sigma_20 / sigma_30: 训练窗末段 X 的 std
        sigma_20_self = X_train[-20:, :].std(axis=0, ddof=1)
        sigma_30_self = X_train[-30:, :].std(axis=0, ddof=1)

        # sigma_20_quantile: 信号日 σ_20 在该 pair 训练窗内 233 个滚动 σ_20 中的分位数
        # 用 cumsum 矩阵化算滚动 std
        Wroll = 20
        cs1 = np.concatenate([np.zeros((1, n_pair)), np.cumsum(X_train, axis=0)], axis=0)
        cs2 = np.concatenate([np.zeros((1, n_pair)), np.cumsum(X_train**2, axis=0)], axis=0)
        sums = cs1[Wroll:] - cs1[:-Wroll]
        sums2 = cs2[Wroll:] - cs2[:-Wroll]
        means_roll = sums / Wroll
        vars_roll = (sums2 - Wroll * means_roll**2) / (Wroll - 1)
        sigma_20_series = np.sqrt(np.maximum(vars_roll, 0.0))   # (W-19, n_pair)
        sigma_20_now = sigma_20_series[-1, :]
        sigma_20_quantile = (sigma_20_series < sigma_20_now[None, :]).mean(axis=0)

        # X_t_quantile: X_t 在过去 60 日 X 序列的分位数
        # 用 X_train[-60:] (训练窗末段 60 日)
        Wq = 60
        X_60 = X_train[-Wq:, :]   # (60, n_pair)
        # 信号日 X_t = 训练窗最后一行的 "下一日" - 但训练窗就是 t-W .. t-1
        # 这里 X_t 是 pair_log 中已存的 (mu - signal*sigma/kappa 反推, 或直接取 X_t 列)
        # 我们重新算更稳: X_t = log_arr[t_idx, i] - log_arr[t_idx, j]
        X_t = log_arr[t_idx, i_arr] - log_arr[t_idx, j_arr]
        X_t_quantile = (X_60 < X_t[None, :]).mean(axis=0)

        # NZC_60: X 在过去 60 日穿越 μ 的次数 (μ 取 pair_log mu 列)
        mu_vals = sub["mu"].values
        Y = X_60 - mu_vals[None, :]   # (60, n_pair)
        s = np.sign(Y).astype(np.int8)
        cross = (s[1:, :] * s[:-1, :]) == -1
        NZC_60 = cross.sum(axis=0).astype(np.float64)

        # ret_i_20, ret_j_20: 过去 20 日 log 收益
        ret_i_20 = log_arr[t_idx, i_arr] - log_arr[t_idx-20, i_arr]
        ret_j_20 = log_arr[t_idx, j_arr] - log_arr[t_idx-20, j_arr]

        # corr_ij_60: 过去 60 日日收益 Pearson 相关
        ret_i_60 = ret_arr[t_idx-60:t_idx, i_arr]   # (60, n_pair)
        ret_j_60 = ret_arr[t_idx-60:t_idx, j_arr]
        # 矩阵化逐列 corrcoef
        # x_demean / y_demean
        ri_mean = np.nanmean(ret_i_60, axis=0, keepdims=True)
        rj_mean = np.nanmean(ret_j_60, axis=0, keepdims=True)
        ri_c = np.where(np.isnan(ret_i_60), 0, ret_i_60 - ri_mean)
        rj_c = np.where(np.isnan(ret_j_60), 0, ret_j_60 - rj_mean)
        cov_ij = np.sum(ri_c * rj_c, axis=0)
        var_i = np.sum(ri_c * ri_c, axis=0)
        var_j = np.sum(rj_c * rj_c, axis=0)
        denom = np.sqrt(np.clip(var_i, _EPS, None) * np.clip(var_j, _EPS, None))
        corr_ij_60 = cov_ij / denom

        # 市场/行业截面特征
        mkt_ret_5_val = mkt_ret_5[t_idx]
        mkt_vol_20_val = mkt_vol_20[t_idx]
        # mkt_avg_sigma_20: 当日所有 OU 通过 pair 的 σ_20 均值
        # 用我们自己算的 sigma_20_self (基线 pair_log 不含 sigma_20 字段)
        mkt_avg_sigma_20 = float(np.nanmean(sigma_20_self))

        # 行业级: 每 pair 取所在行业的 ind_ret_5, ind_vol_20, ind_avg_signal_zs
        ind_codes = sub["industry"].values   # 来自 pair_log
        ir5 = np.array([ind_ret_5.get(ic, np.array([np.nan]*T_total))[t_idx] for ic in ind_codes])
        iv20 = np.array([ind_vol_20.get(ic, np.array([np.nan]*T_total))[t_idx] for ic in ind_codes])
        # ind_avg_signal_zs: sub 按 industry 分组 |signal| 均值
        ind_sig_map = sub.assign(abs_signal=sub["signal"].abs()).groupby("industry")["abs_signal"].mean()
        iasig = ind_codes.copy().astype(object)
        for idx_ in range(len(iasig)):
            iasig[idx_] = ind_sig_map.get(iasig[idx_], np.nan)
        iasig = iasig.astype(np.float64)

        # 标签: ΔX_{t→t+H}
        fwd_X = log_arr[t_idx + LABEL_HORIZON, i_arr] - log_arr[t_idx + LABEL_HORIZON, j_arr]
        delta_X = fwd_X - X_t

        out_df = sub[["date","stock_i","stock_j","industry","is_legal","is_top20",
                       "mu","kappa","sigma","half_life","b","ADF_p","mean_X","std_X",
                       "valid_rank_ic","cv_mean_ic","signal",
                       "i_idx","j_idx","t_idx"]].copy()
        out_df["sigma_20"] = sigma_20_self
        out_df["sigma_30"] = sigma_30_self
        out_df["X_t"] = X_t
        out_df["abs_dev"] = np.abs(mu_vals - X_t)
        out_df["z_OU"] = out_df["abs_dev"] / out_df["sigma"].clip(lower=_EPS)
        out_df["sigma_20_quantile"] = sigma_20_quantile
        out_df["X_t_quantile"] = X_t_quantile
        out_df["NZC_60"] = NZC_60
        out_df["ret_i_20"] = ret_i_20
        out_df["ret_j_20"] = ret_j_20
        out_df["corr_ij_60"] = corr_ij_60
        out_df["mkt_ret_5"] = mkt_ret_5_val
        out_df["mkt_vol_20"] = mkt_vol_20_val
        out_df["mkt_avg_sigma_20"] = mkt_avg_sigma_20
        out_df["ind_ret_5"] = ir5
        out_df["ind_vol_20"] = iv20
        out_df["ind_avg_signal_zs"] = iasig
        out_df["delta_X"] = delta_X
        rows.append(out_df)

        if (k+1) % 50 == 0 or k == n_section - 1:
            print(f"    截面 {k+1}/{n_section}, 累计 {sum(len(r) for r in rows):,} 行, "
                  f"耗时 {time.time()-t0:.0f}s")

    df = pd.concat(rows, ignore_index=True)
    # 把 inf 替换为 NaN (XGBoost 支持 missing value 但不接受 inf)
    # 主要来源: z_OU = abs_dev / sigma, 当 sigma 极小时可能溢出
    n_inf_before = int(np.isinf(df[FEATURES].to_numpy()).sum())
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    print(f"  特征工程完成, 总 {len(df):,} 行, 耗时 {time.time()-t0:.0f}s, "
          f"inf 替换为 NaN: {n_inf_before:,} 处")
    return df


# =====================================================================
# 3) 5 段 rolling 训练 + 预测
# =====================================================================

def run_xgb_rolling(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], List[pd.DataFrame]]:
    """按 5 段 rolling 训练 + 预测.
    返回: (含 'pred_delta_X' 列的 DataFrame, 训练日志, 特征重要性列表).
    """
    print("\n5 段 rolling XGBoost 训练 + 预测 ...")
    df = df.copy()
    df["pred_delta_X"] = np.nan
    train_logs = []
    feat_imp_list = []

    # 仅 is_legal=True 的样本 + 标签非 NaN 用于训练
    df_legal = df[df["is_legal"]].copy()
    valid_label_mask = ~df_legal["delta_X"].isna()
    df_train_pool = df_legal[valid_label_mask].copy()
    print(f"  is_legal + 标签可用样本: {len(df_train_pool):,}")

    for seg in ROLLING_SEGMENTS:
        seg_name, train_lo, train_hi, test_lo, test_hi = seg
        train_lo = pd.Timestamp(train_lo); train_hi = pd.Timestamp(train_hi)
        test_lo  = pd.Timestamp(test_lo);  test_hi  = pd.Timestamp(test_hi)

        # 训练数据
        m_train = (df_train_pool["date"] >= train_lo) & (df_train_pool["date"] <= train_hi)
        train_seg = df_train_pool[m_train].copy().sort_values("date").reset_index(drop=True)
        n_train_total = len(train_seg)
        if n_train_total == 0:
            print(f"  [{seg_name}] 训练样本 0, 跳过")
            continue

        # 段内 train/val 切分: 按日期, 前 80% train + 5 日 gap + 后 20% val
        unique_dates = sorted(train_seg["date"].unique())
        n_dates = len(unique_dates)
        n_train_dates = int(np.floor(n_dates * TRAIN_FRAC))
        train_end_date = unique_dates[n_train_dates - 1]
        # val 起点 = train_end_date + 5 日 gap (按交易日, 不是自然日)
        gap_pos = min(n_train_dates + TRAIN_VAL_GAP_DAYS, n_dates)
        if gap_pos >= n_dates:
            print(f"  [{seg_name}] val 样本量不足, 跳过 val 用 train 末段")
            val_start_date = train_end_date  # fallback
            m_tr = train_seg["date"] <= train_end_date
            m_va = pd.Series([False]*len(train_seg))
        else:
            val_start_date = unique_dates[gap_pos]
            m_tr = train_seg["date"] <= train_end_date
            m_va = train_seg["date"] >= val_start_date
        n_tr = int(m_tr.sum())
        n_va = int(m_va.sum())

        # 测试数据 (含 NaN 标签的, 因为我们要预测; 不需要 is_legal 过滤吗?
        # 注意: 回测时仍只对 is_legal=True 选 top, 所以测试时只预测 is_legal=True 的)
        m_test = (df_legal["date"] >= test_lo) & (df_legal["date"] <= test_hi)
        test_seg = df_legal[m_test].copy()
        n_test = len(test_seg)

        print(f"\n  [{seg_name}] train: {train_lo.date()} ~ {train_hi.date()}, "
              f"test: {test_lo.date()} ~ {test_hi.date()}")
        print(f"    train: {n_tr:,} / val: {n_va:,} / 共 {n_train_total:,}; test: {n_test:,}")

        if n_tr < 100 or n_test == 0:
            print(f"  [{seg_name}] 样本不足, 跳过")
            continue

        X_tr = train_seg.loc[m_tr, FEATURES].astype(np.float32).values
        y_tr = train_seg.loc[m_tr, "delta_X"].astype(np.float32).values
        X_va = train_seg.loc[m_va, FEATURES].astype(np.float32).values if n_va > 0 else None
        y_va = train_seg.loc[m_va, "delta_X"].astype(np.float32).values if n_va > 0 else None

        t0 = time.time()
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURES)
        valid_sets = [dtrain]; valid_names = ["train"]
        if n_va > 0:
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=FEATURES)
            valid_sets.append(dval); valid_names.append("val")
        callbacks = []
        if n_va > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=LGB_EARLY_STOPPING, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))   # 静默
        booster = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=LGB_N_ROUNDS,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )
        best_iter = booster.best_iteration if n_va > 0 else booster.current_iteration()
        best_score = booster.best_score.get("val", {}).get("rmse", None) if n_va > 0 else None
        elapsed = time.time() - t0
        print(f"    LGB 训练完成, best_iter={best_iter}, "
              f"best_val_rmse={best_score:.5f}" if n_va > 0 else f"  best_iter={best_iter}")
        print(f"    耗时 {elapsed:.1f}s")

        # 预测 test
        X_te = test_seg[FEATURES].astype(np.float32).values
        pred = booster.predict(X_te, num_iteration=best_iter)
        df.loc[test_seg.index, "pred_delta_X"] = pred

        # 训练日志
        train_logs.append({
            "seg": seg_name,
            "train_range": f"{train_lo.date()}~{train_hi.date()}",
            "test_range":  f"{test_lo.date()}~{test_hi.date()}",
            "n_train": n_tr, "n_val": n_va, "n_test": n_test,
            "best_iter": int(best_iter),
            "best_val_rmse": float(best_score) if (n_va > 0 and best_score is not None) else np.nan,
            "elapsed_sec": elapsed,
        })

        # 特征重要性 (gain)
        gain_arr = booster.feature_importance(importance_type="gain")
        score_dict = dict(zip(FEATURES, gain_arr))
        imp_df = pd.DataFrame([
            {"seg": seg_name, "feature": f, "gain": float(score_dict.get(f, 0.0))}
            for f in FEATURES
        ]).sort_values("gain", ascending=False)
        feat_imp_list.append(imp_df)

    return df, train_logs, feat_imp_list


# =====================================================================
# 4) 选 top + 回测 (复用 RPL 框架)
# =====================================================================

def build_holdings(df_legal_pred: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """按 |pred_delta_X| 排序选 top, sign 用 sign(μ-X_t).
    返回 long/short holdings (Series-of-list) + 每日统计."""
    print("\n按 |pred_delta_X| 选 top, 生成 holdings ...")
    long_rows: List[Tuple] = []
    short_rows: List[Tuple] = []
    daily_stats: List[Dict] = []
    for d, sub in df_legal_pred.groupby("date"):
        sub = sub[~sub["pred_delta_X"].isna()].copy()
        n_legal = len(sub)
        if n_legal == 0:
            daily_stats.append({"date": d, "n_legal": 0, "n_top": 0})
            continue
        n_top = min(n_legal, max(TOP_PCT_MIN_N, int(math.ceil(n_legal * TOP_PCT))))
        sub["abs_pred"] = sub["pred_delta_X"].abs()
        sub_top = sub.sort_values("abs_pred", ascending=False).head(n_top)
        sign_pred = np.sign(sub_top["mu"] - sub_top["X_t"]).values
        i_arr = sub_top["stock_i"].values
        j_arr = sub_top["stock_j"].values
        for k in range(len(sub_top)):
            s = sign_pred[k]
            if s > 0:
                long_rows.append((d, i_arr[k]))
                short_rows.append((d, j_arr[k]))
            elif s < 0:
                long_rows.append((d, j_arr[k]))
                short_rows.append((d, i_arr[k]))
        daily_stats.append({"date": d, "n_legal": n_legal, "n_top": len(sub_top)})

    long_df = pd.DataFrame(long_rows, columns=["date","stock_code"])
    short_df = pd.DataFrame(short_rows, columns=["date","stock_code"])
    long_h = long_df.groupby("date")["stock_code"].apply(list)
    short_h = short_df.groupby("date")["stock_code"].apply(list)
    all_dates = sorted(set(long_h.index) | set(short_h.index))
    long_h = long_h.reindex(all_dates, fill_value=[])
    short_h = short_h.reindex(all_dates, fill_value=[])
    return long_h, short_h, daily_stats


# =====================================================================
# main
# =====================================================================

def main():
    print(f"\n{'='*70}")
    print(f"OU 配对·XGBoost 信号  [{EXP_NAME}]  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}")
    print(f"  输出: {EXP_DIR}")
    print(f"{'='*70}\n")

    metrics_path = os.path.join(EXP_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[{EXP_NAME}] 已有 metrics.json, 跳过")
        return
    os.makedirs(EXP_DIR, exist_ok=True)

    t_start = time.time()
    pl_full = load_all_pair_log()
    px = load_price_wides()

    df = add_derived_features(pl_full, px, train_window=252)
    df, train_logs, feat_imp_list = run_xgb_rolling(df)

    # 仅取 is_legal + 有 pred 的样本做 holdings
    df_legal_pred = df[df["is_legal"] & ~df["pred_delta_X"].isna()].copy()
    long_h, short_h, daily_stats = build_holdings(df_legal_pred)
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
    for k in ("head","tail","tail_raw","longshort","benchmark",
              "head_excess","tail_excess","ls_excess"):
        if k in stats:
            stats[k] = _add_calmar(stats[k])

    # 写 metrics.json
    stats_df = pd.DataFrame(daily_stats)
    metrics = {
        "experiment": EXP_NAME,
        "baseline_pair_pool": BASELINE,
        "extra_train_pool":   EXTRA_TRAIN,
        "n_features":         len(FEATURES),
        "features":           FEATURES,
        "holding_period":     HOLDING_PERIOD,
        "label_horizon":      LABEL_HORIZON,
        "commission_rate":    COMMISSION,
        "top_pct":            TOP_PCT,
        "top_pct_min_n":      TOP_PCT_MIN_N,
        "n_signal_days":      int((stats_df["n_top"] > 0).sum()),
        "avg_n_top_per_day":  float(stats_df["n_top"].mean()),
        "lgb_params":         LGB_PARAMS,
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

    # 写特征重要性
    if feat_imp_list:
        all_imp = pd.concat(feat_imp_list, ignore_index=True)
        all_imp.to_csv(os.path.join(EXP_DIR, "feature_importance.csv"), index=False)
        # 跨段平均, 看 top10
        avg_imp = (all_imp.groupby("feature")["gain"].mean()
                          .sort_values(ascending=False).head(10))
        print(f"\n[{EXP_NAME}] 特征重要性 (跨 5 段 gain 均值, top 10):")
        for f, g in avg_imp.items():
            print(f"  {f:<24} gain = {g:.4f}")

    # 写训练日志
    if train_logs:
        pd.DataFrame(train_logs).to_csv(os.path.join(EXP_DIR, "train_log.csv"), index=False)

    # 追加 summary.csv
    fieldnames = [
        "exp", "train_window",
        "ls_annual", "ls_sharpe", "ls_calmar", "ls_mdd",
        "long_annual", "long_sharpe", "long_calmar", "long_excess_annual",
        "short_annual", "short_sharpe",
        "avg_n_signal_stocks", "avg_n_long", "avg_n_short",
        "avg_n_legal_pairs", "avg_n_top20_pairs",
    ]
    save_summary_row(os.path.join(OUTPUT_DIR, "summary.csv"), {
        "exp": EXP_NAME,
        "train_window": "",
        "ls_annual":  stats["longshort"]["annual_return"],
        "ls_sharpe":  stats["longshort"]["sharpe_ratio"],
        "ls_calmar":  stats["longshort"].get("calmar_ratio", 0.0),
        "ls_mdd":     stats["longshort"]["max_drawdown"],
        "long_annual":         stats["head"]["annual_return"],
        "long_sharpe":         stats["head"]["sharpe_ratio"],
        "long_calmar":         stats["head"].get("calmar_ratio", 0.0),
        "long_excess_annual":  stats["head_excess"]["annual_return"],
        "short_annual":        stats["tail"]["annual_return"],
        "short_sharpe":        stats["tail"]["sharpe_ratio"],
    }, fieldnames)
    print(f"[{EXP_NAME}] summary.csv 已追加")
    print(f"\n[{EXP_NAME}] 总耗时 {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
