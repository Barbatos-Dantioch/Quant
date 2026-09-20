#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OU 配对·DEV 系列信号 Replay 实验
================================

复用 OU-P00-zs-cv5-h5 的 pair_log (OU 过滤 + CV5 + 行业内贪心去重选出的 pair 池子
完全沿用基线), 仅在截面 top20 排序阶段换 DEV 系列信号公式, 回测层一致.

设计动机:
    已有的 OU-P-DEV-* 系列把"新信号公式"贯穿了 CV5 选 pair 阶段, 导致 pair 池子
    本身就和基线不同, 没法干净对照"信号公式是否更优". 本实验把 pair 池子完全固定
    为基线输出, 唯一变量 = 截面 top20 的 magnitude 公式.

5 个实验 (magnitude 公式):
    OU-P-RPL-DEV-cv5-h5:      |μ - X|                  (σ 不参与)
    OU-P-RPL-DEV-SQ20-cv5-h5: |μ - X| · √σ_20          (σ_20 弱加权)
    OU-P-RPL-DEV-S20-cv5-h5:  |μ - X| · σ_20           (σ_20 等权加权)
    OU-P-RPL-DEV-SQ30-cv5-h5: |μ - X| · √σ_30          (σ_30 弱加权)
    OU-P-RPL-DEV-S30-cv5-h5:  |μ - X| · σ_30           (σ_30 等权加权)

方向 sign 全部由 sign(μ - X_t) 决定 (与基线 zs 一致), 故 is_legal 集合不变.

输出:
    output/0506_ou_pair/OU-P-RPL-DEV-*/metrics.json
    output/0506_ou_pair/OU-P-RPL-DEV-*/longshort_backtest.png 等
    追加到 output/0506_ou_pair/summary.csv
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from typing import Callable, Dict, List, Tuple

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

# ── 常量 ──
BASELINE = "OU-P00-zs-cv5-h5"
OUTPUT_DIR = "output/0506_ou_pair"
COMMISSION = 0.0007
TOP_PCT = 0.20
TOP_PCT_MIN_N = 50
HOLDING_PERIOD = 5
DATA_START = "2022-09-01"
DATA_END = "2026-05-13"
_EPS = 1e-12


# ── magnitude 公式 ──
# 统一签名: (abs_dev, sigma_20, sigma_30, sigma_ou, kappa)
# 返回: magnitude 数组 (>=0)
# 旧的 DEV 系列 (σ 在分子, 正向加权) 不使用 sigma_ou / kappa, 仅占位以统一接口
def _mag_dev(abs_dev, sigma_20, sigma_30, sigma_ou, kappa):
    return abs_dev


def _mag_dev_sqrt_s20(abs_dev, sigma_20, sigma_30, sigma_ou, kappa):
    return abs_dev * np.sqrt(np.clip(sigma_20, _EPS, None))


def _mag_dev_s20(abs_dev, sigma_20, sigma_30, sigma_ou, kappa):
    return abs_dev * np.clip(sigma_20, _EPS, None)


def _mag_dev_sqrt_s30(abs_dev, sigma_20, sigma_30, sigma_ou, kappa):
    return abs_dev * np.sqrt(np.clip(sigma_30, _EPS, None))


def _mag_dev_s30(abs_dev, sigma_20, sigma_30, sigma_ou, kappa):
    return abs_dev * np.clip(sigma_30, _EPS, None)


def _mag_zs_x_z20(abs_dev, sigma_20, sigma_30, sigma_ou, kappa):
    """signal = signal_zs / σ_20 · |μ-X| → magnitude = κ · (μ-X)² / (σ_OU · σ_20)
    等价表达: |signal_zs| · |μ-X| / σ_20 (基线 magnitude × 短窗 z-score)."""
    denom = np.clip(sigma_ou, _EPS, None) * np.clip(sigma_20, _EPS, None)
    return kappa * (abs_dev ** 2) / denom


EXPERIMENTS = [
    ("OU-P-RPL-DEV-cv5-h5",      "abs_dev",            _mag_dev),
    ("OU-P-RPL-DEV-SQ20-cv5-h5", "abs_dev_sqrt_s20",   _mag_dev_sqrt_s20),
    ("OU-P-RPL-DEV-S20-cv5-h5",  "abs_dev_s20",        _mag_dev_s20),
    ("OU-P-RPL-DEV-SQ30-cv5-h5", "abs_dev_sqrt_s30",   _mag_dev_sqrt_s30),
    ("OU-P-RPL-DEV-S30-cv5-h5",  "abs_dev_s30",        _mag_dev_s30),
    ("OU-P-RPL-ZSxZ20-cv5-h5",   "signal_zs_x_z20",    _mag_zs_x_z20),
]


def prepare_shared_data() -> Dict:
    """加载所有 5 个实验共享的数据 (基线 pair_log + 价格宽表)."""
    print("=" * 60)
    print(f"加载基线 pair_log: {BASELINE}")
    print("=" * 60)
    pl_path = os.path.join(OUTPUT_DIR, BASELINE, "pair_log.parquet")
    pl = pd.read_parquet(pl_path, columns=[
        "date", "stock_i", "stock_j", "industry", "mu", "kappa", "sigma",
        "is_i_short_pool", "is_j_short_pool", "is_legal", "valid_rank_ic",
        "cv_mean_ic", "signal",
    ])
    pl["date"] = pd.to_datetime(pl["date"])
    pl["stock_i"] = pl["stock_i"].astype(str).str.zfill(6)
    pl["stock_j"] = pl["stock_j"].astype(str).str.zfill(6)
    print(f"  pair_log: {len(pl):,} 行, "
          f"is_legal={int(pl['is_legal'].sum()):,}, "
          f"截面 {pl['date'].nunique()}")

    print("\n加载价格面板...")
    panel = load_price_industry_mv(DATA_START, DATA_END)
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
    cal_index = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    stock_index = panel["stock_code"].drop_duplicates().sort_values().reset_index(drop=True)
    stock_codes = stock_index.to_numpy(dtype=object)

    close_wide = panel.pivot(index="date", columns="stock_code", values="close_price").reindex(
        index=cal_index, columns=stock_codes)
    open_wide = panel.pivot(index="date", columns="stock_code", values="open_price").reindex(
        index=cal_index, columns=stock_codes)
    status_wide = panel.pivot(index="date", columns="stock_code", values="status").reindex(
        index=cal_index, columns=stock_codes)

    # log_close ndarray, 供截面快速索引算 X_t / σ 短窗
    close_arr = close_wide.to_numpy(dtype=np.float64)
    log_arr = np.where(close_arr > 0, np.log(close_arr), np.nan)
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    date_to_idx = {d: i for i, d in enumerate(cal_index)}

    # 在 pair_log 中预先映射 i_idx, j_idx, t_idx (一次性, 5 个实验复用)
    pl = pl[pl["stock_i"].isin(code_to_idx) & pl["stock_j"].isin(code_to_idx)].copy()
    pl["i_idx"] = pl["stock_i"].map(code_to_idx).astype(np.int32)
    pl["j_idx"] = pl["stock_j"].map(code_to_idx).astype(np.int32)
    pl["t_idx"] = pl["date"].map(date_to_idx)
    pl = pl.dropna(subset=["t_idx"])
    pl["t_idx"] = pl["t_idx"].astype(np.int32)
    print(f"  对齐后 pair_log: {len(pl):,} 行, "
          f"log_close shape={log_arr.shape}")

    return {
        "pair_log": pl,
        "log_arr": log_arr,
        "open_wide": open_wide,
        "close_wide": close_wide,
        "status_wide": status_wide,
    }


def build_holdings_for_signal(
    legal_pl: pd.DataFrame,
    log_arr: np.ndarray,
    mag_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """按截面计算新 magnitude, 选 top, 返回 long_holdings / short_holdings (Series-of-list) + 每日统计.

    Args:
        legal_pl: pair_log 中 is_legal=True 的子集 (已带 i_idx/j_idx/t_idx)
        log_arr: (T, S) log_close 数组
        mag_func: magnitude 公式 (|μ-X|, σ_20, σ_30) → magnitude

    Returns:
        long_holdings:  pd.Series, index=date, values=list[stock_code]
        short_holdings: pd.Series, index=date, values=list[stock_code]
        daily_stats:    List[Dict], 每日统计 (date, n_legal, n_top, n_skip)
    """
    T_total = log_arr.shape[0]
    long_rows: List[Tuple[pd.Timestamp, str]] = []
    short_rows: List[Tuple[pd.Timestamp, str]] = []
    daily_stats: List[Dict] = []

    # 按 t_idx 分组, 每个截面批量计算
    for t_idx, sub in legal_pl.groupby("t_idx", sort=True):
        if t_idx < 30:  # 需要 30 日历史算 σ_30
            continue
        if t_idx + HOLDING_PERIOD >= T_total:
            continue

        i_arr = sub["i_idx"].values
        j_arr = sub["j_idx"].values

        # X 训练窗末 30 日 + 信号日 X_t
        X_hist30 = log_arr[t_idx - 30: t_idx, i_arr] - log_arr[t_idx - 30: t_idx, j_arr]
        X_t = log_arr[t_idx, i_arr] - log_arr[t_idx, j_arr]
        # NaN 防御
        if np.isnan(X_t).any():
            valid_mask = ~np.isnan(X_t) & ~np.isnan(X_hist30).any(axis=0)
        else:
            valid_mask = ~np.isnan(X_hist30).any(axis=0)
        if not valid_mask.any():
            daily_stats.append({"date": sub["date"].iloc[0], "n_legal": len(sub),
                                "n_top": 0, "n_skip": int((~valid_mask).sum())})
            continue

        sigma_20 = np.nanstd(X_hist30[-20:], axis=0, ddof=1)
        sigma_30 = np.nanstd(X_hist30, axis=0, ddof=1)
        mu = sub["mu"].values
        sigma_ou = sub["sigma"].values
        kappa_arr = sub["kappa"].values
        abs_dev = np.abs(mu - X_t)
        sign_pred = np.sign(mu - X_t)

        mag = mag_func(abs_dev, sigma_20, sigma_30, sigma_ou, kappa_arr)
        # 把 valid_mask=False 处 magnitude 置 -inf, 自动排到末尾
        mag = np.where(valid_mask & (sign_pred != 0), mag, -np.inf)

        # 排序选 top: n_top = min(n_legal, max(50, ceil(0.2 * n_legal))) (与生产一致)
        n_legal = len(sub)
        n_top = min(n_legal, max(TOP_PCT_MIN_N, int(math.ceil(n_legal * TOP_PCT))))
        order = np.argsort(-mag, kind="stable")
        keep = order[:n_top]
        # 过滤 mag = -inf 的样本 (跳过)
        keep = keep[mag[keep] > -np.inf]
        if len(keep) == 0:
            daily_stats.append({"date": sub["date"].iloc[0], "n_legal": n_legal,
                                "n_top": 0, "n_skip": int((~valid_mask).sum())})
            continue

        sig_date = pd.Timestamp(sub["date"].iloc[0])
        stock_i_arr = sub["stock_i"].values
        stock_j_arr = sub["stock_j"].values

        for k in keep:
            s = sign_pred[k]
            if s > 0:
                long_rows.append((sig_date, stock_i_arr[k]))
                short_rows.append((sig_date, stock_j_arr[k]))
            elif s < 0:
                long_rows.append((sig_date, stock_j_arr[k]))
                short_rows.append((sig_date, stock_i_arr[k]))

        daily_stats.append({"date": sig_date, "n_legal": n_legal,
                            "n_top": int(len(keep)),
                            "n_skip": int((~valid_mask).sum())})

    long_df = pd.DataFrame(long_rows, columns=["date", "stock_code"])
    short_df = pd.DataFrame(short_rows, columns=["date", "stock_code"])
    long_holdings = long_df.groupby("date")["stock_code"].apply(list)
    short_holdings = short_df.groupby("date")["stock_code"].apply(list)
    all_dates = sorted(set(long_holdings.index) | set(short_holdings.index))
    long_holdings = long_holdings.reindex(all_dates, fill_value=[])
    short_holdings = short_holdings.reindex(all_dates, fill_value=[])
    return long_holdings, short_holdings, daily_stats


def replay_one(exp_name: str, signal_label: str, mag_func: Callable,
               shared: Dict):
    """跑单个 replay 实验."""
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"\n[{exp_name}] 已有 metrics.json, 跳过")
        return

    print(f"\n{'='*60}")
    print(f"[{exp_name}] signal={signal_label}, baseline_pool={BASELINE}, h={HOLDING_PERIOD}")
    print(f"{'='*60}")
    t0 = time.time()
    os.makedirs(exp_dir, exist_ok=True)

    legal_pl = shared["pair_log"][shared["pair_log"]["is_legal"]].copy()
    long_holdings, short_holdings, daily_stats = build_holdings_for_signal(
        legal_pl=legal_pl, log_arr=shared["log_arr"], mag_func=mag_func,
    )

    stats_df = pd.DataFrame(daily_stats)
    print(f"  截面 {len(stats_df)} 个, 平均 n_top {stats_df['n_top'].mean():.1f}, "
          f"耗时 {time.time()-t0:.0f}s")

    print(f"  启动多空回测 (等权 h={HOLDING_PERIOD})...")
    bt = run_longshort_backtest(
        long_holdings=long_holdings,
        short_holdings=short_holdings,
        open_prices=shared["open_wide"],
        close_prices=shared["close_wide"],
        status_data=shared["status_wide"],
        output_dir=exp_dir,
        commission_rate=COMMISSION,
        holding_period=HOLDING_PERIOD,
    )
    stats = bt["statistics"]

    # 补 calmar
    def _add_calmar(s):
        ann = s.get("annual_return", 0.0)
        mdd = s.get("max_drawdown", 0.0)
        s["calmar_ratio"] = round(ann / abs(mdd), 3) if mdd != 0 else 0.0
        return s
    for k in ("head", "tail", "tail_raw", "longshort", "benchmark",
              "head_excess", "tail_excess", "ls_excess"):
        if k in stats:
            stats[k] = _add_calmar(stats[k])

    metrics = {
        "experiment": exp_name,
        "baseline_pair_pool": BASELINE,
        "signal_formula": signal_label,
        "holding_period": HOLDING_PERIOD,
        "commission_rate": COMMISSION,
        "top_pct": TOP_PCT,
        "top_pct_min_n": TOP_PCT_MIN_N,
        "n_signal_days": int((stats_df["n_top"] > 0).sum()),
        "avg_n_top_per_day": float(stats_df["n_top"].mean()),
        "long":      stats["head"],
        "short":     stats["tail"],
        "longshort": stats["longshort"],
        "benchmark": stats["benchmark"],
        "long_excess": stats["head_excess"],
        "ls_excess":   stats["ls_excess"],
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{exp_name}] metrics.json 已写入")

    # 追加 summary
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
        "train_window": "",  # replay 实验沿用基线 252, 此处留空避免歧义
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
    print(f"[{exp_name}] summary.csv 已追加")
    print(f"[{exp_name}] 总耗时 {time.time()-t0:.0f}s")


def main():
    print(f"\n{'='*60}")
    print(f"OU 配对·DEV 信号 Replay  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  基线 pair 池: {BASELINE}")
    print(f"{'='*60}")

    shared = prepare_shared_data()

    for exp_name, signal_label, mag_func in EXPERIMENTS:
        replay_one(exp_name, signal_label, mag_func, shared)

    print(f"\n{'='*60}\n所有 Replay 实验完成\n{'='*60}")


if __name__ == "__main__":
    main()
