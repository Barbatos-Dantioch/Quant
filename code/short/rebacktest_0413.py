#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 0413_mlp 中回测范围超出 2024-2025 的实验，截取 factor_df 重新回测。
不重新训练，只更新 metrics.json / factor_df.pkl / yearly_longshort.png。
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "func"))
os.chdir("/root/quant")

import experiment_engine as ree
from run_0413_mlp import (
    build_signals_from_factor,
    MV_TAIL_QUANTILE, OUTPUT_DIR, PRICE_PATH,
)
from func.model_backtest_framework_longshort import analyze_longshort_holdings

sys.path.insert(0, os.path.join(SCRIPT_DIR, "short"))
from run_ensemble_experiments import (
    _save_longshort_figure as save_4panel_figure,
    _save_yearly_longshort_figure as save_yearly_figure,
)

import matplotlib
matplotlib.use("Agg")

BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2025-12-31"

SHORT_PANEL_PATH = "Data/short/panel_origin.pkl"

# (output_dir, exp_id) 列表
NEED_REBACKTEST = [
    ("output/0409_rolling", "RW-02"),
    ("output/0409_rolling", "RW-03"),
]


def load_price_data():
    pt = pd.read_pickle(PRICE_PATH)
    pt["date"] = pd.to_datetime(pt["date"])
    close_df = pt.pivot(index="date", columns="stock_code", values="close_price").sort_index()
    status_df = pt.pivot(index="date", columns="stock_code", values="status").sort_index()
    price_data = {
        "open":   pt.pivot(index="date", columns="stock_code", values="open_price").sort_index(),
        "close":  close_df,
        "status": status_df,
    }
    # 基准：正常交易股票(status==0)的等权日收益率，clip 掉极端值
    daily_ret = close_df.pct_change().clip(-0.11, 0.11)
    normal = (status_df == 0) | status_df.isna()
    bm = daily_ret.where(normal).mean(axis=1).fillna(0)
    price_data["benchmark"] = bm
    del pt, daily_ret, normal
    gc.collect()
    return price_data


def rebacktest_one(exp_dir, exp_id, short_codes, price_data):
    fdf_path = os.path.join(exp_dir, "factor_df.pkl")
    if not os.path.exists(fdf_path):
        print(f"  [{exp_id}] factor_df.pkl 不存在，跳过")
        return

    factor_df = pd.read_pickle(fdf_path)
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    d_min, d_max = factor_df["date"].min(), factor_df["date"].max()
    print(f"  [{exp_id}] 原始范围: {str(d_min)[:10]} ~ {str(d_max)[:10]}, {factor_df['date'].nunique()} 天")

    # 截取 2024-2025
    factor_df = factor_df[
        (factor_df["date"] >= BACKTEST_START) &
        (factor_df["date"] <= BACKTEST_END)
    ].copy()
    print(f"  [{exp_id}] 截取后: {str(factor_df['date'].min())[:10]} ~ {str(factor_df['date'].max())[:10]}, {factor_df['date'].nunique()} 天")

    # RankIC
    ic_vals = []
    for _, g in factor_df.groupby("date"):
        if "return_neutral" in g.columns:
            ic = ree.calculate_rank_ic(g["factor"], g["return_neutral"])
            if not np.isnan(ic):
                ic_vals.append(ic)
    ic_mean = np.mean(ic_vals) if ic_vals else np.nan
    ic_std  = np.std(ic_vals) if ic_vals else np.nan
    ic_ir   = ic_mean / ic_std if ic_std > 0 else np.nan

    # 信号
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

    # 多空回测
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

    # 读取原 config 保留其他字段
    cfg_path = os.path.join(exp_dir, "config.json")
    old_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old_cfg = json.load(f)

    metrics = {
        "experiment": exp_id,
        "rank_ic_mean": round(ic_mean, 6) if not np.isnan(ic_mean) else None,
        "rank_ic_std":  round(ic_std, 6)  if not np.isnan(ic_std)  else None,
        "rank_ic_ir":   round(ic_ir, 6)   if not np.isnan(ic_ir)   else None,
        "overlap_removed": overlap_removed,
        "hidden_dims": old_cfg.get("hidden_dims"),
        "label": old_cfg.get("label"),
        "head": stats["head"],
        "tail": stats["tail"],
        "tail_raw": stats["tail_raw"],
        "longshort": stats["longshort"],
        "benchmark": stats["benchmark"],
        "head_excess": stats["head_excess"],
        "tail_excess": stats["tail_excess"],
    }

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # 覆盖 factor_df（只保留 2024-2025）
    factor_df.to_pickle(fdf_path)

    # 重新出图（与 M-01 格式一致）
    save_4panel_figure(ls_results, os.path.join(exp_dir, "longshort_backtest.png"))
    save_yearly_figure(ls_results, os.path.join(exp_dir, "yearly_longshort.png"))

    ls = stats["longshort"]
    he = stats["head_excess"]
    print(f"  [{exp_id}] IC={ic_mean:.4f}  "
          f"多空年化={ls['annual_return']:.1f}%  "
          f"夏普={ls['sharpe_ratio']:.3f}  "
          f"回撤={ls['max_drawdown']:.1f}%  "
          f"头超={he['annual_return']:.1f}%")

    del factor_df, ls_results
    gc.collect()


def main():
    print(f"\n{'='*60}")
    print(f"重新回测 (2024-2025)  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  实验: {[e[1] for e in NEED_REBACKTEST]}")
    print(f"{'='*60}")

    # 融券池：从 short panel 中提取所有 stock_code
    short_panel = pd.read_pickle(SHORT_PANEL_PATH)
    short_codes = set(short_panel["stock_code"].astype(str).unique())
    del short_panel
    gc.collect()
    print(f"  融券池: {len(short_codes)} 只")

    price_data = load_price_data()
    print(f"  价格数据加载完成")

    for out_dir, exp_id in NEED_REBACKTEST:
        exp_dir = os.path.join(out_dir, exp_id)
        t0 = time.time()
        rebacktest_one(exp_dir, exp_id, short_codes, price_data)
        print(f"  [{exp_id}] 耗时 {time.time()-t0:.1f}s")

    print(f"\n{'='*60}")
    print("完成")


if __name__ == "__main__":
    main()
