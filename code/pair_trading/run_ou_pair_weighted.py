#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OU 配对实验 - 加权回测对照 (复用基准的 stock_pred / pair_log, 仅重跑回测层)

每个加权实验对应 1 个基准实验, 复用基准的 top20 pair 选择,
仅改"组内等权"为"组内按权重加权" (∑w=1):
    - SW (Signal Weighted): 权重 ∝ |pair_signal|
    - IW (IC Weighted):     权重 ∝ max(0, cv_mean_ic)

实验列表 (4 个基准 × 2 种加权 = 8 个新实验):
    OU-P00-zs-cv5-h5-SW  / -IW  ←  OU-P00-zs-cv5-h5
    OU-P00-120-zs-cv5-h5-SW / -IW  ←  OU-P00-120-zs-cv5-h5
    OU-P-CC5-zs-cv5-h5-SW / -IW   ←  OU-P-CC5-zs-cv5-h5
    OU-P-CRD50-zs-cv5-h5-SW / -IW ←  OU-P-CRD50-zs-cv5-h5

输出:
    output/0506_ou_pair/{exp_name}-{SW|IW}/
    ├── metrics.json
    ├── longshort_backtest.png / head_backtest.png / tail_backtest.png
    ├── yearly_longshort.png
    └── (无 stock_pred / pair_log, 复用基准的)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir("/root/quant")

_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
if _FACTOR1_DIR not in sys.path:
    sys.path.insert(0, _FACTOR1_DIR)
from _pair_runner import (
    load_price_industry_mv,
    run_longshort_backtest_weighted,
    save_summary_row,
)

OUTPUT_DIR = "output/0506_ou_pair"
COMMISSION = 0.0007

# 基准实验配置: name -> holding_period (与 run_ou_pair.py 中 EXPERIMENTS 一致)
BASELINES = {
    "OU-P00-zs-cv5-h5":       5,
    "OU-P00-120-zs-cv5-h5":   5,
    "OU-P-CC5-zs-cv5-h5":     5,
    "OU-P-CRD50-zs-cv5-h5":   5,
}

# 加权方式 (后缀 -> (描述, 字段, 处理函数))
# 处理函数: 把 pair_log/stock_pred 中的字段值转成非负权重 (后续按截面 ∑w=1 归一化)
WEIGHT_MODES = {
    "SW": ("信号加权 |pair_signal|", "abs_pair_signal",
           lambda v: np.abs(v)),
    "IW": ("IC 加权 max(0, cv_mean_ic)", "cv_mean_ic",
           lambda v: np.clip(v, 0.0, None)),
}


def build_weight_dicts(stock_pred: pd.DataFrame, pair_log: pd.DataFrame,
                        weight_field: str, weight_fn) -> Dict[str, pd.Series]:
    """
    为加权回测构造 long_holdings_w / short_holdings_w (Series of dict)。

    工作流:
        1. 从 pair_log 抽出 (date, stock_i, stock_j, signal, cv_mean_ic) 的 top20 子集
        2. join 到 stock_pred (按 date + (stock_i, stock_j) 唯一标识 pair)
        3. 截面内每只票算 raw_weight = weight_fn(pair_field)
        4. 同截面内每只票仅出现 1 次 (贪心去重保证), 按 long/short 分组归一化 ∑w=1

    返回:
        {"long": Series[date -> dict[code, w]], "short": Series[date -> dict[code, w]]}
    """
    sp = stock_pred.copy()
    sp["date"] = pd.to_datetime(sp["date"]).dt.normalize()
    sp["stock_code"] = sp["stock_code"].astype(str).str.zfill(6)
    sp["paired_stock"] = sp["paired_stock"].astype(str).str.zfill(6)

    if weight_field == "abs_pair_signal":
        # stock_pred 已有 pair_signal, 直接 abs
        sp["raw_weight"] = sp["pair_signal"].abs()
    else:
        # 需要从 pair_log join cv_mean_ic, pair 唯一键 = (date, stock_i, stock_j)
        # stock_pred 中: long 行的 (stock_code, paired_stock) = (stock_i, stock_j)
        #                short 行的 (stock_code, paired_stock) = (stock_j, stock_i)
        # 都对应同一 pair, 但 stock_i < stock_j (字典序固定), 所以这里取 min/max 还原 pair key
        sp["pair_i"] = np.minimum(sp["stock_code"].values, sp["paired_stock"].values)
        sp["pair_j"] = np.maximum(sp["stock_code"].values, sp["paired_stock"].values)

        pl = pair_log[pair_log["is_top20"] == True].copy()  # noqa: E712
        pl["date"] = pd.to_datetime(pl["date"]).dt.normalize()
        pl["stock_i"] = pl["stock_i"].astype(str).str.zfill(6)
        pl["stock_j"] = pl["stock_j"].astype(str).str.zfill(6)
        pl_key = pl[["date", "stock_i", "stock_j", weight_field]].rename(
            columns={"stock_i": "pair_i", "stock_j": "pair_j"})

        sp = sp.merge(pl_key, on=["date", "pair_i", "pair_j"], how="left")
        miss = sp[weight_field].isna().sum()
        if miss > 0:
            print(f"  [WARN] {miss}/{len(sp)} 行 {weight_field} 缺失 (将被截断为 0)")
        sp["raw_weight"] = weight_fn(sp[weight_field].fillna(0.0).values)

    # 组内 ∑w=1 归一化 (按 date + side)
    grp = sp.groupby(["date", "side"])["raw_weight"].transform("sum")
    sp["weight"] = np.where(grp > 0, sp["raw_weight"] / grp, 0.0)

    # 转 Series of dict
    out = {}
    for side in ("long", "short"):
        sub = sp[sp["side"] == side]
        s = sub.groupby("date").apply(
            lambda g: dict(zip(g["stock_code"].tolist(), g["weight"].tolist()))
        )
        out[side] = s
    return out


def run_one_weighted(exp_name: str, baseline: str, weight_suffix: str,
                      open_wide: pd.DataFrame, close_wide: pd.DataFrame,
                      status_wide: pd.DataFrame):
    desc, weight_field, weight_fn = WEIGHT_MODES[weight_suffix]
    holding_period = BASELINES[baseline]

    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"\n[{exp_name}] 已有 metrics.json, 跳过")
        return

    base_dir = os.path.join(OUTPUT_DIR, baseline)
    sp_path = os.path.join(base_dir, "stock_pred.parquet")
    pl_path = os.path.join(base_dir, "pair_log.parquet")
    if not (os.path.exists(sp_path) and os.path.exists(pl_path)):
        print(f"\n[{exp_name}] 基准 {baseline} 缺少 stock_pred/pair_log, 跳过")
        return

    print(f"\n{'='*60}")
    print(f"[{exp_name}] 基准={baseline}, 加权={desc}, holding_period={holding_period}")
    print(f"{'='*60}")
    t0 = time.time()
    os.makedirs(exp_dir, exist_ok=True)

    print("  读取基准 stock_pred / pair_log ...")
    stock_pred = pd.read_parquet(sp_path)
    pair_log = pd.read_parquet(pl_path)
    print(f"    stock_pred {len(stock_pred):,} 行, pair_log {len(pair_log):,} 行")

    print("  构造权重字典 ...")
    w_dicts = build_weight_dicts(stock_pred, pair_log, weight_field, weight_fn)
    long_w = w_dicts["long"]; short_w = w_dicts["short"]
    print(f"    long: {len(long_w)} 个截面, short: {len(short_w)} 个截面")

    # 对齐 long/short 索引 (缺失日期填空 dict)
    all_dates = sorted(set(long_w.index) | set(short_w.index))
    long_w = long_w.reindex(all_dates, fill_value={})
    short_w = short_w.reindex(all_dates, fill_value={})

    print(f"  启动加权多空回测 ...")
    bt = run_longshort_backtest_weighted(
        long_holdings_w=long_w,
        short_holdings_w=short_w,
        open_prices=open_wide,
        close_prices=close_wide,
        status_data=status_wide,
        output_dir=exp_dir,
        commission_rate=COMMISSION,
        holding_period=holding_period,
    )
    stats = bt["statistics"]

    # 计算 calmar 并写 metrics.json
    def _add_calmar(s):
        ann = s.get("annual_return", 0.0); mdd = s.get("max_drawdown", 0.0)
        s["calmar_ratio"] = round(ann / abs(mdd), 3) if mdd != 0 else 0.0
        return s
    for k in ("head", "tail", "tail_raw", "longshort", "benchmark",
              "head_excess", "tail_excess", "ls_excess"):
        if k in stats:
            stats[k] = _add_calmar(stats[k])

    metrics = {
        "experiment": exp_name,
        "baseline": baseline,
        "weight_mode": weight_suffix,
        "weight_desc": desc,
        "weight_field": weight_field,
        "holding_period": holding_period,
        "commission_rate": COMMISSION,
        "long":      stats["head"],
        "short":     stats["tail"],
        "longshort": stats["longshort"],
        "benchmark": stats["benchmark"],
        "long_excess": stats["head_excess"],
        "ls_excess":   stats["ls_excess"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{exp_name}] metrics.json 已写入")

    # summary.csv 一行 (复用主 summary)
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
        "train_window": "",  # 加权实验的 train_window 与基准一致, 这里留空避免冗余
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


def prepare_price_wide() -> Dict:
    """加载价格面板, 构造 open/close/status 宽表 (基准与 run_ou_pair.py 一致)."""
    print("=" * 60)
    print("加载价格面板 (复用 run_ou_pair 的数据规格)")
    print("=" * 60)
    # 与 run_ou_pair.py 一致的时间范围, 保证宽表对齐
    DATA_START = "2022-09-01"
    BACKTEST_END = "2026-05-13"
    panel = load_price_industry_mv(DATA_START, BACKTEST_END)
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
    print(f"  宽表 shape: close={close_wide.shape}")
    return {"open_wide": open_wide, "close_wide": close_wide, "status_wide": status_wide}


def main():
    print(f"\n{'='*60}")
    print(f"OU 配对实验 - 加权回测对照  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"{'='*60}")

    data = prepare_price_wide()

    for baseline in BASELINES:
        for suffix in ("SW", "IW"):
            exp_name = f"{baseline}-{suffix}"
            run_one_weighted(
                exp_name=exp_name,
                baseline=baseline,
                weight_suffix=suffix,
                open_wide=data["open_wide"],
                close_wide=data["close_wide"],
                status_wide=data["status_wide"],
            )

    print(f"\n{'='*60}\n所有加权实验完成\n{'='*60}")


if __name__ == "__main__":
    main()
