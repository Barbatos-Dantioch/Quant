#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 NZC 去重 + 市值差距过滤 候选池 pair_log_full_NZCdedupMV10.parquet
====================================================================

在 build_pair_log_nzcdedup.py(NZCdedup 候选池)基础上, 唯一新增:
  配对时市值差距硬过滤 —— 行业内枚举 pair 后、OU 估计前, 仅保留
  |log mv_i - log mv_j| < MV_GAP_MAX 的 pair (市值倍数 < e^MV_GAP_MAX)。
  其余完全一致: train=252, valid=20, signal_type=zscore, cv_folds=5,
  cv_label_h=5, dedup_rank=nzc, cap=1。

依据: 诊断显示配对内市值差距越小, 平均 pnl 越高(gap<0.5≈+0.013 vs >1.5≈+0.005);
MV_GAP_MAX=1.0(倍数<2.7x)在候选池级保留约 62%, cuts 掉 pnl 最薄的悬殊市值对。

覆盖区间 2021-07-01 ~ 2026-05-13(与 pair_log_full_NZCdedup 对齐), 一次性生成,
输出 output/0506_ou_pair/pair_log_full_NZCdedupMV10.parquet, 供
run_ou_pair_nn_ws.py 经 PAIR_POOL=NZCdedupMV10 读取。

只跑信号生成阶段, 不做回测。预计单进程 ~2 小时。
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

os.chdir("/root/quant")

_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
if _FACTOR1_DIR not in sys.path:
    sys.path.insert(0, _FACTOR1_DIR)

import run_ou_pair as ROP
ROP.BACKTEST_START = "2021-07-01"
ROP.BACKTEST_END   = "2026-05-13"
ROP.DATA_START     = "2020-01-02"

OUTPUT_DIR   = "output/0506_ou_pair"
OUT_PATH     = os.path.join(OUTPUT_DIR, "pair_log_full_NZCdedupMV10.parquet")
TRAIN_WINDOW = 252
VALID_WINDOW = 20
SIGNAL_TYPE  = "zscore"
CV_FOLDS     = 5
CV_LABEL_H   = 5
DEDUP_RANK   = "nzc"
MV_GAP_MAX   = 1.0      # 本实验唯一新增变量: 配对内 |log mv_i - log mv_j| < 1.0 (倍数 < 2.7x)


def main():
    print(f"\n{'='*60}")
    print(f"生成 NZC去重+市值过滤 候选池  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}")
    print(f"  回测范围: {ROP.BACKTEST_START} ~ {ROP.BACKTEST_END}")
    print(f"  输出文件: {OUT_PATH}")
    print(f"  参数: train={TRAIN_WINDOW}, valid={VALID_WINDOW}, signal_type={SIGNAL_TYPE}, "
          f"cv_folds={CV_FOLDS}, cv_label_h={CV_LABEL_H}, dedup_rank={DEDUP_RANK}, "
          f"mv_gap_max={MV_GAP_MAX}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    data = ROP.prepare_data()
    cal = data["cal"]

    date_to_idx = {dt: i for i, dt in enumerate(pd.DatetimeIndex(cal))}
    backtest_dates = data["backtest_dates"]
    print(f"\n回测截面数: {len(backtest_dates)}")

    pair_log_chunks = []
    daily_stats = []
    n_dates = len(backtest_dates)

    for k, sig_date in enumerate(backtest_dates):
        if sig_date not in date_to_idx:
            continue
        t_idx = date_to_idx[sig_date]
        sp_set = data["short_pool_by_date"].get(sig_date, set())
        verbose = (k < 3) or (k % 10 == 0) or (k == n_dates - 1)

        out = ROP.process_one_section(
            section_idx=t_idx, cal=cal,
            log_price_wide=data["log_price_wide"], ret_wide=data["ret_wide"],
            industry_codes=data["industry_codes"], stock_codes=data["stock_codes"],
            short_pool_set=sp_set,
            train_window=TRAIN_WINDOW, valid_window=VALID_WINDOW,
            signal_type=SIGNAL_TYPE, cv_folds=CV_FOLDS, cv_label_h=CV_LABEL_H,
            dedup_rank=DEDUP_RANK,
            mv_wide=data["log_mv_wide"], mv_gap_max=MV_GAP_MAX,
            verbose=verbose,
        )
        if out.get("skip"):
            daily_stats.append({
                "date": sig_date, "skip": True, "reason": out.get("reason", ""),
                "n_pairs_total": out.get("n_pairs_total", 0),
                "n_pairs_ou_pass": out.get("n_pairs_ou_pass", 0),
                "n_pairs_dedup": out.get("n_pairs_dedup", 0),
                "n_pairs_legal": out.get("n_pairs_legal", 0),
                "n_pairs_top20": out.get("n_pairs_top20", 0),
            })
            continue
        pair_log_chunks.append(out["pair_log"])
        daily_stats.append({
            "date": sig_date, "skip": False, "reason": "",
            "n_pairs_total": out["n_pairs_total"],
            "n_pairs_ou_pass": out["n_pairs_ou_pass"],
            "n_pairs_dedup": out["n_pairs_dedup"],
            "n_pairs_legal": out["n_pairs_legal"],
            "n_pairs_top20": out["n_pairs_top20"],
        })

    if not pair_log_chunks:
        print("\n[ERROR] 无任何有效截面, 退出")
        return

    pair_log_full = pd.concat(pair_log_chunks, ignore_index=True)
    pair_log_full["date"] = pd.to_datetime(pair_log_full["date"])
    pair_log_full["stock_i"] = pair_log_full["stock_i"].astype(str).str.zfill(6)
    pair_log_full["stock_j"] = pair_log_full["stock_j"].astype(str).str.zfill(6)
    pair_log_full = pair_log_full.sort_values(
        ["date", "stock_i", "stock_j"]).reset_index(drop=True)
    stats_df = pd.DataFrame(daily_stats)

    out_stats = os.path.join(OUTPUT_DIR, "daily_stats_NZCdedupMV10.parquet")
    pair_log_full.to_parquet(OUT_PATH, index=False)
    stats_df.to_parquet(out_stats, index=False)

    print(f"\n{'='*60}")
    print(f"完成, 总耗时 {time.time()-t0:.0f}s")
    print(f"  pair_log: {len(pair_log_full):,} 行, "
          f"{pair_log_full['date'].nunique()} 个截面, "
          f"{pair_log_full['date'].min().date()} ~ {pair_log_full['date'].max().date()}")
    print(f"  is_legal=True: {int(pair_log_full['is_legal'].sum()):,} 行 "
          f"({pair_log_full['is_legal'].mean()*100:.1f}%)")
    print(f"  is_top20=True: {int(pair_log_full['is_top20'].sum()):,} 行")
    print(f"  已写入: {OUT_PATH}")
    print(f"  已写入: {out_stats}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
