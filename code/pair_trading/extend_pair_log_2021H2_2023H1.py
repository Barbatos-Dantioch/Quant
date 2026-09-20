#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩展 OU-P00-zs-cv5-h5 的 pair_log 到 2021H2 ~ 2023H1
===================================================

为 XGBoost/LightGBM 实验进一步补充更早的训练数据: 生成
2021-07-01 ~ 2023-06-30 (约 486 个交易日) 的 pair_log,
流程与 OU-P00-zs-cv5-h5 完全一致 (zscore + cv5 + h5 + 行业内 cap=1).

与 extend_pair_log_2023H2.py 唯一区别:
  - 回测范围 2021-07-01 ~ 2023-06-30 (接在现有 2023H2 段之前)
  - DATA_START 设 2020-01-02 (数据最早日): 2021-07-01 信号日需 272 日历史,
    2021-01-04 起不足, 必须从 2020-01-02 加载

输出: output/0506_ou_pair/_train_extra_2021H2_2023H1/pair_log.parquet
      (独立目录, 不覆盖基线与现有 2023H2 段)

只跑信号生成阶段, 不做回测.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

os.chdir("/root/quant")

_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
if _FACTOR1_DIR not in sys.path:
    sys.path.insert(0, _FACTOR1_DIR)

# Monkey-patch BACKTEST_START / END 让 prepare_data 加载到 2021H2~2023H1 范围
import run_ou_pair as ROP
ROP.BACKTEST_START = "2021-07-01"
ROP.BACKTEST_END   = "2023-06-30"
# 价格表实际起于 2020-01-02; OU 训练窗 272 日历史 → 2021-07-01 信号日需 ~14 个月历史
# 2020-01-02 到 2021-07-01 约 360 交易日 (> 272), 数据完全支持; 设最早日留足 buffer
ROP.DATA_START     = "2020-01-02"

OUTPUT_DIR = "output/0506_ou_pair/_train_extra_2021H2_2023H1"
TRAIN_WINDOW = 252      # 与 OU-P00-zs-cv5-h5 一致
VALID_WINDOW = 20
SIGNAL_TYPE  = "zscore"
CV_FOLDS     = 5
CV_LABEL_H   = 5


def main():
    print(f"\n{'='*60}")
    print(f"扩展 pair_log (训练补充·2021H2~2023H1)  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PID: {os.getpid()}")
    print(f"  回测范围: {ROP.BACKTEST_START} ~ {ROP.BACKTEST_END}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  其它参数: train={TRAIN_WINDOW}, valid={VALID_WINDOW}, "
          f"signal_type={SIGNAL_TYPE}, cv_folds={CV_FOLDS}, cv_label_h={CV_LABEL_H}")
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
    stats_df = pd.DataFrame(daily_stats)

    out_pl = os.path.join(OUTPUT_DIR, "pair_log.parquet")
    out_stats = os.path.join(OUTPUT_DIR, "daily_stats.parquet")
    pair_log_full.to_parquet(out_pl, index=False)
    stats_df.to_parquet(out_stats, index=False)

    print(f"\n{'='*60}")
    print(f"扩展完成, 总耗时 {time.time()-t0:.0f}s")
    print(f"  pair_log: {len(pair_log_full):,} 行, "
          f"{pair_log_full['date'].nunique()} 个截面")
    print(f"  is_legal=True: {int(pair_log_full['is_legal'].sum()):,} 行 "
          f"({pair_log_full['is_legal'].mean()*100:.1f}%)")
    print(f"  is_top20=True: {int(pair_log_full['is_top20'].sum()):,} 行")
    print(f"  已写入: {out_pl}")
    print(f"  已写入: {out_stats}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
