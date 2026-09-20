#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重建 OU-P00-zs-cv5-h5 实验的每日持仓 (stock_pred.parquet)
=========================================================

背景: 原实验目录只剩 metrics.json, 每日持仓 (stock_pred.parquet) 已被清理。
Barra 因子暴露分析需要每日多/空持仓股票, 故复用 run_ou_pair 的信号阶段重跑该
单个配置, 输出到独立目录, 不触碰原 metrics.json。

复现配置 (与 EXPERIMENTS 中 OU-P00-zs-cv5-h5 严格一致):
    train_window=252, signal_type=zscore, holding_period=5, cv_folds=5, cv_label_h=5
    其余参数 (corr_filter/nzc_filter/pairing=industry/dedup_max=1/min_uniq=50) 用默认值。

输出: output/0506_ou_pair/barra/OU-P00-zs-cv5-h5/
      ├── stock_pred.parquet   每日股票级多/空预测 (date, stock_code, side, ...)
      ├── pair_log.parquet
      ├── daily_stats.parquet
      └── metrics.json (实验名记为 barra/OU-P00-zs-cv5-h5)
"""
import os
import sys

_PT_DIR = "/root/quant/xgbcode/pair_trading"
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
os.chdir("/root/quant")

import run_ou_pair as base


def main():
    print("加载共享数据 (prepare_data) ...")
    data = base.prepare_data()

    # exp_name 用子目录, 使 exp_dir = output/0506_ou_pair/barra/OU-P00-zs-cv5-h5,
    # 与原实验目录 output/0506_ou_pair/OU-P00-zs-cv5-h5 隔离, 不覆盖原 metrics.json
    base.run_one_experiment(
        exp_name="barra/OU-P00-zs-cv5-h5",
        train_window=252,
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
        signal_type="zscore",
        holding_period=5,
        cv_folds=5,
        cv_label_h=5,
    )
    print("\n持仓重建完成: output/0506_ou_pair/barra/OU-P00-zs-cv5-h5/stock_pred.parquet")


if __name__ == "__main__":
    main()
