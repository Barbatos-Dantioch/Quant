#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""realized_skew 极端度衍生因子：捕捉 U 型（两端最优）关系。

定义：
    rank_pct = 当日截面 realized_skew 百分位排名，取值 [0, 1]
    realized_skew_extreme = 2 * |rank_pct - 0.5|

    - 截面中位（rank_pct=0.5）→ 0（最小，对应平淡交易日）
    - 两端极端（rank_pct=0 或 1）→ 1（最大，对应强负偏或强正偏）
    - 不做符号翻转：值越大表示日内尾部事件越强

输出：
    /root/quant/output/Tick/realized_skew_extreme/
"""
import gc
import json
import os
import sys

import pandas as pd

FUNC_DIR = '/root/quant/xgbcode/func'
TICK_DIR = '/root/quant/xgbcode/Tick'
for d in (FUNC_DIR, TICK_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import experiment_engine as eng
from backtest_evaluator import BacktestEvaluator
from backtest_tick_factors import (
    build_all_a_benchmark_close_to_close,
    build_all_a_benchmark_forward,
    build_factor_df,
    build_forward_return_wide,
    forward_return_wide_to_long,
    safe_get,
)
from factor_transforms import compute_extreme_quantile_dev

FACTOR_PANEL_PATH = '/root/quant/Data/Tick/factors/tick_factor_panel.pkl'
PANEL_TRADE_PATH = '/root/quant/Data/all/panel_trade.pkl'
OUTPUT_DIR = '/root/quant/output/Tick'
DERIVED_PANEL_PATH = '/root/quant/Data/Tick/factors/tick_factor_panel_extreme.pkl'

BASE_COL = 'realized_skew'
DERIVED_COL = 'realized_skew_extreme'
N_GROUPS = 10
REBALANCE_DAYS = 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DERIVED_PANEL_PATH), exist_ok=True)

    print(f'加载因子面板，构造 {DERIVED_COL}...', flush=True)
    panel = pd.read_pickle(FACTOR_PANEL_PATH)
    panel['date'] = pd.to_datetime(panel['date'])
    panel['stock_code'] = panel['stock_code'].astype(str)
    panel = compute_extreme_quantile_dev(panel, BASE_COL, DERIVED_COL)
    panel.to_pickle(DERIVED_PANEL_PATH)
    print(f'  已保存衍生面板 -> {DERIVED_PANEL_PATH}', flush=True)

    valid = panel[[DERIVED_COL]].dropna()
    print(f'  {DERIVED_COL}: mean={valid[DERIVED_COL].mean():.4f}, '
          f'min={valid[DERIVED_COL].min():.4f}, max={valid[DERIVED_COL].max():.4f}', flush=True)

    evaluator = BacktestEvaluator(
        panel_trade_path=PANEL_TRADE_PATH,
        avg_return_daily_path=None,
        n_groups=N_GROUPS,
        rebalance_days=REBALANCE_DAYS,
    )
    fwd_ret_wide = build_forward_return_wide(evaluator.open_prices)
    fwd_return = forward_return_wide_to_long(fwd_ret_wide)
    evaluator.benchmark_returns = build_all_a_benchmark_close_to_close(evaluator.close_prices)
    forward_benchmark = build_all_a_benchmark_forward(fwd_ret_wide)
    eng.avg_return = forward_benchmark.rename('avg_return').reset_index().rename(columns={'index': 'date'})

    factor_df = build_factor_df(panel, DERIVED_COL, fwd_return, flip_sign=False)
    print(f'  有效样本: {len(factor_df)} 行, {factor_df["date"].nunique()} 天', flush=True)

    factor_dir = os.path.join(OUTPUT_DIR, DERIVED_COL)
    os.makedirs(factor_dir, exist_ok=True)
    group_save_path = os.path.join(factor_dir, 'group.png')
    holdings_save_path = os.path.join(factor_dir, 'holdings.png')

    group_metrics = evaluator.evaluate_factor_df(factor_df, save_path=group_save_path, train_windows=0)
    holdings_metrics = evaluator.evaluate_head_holdings(factor_df, save_path=holdings_save_path)

    row = {
        'factor_name': DERIVED_COL,
        'base_factor': BASE_COL,
        'flip_sign': False,
        'sample_rows': len(factor_df),
        'date_count': factor_df['date'].nunique(),
        'rank_ic_mean': safe_get(group_metrics, 'rank_ic_mean'),
        'rank_ic_ir': safe_get(group_metrics, 'rank_ic_ir'),
        'head_excess_return_annualized': safe_get(group_metrics, 'head_excess_return_annualized'),
        'head_excess_ir': safe_get(group_metrics, 'head_excess_ir'),
        'head_excess_max_drawdown': safe_get(group_metrics, 'head_excess_max_drawdown'),
        'head_turnover_mean': safe_get(group_metrics, 'head_turnover_mean'),
        'head_holdings_strategy_annual_return_pct': safe_get(holdings_metrics, 'head_holdings_strategy_annual_return_pct'),
        'head_holdings_excess_annual_return_pct': safe_get(holdings_metrics, 'head_holdings_excess_annual_return_pct'),
        'head_holdings_excess_sharpe_ratio': safe_get(holdings_metrics, 'head_holdings_excess_sharpe_ratio'),
        'head_holdings_excess_max_drawdown_pct': safe_get(holdings_metrics, 'head_holdings_excess_max_drawdown_pct'),
    }
    metrics_path = os.path.join(factor_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2, default=str)

    print(f'\n=== {DERIVED_COL} 回测结果 ===', flush=True)
    print(f'  rank_ic_mean={row["rank_ic_mean"]:.4f}, rank_ic_ir={row["rank_ic_ir"]:.4f}', flush=True)
    print(f'  head_excess_ann={row["head_excess_return_annualized"]:.4f}, '
          f'head_excess_ir={row["head_excess_ir"]:.4f}', flush=True)
    print(f'  结果目录 -> {factor_dir}', flush=True)

    del factor_df, panel
    gc.collect()


if __name__ == '__main__':
    main()
