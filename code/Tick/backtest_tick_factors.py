#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tick 逐笔因子回测主脚本。

对 tick_factor_panel.pkl 中的 29 个因子逐个做每日调仓回测：
    1. evaluate_factor_df    : 分组 IC / 头组超额收益回测（横截面）
    2. evaluate_head_holdings: 头组每日持仓净值回测

口径约定（已与用户确认）：
    - 因子样本（选股对象）：tick_factor_panel.pkl，已剔除 ST
    - forward return：T 日收盘后可得因子 -> T+1 开盘建仓 -> T+2 开盘卖出，
      return = open_price[T+2] / open_price[T+1] - 1，基于 open_prices 宽表
      （完整交易日历）按行位移计算，跨停牌缺口的样本记为 NaN（详见
      build_forward_return_wide 说明：曾因长表 groupby.shift 按"下一条记录"
      取值，把复牌跳空收益错误分配到停牌前一天，已修复并用数据验证）
    - 基准分两处，口径与各自的收益窗口对齐，不再共用一份序列：
      * eng.avg_return（横截面分组回测 group.png 用）：全A等权
        T+1开盘->T+2开盘远期收益，与个股 return 同一时间窗口，避免时间错位
        （2025-04 关税冲击暴涨暴跌期间，若用收-收基准会与远期收益错位 1-2 天，
        导致全部分组同时"虚假跑输/跑赢"基准，已用数据验证并修复）
      * evaluator.benchmark_returns（头组持仓回测 holdings.png 用）：全A等权
        close_price.pct_change() 收-收日收益 —— analyze_model_holdings 内部本身
        是按自然日逐日模拟隔夜+日内收益，天然收-收口径，与此基准对齐，无需改动
      两者均来自 panel_trade.pkl 全样本，不做 status 过滤、不剔 ST、不做截尾，
      仅用 skipna 跳过新股/退市 NaN
    - 回测参数：n_groups=10, rebalance_days=1（每日调仓）
    - 符号翻转：首轮全量回测（不翻转）后发现 13 个因子 rank_ic_mean 为负
      （即原始定义方向与实际预测方向相反），经用户确认对这 13 个因子做符号
      翻转（factor = -factor）后再按原始因子值升序分组（第9组=最大值组）

输出：
    /root/quant/output/Tick/tick_factor_summary.csv           29 个因子汇总指标
    /root/quant/output/Tick/{factor_name}/group.png         分组回测净值图
    /root/quant/output/Tick/{factor_name}/holdings.png      头组持仓净值图
    /root/quant/output/Tick/{factor_name}/metrics.json      单因子回测指标
"""
import os
import sys
import gc
import json
import numpy as np
import pandas as pd

FUNC_DIR = '/root/quant/xgbcode/func'
if FUNC_DIR not in sys.path:
    sys.path.insert(0, FUNC_DIR)

import experiment_engine as eng
from backtest_evaluator import BacktestEvaluator

FACTOR_PANEL_PATH = '/root/quant/Data/Tick/factors/tick_factor_panel.pkl'
PANEL_TRADE_PATH = '/root/quant/Data/all/panel_trade.pkl'
OUTPUT_DIR = '/root/quant/output/Tick'
N_GROUPS = 10
REBALANCE_DAYS = 1

FACTOR_COLS = [
    'active_net_ratio', 'large_active_net_ratio', 'order_imbalance_amt',
    'cancel_imbalance', 'fast_cancel_ratio', 'fill_amt_ratio',
    'aggressor_ratio', 'tail_active_net_ratio', 'buy_impact_1m', 'realized_skew',
    'minute_buy_absorption', 'static_absorption_buy', 'absorption_asymmetry',
    'side_switch_rate', 'p_sellcancel_to_activebuy',
    'cancel_burstiness', 'drift_after_buy_cancel', 'order_arrival_accel',
    'auction_follow_through',
    'cancel_life_median', 'cancel_life_dispersion', 'pre_trade_cancel_ratio',
    'cancel_to_fill_ratio', 'partial_fill_cancel_ratio', 'deep_cancel_ratio',
    'aggressive_cancel_ratio', 'cancel_flow_toxicity', 'refill_after_cancel_ratio',
    'large_cancel_concentration',
]

# 首轮全量回测（未翻转）后 rank_ic_mean 为负的 13 个因子，已与用户确认做符号翻转
_ROUND1_FLIP_SIGN_FACTORS = {
    'p_sellcancel_to_activebuy', 'cancel_life_dispersion', 'refill_after_cancel_ratio',
    'deep_cancel_ratio', 'fill_amt_ratio', 'buy_impact_1m', 'active_net_ratio',
    'large_active_net_ratio', 'side_switch_rate', 'drift_after_buy_cancel',
    'order_imbalance_amt', 'cancel_life_median', 'tail_active_net_ratio',
}
# 第二轮：用户复核分组图后，发现以下 6 个因子头组表现与 IC 符号不一致
# （如 order_imbalance_amt 翻转后 rank_ic_mean 微弱为正但头组 head_excess_ir=-2.2，
# 实际最差），要求在第一轮基础上再次取反
# 第三轮：cancel_life_dispersion 恢复取反（从第二轮撤销列表中移除）
_ROUND2_RE_FLIP_FACTORS = {
    'aggressive_cancel_ratio', 'buy_impact_1m',
    'fill_amt_ratio', 'large_cancel_concentration', 'order_imbalance_amt',
}
FLIP_SIGN_FACTORS = _ROUND1_FLIP_SIGN_FACTORS ^ _ROUND2_RE_FLIP_FACTORS  # 对称差 = 再次取反


def build_forward_return_wide(open_prices):
    """构造 T+1开盘->T+2开盘 远期收益宽表，基于真实交易日历按行(日期)位移。

    修复说明：panel_trade.pkl 是稀疏面板——个股停牌期间整行缺失（不是 NaN 占位）。
    若用长表 groupby('stock_code').shift(-1/-2)，会按"该股票下一条有记录的行"取值，
    跨越停牌缺口时会把复牌跳空收益错误地分配到停牌前一天的样本上（已用数据验证，
    2025-11 一批停牌股复牌导致全市场当日均值偏移超1%）。
    正确做法：用 open_prices 宽表（index 为完整交易日历，每行代表一个真实交易日），
    按"行"（即按日期）位移，缺失的股票天然是 NaN，不会跨日期错位。

    个股远期收益（build_factor_df 用）与全A基准（build_all_a_benchmark_forward 用）
    均从这一份宽表派生，避免两处实现分叉导致口径不一致。

    参数
    ----
    open_prices : pd.DataFrame，index=date（完整交易日历），columns=stock_code，值=开盘价
        （直接复用 BacktestEvaluator.open_prices，避免重复构造宽表）
    """
    valid_open = open_prices.where(open_prices > 0)
    open_t1 = valid_open.shift(-1)
    open_t2 = valid_open.shift(-2)
    return (open_t2 / open_t1 - 1.0).where(open_t1.notna() & open_t2.notna())


def forward_return_wide_to_long(fwd_ret_wide):
    """宽表 -> 长表 [date, stock_code, fwd_return]，用于与因子面板 merge。"""
    long_df = fwd_ret_wide.stack(dropna=False).rename('fwd_return').reset_index()
    long_df.columns = ['date', 'stock_code', 'fwd_return']
    return long_df


def build_all_a_benchmark_close_to_close(close_prices):
    """真实全A等权收-收日收益：close_price.pct_change() 跨股票均值，仅 skipna 跳过 NaN。

    用于头组持仓回测基准（evaluator.benchmark_returns）：
    analyze_model_holdings 内部按自然日逐日模拟隔夜+日内收益，天然收-收口径，与此对齐。

    close_price<=0 属于物理无意义价格（如退市/异常状态残留的 0 值），
    视为缺失值以避免 pct_change 产生 inf；不做 status 过滤、不截尾。
    """
    valid_close = close_prices.where(close_prices > 0)
    daily_ret = valid_close.pct_change()
    benchmark = daily_ret.mean(axis=1, skipna=True).fillna(0.0).astype(np.float32)
    return benchmark


def build_all_a_benchmark_forward(fwd_ret_wide):
    """真实全A等权 T+1开盘->T+2开盘 远期收益，跨股票均值，仅 skipna 跳过 NaN。

    用于横截面分组回测基准（eng.avg_return）：与个股 factor_df['return']
    （同样是 T+1开盘->T+2开盘）严格对齐同一时间窗口，避免与收-收基准的
    1-2 天错位在行情剧烈波动时段（如 2025-04）产生虚假的全组同向超额。

    参数
    ----
    fwd_ret_wide : build_forward_return_wide() 的输出，与个股 return 出自同一份计算，
        确保基准与样本口径完全一致（不再各自实现一遍位移逻辑）。
    """
    benchmark = fwd_ret_wide.mean(axis=1, skipna=True).fillna(0.0).astype(np.float32)
    return benchmark


def build_factor_df(panel, factor_col, fwd_return, flip_sign=False):
    """构造单因子回测所需的 [date, stock_code, factor, return]。

    flip_sign=True 时对 factor 取相反数（用于首轮回测中 rank_ic_mean 为负的因子，
    使其头组（因子值最大组）与实际有效方向一致）。
    """
    sub = panel[['date', 'stock_code', factor_col]].rename(columns={factor_col: 'factor'})
    if flip_sign:
        sub['factor'] = -sub['factor']
    merged = sub.merge(fwd_return, on=['date', 'stock_code'], how='inner')
    merged = merged.rename(columns={'fwd_return': 'return'})
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['factor', 'return'])
    return merged.sort_values(['date', 'stock_code']).reset_index(drop=True)


def safe_get(d, key, default=np.nan):
    return d.get(key, default) if isinstance(d, dict) else default


def factor_output_dir(factor_col):
    return os.path.join(OUTPUT_DIR, factor_col)


def save_factor_metrics(factor_dir, row):
    metrics_path = os.path.join(factor_dir, 'metrics.json')
    payload = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('加载因子面板...', flush=True)
    panel = pd.read_pickle(FACTOR_PANEL_PATH)
    panel['date'] = pd.to_datetime(panel['date'])
    panel['stock_code'] = panel['stock_code'].astype(str)
    print(f'  因子面板: {len(panel)} 行, {panel["date"].nunique()} 天, {panel["stock_code"].nunique()} 只股票', flush=True)

    print('初始化 BacktestEvaluator (真实全A基准)...', flush=True)
    evaluator = BacktestEvaluator(
        panel_trade_path=PANEL_TRADE_PATH,
        avg_return_daily_path=None,
        n_groups=N_GROUPS,
        rebalance_days=REBALANCE_DAYS,
    )
    # forward return 宽表：基于真实交易日历按行位移，个股收益与全A基准共用同一份，
    # 避免长表 groupby.shift 跨停牌缺口错位（详见 build_forward_return_wide 说明）
    fwd_ret_wide = build_forward_return_wide(evaluator.open_prices)
    fwd_return = forward_return_wide_to_long(fwd_ret_wide)

    # 头组持仓基准：收-收口径，与 analyze_model_holdings 内部逐日模拟对齐
    close_to_close_benchmark = build_all_a_benchmark_close_to_close(evaluator.close_prices)
    evaluator.benchmark_returns = close_to_close_benchmark
    # 横截面分组基准：T+1开盘->T+2开盘远期口径，与个股 factor_df['return'] 对齐
    forward_benchmark = build_all_a_benchmark_forward(fwd_ret_wide)
    eng.avg_return = forward_benchmark.rename('avg_return').reset_index().rename(columns={'index': 'date'})
    print(f'  头组持仓基准(收-收): {len(close_to_close_benchmark)} 天, '
          f'横截面基准(远期开-开): {len(forward_benchmark)} 天', flush=True)

    summary_rows = []
    for i, factor_col in enumerate(FACTOR_COLS, 1):
        flip_sign = factor_col in FLIP_SIGN_FACTORS
        print(f'[{i}/{len(FACTOR_COLS)}] 回测因子: {factor_col}' + ('（已翻转符号）' if flip_sign else ''), flush=True)
        factor_df = build_factor_df(panel, factor_col, fwd_return, flip_sign=flip_sign)
        if len(factor_df) == 0:
            print(f'  [跳过] 无有效样本', flush=True)
            continue

        factor_dir = factor_output_dir(factor_col)
        os.makedirs(factor_dir, exist_ok=True)
        group_save_path = os.path.join(factor_dir, 'group.png')
        holdings_save_path = os.path.join(factor_dir, 'holdings.png')

        try:
            group_metrics = evaluator.evaluate_factor_df(factor_df, save_path=group_save_path, train_windows=0)
        except Exception as e:
            print(f'  [错误] 分组回测失败: {e}', flush=True)
            group_metrics = {}

        try:
            holdings_metrics = evaluator.evaluate_head_holdings(factor_df, save_path=holdings_save_path)
        except Exception as e:
            print(f'  [错误] 头组持仓回测失败: {e}', flush=True)
            holdings_metrics = {}

        row = {
            'factor_name': factor_col,
            'flip_sign': flip_sign,
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
        summary_rows.append(row)
        save_factor_metrics(factor_dir, row)
        print(f'  rank_ic_mean={row["rank_ic_mean"]:.4f}, '
              f'head_excess_ann={row["head_excess_return_annualized"]}, '
              f'holdings_excess_ann={row["head_holdings_excess_annual_return_pct"]}', flush=True)

        del factor_df
        gc.collect()

    summary_df = pd.DataFrame(summary_rows).sort_values('rank_ic_mean', ascending=False, na_position='last')
    summary_path = os.path.join(OUTPUT_DIR, 'tick_factor_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'\n汇总表已保存 -> {summary_path}', flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
