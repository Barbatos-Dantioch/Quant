# -*- coding: utf-8 -*-
"""
多空模型回测框架
同时对头组(多头)和尾组(空头)进行持仓回测，计算多空组合收益

输入说明:
    - head_holdings_data: DataFrame/Series, T日信号 → T+1日多头持仓股票列表
    - tail_holdings_data: DataFrame/Series, T日信号 → T+1日空头持仓股票列表
    - open_prices: DataFrame, 开盘价 (日期 × 股票)
    - close_prices: DataFrame, 收盘价 (日期 × 股票)
    - hedge_ratio_data: 可选, 每个日期按 pair 顺序给出的空头/多头对冲比例

多空逻辑:
    - 头组(多头): 买入头组股票, 收益 = 头组股票实际涨跌 - 换仓成本
    - 尾组(空头): 做空尾组股票, 调仓日卖出新加入尾组的股票(开空仓),
      买入被剔除尾组的股票(平空仓), 收益 = -(尾组股票实际涨跌) - 换仓成本
    - 多空组合: 默认头组收益 + 空头收益；提供对冲比例时按 pair 计算
      多头收益 - 对冲比例 × 空头收益，并对所有 pair 等权平均

使用示例:
    results = analyze_longshort_holdings(
        head_holdings_data=head_df,
        tail_holdings_data=tail_df,
        open_prices=open_df,
        close_prices=close_df,
        method='periodic_rebalance',
        holding_period=5
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _to_holdings_series(data):
    """将输入统一为 Series[list] 格式"""
    s = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    return s.apply(lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else [])


def _to_ratio_series(data):
    """将对冲比例输入统一为按日期索引的 Series。"""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        return data.apply(lambda row: row.tolist(), axis=1)
    return data


def _normalize_ratios(value, n_pairs):
    """把某日的对冲比例展开为与 pair 顺序对应的浮点列表。"""
    if n_pairs == 0:
        return []
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return [1.0] * n_pairs
    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = list(value)
    else:
        values = [value]
    if len(values) == 1 and n_pairs > 1:
        values *= n_pairs
    if len(values) != n_pairs:
        raise ValueError(
            f"hedge_ratio_data 的比例数量 ({len(values)}) 必须等于当日 pair 数量 ({n_pairs})"
        )
    ratios = [float(x) for x in values]
    if any(not np.isfinite(x) or x < 0 for x in ratios):
        raise ValueError("对冲比例必须是有限的非负数")
    return ratios


def _single_return(stock, price_from, price_to):
    """计算单只股票收益；价格无效时返回 None。"""
    if stock not in price_from.index or stock not in price_to.index:
        return None
    pf, pt = price_from[stock], price_to[stock]
    if not (pd.notna(pf) and pd.notna(pt) and pf > 0 and pt > 0):
        return None
    return (pt - pf) / pf


def _pair_segment_return(pairs, price_from, price_to):
    """某一时间段的 pair 收益平均值: long - ratio * short。"""
    values = []
    for long_stock, short_stock, ratio in pairs:
        long_ret = _single_return(long_stock, price_from, price_to)
        short_ret = _single_return(short_stock, price_from, price_to)
        if long_ret is not None and short_ret is not None:
            values.append(long_ret - ratio * short_ret)
    return np.mean(values) if values else 0.0


def _pair_full_return(pairs, prev_close, curr_open, curr_close):
    """未换仓时逐 pair 复合隔夜/日内收益后再取平均。"""
    values = []
    for long_stock, short_stock, ratio in pairs:
        long_overnight = _single_return(long_stock, prev_close, curr_open)
        short_overnight = _single_return(short_stock, prev_close, curr_open)
        long_intraday = _single_return(long_stock, curr_open, curr_close)
        short_intraday = _single_return(short_stock, curr_open, curr_close)
        if None in (long_overnight, short_overnight, long_intraday, short_intraday):
            continue
        long_ret = (1 + long_overnight) * (1 + long_intraday) - 1
        short_ret = (1 + short_overnight) * (1 + short_intraday) - 1
        values.append(long_ret - ratio * short_ret)
    return np.mean(values) if values else 0.0


def _build_pair_schedule(head_series, tail_series, ratio_series, dates,
                         open_p, close_p, method, period, status_data=None):
    """生成 (多头股票, 空头股票, 对冲比例) 的调仓计划。"""
    schedule = {}
    for i in range(len(dates) - 1):
        sig_d, trd_d = dates[i], dates[i + 1]
        if method != 'daily_rebalance' and i % period != 0:
            continue
        long_stocks = head_series.loc[sig_d]
        short_stocks = tail_series.loc[sig_d]
        if len(long_stocks) != len(short_stocks):
            raise ValueError(f"{sig_d} 的多头和空头数量不一致，无法按 pair 对齐")
        ratio_value = (ratio_series.loc[sig_d]
                       if ratio_series is not None and sig_d in ratio_series.index
                       else None)
        ratios = _normalize_ratios(ratio_value, len(long_stocks))
        op = open_p.loc[trd_d]
        cp = close_p.loc[trd_d]
        st = (status_data.loc[trd_d]
              if status_data is not None and trd_d in status_data.index
              else None)
        pairs = []
        for long_stock, short_stock, ratio in zip(long_stocks, short_stocks, ratios):
            if (long_stock not in op.index or long_stock not in cp.index
                    or short_stock not in op.index or short_stock not in cp.index):
                continue
            if not (pd.notna(op[long_stock]) and pd.notna(cp[long_stock])
                    and op[long_stock] > 0 and pd.notna(op[short_stock])
                    and pd.notna(cp[short_stock]) and op[short_stock] > 0):
                continue
            if st is not None and (st.get(long_stock, np.nan) != 0
                                   or st.get(short_stock, np.nan) != 0):
                continue
            pairs.append((long_stock, short_stock, ratio))
        if pairs:
            schedule[trd_d] = pairs
    return schedule


def _build_schedule(series, dates, open_p, close_p, method, period,
                    status_data=None):
    """
    生成调仓计划: {交易日 -> [有效股票]}
    T日信号 → T+1日持仓, 仅保留价格有效且 status==0 的股票
    """
    schedule = {}
    for i in range(len(dates) - 1):
        sig_d, trd_d = dates[i], dates[i + 1]
        if method != 'daily_rebalance' and i % period != 0:
            continue

        stocks = series.loc[sig_d]
        if not isinstance(stocks, list):
            stocks = [stocks] if pd.notna(stocks) else []

        op = open_p.loc[trd_d]
        cp = close_p.loc[trd_d]
        st = (status_data.loc[trd_d]
              if status_data is not None and trd_d in status_data.index
              else None)

        valid = []
        for stk in stocks:
            if stk not in op.index or stk not in cp.index:
                continue
            if not (pd.notna(op[stk]) and pd.notna(cp[stk]) and op[stk] > 0):
                continue
            if st is not None and st.get(stk, np.nan) != 0:
                continue
            valid.append(stk)

        if valid:
            schedule[trd_d] = valid
    return schedule


def _equal_weight_return(stocks, price_from, price_to):
    """等权平均收益: mean((to - from) / from), 空列表返回 0"""
    if not stocks:
        return 0.0
    rets = []
    for stk in stocks:
        if stk in price_from.index and stk in price_to.index:
            pf, pt = price_from[stk], price_to[stk]
            if pd.notna(pf) and pd.notna(pt) and pf > 0:
                rets.append((pt - pf) / pf)
    return np.mean(rets) if rets else 0.0


def _turnover_cost(old, new, rate):
    """换手成本 = 换手比例 × 双边费率"""
    os, ns = set(old), set(new)
    if not os and not ns:
        return 0.0
    total = max(len(os), len(ns), 1)
    return len(os.symmetric_difference(ns)) / (2 * total) * 2 * rate


def _calc_stats(nav_s, name):
    """计算统计指标, 返回 (stats_dict, returns_series)"""
    ret_s = nav_s.pct_change().dropna()
    total = (nav_s.iloc[-1] / nav_s.iloc[0] - 1) * 100
    yrs = len(nav_s) / 250
    ann_r = (((nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1) * 100
             if yrs > 0 else 0)
    ann_v = ret_s.std() * np.sqrt(250) * 100
    sr = (ann_r - 3) / ann_v if ann_v > 0 else 0
    dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100
    wr = ((ret_s > 0).sum() / len(ret_s) * 100) if len(ret_s) > 0 else 0
    return {
        'name': name,
        'total_return': round(total, 2),
        'annual_return': round(ann_r, 2),
        'annual_volatility': round(ann_v, 2),
        'sharpe_ratio': round(sr, 3),
        'max_drawdown': round(dd, 2),
        'win_rate': round(wr, 1),
    }, ret_s


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def analyze_longshort_holdings(head_holdings_data, tail_holdings_data,
                               open_prices, close_prices,
                               benchmark_data=None, status_data=None,
                               method='daily_rebalance', holding_period=1,
                               commission_rate=0.0007, verbose=True,
                               hedge_ratio_data=None, borrow_rate=0.03):
    """
    多空模型持仓回测

    参数:
        head_holdings_data: 头组持仓信号 (T日信号→T+1日持仓)
                            DataFrame 或 Series, 元素为股票代码列表
        tail_holdings_data: 尾组持仓信号 (T日信号→T+1日持仓)
                            DataFrame 或 Series, 元素为股票代码列表
        open_prices:  开盘价 DataFrame (日期 × 股票)
        close_prices: 收盘价 DataFrame (日期 × 股票)
        benchmark_data: Series, 可选, 基准日收益率
        status_data:  DataFrame, 可选, 股票状态矩阵 (买入/开仓需 status==0)
        method:       'daily_rebalance' | 'periodic_rebalance'
        holding_period: 调仓周期天数, 仅 periodic_rebalance 时有效
        commission_rate: 单边手续费率, 默认 0.0007
        verbose: 是否打印进度/绘图
        hedge_ratio_data: 每个日期的 pair 对冲比例，按 head/tail 列表顺序排列；默认 1:1
        borrow_rate: 年化融券费率，空头比例为 1 时默认 3%

    返回:
        dict:
            head_nav        - 头组净值 Series
            tail_nav        - 尾组实际净值 Series (跟踪尾组股票真实涨跌, 不含做空)
            longshort_nav   - 多空组合净值 Series
            benchmark_nav   - 基准净值 Series
            head_excess_nav - 头组超额净值 Series
            head_returns    - 头组日收益率 Series
            longshort_returns - 多空组合日收益率 Series
            statistics      - 各维度统计指标 dict
            head_schedule   - 头组调仓计划 dict
            tail_schedule   - 尾组调仓计划 dict
            nav_df          - 所有净值汇总 DataFrame

    收益逻辑:
        1. T日的持仓信号决定T+1日的实际持仓
        2. 隔夜收益使用调仓前的持仓计算
        3. 日内收益使用调仓后的持仓计算
        4. 换手成本在调仓日扣除
        5. 多空收益按 pair 等权平均；空头融券费按对冲比例折算
    """

    # ── 数据预处理 ──
    head_s = _to_holdings_series(head_holdings_data)
    tail_s = _to_holdings_series(tail_holdings_data)
    ratio_s = _to_ratio_series(hedge_ratio_data)

    common_dates = sorted(
        set(head_s.index) & set(tail_s.index)
        & set(open_prices.index) & set(close_prices.index)
    )
    head_s = head_s.loc[common_dates]
    tail_s = tail_s.loc[common_dates]
    if ratio_s is not None:
        ratio_s = ratio_s.loc[ratio_s.index.intersection(common_dates)]
    open_prices = open_prices.loc[common_dates]
    close_prices = close_prices.loc[common_dates]

    if verbose:
        print(f"数据对齐：{len(common_dates)} 个交易日")

    # ── 生成调仓计划 ──
    if verbose:
        print("生成头组交易信号...")
    head_sch = _build_schedule(head_s, common_dates, open_prices, close_prices,
                               method, holding_period, status_data)
    if verbose:
        print("生成尾组交易信号...")
    tail_sch = _build_schedule(tail_s, common_dates, open_prices, close_prices,
                               method, holding_period, status_data)
    pair_mode = hedge_ratio_data is not None
    pair_sch = (_build_pair_schedule(head_s, tail_s, ratio_s, common_dates,
                                     open_prices, close_prices, method,
                                     holding_period, status_data)
                if pair_mode else {})
    if verbose:
        print(f"头组 {len(head_sch)} 个调仓日，尾组 {len(tail_sch)} 个调仓日")

    # ── 回测主循环 ──
    h_nav = t_nav = t_raw_nav = ls_nav = bm_nav = 1.0
    h_curr, t_curr = [], []       # 当前持仓
    p_curr = []                   # 当前 pair: (多头, 空头, 对冲比例)
    records = []

    it = (tqdm(enumerate(common_dates), total=len(common_dates), desc="回测进度")
          if verbose else enumerate(common_dates))

    for i, date in it:
        if i == 0:
            records.append({'date': date, 'head': 1.0, 'tail': 1.0,
                            'tail_raw': 1.0,
                            'longshort': 1.0, 'benchmark': 1.0})
            continue

        yday = common_dates[i - 1]
        prev_close_row = close_prices.loc[yday]
        curr_open_row = open_prices.loc[date]
        curr_close_row = close_prices.loc[date]

        # ── 头组(多头) ──
        hp = head_sch.get(date)
        h_rebal = hp is not None and set(hp) != set(h_curr)
        # 隔夜用旧持仓, 日内用新持仓(若有调仓)
        h_overnight_h = h_curr
        h_intraday_h = hp if hp else h_curr

        h_overnight = _equal_weight_return(h_overnight_h, prev_close_row,
                                           curr_open_row)
        h_cost = (_turnover_cost(h_curr, hp, commission_rate)
                  if h_rebal and hp else 0.0)
        if h_rebal and hp:
            h_curr = hp
        h_intraday = _equal_weight_return(h_intraday_h, curr_open_row,
                                          curr_close_row)
        h_ret = ((1 + h_overnight) * (1 + h_intraday) - 1) - h_cost

        # ── 尾组(空头) ──
        tp = tail_sch.get(date)
        t_rebal = tp is not None and set(tp) != set(t_curr)
        t_overnight_h = t_curr
        t_intraday_h = tp if tp else t_curr

        t_overnight = _equal_weight_return(t_overnight_h, prev_close_row,
                                           curr_open_row)
        t_cost = (_turnover_cost(t_curr, tp, commission_rate)
                  if t_rebal and tp else 0.0)
        if t_rebal and tp:
            t_curr = tp
        t_intraday = _equal_weight_return(t_intraday_h, curr_open_row,
                                          curr_close_row)

        t_raw = (1 + t_overnight) * (1 + t_intraday) - 1

        # ── 多空组合：pair 级收益平均，并按空头比例扣融券费 ──
        if pair_mode:
            pp = pair_sch.get(date)
            overnight_pairs = p_curr
            intraday_pairs = pp if pp is not None else p_curr
            pair_rebal = pp is not None and pp != p_curr
            if overnight_pairs and not pair_rebal:
                gross_pair_ret = _pair_full_return(
                    overnight_pairs, prev_close_row, curr_open_row, curr_close_row)
            else:
                pair_overnight = _pair_segment_return(
                    overnight_pairs, prev_close_row, curr_open_row)
                pair_intraday = _pair_segment_return(
                    intraday_pairs, curr_open_row, curr_close_row)
                gross_pair_ret = (1 + pair_overnight) * (1 + pair_intraday) - 1
            if pp is not None:
                p_curr = pp
            active_pairs = intraday_pairs
            avg_short_ratio = (np.mean([x[2] for x in active_pairs])
                               if active_pairs else 0.0)
            borrow_cost = avg_short_ratio * borrow_rate / 252.0
            short_ret = -t_raw - t_cost - borrow_cost
            ls_ret = gross_pair_ret - h_cost - t_cost - borrow_cost
        else:
            active_short = tp if tp is not None else t_curr
            avg_short_ratio = 1.0 if active_short else 0.0
            borrow_cost = avg_short_ratio * borrow_rate / 252.0
            short_ret = -t_raw - t_cost - borrow_cost
            ls_ret = h_ret + short_ret

        # ── 基准收益 ──
        bm_ret = 0.0
        if benchmark_data is not None:
            if date in benchmark_data.index:
                bm_ret = benchmark_data.loc[date]
                if abs(bm_ret) > 0.5:
                    bm_ret /= 100.0
        else:
            try:
                mask = ((prev_close_row > 0) & pd.notna(prev_close_row)
                        & pd.notna(curr_close_row))
                if mask.any():
                    bm_ret = ((curr_close_row[mask] - prev_close_row[mask])
                              / prev_close_row[mask]).mean()
            except Exception:
                pass

        # 更新净值
        h_nav *= (1 + h_ret)
        t_nav *= (1 + short_ret)   # 尾组做空收益净值
        t_raw_nav *= (1 + t_raw)   # 尾组股票实际涨跌净值（供图表使用）
        ls_nav *= (1 + ls_ret)
        bm_nav *= (1 + bm_ret)

        records.append({'date': date, 'head': h_nav, 'tail': t_nav,
                        'tail_raw': t_raw_nav,
                        'longshort': ls_nav, 'benchmark': bm_nav})

    # ── 结果汇总 ──
    nav_df = pd.DataFrame(records).set_index('date')

    stats = {}
    stats['head'], h_rets = _calc_stats(nav_df['head'], '头组(多头)')
    stats['tail'], t_rets = _calc_stats(nav_df['tail'], '尾组(做空)')
    stats['tail_raw'], _ = _calc_stats(nav_df['tail_raw'], '尾组(实际)')
    stats['longshort'], ls_rets = _calc_stats(nav_df['longshort'], '多空组合')
    stats['benchmark'], bm_rets = _calc_stats(nav_df['benchmark'], '基准')

    head_excess_nav = nav_df['head'] / nav_df['benchmark']
    stats['head_excess'], _ = _calc_stats(head_excess_nav, '头组超额')
    tail_excess_nav = nav_df['tail'] / nav_df['benchmark']
    stats['tail_excess'], _ = _calc_stats(tail_excess_nav, '尾组做空超额')
    ls_excess_nav = nav_df['longshort'] / nav_df['benchmark']
    stats['ls_excess'], _ = _calc_stats(ls_excess_nav, '多空超额')

    # ── 绘图 ──
    if verbose:
        fig = plt.figure(figsize=(18, 10))

        # 1. 头组 / 尾组 / 基准 净值
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(nav_df.index, nav_df['head'], lw=2, color='red',
                 label='头组(多头)')
        ax1.plot(nav_df.index, nav_df['tail'], lw=2, color='green',
                 label='尾组(实际)')
        ax1.plot(nav_df.index, nav_df['benchmark'], lw=2, color='gray',
                 alpha=0.7, label='基准')
        ax1.set_title('头组 / 尾组 / 基准 净值', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 多空组合 & 头组超额
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(nav_df.index, nav_df['longshort'], lw=2, color='blue',
                 label='多空组合')
        ax2.plot(head_excess_nav.index, head_excess_nav, lw=1.5,
                 color='orange', alpha=0.8, label='头组超额')
        ax2.axhline(y=1, color='gray', ls='--', alpha=0.5)
        ax2.set_title('多空组合 & 头组超额', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 多空组合回撤
        ax3 = plt.subplot(2, 2, 3)
        ls_cummax = nav_df['longshort'].cummax()
        ls_dd = (nav_df['longshort'] - ls_cummax) / ls_cummax * 100
        ax3.fill_between(ls_dd.index, ls_dd, 0, alpha=0.3, color='red')
        ax3.plot(ls_dd.index, ls_dd, 'r-', lw=1)
        ax3.set_title('多空组合回撤', fontsize=13)
        ax3.set_ylabel('回撤 (%)')
        ax3.grid(True, alpha=0.3)

        # 4. 统计表
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        headers = ['指标', '头组', '尾组', '多空', '基准', '头组超额']
        ms = ['annual_return', 'annual_volatility', 'sharpe_ratio',
              'max_drawdown', 'win_rate']
        mns = ['年化收益(%)', '年化波动(%)', '夏普比率',
               '最大回撤(%)', '胜率(%)']
        tdata = [
            [mn,
             f"{stats['head'][m]:.2f}",
             f"{stats['tail'][m]:.2f}",
             f"{stats['longshort'][m]:.2f}",
             f"{stats['benchmark'][m]:.2f}",
             f"{stats['head_excess'][m]:.2f}"]
            for m, mn in zip(ms, mns)
        ]
        tbl = ax4.table(cellText=tdata, colLabels=headers,
                        cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.3, 1.5)

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 70)
        print(f"多空模型回测结果（{method}，持仓周期 {holding_period}）")
        print("=" * 70)
        sd = pd.DataFrame([stats['head'], stats['tail'], stats['longshort'],
                           stats['benchmark'], stats['head_excess']])
        sd.set_index('name', inplace=True)
        print(sd)

    return {
        'head_nav': nav_df['head'],
        'tail_nav': nav_df['tail'],
        'longshort_nav': nav_df['longshort'],
        'benchmark_nav': nav_df['benchmark'],
        'head_excess_nav': head_excess_nav,
        'head_returns': h_rets,
        'longshort_returns': ls_rets,
        'statistics': stats,
        'head_schedule': head_sch,
        'tail_schedule': tail_sch,
        'nav_df': nav_df,
    }


if __name__ == '__main__':
    print("多空模型回测框架已加载")
    print("使用方法:")
    print("  from model_backtest_framework_longshort import analyze_longshort_holdings")
    print("  results = analyze_longshort_holdings(head_df, tail_df, open_prices, close_prices)")
