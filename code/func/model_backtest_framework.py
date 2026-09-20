# -*- coding: utf-8 -*-
"""
模型回测框架
用于对模型输出的拟持仓股票进行回测分析

输入说明:
    - holdings_data: DataFrame, index为日期, 元素为股票代码列表
                    表示T日的持仓信号对应T+1日应该持有的股票
    - open_prices: DataFrame, index为日期, columns为股票代码, 值为开盘价
    - close_prices: DataFrame, index为日期, columns为股票代码, 值为收盘价
    
使用示例:
    results = analyze_model_holdings(
        holdings_data=holdings_df,
        open_prices=open_df,
        close_prices=close_df,
        method='periodic_rebalance',
        holding_period=5,
        commission_rate=0.0002
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def analyze_model_holdings(holdings_data, open_prices, close_prices, benchmark_data=None,
                          status_data=None,
                          method='daily_rebalance', holding_period=1, 
                          commission_rate=0.0007, verbose=True):
    """
    模型持仓回测分析
    
    参数:
        holdings_data: DataFrame 或 Series, T日的股票名称列表表示T+1日应持有的股票
                      - DataFrame: index为日期, 某一列为股票列表
                      - Series: index为日期, 值为股票列表
        open_prices: DataFrame, 开盘价 (日期 × 股票)
        close_prices: DataFrame, 收盘价 (日期 × 股票)
        benchmark_data: Series, 可选, 基准指数的日收益率 (index为日期, 值为float收益率)
                        如果为None, 则使用所有可交易股票的等权平均作为基准
        status_data: DataFrame, 可选, 股票状态矩阵 (日期 × 股票)
                    如果提供，则仅在买入日使用 `status == 0` 过滤；卖出阶段不检查 status
        method: str, 调仓方式
            - 'daily_rebalance': 每日调仓
            - 'periodic_rebalance': 周期调仓
        holding_period: int, 持仓周期(天), 仅在periodic_rebalance时有效
        commission_rate: float, 单边手续费率, 默认0.0007
        verbose: bool, 是否打印详细信息
    
    返回:
        dict, 包含以下键:
            - 'nav': Series, 净值曲线
            - 'returns': Series, 收益率序列
            - 'statistics': dict, 统计指标
            - 'holdings_schedule': dict, 持仓计划
            - 'benchmark_nav': Series, 基准净值(等权所有股票)
    
    逻辑说明:
        1. T日的持仓信号决定T+1日的实际持仓
        2. 隔夜收益使用调仓前的持仓计算
        3. 日内收益使用调仓后的持仓计算
        4. 换手成本在调仓日扣除
    """
    
    # 处理输入数据
    if isinstance(holdings_data, pd.DataFrame):
        # 如果是DataFrame,取第一列
        holdings_series = holdings_data.iloc[:, 0]
    else:
        holdings_series = holdings_data
    
    # 确保holdings_series中的元素是列表
    holdings_series = holdings_series.apply(lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else [])
    
    # 数据对齐
    common_dates = sorted(set(holdings_series.index) & 
                         set(open_prices.index) & 
                         set(close_prices.index))
    
    holdings_series = holdings_series.loc[common_dates]
    open_prices = open_prices.loc[common_dates]
    close_prices = close_prices.loc[common_dates]
    
    if verbose:
        print(f"数据对齐：{len(common_dates)}个交易日")
    
    # 初始化
    nav = 1.0
    benchmark_nav = 1.0
    
    nav_records = []
    
    # 生成所有交易信号（T日信号决定T+1日持仓）
    if verbose:
        print("生成交易信号...")
    holdings_schedule = {}  # date -> holdings (股票列表)
    
    for i in range(len(common_dates) - 1):
        signal_date = common_dates[i]
        trade_date = common_dates[i + 1]
        
        if method == 'daily_rebalance' or i % holding_period == 0:
            # 获取T日的持仓信号
            target_stocks = holdings_series.loc[signal_date]
            
            if not isinstance(target_stocks, list):
                target_stocks = [target_stocks] if pd.notna(target_stocks) else []
            
            # 检查trade_date的可交易性
            next_open = open_prices.loc[trade_date]
            next_close = close_prices.loc[trade_date]
            status_row = None
            if status_data is not None and trade_date in status_data.index:
                status_row = status_data.loc[trade_date]
            
            # 过滤出在价格数据中存在且有效的股票
            valid_stocks = []
            for stock in target_stocks:
                if stock in next_open.index and stock in next_close.index:
                    buy_status_ok = True
                    if status_row is not None:
                        buy_status_ok = status_row.get(stock, np.nan) == 0
                    # status 规则只在买入日生效：买入必须 status == 0；后续持有/卖出不再检查 status。
                    if pd.notna(next_open[stock]) and pd.notna(next_close[stock]) and next_open[stock] > 0 and buy_status_ok:
                        valid_stocks.append(stock)
            
            if len(valid_stocks) > 0:
                holdings_schedule[trade_date] = valid_stocks
    
    if verbose:
        print(f"生成了{len(holdings_schedule)}个交易计划")
    
    # 持仓状态跟踪
    current_holdings = []  # 当前实际持仓
    
    # 回测主循环
    iterator = tqdm(enumerate(common_dates), total=len(common_dates), desc="回测进度") if verbose else enumerate(common_dates)
    
    for i, date in iterator:
        if i == 0:
            # 第一天
            nav_records.append({
                'date': date,
                'nav': 1.0,
                'benchmark': 1.0
            })
            continue
        
        yesterday = common_dates[i-1]
        
        # 检查今天是否有新的交易计划
        new_plan = holdings_schedule.get(date, None)
        
        # 判断是否需要调仓
        need_rebalance = (new_plan is not None) and (set(new_plan) != set(current_holdings))
        
        # 用于计算收益的持仓
        holdings_for_overnight = current_holdings  # 隔夜收益用当前持仓
        holdings_for_intraday = new_plan if new_plan else current_holdings  # 日内收益用新持仓（如果有）
        
        # 初始化收益
        daily_return = 0.0
        benchmark_return = 0.0
        
        # 1. 计算隔夜收益（使用当前持仓）
        if holdings_for_overnight:
            rets = []
            # 预先获取当日和昨日的价格行，避免循环中重复索引
            try:
                prev_close_row = close_prices.loc[yesterday]
                curr_open_row = open_prices.loc[date]
                
                for stock in holdings_for_overnight:
                    if stock in prev_close_row.index: # 确保股票在价格数据中
                        prev_close = prev_close_row[stock]
                        curr_open = curr_open_row[stock]
                        if pd.notna(prev_close) and pd.notna(curr_open) and prev_close > 0:
                            ret = (curr_open - prev_close) / prev_close
                            rets.append(ret)
            except:
                pass
            
            if rets:
                overnight_return = np.mean(rets)
                daily_return += overnight_return
        
        # 2. 计算换手成本（如果调仓）
        if need_rebalance and new_plan:
            old_stocks = set(current_holdings)
            new_stocks = set(new_plan)
            
            if old_stocks or new_stocks:
                # 换手率计算
                total_stocks = max(len(old_stocks), len(new_stocks), 1)
                turnover = len(old_stocks.symmetric_difference(new_stocks)) / (2 * total_stocks)
                
                # 双边成本
                cost = turnover * 2 * commission_rate
                daily_return -= cost
            
            # 只有在真正调仓时才更新current_holdings
            if new_plan:
                current_holdings = new_plan
        
        # 3. 计算日内收益（使用今日持仓）
        if holdings_for_intraday:
            rets = []
            # 预先获取当日价格行
            try:
                curr_open_row = open_prices.loc[date]
                curr_close_row = close_prices.loc[date]
                
                for stock in holdings_for_intraday:
                    if stock in curr_open_row.index:
                        curr_open = curr_open_row[stock]
                        curr_close = curr_close_row[stock]
                        if pd.notna(curr_open) and pd.notna(curr_close) and curr_open > 0:
                            ret = (curr_close - curr_open) / curr_open
                            rets.append(ret)
            except:
                pass
            
            if rets:
                intraday_return = np.mean(rets)
                daily_return += intraday_return
        
        # 4. 计算基准收益
        if benchmark_data is not None:
            # 使用传入的基准数据 (如中证1000)
            if date in benchmark_data.index:
                benchmark_return = benchmark_data.loc[date]
                # 如果是百分数形式 (如 1.5 代表 1.5%), 则需要除以100
                # 这里假设传入的是小数形式 (如 0.015 代表 1.5%)
                # 检查数据量级，如果大部分值 > 1，可能是百分数
                if abs(benchmark_return) > 0.5: # 简单启发式判断
                     benchmark_return /= 100.0
            else:
                benchmark_return = 0.0
        else:
            # 默认基准：所有可交易股票的等权组合 - 向量化优化
            try:
                # 获取当日和昨日的价格向量
                curr_close_all = close_prices.loc[date]
                prev_close_all = close_prices.loc[yesterday]
                
                # 计算个股收益率: (今日收盘 - 昨日收盘) / 昨日收盘
                # 过滤掉昨日收盘价为0或NaN的情况
                valid_mask = (prev_close_all > 0) & pd.notna(prev_close_all) & pd.notna(curr_close_all)
                
                if valid_mask.any():
                    stock_rets = (curr_close_all[valid_mask] - prev_close_all[valid_mask]) / prev_close_all[valid_mask]
                    benchmark_return = stock_rets.mean()
                else:
                    benchmark_return = 0.0
            except Exception as e:
                # print(f"基准计算错误 {date}: {e}")
                benchmark_return = 0.0
        
        # 更新净值
        nav *= (1 + daily_return)
        benchmark_nav *= (1 + benchmark_return)
        
        # 记录
        nav_records.append({
            'date': date,
            'nav': nav,
            'benchmark': benchmark_nav
        })
    
    # 转换为DataFrame
    nav_df = pd.DataFrame(nav_records)
    nav_df.set_index('date', inplace=True)
    
    # 计算统计
    returns_series = nav_df['nav'].pct_change().dropna()
    benchmark_returns = nav_df['benchmark'].pct_change().dropna()
    
    def calculate_statistics(nav_series, returns_series, name):
        """计算统计指标"""
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
        years = len(nav_series) / 250
        annual_return = ((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        annual_vol = returns_series.std() * np.sqrt(250) * 100
        sharpe = (annual_return - 3) / annual_vol if annual_vol > 0 else 0
        
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        win_rate = (returns_series > 0).sum() / len(returns_series) * 100 if len(returns_series) > 0 else 0
        
        return {
            'name': name,
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'annual_volatility': round(annual_vol, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 1)
        }
    
    statistics = {}
    statistics['strategy'] = calculate_statistics(nav_df['nav'], returns_series, '策略组合')
    statistics['benchmark'] = calculate_statistics(nav_df['benchmark'], benchmark_returns, '基准组合')
    
    # 计算超额收益
    excess_nav = nav_df['nav'] / nav_df['benchmark']
    excess_returns = excess_nav.pct_change().dropna()
    statistics['excess'] = calculate_statistics(excess_nav, excess_returns, '超额收益')
    
    # 绘图
    if verbose:
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 净值曲线对比
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(nav_df.index, nav_df['nav'], label='策略组合', linewidth=2, color='blue')
        ax1.plot(nav_df.index, nav_df['benchmark'], label='基准组合', linewidth=2, color='gray', alpha=0.7)
        ax1.set_title('净值曲线对比', fontsize=14)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('净值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 超额收益曲线
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(excess_nav.index, excess_nav, 'g-', linewidth=2, label='超额收益')
        ax2.set_title('超额收益曲线', fontsize=14)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('超额净值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # 3. 回撤曲线
        ax3 = plt.subplot(2, 2, 3)
        cummax = nav_df['nav'].cummax()
        drawdown = (nav_df['nav'] - cummax) / cummax * 100
        ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(drawdown.index, drawdown, 'r-', linewidth=1)
        ax3.set_title('回撤曲线', fontsize=14)
        ax3.set_xlabel('日期')
        ax3.set_ylabel('回撤 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计表格
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        headers = ['指标', '策略组合', '基准组合', '超额']
        
        metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        metric_names = ['年化收益(%)', '年化波动(%)', '夏普比率', '最大回撤(%)', '胜率(%)']
        
        for metric, name in zip(metrics, metric_names):
            row = [
                name,
                f"{statistics['strategy'][metric]:.2f}",
                f"{statistics['benchmark'][metric]:.2f}",
                f"{statistics['excess'][metric]:.2f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        plt.show()
        
        # 输出结果
        print("\n" + "="*60)
        print(f"模型持仓回测结果（{method}）")
        print("="*60)
        stats_df = pd.DataFrame([statistics['strategy'], statistics['benchmark'], statistics['excess']])
        stats_df.set_index('name', inplace=True)
        print(stats_df)
    
    return {
        'nav': nav_df['nav'],
        'returns': returns_series,
        'statistics': statistics,
        'holdings_schedule': holdings_schedule,
        'benchmark_nav': nav_df['benchmark'],
        'excess_nav': excess_nav
    }


if __name__ == '__main__':
    # 使用示例
    print("模型回测框架已加载")
    print("使用方法:")
    print("  from model_backtest_framework import analyze_model_holdings")
    print("  results = analyze_model_holdings(holdings_data, open_prices, close_prices)")

