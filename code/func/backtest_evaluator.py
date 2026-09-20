#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测评估模块。

提供 BacktestEvaluator 类，统一封装"因子分组回测"和"头组持仓回测"两套评估流程。
各实验脚本创建实例后，通过 evaluate_factor_df / evaluate_head_holdings 两个方法
获取标准化的 metrics dict，不再需要各自维护回测数据加载和指标计算逻辑。

外部依赖：
    - experiment_engine  : 因子分组回测（eng.backtest）
    - model_backtest_framework : 头组持仓净值模拟（holdings_bt.analyze_model_holdings）
"""

from __future__ import annotations

import gc
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Linux 环境通常无 SimHei，按可用字体自动选择 CJK 字体，避免 holdings 图中文乱码
_CJK_FONT_CANDIDATES = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
_AVAILABLE_FONTS = {f.name for f in _fm.fontManager.ttflist}
_CJK_FONT = next((f for f in _CJK_FONT_CANDIDATES if f in _AVAILABLE_FONTS), None)
if _CJK_FONT:
    plt.rcParams['font.sans-serif'] = [_CJK_FONT] + plt.rcParams.get('font.sans-serif', [])
plt.rcParams['axes.unicode_minus'] = False

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import experiment_engine as eng
import model_backtest_framework as holdings_bt


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = -999.0) -> float:
    """把 None / NaN / inf 全部收敛到可比较的标量。"""
    if value is None:
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def empty_head_holdings_metrics() -> Dict[str, Any]:
    """头组持仓回测失败或无信号时的空指标模板。"""
    return {
        "head_holdings_signal_count": 0,
        "head_holdings_strategy_total_return_pct": np.nan,
        "head_holdings_strategy_annual_return_pct": np.nan,
        "head_holdings_strategy_annual_volatility_pct": np.nan,
        "head_holdings_strategy_sharpe_ratio": np.nan,
        "head_holdings_strategy_max_drawdown_pct": np.nan,
        "head_holdings_strategy_win_rate_pct": np.nan,
        "head_holdings_excess_total_return_pct": np.nan,
        "head_holdings_excess_annual_return_pct": np.nan,
        "head_holdings_excess_annual_volatility_pct": np.nan,
        "head_holdings_excess_sharpe_ratio": np.nan,
        "head_holdings_excess_max_drawdown_pct": np.nan,
        "head_holdings_excess_win_rate_pct": np.nan,
        "head_holdings_excess_return_by_year_pct": {},
    }


def normalize_metrics(raw_metrics: Dict[str, Any], train_windows: int) -> Dict[str, Any]:
    """把 experiment_engine.backtest 返回的字段名收敛到统一口径。"""
    return {
        "rank_ic_mean": _safe_float(raw_metrics.get("rank_ic_mean")),
        "rank_ic_ir": _safe_float(raw_metrics.get("rank_ic_ir")),
        "top_group_excess_return": _safe_float(raw_metrics.get("top_group_excess_return"), np.nan),
        "head_excess_return_annualized": _safe_float(raw_metrics.get("head_excess_return_annualized"), np.nan),
        "head_excess_volatility_annualized": _safe_float(raw_metrics.get("head_excess_volatility_annualized"), np.nan),
        "head_excess_ir": _safe_float(raw_metrics.get("head_excess_ir"), np.nan),
        "head_excess_max_drawdown": _safe_float(raw_metrics.get("head_excess_max_drawdown"), np.nan),
        "head_cum_win_rate": _safe_float(raw_metrics.get("head_cum_excess_win_rate"), np.nan),
        "head_cum_advantage": _safe_float(raw_metrics.get("head_cum_excess_advantage"), np.nan),
        "head_turnover_mean": _safe_float(raw_metrics.get("head_turnover_mean"), np.nan),
        "head_turnover_std": _safe_float(raw_metrics.get("head_turnover_std"), np.nan),
        "head_turnover_last": _safe_float(raw_metrics.get("head_turnover_last"), np.nan),
        "head_turnover_count": int(raw_metrics.get("head_turnover_count", 0)),
        "train_windows": int(train_windows),
    }


# ---------------------------------------------------------------------------
# BacktestEvaluator
# ---------------------------------------------------------------------------

class BacktestEvaluator:
    """
    统一回测评估器。

    Parameters
    ----------
    panel_trade_path : str
        panel_trade.pkl 的路径，用于加载开/收盘价和 status。
    avg_return_daily_path : Optional[str]
        avg_return_daily.pkl 的路径，用于构造日频基准收益。
        传 None 时跳过该资源加载，`self.benchmark_returns` 保持为 None，
        由调用方自行构造基准后赋值覆盖。
    n_groups : int
        因子分组回测的分组数。
    rebalance_days : int
        头组持仓回测的调仓间隔天数。
    commission_rate : float
        头组持仓回测的单边手续费率。
    raw_panel : pd.DataFrame, optional
        当 panel_trade_path 不存在时的回退数据源（例如 eng.panel）。
    """

    def __init__(
        self,
        panel_trade_path: str,
        avg_return_daily_path: Optional[str],
        n_groups: int = 10,
        rebalance_days: int = 5,
        commission_rate: float = 0.0007,
        raw_panel: Optional[pd.DataFrame] = None,
    ):
        self.n_groups = n_groups
        self.rebalance_days = rebalance_days
        self.commission_rate = commission_rate

        self.open_prices: Optional[pd.DataFrame] = None
        self.close_prices: Optional[pd.DataFrame] = None
        self.status: Optional[pd.DataFrame] = None
        self.benchmark_returns: Optional[pd.Series] = None

        self._load_resources(panel_trade_path, avg_return_daily_path, raw_panel)

    # ------------------------------------------------------------------
    # 资源加载
    # ------------------------------------------------------------------

    def _load_resources(
        self,
        panel_trade_path: str,
        avg_return_daily_path: Optional[str],
        raw_panel: Optional[pd.DataFrame],
    ) -> None:
        if os.path.exists(panel_trade_path):
            src = pd.read_pickle(panel_trade_path)
            price_panel = src[["date", "stock_code", "open_price"]].copy()
            close_panel = src[["date", "stock_code", "close_price"]].copy()
            status_panel = src[["date", "stock_code", "status"]].copy()
            del src
            gc.collect()
        elif raw_panel is not None:
            price_panel = raw_panel[["date", "stock_code", "open_price"]].copy()
            close_panel = raw_panel[["date", "stock_code", "close_price"]].copy()
            status_panel = raw_panel[["date", "stock_code", "status"]].copy()
        else:
            raise FileNotFoundError(
                f"panel_trade_path 不存在且未提供 raw_panel 回退: {panel_trade_path}"
            )

        # 清洗 + 去重 + pivot 为宽表
        for df in (price_panel, close_panel, status_panel):
            df.dropna(subset=["date", "stock_code"], inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df["stock_code"] = df["stock_code"].astype(str)
            df.sort_values(["date", "stock_code"], inplace=True)
            df.drop_duplicates(["date", "stock_code"], keep="last", inplace=True)

        self.open_prices = (
            price_panel.pivot(index="date", columns="stock_code", values="open_price")
            .sort_index().astype(np.float32)
        )
        self.close_prices = (
            close_panel.pivot(index="date", columns="stock_code", values="close_price")
            .sort_index().astype(np.float32)
        )
        self.status = (
            status_panel.pivot(index="date", columns="stock_code", values="status")
            .sort_index()
        )
        del price_panel, close_panel, status_panel
        gc.collect()

        # 基准收益：avg_return_daily_path 为 None 时跳过，留给调用方自行赋值 self.benchmark_returns
        if avg_return_daily_path is None:
            self.benchmark_returns = None
            return

        avg_return_daily = pd.read_pickle(avg_return_daily_path)
        if not isinstance(avg_return_daily, pd.DataFrame):
            raise TypeError("avg_return_daily.pkl 类型不支持，无法构造持仓回测基准。")

        benchmark_df = avg_return_daily.copy()
        if "date" not in benchmark_df.columns:
            benchmark_df = benchmark_df.reset_index()
        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
        benchmark_df = benchmark_df.sort_values("date").drop_duplicates("date", keep="last")

        if "close_price" in benchmark_df.columns:
            price_col = "close_price"
        elif "open_price" in benchmark_df.columns:
            price_col = "open_price"
        else:
            raise ValueError("avg_return_daily.pkl 缺少 open_price / close_price 列。")

        benchmark_prices = benchmark_df.set_index("date")[price_col].astype(np.float64)
        self.benchmark_returns = (
            benchmark_prices.pct_change()
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
        del avg_return_daily, benchmark_df
        gc.collect()

    # ------------------------------------------------------------------
    # 因子分组回测
    # ------------------------------------------------------------------

    def evaluate_factor_df(
        self,
        factor_df: pd.DataFrame,
        save_path: str,
        train_windows: int,
    ) -> Dict[str, Any]:
        raw = eng.backtest(
            factor_df,
            n=self.n_groups,
            save_path=save_path,
            rebalance_days=self.rebalance_days,
        )
        return normalize_metrics(raw, train_windows=train_windows)

    # ------------------------------------------------------------------
    # 头组持仓回测
    # ------------------------------------------------------------------

    def evaluate_head_holdings(
        self,
        factor_df: pd.DataFrame,
        save_path: str,
    ) -> Dict[str, Any]:
        results = self._build_backtest_results(factor_df)
        if results is None:
            return empty_head_holdings_metrics()
        save_backtest_figure(results, save_path)

        strategy_stats = results.get("statistics", {}).get("strategy", {})
        excess_stats = results.get("statistics", {}).get("excess", {})
        holdings_schedule = results.get("holdings_schedule", {})
        yearly_excess_returns = _calculate_yearly_excess_returns(
            results.get("excess_nav", pd.Series(dtype=float))
        )
        return {
            "head_holdings_signal_count": int(len(holdings_schedule)) if isinstance(holdings_schedule, dict) else 0,
            "head_holdings_strategy_total_return_pct": _safe_float(strategy_stats.get("total_return"), np.nan),
            "head_holdings_strategy_annual_return_pct": _safe_float(strategy_stats.get("annual_return"), np.nan),
            "head_holdings_strategy_annual_volatility_pct": _safe_float(strategy_stats.get("annual_volatility"), np.nan),
            "head_holdings_strategy_sharpe_ratio": _safe_float(strategy_stats.get("sharpe_ratio"), np.nan),
            "head_holdings_strategy_max_drawdown_pct": _safe_float(strategy_stats.get("max_drawdown"), np.nan),
            "head_holdings_strategy_win_rate_pct": _safe_float(strategy_stats.get("win_rate"), np.nan),
            "head_holdings_excess_total_return_pct": _safe_float(excess_stats.get("total_return"), np.nan),
            "head_holdings_excess_annual_return_pct": _safe_float(excess_stats.get("annual_return"), np.nan),
            "head_holdings_excess_annual_volatility_pct": _safe_float(excess_stats.get("annual_volatility"), np.nan),
            "head_holdings_excess_sharpe_ratio": _safe_float(excess_stats.get("sharpe_ratio"), np.nan),
            "head_holdings_excess_max_drawdown_pct": _safe_float(excess_stats.get("max_drawdown"), np.nan),
            "head_holdings_excess_win_rate_pct": _safe_float(excess_stats.get("win_rate"), np.nan),
            "head_holdings_excess_return_by_year_pct": yearly_excess_returns,
        }

    def build_head_group_holdings_signal(
        self,
        factor_df: pd.DataFrame,
    ) -> pd.Series:
        """从 factor_df 中提取头组股票名单，构造按日期索引的信号序列。"""
        signal_pairs = []
        for dt, rd in factor_df.groupby("date", sort=True):
            if len(rd) < self.n_groups * 2:
                continue
            ranked = rd.loc[:, ["stock_code", "factor"]].copy()
            ranked["group"] = pd.qcut(
                ranked["factor"].rank(method="first"),
                q=self.n_groups,
                labels=False,
                duplicates="drop",
            )
            head_codes = (
                ranked.loc[ranked["group"] == (self.n_groups - 1), "stock_code"]
                .dropna().astype(str).drop_duplicates().tolist()
            )
            signal_pairs.append((pd.to_datetime(dt), head_codes))

        if not signal_pairs:
            return pd.Series(dtype=object)
        signal_index, signal_values = zip(*signal_pairs)
        return pd.Series(signal_values, index=pd.DatetimeIndex(signal_index), name="head_holdings")

    def _build_backtest_results(
        self,
        factor_df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        holdings_signal = self.build_head_group_holdings_signal(factor_df)
        if holdings_signal.empty:
            return None

        common_dates = sorted(
            set(pd.to_datetime(holdings_signal.index))
            & set(self.open_prices.index)
            & set(self.close_prices.index)
            & set(pd.to_datetime(self.benchmark_returns.index))
        )
        if len(common_dates) <= self.rebalance_days + 1:
            return None

        results = holdings_bt.analyze_model_holdings(
            holdings_data=holdings_signal,
            open_prices=self.open_prices,
            close_prices=self.close_prices,
            benchmark_data=self.benchmark_returns,
            status_data=self.status,
            method="periodic_rebalance",
            holding_period=self.rebalance_days,
            commission_rate=self.commission_rate,
            verbose=False,
        )
        if not isinstance(results, dict):
            return None

        nav = results.get("nav", pd.Series(dtype=float))
        benchmark_nav = results.get("benchmark_nav", pd.Series(dtype=float))
        excess_nav = results.get("excess_nav", pd.Series(dtype=float))
        if len(nav) == 0 or len(benchmark_nav) == 0 or len(excess_nav) == 0:
            return None
        return {
            "nav": nav,
            "benchmark_nav": benchmark_nav,
            "excess_nav": excess_nav,
            "statistics": results.get("statistics", {}),
            "holdings_schedule": results.get("holdings_schedule", {}),
        }


# ---------------------------------------------------------------------------
# 静态辅助函数（不依赖实例状态）
# ---------------------------------------------------------------------------

def _calculate_yearly_excess_returns(excess_nav: pd.Series) -> Dict[str, float]:
    """逐年超额收益：每个自然年从 1 开始重新复利。"""
    if len(excess_nav) == 0:
        return {}
    yearly_returns: Dict[str, float] = {}
    excess_returns = excess_nav.pct_change().dropna()
    if len(excess_returns) == 0:
        return yearly_returns
    for year, ser in excess_returns.groupby(excess_returns.index.year):
        year_return = float((1.0 + ser).prod() - 1.0) * 100.0
        yearly_returns[str(int(year))] = round(year_return, 2)
    return yearly_returns


def save_backtest_figure(results: Dict[str, Any], save_path: str) -> None:
    """绘制头组持仓回测四宫格图：净值、超额、回撤、统计表。"""
    nav_series = results["nav"]
    benchmark_series = results["benchmark_nav"]
    excess_series = results["excess_nav"]
    statistics = results["statistics"]

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(nav_series.index, nav_series, label="头组持仓组合", linewidth=2, color="blue")
    ax1.plot(benchmark_series.index, benchmark_series, label="基准组合", linewidth=2, color="gray", alpha=0.7)
    ax1.set_title("头组持仓回测-净值曲线", fontsize=14)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("净值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(excess_series.index, excess_series, "g-", linewidth=2, label="头组持仓超额")
    ax2.set_title("头组持仓回测-超额收益曲线", fontsize=14)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("超额净值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    ax3 = plt.subplot(2, 2, 3)
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax * 100
    ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
    ax3.plot(drawdown.index, drawdown, "r-", linewidth=1)
    ax3.set_title("头组持仓回测-回撤曲线", fontsize=14)
    ax3.set_xlabel("日期")
    ax3.set_ylabel("回撤 (%)")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("tight")
    ax4.axis("off")
    table_data = []
    headers = ["指标", "策略组合", "基准组合", "超额"]
    metric_keys = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "win_rate"]
    metric_names = ["年化收益(%)", "年化波动(%)", "夏普比率", "最大回撤(%)", "胜率(%)"]
    for key, name in zip(metric_keys, metric_names):
        row = [
            name,
            f"{statistics['strategy'][key]:.2f}",
            f"{statistics['benchmark'][key]:.2f}",
            f"{statistics['excess'][key]:.2f}",
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
