#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可复用的单实验输出模块。

职责：
- 缓存检测与复用
- 因子表 / 回测图 / 指标 / 报告的统一保存
- 阶段汇总表生成
- 指标排序与格式化工具

零项目内依赖，仅依赖 os / json / numpy / pandas。
BacktestEvaluator 等评估器通过参数传入（duck typing）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IO 工具函数
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# 指标工具函数
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


def metric_key(metrics: Dict[str, Any]) -> tuple:
    """排序元组：(head_cum_win_rate, head_cum_advantage, rank_ic_mean)，逐位比较。"""
    return (
        _safe_float(metrics.get("head_cum_win_rate"), default=-1.0),
        _safe_float(metrics.get("head_cum_advantage"), default=-999.0),
        _safe_float(metrics.get("rank_ic_mean"), default=-999.0),
    )


def metrics_to_row(metrics: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """将 metrics + metadata 拍平为一行 dict，供 results.csv 使用。

    metadata 字段：stage, name, desc, family, runner, backbone_name, elapsed_sec, error
    """
    return {
        "stage": metadata.get("stage", ""),
        "name": metadata.get("name", ""),
        "desc": metadata.get("desc", ""),
        "family": metadata.get("family", ""),
        "runner": metadata.get("runner", ""),
        "backbone": metadata.get("backbone_name", ""),
        "elapsed_sec": round(float(metadata.get("elapsed_sec", 0.0)), 2),
        "rank_ic_mean": _safe_float(metrics.get("rank_ic_mean")),
        "rank_ic_ir": _safe_float(metrics.get("rank_ic_ir")),
        "top_group_excess_return": _safe_float(metrics.get("top_group_excess_return"), np.nan),
        "head_excess_return_annualized": _safe_float(metrics.get("head_excess_return_annualized"), np.nan),
        "head_excess_volatility_annualized": _safe_float(metrics.get("head_excess_volatility_annualized"), np.nan),
        "head_excess_ir": _safe_float(metrics.get("head_excess_ir"), np.nan),
        "head_excess_max_drawdown": _safe_float(metrics.get("head_excess_max_drawdown"), np.nan),
        "head_holdings_excess_annual_return_pct": _safe_float(metrics.get("head_holdings_excess_annual_return_pct"), np.nan),
        "head_holdings_excess_annual_volatility_pct": _safe_float(metrics.get("head_holdings_excess_annual_volatility_pct"), np.nan),
        "head_holdings_excess_sharpe_ratio": _safe_float(metrics.get("head_holdings_excess_sharpe_ratio"), np.nan),
        "head_holdings_excess_max_drawdown_pct": _safe_float(metrics.get("head_holdings_excess_max_drawdown_pct"), np.nan),
        "head_cum_win_rate": _safe_float(metrics.get("head_cum_win_rate"), np.nan),
        "head_cum_advantage": _safe_float(metrics.get("head_cum_advantage"), np.nan),
        "head_turnover_mean": _safe_float(metrics.get("head_turnover_mean"), np.nan),
        "head_turnover_last": _safe_float(metrics.get("head_turnover_last"), np.nan),
        "head_turnover_count": int(metrics.get("head_turnover_count", 0)),
        "train_windows": int(metrics.get("train_windows", 0)),
        "error": metadata.get("error", ""),
    }


def empty_experiment_metrics() -> Dict[str, Any]:
    """因子为空或评估失败时的完整空指标模板。"""
    return {
        "rank_ic_mean": -999.0,
        "rank_ic_ir": -999.0,
        "top_group_excess_return": np.nan,
        "head_excess_return_annualized": np.nan,
        "head_excess_volatility_annualized": np.nan,
        "head_excess_ir": np.nan,
        "head_excess_max_drawdown": np.nan,
        "head_cum_win_rate": np.nan,
        "head_cum_advantage": np.nan,
        "head_turnover_mean": np.nan,
        "head_turnover_std": np.nan,
        "head_turnover_last": np.nan,
        "head_turnover_count": 0,
        "train_windows": 0,
        # 头组持仓回测子集
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


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def build_experiment_report(
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    elapsed_sec: float,
    error: str = "",
    config_sections: Optional[Dict[str, Any]] = None,
    rebalance_days: int = 5,
    commission_rate: float = 0.0007,
) -> str:
    """生成单实验 report.md 内容。

    Parameters
    ----------
    metadata : dict
        必须包含 name, stage, desc, family, runner, backbone_name。
    config_sections : dict, optional
        每个 key 作为章节标题，value 序列化为 JSON 块写入 report。
    """
    name = metadata.get("name", "")
    stage = metadata.get("stage", "")
    desc = metadata.get("desc", "")
    family = metadata.get("family", "")
    runner = metadata.get("runner", "")
    backbone_name = metadata.get("backbone_name", "")

    yearly_excess_returns = metrics.get("head_holdings_excess_return_by_year_pct", {})
    yearly_excess_lines: List[str] = []
    if isinstance(yearly_excess_returns, dict) and yearly_excess_returns:
        yearly_excess_lines.extend(["### 头组年度超额收益(%)", ""])
        for year, value in sorted(yearly_excess_returns.items(), key=lambda item: item[0]):
            yearly_excess_lines.append(f"- `{year}`: `{_safe_float(value, np.nan):.2f}`")
        yearly_excess_lines.append("")

    lines = [
        f"# {name}",
        "",
        f"- 阶段: `{stage}`",
        f"- 描述: {desc}",
        f"- family: `{family}`",
        f"- runner: `{runner}`",
        f"- backbone: `{backbone_name}`",
        f"- elapsed_sec: `{elapsed_sec:.2f}`",
        "",
        "## 核心指标",
        "",
        f"- `head_cum_win_rate`: `{_safe_float(metrics.get('head_cum_win_rate'), np.nan):.6f}`",
        f"- `head_cum_advantage`: `{_safe_float(metrics.get('head_cum_advantage'), np.nan):.6f}`",
        f"- `Rank IC`: `{_safe_float(metrics.get('rank_ic_mean'), np.nan):.6f}`",
        f"- `rank_ic_ir`: `{_safe_float(metrics.get('rank_ic_ir'), np.nan):.6f}`",
        f"- `top_group_excess_return`: `{_safe_float(metrics.get('top_group_excess_return'), np.nan):.6f}`",
        f"- `年化头组超额收益`: `{_safe_float(metrics.get('head_excess_return_annualized'), np.nan):.6f}`",
        f"- `头组超额波动`: `{_safe_float(metrics.get('head_excess_volatility_annualized'), np.nan):.6f}`",
        f"- `头组超额收益IR`: `{_safe_float(metrics.get('head_excess_ir'), np.nan):.6f}`",
        f"- `最大回撤`: `{_safe_float(metrics.get('head_excess_max_drawdown'), np.nan):.6f}`",
        f"- `head_turnover_mean`: `{_safe_float(metrics.get('head_turnover_mean'), np.nan):.6f}`",
        f"- `head_turnover_std`: `{_safe_float(metrics.get('head_turnover_std'), np.nan):.6f}`",
        f"- `head_turnover_last`: `{_safe_float(metrics.get('head_turnover_last'), np.nan):.6f}`",
        f"- `head_turnover_count`: `{int(metrics.get('head_turnover_count', 0))}`",
        f"- `train_windows`: `{int(metrics.get('train_windows', 0))}`",
        "",
        "## 头组持仓回测",
        "",
        f"- 信号定义：每 {rebalance_days} 个交易日取一次头组股票，`T` 日信号决定 `T+1` 日持仓，并复用 `model_backtest_framework.py` 按日更新净值",
        f"- 调仓频率：`{rebalance_days}` 个交易日",
        f"- 手续费：`{commission_rate}`（单边）",
        "- 权重口径：每天按当日有效持仓等权聚合收益，不保留上一日自然漂移后的权重",
        "- 买入过滤：仅买入日检查 `status == 0`，卖出阶段不检查 `status`",
        "- 基准：`avg_return_daily.pkl` 转换得到的日频基准收益（优先 `close_price` 的 close-to-close）",
        f"- `头组持仓信号日期数`: `{int(metrics.get('head_holdings_signal_count', 0))}`",
        f"- `策略累计收益(%)`: `{_safe_float(metrics.get('head_holdings_strategy_total_return_pct'), np.nan):.2f}`",
        f"- `策略年化收益(%)`: `{_safe_float(metrics.get('head_holdings_strategy_annual_return_pct'), np.nan):.2f}`",
        f"- `策略年化波动(%)`: `{_safe_float(metrics.get('head_holdings_strategy_annual_volatility_pct'), np.nan):.2f}`",
        f"- `策略Sharpe`: `{_safe_float(metrics.get('head_holdings_strategy_sharpe_ratio'), np.nan):.4f}`",
        f"- `策略最大回撤(%)`: `{_safe_float(metrics.get('head_holdings_strategy_max_drawdown_pct'), np.nan):.2f}`",
        f"- `策略胜率(%)`: `{_safe_float(metrics.get('head_holdings_strategy_win_rate_pct'), np.nan):.2f}`",
        f"- `超额累计收益(%)`: `{_safe_float(metrics.get('head_holdings_excess_total_return_pct'), np.nan):.2f}`",
        f"- `超额年化收益(%)`: `{_safe_float(metrics.get('head_holdings_excess_annual_return_pct'), np.nan):.2f}`",
        f"- `超额年化波动(%)`: `{_safe_float(metrics.get('head_holdings_excess_annual_volatility_pct'), np.nan):.2f}`",
        f"- `超额Sharpe`: `{_safe_float(metrics.get('head_holdings_excess_sharpe_ratio'), np.nan):.4f}`",
        f"- `超额最大回撤(%)`: `{_safe_float(metrics.get('head_holdings_excess_max_drawdown_pct'), np.nan):.2f}`",
        f"- `超额胜率(%)`: `{_safe_float(metrics.get('head_holdings_excess_win_rate_pct'), np.nan):.2f}`",
        "",
        *yearly_excess_lines,
    ]

    if config_sections:
        for section_title, section_data in config_sections.items():
            lines.extend([
                f"## {section_title}",
                "",
                "```json",
                json.dumps(section_data, ensure_ascii=False, indent=2),
                "```",
                "",
            ])

    if error:
        lines.extend(["## Error", "", f"`{error}`", ""])
    return "\n".join(lines)


def summarize_table(rows: List[Dict[str, Any]], title: str, stage_note: str = "") -> str:
    """生成阶段汇总 markdown 表格。"""
    lines = [f"# {title}", ""]
    if stage_note:
        lines.extend([stage_note, ""])
    lines.append("| name | family | backbone | head_cum_win_rate | head_cum_advantage | Rank IC | top_excess | 年化头组超额收益 | 头组超额波动 | 头组超额收益IR | 最大回撤 | turnover_mean | elapsed_sec | error |")
    lines.append("|------|--------|----------|-------------------|--------------------|---------|-----------:|------------------:|------------:|----------------:|---------:|--------------:|------------:|-------|")
    for row in rows:
        lines.append(
            "| {name} | {family} | {backbone} | {head_cum_win_rate:.4f} | {head_cum_advantage:.4f} | {rank_ic_mean:.4f} | {top_group_excess_return:.4f} | {head_excess_return_annualized:.4f} | {head_excess_volatility_annualized:.4f} | {head_excess_ir:.4f} | {head_excess_max_drawdown:.4f} | {head_turnover_mean:.4f} | {elapsed_sec:.0f} | {error} |".format(
                name=row.get("name", ""),
                family=row.get("family", ""),
                backbone=row.get("backbone", ""),
                head_cum_win_rate=_safe_float(row.get("head_cum_win_rate"), np.nan),
                head_cum_advantage=_safe_float(row.get("head_cum_advantage"), np.nan),
                rank_ic_mean=_safe_float(row.get("rank_ic_mean"), np.nan),
                top_group_excess_return=_safe_float(row.get("top_group_excess_return"), np.nan),
                head_excess_return_annualized=_safe_float(row.get("head_excess_return_annualized"), np.nan),
                head_excess_volatility_annualized=_safe_float(row.get("head_excess_volatility_annualized"), np.nan),
                head_excess_ir=_safe_float(row.get("head_excess_ir"), np.nan),
                head_excess_max_drawdown=_safe_float(row.get("head_excess_max_drawdown"), np.nan),
                head_turnover_mean=_safe_float(row.get("head_turnover_mean"), np.nan),
                elapsed_sec=_safe_float(row.get("elapsed_sec"), 0.0),
                error=row.get("error", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心输出函数
# ---------------------------------------------------------------------------

_DEFAULT_REQUIRED_FILES = ["factor_df.pkl", "metrics.json", "head_holdings_backtest.png"]
_DEFAULT_REQUIRED_KEYS = ["head_holdings_excess_annual_return_pct"]


def load_cached_metrics(
    output_dir: str,
    required_files: Optional[Sequence[str]] = None,
    required_keys: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """检查 output_dir 下是否有完整的缓存结果。

    Returns
    -------
    dict or None
        缓存完整时返回 metrics dict，否则返回 None。
    """
    if required_files is None:
        required_files = _DEFAULT_REQUIRED_FILES
    if required_keys is None:
        required_keys = _DEFAULT_REQUIRED_KEYS

    for fname in required_files:
        if not os.path.exists(os.path.join(output_dir, fname)):
            return None

    metrics = load_json(os.path.join(output_dir, "metrics.json"), default={}) or {}
    for key in required_keys:
        if key not in metrics:
            return None
    return metrics


def save_experiment_outputs(
    output_dir: str,
    factor_df: Optional[pd.DataFrame],
    evaluator: Any,
    train_windows: int,
    metadata: Dict[str, Any],
    config_sections: Optional[Dict[str, Any]] = None,
    dual_breakdown_df: Optional[pd.DataFrame] = None,
    elapsed_sec: float = 0.0,
    error: str = "",
    rebalance_days: int = 5,
    commission_rate: float = 0.0007,
) -> Dict[str, Any]:
    """保存单实验全部输出（不含 config.json），返回 metrics dict。

    产出文件：
    - factor_df.pkl
    - dual_breakdown.pkl（仅当 dual_breakdown_df 非空）
    - backtest.png（由 evaluator.evaluate_factor_df 产出）
    - head_holdings_backtest.png（由 evaluator.evaluate_head_holdings 产出）
    - metrics.json
    - report.md

    Parameters
    ----------
    evaluator
        需要有 evaluate_factor_df(factor_df, save_path, train_windows) 和
        evaluate_head_holdings(factor_df, save_path) 两个方法（duck typing）。
    metadata : dict
        必须包含 name, stage, desc, family, runner, backbone_name。
    """
    ensure_dir(output_dir)
    factor_path = os.path.join(output_dir, "factor_df.pkl")
    dual_breakdown_path = os.path.join(output_dir, "dual_breakdown.pkl")
    png_path = os.path.join(output_dir, "backtest.png")
    holdings_png_path = os.path.join(output_dir, "head_holdings_backtest.png")
    metrics_path = os.path.join(output_dir, "metrics.json")
    report_path = os.path.join(output_dir, "report.md")

    if factor_df is None or len(factor_df) == 0:
        metrics = empty_experiment_metrics()
    else:
        factor_df.to_pickle(factor_path)
        if isinstance(dual_breakdown_df, pd.DataFrame) and len(dual_breakdown_df) == len(factor_df):
            dual_breakdown_df.to_pickle(dual_breakdown_path)
        metrics = evaluator.evaluate_factor_df(factor_df, png_path, train_windows=train_windows)
        metrics.update(evaluator.evaluate_head_holdings(factor_df, holdings_png_path))

    metrics["elapsed_sec"] = round(elapsed_sec, 2)
    write_json(metrics_path, metrics)
    write_text(report_path, build_experiment_report(
        metadata, metrics, elapsed_sec, error=error,
        config_sections=config_sections,
        rebalance_days=rebalance_days,
        commission_rate=commission_rate,
    ))
    return metrics
