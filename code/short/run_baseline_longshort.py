#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 多空回测实验脚本

基于 baseline 设计（baseline_config.md），改动两点：
1. 回测使用多空框架：做多头组（top 10%）+ 做空尾组（bottom 10%）
2. 使用全部特征进行训练和预测

数据源：Data/short/panel.pkl
- 特征列：所有 f* 开头的列（约 1088 列）
- 排除列：index, stock_code, trade_date, date, c_pct_5, market_value, is_margin_buy

模型训练流程与 baseline 完全一致：
- 标签：市值中性化连续标签（逐日横截面回归 log(市值) 取残差再标准化）
- 模型：XGBoost 回归 (reg:squarederror)
- 训练：固定 4 窗滚动 + 3 种子集成 + Rank IC 早停（30 轮 / 0.001）
- 特征：不做任何变换，缺失值填 0

复用 func/ 模块：
- experiment_engine: split_tv, prep_xy, calculate_rank_ic, backtest,
                     neutralize_label_mv, train_silent, half_year
- backtest_evaluator: BacktestEvaluator（若价格数据可用）
- model_backtest_framework_longshort: analyze_longshort_holdings（若价格数据可用）
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

# ─── 路径设置 ───
WORK_DIR = "/root/quant"
if WORK_DIR not in sys.path:
    sys.path.append(WORK_DIR)
XGBCODE_DIR = os.path.join(WORK_DIR, "xgbcode")
if XGBCODE_DIR not in sys.path:
    sys.path.append(XGBCODE_DIR)
FUNC_DIR = os.path.join(XGBCODE_DIR, "func")
if FUNC_DIR not in sys.path:
    sys.path.append(FUNC_DIR)
os.chdir(WORK_DIR)

import experiment_engine as eng
from experiment_engine import (
    RANDOM_SEED, params, prep_xy, calculate_rank_ic,
    neutralize_label_mv, half_year,
)

np.random.seed(RANDOM_SEED)

# 覆盖 experiment_engine 中的字体设置（SimHei 在 Linux 上不可用）
import matplotlib.font_manager as _fm
_cjk_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Microsoft YaHei']
_available = {f.name for f in _fm.fontManager.ttflist}
_font = next((f for f in _cjk_fonts if f in _available), None)
if _font:
    plt.rcParams['font.sans-serif'] = [_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════════════════
# 实验常量
# ═══════════════════════════════════════════════════════════════════════
PANEL_PATH = os.path.join(WORK_DIR, "Data", "short", "panel.pkl")
OUTPUT_DIR = os.path.join(WORK_DIR, "output", "baseline_longshort")
PANEL_TRADE_PATH = os.path.join(WORK_DIR, "Data", "all", "panel_trade.pkl")
AVG_RETURN_PATH = os.path.join(WORK_DIR, "Data", "avg_return_daily.pkl")

# short panel 中需要排除的非特征列
NON_FEATURE_COLS = {
    "index", "stock_code", "trade_date", "date",
    "c_pct_5", "market_value", "is_margin_buy",
}

ENSEMBLE_SEEDS = [33, 42, 101]
N_GROUPS = 10
WINDOW_SIZE = 4
TRAIN_RATIO = 0.8
EARLY_STOP_ROUNDS = 30
MIN_IMPROVE = 0.001
NUM_BOOST_ROUND = 500
REBALANCE_DAYS = 5
COMMISSION_RATE = 0.0007
START_DATE = "2020-01-01"
END_DATE = "2026-03-01"

# ─── CPU 并行配置 ───
# MAX_PARALLEL_WINDOWS: 同时并行训练的滚动窗口数，设为 1 则退化为串行
# NTHREAD_PER_MODEL:    每个 XGBoost 模型使用的线程数，设为 0 则由 XGBoost 自动决定
# 参考: 当前服务器 4 Socket × 56 线程 = 224 核, 4 路并行每模型 56 线程恰好对齐 NUMA
MAX_PARALLEL_WINDOWS = 4
NTHREAD_PER_MODEL = max(1, os.cpu_count() // MAX_PARALLEL_WINDOWS)

MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "gamma": 5.0,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "nthread": NTHREAD_PER_MODEL,
}


# ═══════════════════════════════════════════════════════════════════════
# 数据加载（替代 eng.load_data，适配 short panel 的列结构）
# ═══════════════════════════════════════════════════════════════════════

def load_short_panel() -> tuple[pd.DataFrame, List[str], List[pd.DataFrame]]:
    """
    加载 Data/short/panel.pkl 并完成预处理。
    返回 (panel, feature_cols, dfs)。

    与 eng.load_data() 的区别：
    - 数据路径为 Data/short/panel.pkl
    - NON_FEATURE_COLS 增加了 index, trade_date, is_margin_buy
    """
    print(f"  加载 {PANEL_PATH} ...")
    panel = pd.read_pickle(PANEL_PATH)
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.dropna(inplace=True)
    panel.reset_index(drop=True, inplace=True)

    assert "c_pct_5" in panel.columns, "panel 中缺少 c_pct_5 列"
    assert "date" in panel.columns, "panel 中缺少 date 列"
    assert "market_value" in panel.columns, "panel 中缺少 market_value 列"

    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    for col in feature_cols:
        if panel[col].dtype != np.float32:
            panel[col] = panel[col].astype(np.float32)

    dfs = half_year(panel, START_DATE, END_DATE)

    print(f"  数据加载完成: {panel.shape[0]} 行, {len(feature_cols)} 特征, {len(dfs)} 半年区间")
    print(f"  日期范围: {panel['date'].min()} ~ {panel['date'].max()}")
    print(f"  股票数: {panel['stock_code'].nunique()}")

    # 从 panel 计算全市场等权日均收益，供 eng.backtest 使用
    avg_ret = (
        panel.groupby("date")["c_pct_5"].mean()
        .reset_index()
        .rename(columns={"c_pct_5": "avg_return"})
    )

    # 设置 engine 全局状态，使 neutralize_label_mv / train_silent / backtest 等可正常工作
    eng.panel = panel
    eng.dfs = dfs
    eng.feature_cols = feature_cols
    eng.avg_return = avg_ret
    params.FEATURE_COLS = feature_cols

    return panel, feature_cols, dfs


# ═══════════════════════════════════════════════════════════════════════
# 模型训练 & 预测
# ═══════════════════════════════════════════════════════════════════════

def train_ensemble(
    tr: pd.DataFrame, va: pd.DataFrame, fcols: List[str],
) -> List[xgb.Booster]:
    """复用 eng.train_silent 训练 3 种子集成模型。"""
    models: List[xgb.Booster] = []
    for seed in ENSEMBLE_SEEDS:
        mp = dict(MODEL_PARAMS)
        mp["seed"] = seed
        model, _ = eng.train_silent(
            tr, va, mp, fcols=fcols,
            label_fn=neutralize_label_mv,
            early_stop_rounds=EARLY_STOP_ROUNDS,
            min_improve=MIN_IMPROVE,
            num_boost_round=NUM_BOOST_ROUND,
        )
        models.append(model)
    return models


def predict_ensemble(
    models: List[xgb.Booster], df: pd.DataFrame, fcols: List[str],
) -> np.ndarray:
    """3 种子独立预测后取均值。"""
    x, _ = prep_xy(df, fcols=fcols, lcol="c_pct_5")
    dmat = xgb.DMatrix(x)
    preds = [m.predict(dmat) for m in models]
    pred = np.mean(np.vstack(preds), axis=0)
    del dmat, x
    gc.collect()
    return pred.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# 滚动训练（多窗口并行）
# ═══════════════════════════════════════════════════════════════════════

_print_lock = threading.Lock()


def _process_one_window(
    window_idx: int,
    train_dfs: List[pd.DataFrame],
    test_raw: pd.DataFrame,
    fcols: List[str],
    total_windows: int,
) -> pd.DataFrame:
    """处理单个滚动窗口：训练集成模型 → 预测 → 构造因子帧。线程安全。"""
    with _print_lock:
        print(f"  窗口 {window_idx}/{total_windows} 开始 ...", flush=True)

    dtr = pd.concat(train_dfs, ignore_index=True)
    tr, va = eng.split_tv(dtr, TRAIN_RATIO)

    models = train_ensemble(tr, va, fcols)
    pred = predict_ensemble(models, test_raw, fcols)

    test_neutralized = neutralize_label_mv(test_raw)
    _, y_neutral = prep_xy(test_neutralized, fcols=fcols, lcol="c_pct_5")

    out = test_raw[["date", "stock_code", "c_pct_5"]].copy()
    out.rename(columns={"c_pct_5": "return"}, inplace=True)
    out["factor"] = pred
    out["return_neutral"] = y_neutral
    result = out[["date", "stock_code", "factor", "return_neutral", "return"]].copy()

    with _print_lock:
        print(f"  窗口 {window_idx}/{total_windows} 完成, 样本数 {len(test_raw)}", flush=True)

    del models, dtr, tr, va, pred, test_neutralized, out
    gc.collect()
    return result


def rolling_train_predict(
    dfs: List[pd.DataFrame], fcols: List[str],
) -> Optional[pd.DataFrame]:
    """
    固定 4 窗滚动训练 + 预测，多窗口并行执行。

    并行安全性说明：
    - 各窗口的训练数据互相独立，不共享可变状态
    - neutralize_label_mv / train_silent 内部只读取 eng.panel，不修改
    - XGBoost 在 C++ 层释放 GIL，ThreadPoolExecutor 可有效并行
    - ThreadPoolExecutor.map 保证输出顺序与输入一致，结果与串行完全相同
    """
    # 预计算所有窗口的 (训练数据列表, 测试数据) 任务
    tasks: List[tuple] = []
    tdl: List[pd.DataFrame] = []
    total_windows = len(dfs) - 1

    for i in range(1, len(dfs)):
        tdl.append(dfs[i - 1])
        if len(tdl) > WINDOW_SIZE:
            tdl.pop(0)
        if len(tdl) < WINDOW_SIZE:
            continue
        # list(tdl) 快照当前训练窗口列表，避免后续迭代中 tdl 变化影响已提交任务
        tasks.append((i, list(tdl), dfs[i], fcols, total_windows))

    if not tasks:
        return None

    print(f"  共 {len(tasks)} 个窗口, {MAX_PARALLEL_WINDOWS} 路并行, "
          f"每模型 nthread={NTHREAD_PER_MODEL}", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WINDOWS) as pool:
        factor_frames = list(pool.map(lambda args: _process_one_window(*args), tasks))

    return pd.concat(factor_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# 持仓信号生成（头组 + 尾组）
# ═══════════════════════════════════════════════════════════════════════

def build_holdings_signals(
    factor_df: pd.DataFrame, n_groups: int,
) -> tuple[pd.Series, pd.Series]:
    """
    按日期横截面将股票分为 n_groups 组，
    提取头组（因子最高 10%）和尾组（因子最低 10%）。
    """
    head_pairs: List[tuple] = []
    tail_pairs: List[tuple] = []

    for dt, rd in factor_df.groupby("date", sort=True):
        if len(rd) < n_groups * 2:
            continue
        ranked = rd[["stock_code", "factor"]].copy()
        ranked["group"] = pd.qcut(
            ranked["factor"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        head_codes = (
            ranked.loc[ranked["group"] == n_groups - 1, "stock_code"]
            .dropna().astype(str).drop_duplicates().tolist()
        )
        tail_codes = (
            ranked.loc[ranked["group"] == 0, "stock_code"]
            .dropna().astype(str).drop_duplicates().tolist()
        )
        dt_ts = pd.to_datetime(dt)
        head_pairs.append((dt_ts, head_codes))
        tail_pairs.append((dt_ts, tail_codes))

    if not head_pairs:
        return pd.Series(dtype=object), pd.Series(dtype=object)

    h_idx, h_vals = zip(*head_pairs)
    t_idx, t_vals = zip(*tail_pairs)
    return (
        pd.Series(list(h_vals), index=pd.DatetimeIndex(h_idx), name="head"),
        pd.Series(list(t_vals), index=pd.DatetimeIndex(t_idx), name="tail"),
    )


# ═══════════════════════════════════════════════════════════════════════
# 多空持仓回测（需要价格数据，不可用时跳过）
# ═══════════════════════════════════════════════════════════════════════

def _load_price_data() -> tuple:
    """从 panel_trade.pkl 加载价格宽表，返回 (open_prices, close_prices, status)。"""
    src = pd.read_pickle(PANEL_TRADE_PATH)
    src["date"] = pd.to_datetime(src["date"])
    src["stock_code"] = src["stock_code"].astype(str)
    src.sort_values(["date", "stock_code"], inplace=True)
    src.drop_duplicates(["date", "stock_code"], keep="last", inplace=True)

    open_prices = (
        src.pivot(index="date", columns="stock_code", values="open_price")
        .sort_index().astype(np.float32)
    )
    close_prices = (
        src.pivot(index="date", columns="stock_code", values="close_price")
        .sort_index().astype(np.float32)
    )
    status = (
        src.pivot(index="date", columns="stock_code", values="status")
        .sort_index()
    )
    del src
    gc.collect()
    return open_prices, close_prices, status


def try_run_longshort_backtest(
    factor_df: pd.DataFrame, output_dir: str,
) -> Optional[dict]:
    """
    执行持仓级多空回测。
    需要 panel_trade.pkl 提供价格数据；若无 avg_return_daily.pkl，
    回测框架自动用全市场等权日收益作为基准。
    """
    if not os.path.exists(PANEL_TRADE_PATH):
        print(f"  [跳过] 未找到 {PANEL_TRADE_PATH}，无法执行持仓级多空回测")
        return None

    from model_backtest_framework_longshort import analyze_longshort_holdings

    open_prices, close_prices, status = _load_price_data()

    # 基准：有 avg_return_daily.pkl 时使用，否则传 None 让框架自动算全市场等权
    benchmark_returns = None
    if os.path.exists(AVG_RETURN_PATH):
        avg_df = pd.read_pickle(AVG_RETURN_PATH)
        if isinstance(avg_df, pd.DataFrame):
            if "date" not in avg_df.columns:
                avg_df = avg_df.reset_index()
            avg_df["date"] = pd.to_datetime(avg_df["date"])
            avg_df = avg_df.sort_values("date").drop_duplicates("date", keep="last")
            price_col = next(
                (c for c in ["close_price", "open_price"] if c in avg_df.columns), None,
            )
            if price_col:
                bm_prices = avg_df.set_index("date")[price_col].astype(np.float64)
                benchmark_returns = (
                    bm_prices.pct_change()
                    .replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    .astype(np.float32)
                )
            del avg_df
            gc.collect()
    else:
        print("  未找到 avg_return_daily.pkl，基准将使用全市场等权日收益")

    head_signal, tail_signal = build_holdings_signals(factor_df, N_GROUPS)

    if head_signal.empty or tail_signal.empty:
        print("  [跳过] 持仓信号为空")
        return None

    print(f"  头组信号天数: {len(head_signal)},  尾组信号天数: {len(tail_signal)}")

    ls_results = analyze_longshort_holdings(
        head_holdings_data=head_signal,
        tail_holdings_data=tail_signal,
        open_prices=open_prices,
        close_prices=close_prices,
        benchmark_data=benchmark_returns,
        status_data=status,
        method="periodic_rebalance",
        holding_period=REBALANCE_DAYS,
        commission_rate=COMMISSION_RATE,
        verbose=False,
    )

    save_longshort_figure(ls_results, os.path.join(output_dir, "longshort_backtest.png"))
    del open_prices, close_prices, status
    gc.collect()
    return ls_results


# ═══════════════════════════════════════════════════════════════════════
# 结果保存
# ═══════════════════════════════════════════════════════════════════════

def save_longshort_figure(results: dict, save_path: str) -> None:
    """绘制多空回测四宫格图并保存到文件。"""
    nav_df = results["nav_df"]
    stats = results["statistics"]

    fig = plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(nav_df.index, nav_df["head"], lw=2, color="red", label="头组(多头)")
    ax1.plot(nav_df.index, nav_df["tail"], lw=2, color="green", label="尾组(做空)")
    ax1.plot(nav_df.index, nav_df["benchmark"], lw=2, color="gray", alpha=0.7, label="基准")
    ax1.set_title("头组 / 尾组 / 基准 净值", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    head_excess = nav_df["head"] / nav_df["benchmark"]
    ax2.plot(nav_df.index, nav_df["longshort"], lw=2, color="blue", label="多空组合")
    ax2.plot(head_excess.index, head_excess, lw=1.5, color="orange", alpha=0.8, label="头组超额")
    ax2.axhline(y=1, color="gray", ls="--", alpha=0.5)
    ax2.set_title("多空组合 & 头组超额", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    ls_dd_nav = nav_df["longshort"] / nav_df["longshort"].cummax()
    ls_dd = (ls_dd_nav - 1) * 100
    ax3.fill_between(ls_dd.index, ls_dd, 0, alpha=0.3, color="red")
    ax3.plot(ls_dd.index, ls_dd, "r-", lw=1)
    ax3.set_title("多空组合回撤", fontsize=13)
    ax3.set_ylabel("回撤 (%)")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    headers = ["指标", "头组超额", "尾组做空超额", "多空组合", "基准"]
    metric_keys = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "win_rate"]
    metric_names = ["年化收益(%)", "年化波动(%)", "夏普比率", "最大回撤(%)", "胜率(%)"]
    tdata = [
        [mn,
         f"{stats['head_excess'][m]:.2f}",
         f"{stats['tail_excess'][m]:.2f}",
         f"{stats['longshort'][m]:.2f}",
         f"{stats['benchmark'][m]:.2f}"]
        for m, mn in zip(metric_keys, metric_names)
    ]
    tbl = ax4.table(cellText=tdata, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def compute_rank_ic(factor_df: pd.DataFrame) -> Dict[str, float]:
    """逐日计算 Rank IC，返回均值、标准差和 IR。"""
    ics: List[float] = []
    for _, rd in factor_df.groupby("date"):
        if len(rd) < 10:
            continue
        ic = calculate_rank_ic(rd["factor"], rd["return_neutral"])
        if not np.isnan(ic):
            ics.append(float(ic))
    if not ics:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan, "rank_ic_ir": np.nan}
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics))
    ir = mean_ic / std_ic if std_ic > 0 else np.nan
    return {
        "rank_ic_mean": round(mean_ic, 6),
        "rank_ic_std": round(std_ic, 6),
        "rank_ic_ir": round(ir, 6),
    }


def save_metrics(
    factor_df: pd.DataFrame,
    ls_results: Optional[dict],
    elapsed_sec: float,
    output_dir: str,
) -> Dict[str, Any]:
    """汇总 Rank IC + 多空回测指标（如有），保存 JSON 并返回。"""
    ic_metrics = compute_rank_ic(factor_df)
    metrics: Dict[str, Any] = {**ic_metrics, "elapsed_sec": round(elapsed_sec, 1)}

    if ls_results is not None:
        stats = ls_results["statistics"]
        metrics.update({
            "head": stats["head"],
            "tail": stats["tail"],
            "longshort": stats["longshort"],
            "benchmark": stats["benchmark"],
            "head_excess": stats["head_excess"],
        })

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    return metrics


def save_report(
    metrics: Dict[str, Any], output_dir: str, has_longshort: bool,
) -> None:
    """生成文本摘要报告。"""
    ic = metrics.get("rank_ic_mean", np.nan)
    ir = metrics.get("rank_ic_ir", np.nan)

    lines = [
        "# Baseline 多空回测报告",
        "",
        "## Rank IC",
        f"- 均值: {ic:.6f}",
        f"- IR:   {ir:.6f}",
        "",
    ]

    if has_longshort:
        h = metrics["head"]
        t = metrics["tail"]
        ls = metrics["longshort"]
        he = metrics["head_excess"]
        lines += [
            "## 多空回测",
            "| 指标 | 头组超额 | 尾组做空超额 | 多空组合 | 基准 |",
            "|---|---|---|---|---|",
            f"| 年化收益(%) | {h['annual_return']:.2f} | {t['annual_return']:.2f} | {ls['annual_return']:.2f} | {he['annual_return']:.2f} |",
            f"| 年化波动(%) | {h['annual_volatility']:.2f} | {t['annual_volatility']:.2f} | {ls['annual_volatility']:.2f} | {he['annual_volatility']:.2f} |",
            f"| 夏普比率 | {h['sharpe_ratio']:.3f} | {t['sharpe_ratio']:.3f} | {ls['sharpe_ratio']:.3f} | {he['sharpe_ratio']:.3f} |",
            f"| 最大回撤(%) | {h['max_drawdown']:.2f} | {t['max_drawdown']:.2f} | {ls['max_drawdown']:.2f} | {he['max_drawdown']:.2f} |",
            f"| 胜率(%) | {h['win_rate']:.1f} | {t['win_rate']:.1f} | {ls['win_rate']:.1f} | {he['win_rate']:.1f} |",
            "",
        ]
    else:
        lines += ["## 多空回测", "", "（缺少价格数据，持仓级多空回测已跳过）", ""]

    lines.append(f"训练耗时: {metrics.get('elapsed_sec', 0):.1f}s")
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print(f"Baseline 多空回测实验  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. 加载数据 ──
    print("\n[1/5] 加载数据 ...")
    panel, fcols, dfs = load_short_panel()
    print(f"  特征数: {len(fcols)},  半年区间数: {len(dfs)}")

    # ── 2. 保存实验配置 ──
    config = {
        "data_path": PANEL_PATH,
        "model_params": MODEL_PARAMS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "window_size": WINDOW_SIZE,
        "train_ratio": TRAIN_RATIO,
        "early_stop_rounds": EARLY_STOP_ROUNDS,
        "min_improve": MIN_IMPROVE,
        "num_boost_round": NUM_BOOST_ROUND,
        "n_groups": N_GROUPS,
        "rebalance_days": REBALANCE_DAYS,
        "commission_rate": COMMISSION_RATE,
        "feature_count": len(fcols),
        "non_feature_cols": sorted(NON_FEATURE_COLS),
        "label": "市值中性化连续标签（逐日横截面回归 log(市值) 取残差再标准化）",
        "backtest": "多空回测（做多头组 top10% + 做空尾组 bottom10%）",
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # ── 3. 滚动训练（支持缓存复用） ──
    factor_pkl = os.path.join(OUTPUT_DIR, "factor_df.pkl")
    if os.path.exists(factor_pkl):
        print("\n[2/5] 发现缓存, 跳过训练")
        factor_df = pd.read_pickle(factor_pkl)
        train_elapsed = 0.0
        print(f"  已加载 factor_df, 样本数: {len(factor_df)}")
    else:
        print("\n[2/5] 滚动训练 ...")
        t0 = time.time()
        factor_df = rolling_train_predict(dfs, fcols)
        train_elapsed = time.time() - t0
        if factor_df is None or factor_df.empty:
            raise RuntimeError("滚动训练未产出任何因子数据")
        print(f"  训练完成, 耗时 {train_elapsed:.1f}s, 因子样本数: {len(factor_df)}")
        factor_df.to_pickle(factor_pkl)

    # ── 4. 分组回测 ──
    print("\n[3/5] 分组回测 ...")
    bt_metrics = eng.backtest(
        factor_df, n=N_GROUPS,
        save_path=os.path.join(OUTPUT_DIR, "backtest.png"),
    )
    print(f"  Rank IC:     {bt_metrics.get('rank_ic_mean', np.nan):.6f}")
    print(f"  头组超额:    {bt_metrics.get('top_group_excess_return', np.nan):.6f}")

    # ── 5. 持仓级多空回测（需要价格数据） ──
    print("\n[4/5] 多空回测 ...")
    ls_results = try_run_longshort_backtest(factor_df, OUTPUT_DIR)

    # ── 6. 保存结果 ──
    print("\n[5/5] 保存结果 ...")
    metrics = save_metrics(factor_df, ls_results, train_elapsed, OUTPUT_DIR)
    save_report(metrics, OUTPUT_DIR, has_longshort=(ls_results is not None))

    # ── 打印摘要 ──
    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"  Rank IC 均值:      {metrics['rank_ic_mean']:.6f}")
    print(f"  Rank IC IR:        {metrics['rank_ic_ir']:.6f}")

    if ls_results is not None:
        ls = metrics["longshort"]
        he = metrics["head_excess"]
        print(f"  头组年化收益:      {metrics['head']['annual_return']:.2f}%")
        print(f"  尾组年化收益:      {metrics['tail']['annual_return']:.2f}%")
        print(f"  多空年化收益:      {ls['annual_return']:.2f}%")
        print(f"  多空夏普比率:      {ls['sharpe_ratio']:.3f}")
        print(f"  多空最大回撤:      {ls['max_drawdown']:.2f}%")
        print(f"  头组超额年化:      {he['annual_return']:.2f}%")
    else:
        print("  （持仓级多空回测已跳过，缺少价格数据）")

    print(f"  训练耗时:          {train_elapsed:.1f}s")
    print(f"  输出目录:          {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"FATAL: {err}")
        traceback.print_exc()
        raise
