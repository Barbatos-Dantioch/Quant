#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
尾组选股池实验（回测起点提前到 2022）

复现 0409_tail_pool / TP-10 设计，但：
  1. 重新训练 fusion 模型（1 回归 + 2 分类 XGB），START_DATE=2020，
     滚动窗口 4 个半年（2 年）训练 → 首个样本外预测落在 2022H1，
     回测区间扩展为 2022 - 2026。
  2. 融券池改为「逐日动态」：用 all panel 的 is_margin_buy==1（两融标的
     逐日逐票标记）作为当日融券池，避免使用静态全集带来的前视/可得性偏差。

模型超参取自 output/0409_cls_tuning/CE-01/config.json（与 TP-10 同款）。

复用：
  - run_ensemble_experiments: train_fusion, rolling_train_predict,
    _load_price_data, _save_longshort_figure, _save_yearly_longshort_figure,
    compute_rank_ic, XGB_REG_PARAMS / XGB_CLS_PARAMS
  - experiment_engine: half_year, backtest, calculate_rank_ic
  - model_backtest_framework_longshort: analyze_longshort_holdings
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 路径设置 ───
WORK_DIR = "/root/quant"
for p in (WORK_DIR, os.path.join(WORK_DIR, "xgbcode"),
          os.path.join(WORK_DIR, "xgbcode", "func"),
          os.path.join(WORK_DIR, "xgbcode", "short")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(WORK_DIR)

import experiment_engine as eng
import run_ensemble_experiments as ens
from run_ensemble_experiments import (
    train_fusion, rolling_train_predict, _load_price_data,
    _save_longshort_figure, _save_yearly_longshort_figure, compute_rank_ic,
)

# ═══════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════
ALL_PANEL_PATH = os.path.join(WORK_DIR, "Data", "all", "panel_origin.pkl")
OUTPUT_ROOT = os.path.join(WORK_DIR, "output", "0531_tail_pool_2022")
EXP_NAME = "TP-10"

START_DATE = "2020-01-01"   # 初始训练窗起点（2020-2021 作训练 → 首预测 2022H1）
END_DATE = "2026-03-01"
BACKTEST_START = "2022-01-01"

MV_TAIL_QUANTILE = 0.75     # 尾组限定全市场市值前 75%
N_GROUPS = 10               # 头组 top 10% / 尾组 bottom 10%
REBALANCE_DAYS = 5
COMMISSION_RATE = 0.0007
MAX_PARALLEL = 2            # all panel 训练窗并行路数（降到 2 以控制峰值内存，避免 OOM）

# 非特征列（与 run_ensemble_experiments 保持一致）
NON_FEATURE_COLS = {
    "index", "stock_code", "trade_date", "date",
    "c_pct_5", "market_value", "is_margin_buy",
}

# CE-01 超参（output/0409_cls_tuning/CE-01/config.json）
CE01_REG_OVERRIDE = {"max_depth": 6, "colsample_bytree": 0.5, "reg_alpha": 2.0}
CE01_CLS_OVERRIDE = {"max_depth": 6, "colsample_bytree": 0.5, "gamma": 3.0}

# ─── 日志 ───
LOG_PATH = os.path.join(OUTPUT_ROOT, "run.log")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


class _Logger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "a", encoding="utf-8")

    def write(self, msg):
        try:
            self.terminal.write(msg)
        except UnicodeEncodeError:
            self.terminal.write(msg.encode("ascii", "replace").decode("ascii"))
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = _Logger(LOG_PATH)
sys.stderr = _Logger(LOG_PATH)


# ═══════════════════════════════════════════════════════════════════════
# 数据加载（一次性读取 all panel，同时产出训练 dfs 与逐日融券池/市值元数据）
# ═══════════════════════════════════════════════════════════════════════

def load_all_panel():
    """
    返回 (feature_cols, dfs, meta)。
    - dfs: 半年切窗后的训练数据（仅保留 date/stock_code/c_pct_5/market_value + 特征）
    - meta: DataFrame[date, stock_code, market_value, is_margin_buy]，供回测期信号构建
    """
    print(f"  加载 all panel: {ALL_PANEL_PATH} ...", flush=True)
    panel = pd.read_pickle(ALL_PANEL_PATH)
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.dropna(inplace=True)
    panel.reset_index(drop=True, inplace=True)
    panel["date"] = pd.to_datetime(panel["date"])

    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    for col in feature_cols:
        if panel[col].dtype != np.float32:
            panel[col] = panel[col].astype(np.float32)

    n_rows = len(panel)
    n_stocks = panel["stock_code"].nunique()

    # 逐日融券池 + 市值元数据（仅回测期需要，提前到全期，后面按日期过滤）
    meta = panel[["date", "stock_code", "market_value", "is_margin_buy"]].copy()
    meta["stock_code"] = meta["stock_code"].astype(str)
    meta.drop_duplicates(["date", "stock_code"], keep="last", inplace=True)

    dfs = eng.half_year(panel, START_DATE, END_DATE)
    keep_cols = ["date", "stock_code", "c_pct_5", "market_value"] + feature_cols
    dfs = [d[keep_cols].reset_index(drop=True) for d in dfs]

    del panel
    gc.collect()

    print(f"  all 完成: {n_rows:,} 行, {len(feature_cols)} 特征, "
          f"{n_stocks} 只股票, {len(dfs)} 半年区间 (panel 已释放)", flush=True)
    return feature_cols, dfs, meta


# ═══════════════════════════════════════════════════════════════════════
# 信号构建：头组=全市场 top 10%；尾组=当日融券池 ∩ 市值前 75% 的 bottom 10%
# （逻辑对齐 run_0413_mlp.build_signals_from_factor，仅把静态融券名单替换为
#   逐日 is_margin_buy==1 动态池）
# ═══════════════════════════════════════════════════════════════════════

def build_signals_dynamic_pool(factor_df, n_groups_head=10, n_groups_tail=10,
                               mv_quantile=0.75):
    head_records, tail_records = [], []

    for dt, grp in factor_df.groupby("date"):
        grp_h = grp.dropna(subset=["factor"]).copy()
        if len(grp_h) < n_groups_head * 2:
            continue
        # 头组：全市场 factor 最高的 1/n
        grp_h["group_h"] = pd.qcut(
            grp_h["factor"].rank(method="first"),
            q=n_groups_head, labels=False, duplicates="drop",
        )
        head_codes = (
            grp_h.loc[grp_h["group_h"] == n_groups_head - 1, "stock_code"]
            .dropna().astype(str).tolist()
        )
        head_records.append((dt, head_codes))

        # 尾组：当日融券池(is_margin_buy==1) + 市值前 mv_quantile
        grp_t = grp_h[grp_h["is_margin_buy"] == 1].copy()
        mv_thresh = grp_h["market_value"].quantile(1 - mv_quantile)
        grp_t = grp_t[grp_t["market_value"] >= mv_thresh]
        if len(grp_t) < n_groups_tail * 2:
            tail_records.append((dt, []))
            continue
        grp_t["group_t"] = pd.qcut(
            grp_t["factor"].rank(method="first"),
            q=n_groups_tail, labels=False, duplicates="drop",
        )
        tail_codes = (
            grp_t.loc[grp_t["group_t"] == 0, "stock_code"]
            .dropna().astype(str).tolist()
        )
        tail_records.append((dt, tail_codes))

    def _to_series(records):
        if not records:
            return pd.Series(dtype=object)
        idx, vals = zip(*records)
        return pd.Series(list(vals), index=pd.DatetimeIndex(idx))

    return _to_series(head_records), _to_series(tail_records)


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print(f"尾组选股池实验 (2022 起回测)  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  头组: 全市场 top {100 // N_GROUPS}%, "
          f"尾组: 逐日融券池 bottom {100 // N_GROUPS}%, 市值前 {int(MV_TAIL_QUANTILE*100)}%")
    print(f"  训练起点 {START_DATE} / 回测起点 {BACKTEST_START} / 数据末端 {END_DATE}")
    print(f"  模型: fusion(CE-01 超参), 滚动窗口 {ens.WINDOW_SIZE} 半年")
    print(f"  输出: {OUTPUT_ROOT}")
    print("=" * 70)

    # 应用 CE-01 超参（回归走模块级 XGB_REG_PARAMS；分类走 cfg override）
    ens.XGB_REG_PARAMS.update(CE01_REG_OVERRIDE)

    exp_dir = os.path.join(OUTPUT_ROOT, EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    cache_dir = os.path.join(OUTPUT_ROOT, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    factor_pkl = os.path.join(exp_dir, "factor_df.pkl")

    # ── 训练（有缓存则跳过）──
    if os.path.exists(factor_pkl):
        print("\n  发现 factor_df 缓存, 跳过训练", flush=True)
        factor_df = pd.read_pickle(factor_pkl)
        meta = None
        # 信号构建仍需 is_margin_buy / market_value；若 factor_df 已含则复用
        if "is_margin_buy" not in factor_df.columns:
            _, _, meta = load_all_panel_meta_only()
    else:
        feature_cols, dfs, meta = load_all_panel()
        eng.panel = None
        eng.dfs = dfs
        eng.feature_cols = feature_cols

        cfg = {
            "name": EXP_NAME,
            "group": "fusion",
            "top_k": 0.10,
            "alpha": 0.5,
            "cls_type": "xgb",
            "cls_params_override": CE01_CLS_OVERRIDE,
            "cls_boost_round": 400,
            "cls_early_stop": 30,
            "_cache_dir": cache_dir,
            "_panel_tag": "all_tp2022_ce01",
        }

        print("\n  开始滚动训练 ...", flush=True)
        t0 = time.time()
        factor_df = rolling_train_predict(dfs, feature_cols, train_fusion, cfg,
                                          max_parallel=MAX_PARALLEL)
        t_train = time.time() - t0
        del dfs
        eng.dfs = []
        gc.collect()
        if factor_df is None or factor_df.empty:
            print("  训练失败, 退出")
            return
        print(f"  训练完成, 耗时 {t_train:.0f}s, 样本数 {len(factor_df)}", flush=True)

    # ── 合并逐日融券池/市值，过滤回测期 ──
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    factor_df["stock_code"] = factor_df["stock_code"].astype(str)
    if "is_margin_buy" not in factor_df.columns:
        factor_df = factor_df.drop(columns=["market_value"], errors="ignore")
        factor_df = factor_df.merge(meta, on=["date", "stock_code"], how="left")
    factor_df = factor_df[factor_df["date"] >= BACKTEST_START].copy()
    factor_df["is_margin_buy"] = factor_df["is_margin_buy"].fillna(0).astype(int)
    # 训练完立即落盘（含融券池/市值），避免后续回测崩溃丢失训练结果
    factor_df.to_pickle(factor_pkl)
    del meta
    gc.collect()

    # ── RankIC ──
    ic_metrics = compute_rank_ic(factor_df)
    print(f"  RankIC mean={ic_metrics['rank_ic_mean']}  "
          f"IR={ic_metrics['rank_ic_ir']}", flush=True)

    # ── 信号构建 ──
    head_signal, tail_signal = build_signals_dynamic_pool(
        factor_df, n_groups_head=N_GROUPS, n_groups_tail=N_GROUPS,
        mv_quantile=MV_TAIL_QUANTILE,
    )

    # ── 头尾去重（与尾组重合的从两边都剔除）──
    overlap_removed = 0
    common_dates = sorted(set(head_signal.index) & set(tail_signal.index))
    for dt in common_dates:
        h_set, t_set = set(head_signal.loc[dt]), set(tail_signal.loc[dt])
        overlap = h_set & t_set
        if overlap:
            overlap_removed += len(overlap)
            head_signal.loc[dt] = [s for s in head_signal.loc[dt] if s not in overlap]
            tail_signal.loc[dt] = [s for s in tail_signal.loc[dt] if s not in overlap]
    print(f"  头尾去重 {overlap_removed} 只", flush=True)

    # ── 多空回测 ──
    from model_backtest_framework_longshort import analyze_longshort_holdings
    open_prices, close_prices, status = _load_price_data()
    ls_results = analyze_longshort_holdings(
        head_holdings_data=head_signal, tail_holdings_data=tail_signal,
        open_prices=open_prices, close_prices=close_prices,
        benchmark_data=None, status_data=status,
        method="periodic_rebalance", holding_period=REBALANCE_DAYS,
        commission_rate=COMMISSION_RATE, verbose=False,
    )
    _save_longshort_figure(ls_results, os.path.join(exp_dir, "longshort_backtest.png"))
    _save_yearly_longshort_figure(ls_results, os.path.join(exp_dir, "yearly_longshort.png"))
    del open_prices, close_prices, status
    gc.collect()

    # ── 分组回测图（需 eng.avg_return）──
    avg_ret = (factor_df.groupby("date")["return"].mean()
               .reset_index().rename(columns={"return": "avg_return"}))
    eng.avg_return = avg_ret
    eng.backtest(factor_df, n=N_GROUPS,
                 save_path=os.path.join(exp_dir, "backtest.png"),
                 rebalance_days=REBALANCE_DAYS)

    # ── 保存 metrics ──
    stats = ls_results["statistics"]
    metrics = {
        "experiment": EXP_NAME,
        "backtest_start": BACKTEST_START,
        "n_groups_tail": N_GROUPS,
        "tail_pct": 100 // N_GROUPS,
        "mv_tail_quantile": MV_TAIL_QUANTILE,
        "dynamic_short_pool": True,
        **ic_metrics,
        "overlap_removed": overlap_removed,
    }
    for k in ["head", "tail", "tail_raw", "longshort", "benchmark",
              "head_excess", "tail_excess"]:
        if k in stats:
            metrics[k] = stats[k]
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    ls = stats.get("longshort", {})
    he = stats.get("head_excess", {})
    print(f"\n  [{EXP_NAME}] 完成: IC={ic_metrics['rank_ic_mean']}  "
          f"多空年化={ls.get('annual_return', float('nan')):.1f}%  "
          f"夏普={ls.get('sharpe_ratio', float('nan')):.3f}  "
          f"回撤={ls.get('max_drawdown', float('nan')):.1f}%  "
          f"头超={he.get('annual_return', float('nan')):.1f}%", flush=True)

    print(f"\n{'='*70}")
    print(f"全部完成  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"总耗时 {(time.time()-t_start)/3600:.2f} 小时")
    print(f"输出: {exp_dir}")
    print("=" * 70)


def load_all_panel_meta_only():
    """仅用于缓存命中但 factor_df 缺 is_margin_buy 的兜底：重新取 meta。"""
    return load_all_panel()


if __name__ == "__main__":
    main()
