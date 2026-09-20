#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单因子检验 (目标 = 多空净值)
============================

对模型使用的每个因子(noMktInd 22 个)做单因子多空回测:
  - 每日按该因子原值降序选 top-k(n_top 公式与策略一致: max(50, ceil(0.2·n_legal)))
  - 方向 sign(μ-X_t), 持仓 5 日等权, 与正式策略口径一致
  - 回测期 2024-01-02~2026-05-13 (与 LGB/NN 实验同期, 便于横向比)

复用: base(特征常量/价格表), build_holdings_by_score(按分降序选), 
      _mean_daily_rank_ic(标注因子方向), _pair_runner.run_longshort_backtest.

附 rank-IC 列: IC = 日均 Spearman(因子, r), r=sign(μ-X)·ΔX (策略实现收益);
IC<0 表示该因子需反向使用(降序选则多空净值为负)。

输出: output/0506_ou_pair/factor_test/
  - 各因子子目录: 多空/多头/空头/分年 回测图 + (无单独 metrics)
  - single_factor_summary.csv / .md : 全部因子指标汇总 (按夏普降序)
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir("/root/quant")
os.environ.setdefault("DROP_GROUPS", "mktind")   # 22 因子

_PT_DIR = "/root/quant/xgbcode/pair_trading"
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

import run_ou_pair_lgb_exp_ws as base
from run_ou_pair_nn_ws import build_holdings_by_score, _mean_daily_rank_ic
from _pair_runner import run_longshort_backtest

FEATURES = base.FEATURES
OUTPUT_DIR = base.OUTPUT_DIR
COMMISSION = base.COMMISSION
HOLDING_PERIOD = base.HOLDING_PERIOD
TEST_LO = pd.Timestamp("2024-01-02")    # 与策略 5 段测试期一致
TEST_HI = pd.Timestamp("2026-05-13")
FT_DIR = os.path.join(OUTPUT_DIR, "factor_test")


def _load_feat_cache() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, ".cache", "feat_df_*.parquet")))
    if not paths:
        raise FileNotFoundError("未找到特征缓存 feat_df_*.parquet; 先跑一次 run_ou_pair_nn_ws.py 生成")
    print(f"读取特征缓存: {paths[-1]}")
    return pd.read_parquet(paths[-1])


def _add_calmar(s: Dict) -> Dict:
    ann = s.get("annual_return", 0.0); mdd = s.get("max_drawdown", 0.0)
    s["calmar_ratio"] = round(ann / abs(mdd), 3) if mdd != 0 else 0.0
    return s


def main():
    t0 = time.time()
    os.makedirs(FT_DIR, exist_ok=True)
    px = base.load_price_wides()
    df = _load_feat_cache()
    df["date"] = pd.to_datetime(df["date"])

    # 回测期 + is_legal
    period = df[(df["is_legal"]) & (df["date"] >= TEST_LO) & (df["date"] <= TEST_HI)].copy()
    print(f"回测期 is_legal 样本: {len(period):,}, "
          f"{period['date'].min().date()}~{period['date'].max().date()}, "
          f"{period['date'].nunique()} 截面")

    # rank-IC 用: r = sign(μ-X)·ΔX, 仅标签有效行
    ic_pool = period[~period["delta_X"].isna()].copy()
    ic_pool["_r"] = np.sign(ic_pool["mu"] - ic_pool["X_t"]) * ic_pool["delta_X"]
    ic_grp = ic_pool["date"].values.astype("datetime64[ns]").astype(np.int64)
    r_arr = ic_pool["_r"].to_numpy(np.float64)

    rows: List[Dict] = []
    for f in FEATURES:
        ic = _mean_daily_rank_ic(ic_pool[f].to_numpy(np.float64), r_arr, ic_grp)

        dd = period[["date", "stock_i", "stock_j", "mu", "X_t"]].copy()
        dd["pred_delta_X"] = period[f].values
        long_h, short_h, _ = build_holdings_by_score(dd)

        f_dir = os.path.join(FT_DIR, f)
        bt = run_longshort_backtest(
            long_holdings=long_h, short_holdings=short_h,
            open_prices=px["open_wide"], close_prices=px["close_wide"],
            status_data=px["status_wide"], output_dir=f_dir,
            commission_rate=COMMISSION, holding_period=HOLDING_PERIOD,
        )
        st = bt["statistics"]
        ls = _add_calmar(st["longshort"]); le = st.get("head_excess", {})
        rows.append({
            "factor": f, "rank_ic": round(ic, 4),
            "ls_annual": round(ls["annual_return"], 2),
            "ls_sharpe": round(ls["sharpe_ratio"], 3),
            "ls_calmar": round(ls.get("calmar_ratio", 0.0), 3),
            "ls_mdd": round(ls["max_drawdown"], 2),
            "long_excess": round(le.get("annual_return", float("nan")), 2),
        })
        print(f"  [{f:<20}] IC={ic:+.4f}  多空年化={ls['annual_return']:.2f}  夏普={ls['sharpe_ratio']:.3f}")

    summ = pd.DataFrame(rows).sort_values("ls_sharpe", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(FT_DIR, "single_factor_summary.csv")
    summ.to_csv(csv_path, index=False)
    # markdown
    md = ["# 单因子检验 (多空净值, 2024-01~2026-05, 按因子降序选 top, dir=sign(μ-X))", "",
          "| 因子 | rank-IC | 多空年化 | 夏普 | Calmar | MDD | long超额 |",
          "|------|--------:|--------:|-----:|-------:|----:|--------:|"]
    for _, r in summ.iterrows():
        md.append(f"| {r['factor']} | {r['rank_ic']:+.4f} | {r['ls_annual']:.2f} | "
                  f"{r['ls_sharpe']:.3f} | {r['ls_calmar']:.3f} | {r['ls_mdd']:.2f} | {r['long_excess']:.2f} |")
    with open(os.path.join(FT_DIR, "single_factor_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"\n汇总已写入: {csv_path}")
    print(summ.to_string(index=False))
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
