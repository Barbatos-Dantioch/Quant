#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐年 regime 归因: 反转 edge vs 动量 edge vs 价差走阔
====================================================

只用缓存特征 df(无需回测/价格表), 对所有 is_legal pair 逐年统计:
  反转: r_rev = sign(μ-X)·ΔX        (赌价差回归 μ 的实现收益)
  动量: r_mom = sign(X_t-X_{t-20})·ΔX (赌价差延续的实现收益; X_t-X_{t-20}=ret_i_20-ret_j_20)
  走阔: 1[|X_{t+5}-μ| > |X_t-μ|]      (价差远离均衡=背离/趋势的比例)

用于检验: 2026 配对差是否因"反转失效 / 动量主导(抱团趋势)"。
输出: output/0506_ou_pair/regime_attribution/ (yearly_attribution.csv/.md + 图)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/root/quant")
OUTPUT_DIR = "output/0506_ou_pair"
OUT = os.path.join(OUTPUT_DIR, "regime_attribution")


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, ".cache", "feat_df_*.parquet")))
    if not paths:
        raise FileNotFoundError("未找到特征缓存 feat_df_*.parquet")
    print(f"读取: {paths[-1]}")
    df = pd.read_parquet(paths[-1], columns=[
        "date", "is_legal", "mu", "X_t", "delta_X", "ret_i_20", "ret_j_20", "z_OU"])
    df["date"] = pd.to_datetime(df["date"])
    d = df[(df["is_legal"]) & (~df["delta_X"].isna())].copy()

    mu = d["mu"].to_numpy(); X = d["X_t"].to_numpy(); dX = d["delta_X"].to_numpy()
    spread_mom = (d["ret_i_20"] - d["ret_j_20"]).to_numpy()   # = X_t - X_{t-20}
    r_rev = np.sign(mu - X) * dX
    r_mom = np.sign(spread_mom) * dX
    widen = (np.abs(X + dX - mu) > np.abs(X - mu)).astype(float)
    d = d.assign(_yr=d["date"].dt.year, _rrev=r_rev, _rmom=r_mom, _widen=widen)

    rows = []
    for yr, g in d.groupby("_yr"):
        rows.append({
            "year": int(yr), "n_signals": len(g), "n_days": g["date"].nunique(),
            "rev_mean(1e4)": round(g["_rrev"].mean() * 1e4, 2),
            "rev_win%": round((g["_rrev"] > 0).mean() * 100, 1),
            "mom_mean(1e4)": round(g["_rmom"].mean() * 1e4, 2),
            "widen%": round(g["_widen"].mean() * 100, 1),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "yearly_attribution.csv"), index=False)

    # ── 选出来的 pair: 每日按 z_OU 取 top20%/min50 (策略 n_top 口径), 逐年反转胜率/均值 ──
    def _topk(g):
        n = len(g); nt = min(n, max(50, int(np.ceil(n * 0.20))))
        return g.nlargest(nt, "z_OU")
    sel = d.groupby("date", group_keys=False).apply(_topk)
    sel_rows = []
    for yr, g in sel.groupby("_yr"):
        sel_rows.append({
            "year": int(yr), "n_selected": len(g),
            "sel_rev_mean(1e4)": round(g["_rrev"].mean() * 1e4, 2),
            "sel_rev_win%": round((g["_rrev"] > 0).mean() * 100, 1),
        })
    sel_summ = pd.DataFrame(sel_rows)
    merged = summ.merge(sel_summ, on="year")
    merged.to_csv(os.path.join(OUT, "yearly_attribution.csv"), index=False)

    md = ["# 逐年 regime 归因 (全 is_legal pair, 单位 1e4=万分之 log-spread)", "",
          "| 年份 | 信号数 | 交易日 | 反转均值 | 反转胜率% | 动量均值 | 走阔% |",
          "|------|------:|-----:|--------:|---------:|--------:|------:|"]
    for _, r in summ.iterrows():
        md.append(f"| {r['year']} | {r['n_signals']:,} | {r['n_days']} | "
                  f"{r['rev_mean(1e4)']:+.2f} | {r['rev_win%']:.1f} | "
                  f"{r['mom_mean(1e4)']:+.2f} | {r['widen%']:.1f} |")
    with open(os.path.join(OUT, "yearly_attribution.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # 图: 反转 vs 动量 年均收益 + 走阔比例
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    yrs = summ["year"].astype(str)
    x = np.arange(len(yrs)); w = 0.38
    ax[0].bar(x - w/2, summ["rev_mean(1e4)"], w, label="反转 sign(μ-X)·ΔX", color="steelblue")
    ax[0].bar(x + w/2, summ["mom_mean(1e4)"], w, label="动量 sign(ΔX_20)·ΔX", color="indianred")
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(yrs)
    ax[0].set_title("逐年 反转 vs 动量 实现收益均值 (1e4)"); ax[0].legend()
    ax[1].plot(x, summ["widen%"], "o-", color="darkorange"); ax[1].set_xticks(x); ax[1].set_xticklabels(yrs)
    ax[1].axhline(50, color="gray", ls="--", lw=0.8, label="50% (无偏)")
    ax[1].set_title("逐年 价差走阔比例 % (越高=越背离)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "regime_attribution.png"), dpi=130)

    print("=== 全 legal pair ===")
    print(summ.to_string(index=False))
    print("\n=== 选出的 pair (每日 z_OU top20%/min50) ===")
    print(sel_summ.to_string(index=False))
    print(f"\n已写入: {OUT}/")


if __name__ == "__main__":
    main()
