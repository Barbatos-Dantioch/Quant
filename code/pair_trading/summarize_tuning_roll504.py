#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 ROLL504 调参实验结果
=========================

扫描 output/0506_ou_pair/tuning_roll504/ 下所有实验的 metrics.json (无并发冲突,
比共享 summary.csv 可靠), 提取多空/超额关键指标, 生成干净对比表 summary_all.csv,
并按多空夏普降序打印. 附带根目录的 ROLL504 基线作中心对照点.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

ROOT = "output/0506_ou_pair"
TUNDIR = os.path.join(ROOT, "OU-lgb", "tuning")           # 调参网格根 (维度/短名 两层)
BASELINE_METRICS = os.path.join(TUNDIR, "roll504", "metrics.json")   # 调参中心基线 (单层, 单独读)


def _extract(mp: str) -> dict:
    with open(mp, encoding="utf-8") as f:
        m = json.load(f)
    ls = m.get("longshort", {})
    lse = m.get("ls_excess", {})
    le = m.get("long_excess", {})
    lg = m.get("long", {})
    return {
        "experiment": m.get("experiment", os.path.basename(os.path.dirname(mp))),
        "n_features": m.get("n_features", ""),
        "ls_annual": ls.get("annual_return"),
        "ls_sharpe": ls.get("sharpe_ratio"),
        "ls_calmar": ls.get("calmar_ratio"),
        "ls_mdd": ls.get("max_drawdown"),
        "ls_excess_annual": lse.get("annual_return"),
        "long_annual": lg.get("annual_return"),
        "long_excess_annual": le.get("annual_return"),
    }


def main():
    rows = []
    # 基线 (中心对照点), 若存在
    if os.path.exists(BASELINE_METRICS):
        r = _extract(BASELINE_METRICS)
        r["experiment"] = "ROLL504-baseline(单种子)"
        rows.append(r)

    # 归类后实验在两层目录: tuning_roll504/<分类>/<短名>/metrics.json
    for mp in sorted(glob.glob(os.path.join(TUNDIR, "*", "*", "metrics.json"))):
        r = _extract(mp)
        short = os.path.basename(os.path.dirname(mp))
        cat = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        r["experiment"] = f"{cat}/{short}"
        rows.append(r)

    if not rows:
        print(f"未找到任何 metrics.json (查找路径: {TUNDIR}/*/metrics.json)")
        return

    df = pd.DataFrame(rows).sort_values("ls_sharpe", ascending=False, na_position="last")

    os.makedirs(TUNDIR, exist_ok=True)
    out = os.path.join(TUNDIR, "summary_all.csv")
    df.to_csv(out, index=False)
    print(f"汇总 {len(df)} 个实验 (含基线) -> {out}\n")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
