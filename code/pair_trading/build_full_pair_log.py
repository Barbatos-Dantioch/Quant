#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并完整训练池 pair_log
=======================

把以下三段 pair_log 合并为一份完整训练池 (2021-07-01 ~ 2026-05-13):
  1. 基线  OU-P00-zs-cv5-h5            (2024-01-01 ~ 2026-05-13, 回测期)
  2. 扩展段 _train_extra_2023H2         (2023-07-01 ~ 2023-12-29)
  3. 扩展段 _train_extra_2021H2_2023H1  (2021-07-01 ~ 2023-06-30)

输出: output/0506_ou_pair/pair_log_full.parquet (唯一训练池数据源)

合并成功后, 删除两个中间扩展段目录 (_train_extra_2023H2 /
_train_extra_2021H2_2023H1); 基线 OU-P00 目录保留不动.

其它实验直接读取 pair_log_full.parquet, 按需切片所需时间段.
"""
from __future__ import annotations

import os
import shutil
import sys

import pandas as pd

os.chdir("/root/quant")

OUTPUT_DIR = "output/0506_ou_pair"
BASELINE = "OU-P00-zs-cv5-h5"
EXTRA_SEGMENTS = [          # 合并成功后删除的中间扩展段
    "_train_extra_2021H2_2023H1",
    "_train_extra_2023H2",
]
FULL_PATH = os.path.join(OUTPUT_DIR, "pair_log_full.parquet")


def _load_segment(rel_dir: str) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, rel_dir, "pair_log.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 pair_log 段: {path}")
    df = pd.read_parquet(path)
    print(f"  {rel_dir}: {len(df):,} 行, {df['date'].min()} ~ {df['date'].max()}")
    return df


def main():
    print(f"\n{'='*60}")
    print("合并完整训练池 pair_log")
    print(f"{'='*60}\n")

    # 按时间先后 (早 → 晚) 读取各段
    print("读取各段 ...")
    pl_early = _load_segment(EXTRA_SEGMENTS[0])   # 2021H2~2023H1
    pl_2023h2 = _load_segment(EXTRA_SEGMENTS[1])  # 2023H2
    pl_base = _load_segment(BASELINE)             # 基线 (回测期)
    parts = [pl_early, pl_2023h2, pl_base]

    # 列对齐: 各段 schema 应一致 (同一个 run_ou_pair.process_one_section), 取公共列
    common_cols = list(parts[0].columns)
    for p in parts[1:]:
        common_cols = [c for c in common_cols if c in p.columns]
    parts = [p[common_cols] for p in parts]

    pl_full = pd.concat(parts, ignore_index=True)
    pl_full["date"] = pd.to_datetime(pl_full["date"])
    pl_full["stock_i"] = pl_full["stock_i"].astype(str).str.zfill(6)
    pl_full["stock_j"] = pl_full["stock_j"].astype(str).str.zfill(6)
    pl_full = pl_full.sort_values(["date", "stock_i", "stock_j"]).reset_index(drop=True)
    print(f"\n合并后: {len(pl_full):,} 行, "
          f"{pl_full['date'].min().date()} ~ {pl_full['date'].max().date()}, "
          f"{pl_full['date'].nunique()} 截面")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pl_full.to_parquet(FULL_PATH, index=False)
    print(f"完整训练池已写入: {FULL_PATH}")

    # 校验写出成功后, 删除中间扩展段 (基线 OU-P00 保留)
    if not os.path.exists(FULL_PATH):
        print("[ERROR] 完整文件未写出, 跳过清理")
        return
    print("\n清理中间扩展段 (基线保留) ...")
    for seg in EXTRA_SEGMENTS:
        seg_dir = os.path.join(OUTPUT_DIR, seg)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir)
            print(f"  已删除: {seg_dir}")
        else:
            print(f"  跳过 (不存在): {seg_dir}")
    print(f"\n{'='*60}")
    print("完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
