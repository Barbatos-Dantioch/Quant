#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并 243 个日频 tick 因子 pkl，并按 is_st 剔除 ST 股票。

输入：
    /root/quant/Data/Tick/factors/daily/{date}.pkl  (243 个文件，每个 31 列)
    /root/quant/Data/Tick/is_st_2025.pkl            (date, stock_code, is_st)

输出：
    /root/quant/Data/Tick/factors/tick_factor_panel_raw.pkl  (剔 ST 前，全部行)
    /root/quant/Data/Tick/factors/tick_factor_panel.pkl      (剔 ST 后，用于选股)
"""
import os
import glob
import pandas as pd

DAILY_DIR = '/root/quant/Data/Tick/factors/daily'
IS_ST_PATH = '/root/quant/Data/Tick/is_st_2025.pkl'
RAW_OUT_PATH = '/root/quant/Data/Tick/factors/tick_factor_panel_raw.pkl'
FINAL_OUT_PATH = '/root/quant/Data/Tick/factors/tick_factor_panel.pkl'


def main():
    files = sorted(glob.glob(os.path.join(DAILY_DIR, '*.pkl')))
    print(f'发现 {len(files)} 个日频因子文件', flush=True)

    dfs = []
    for fp in files:
        df = pd.read_pickle(fp)
        if '_error' in df.columns:
            n_err = df['_error'].notna().sum()
            if n_err > 0:
                print(f'  [警告] {os.path.basename(fp)} 有 {n_err} 行计算出错', flush=True)
            df = df.drop(columns=['_error'])
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'])
    raw['stock_code'] = raw['stock_code'].astype(str)
    print(f'合并完成: {len(raw)} 行, {raw["date"].nunique()} 个交易日, '
          f'{raw["stock_code"].nunique()} 只股票', flush=True)

    os.makedirs(os.path.dirname(RAW_OUT_PATH), exist_ok=True)
    raw.to_pickle(RAW_OUT_PATH)
    print(f'已保存原始面板(剔ST前) -> {RAW_OUT_PATH}', flush=True)

    is_st = pd.read_pickle(IS_ST_PATH)
    is_st['date'] = pd.to_datetime(is_st['date'])
    is_st['stock_code'] = is_st['stock_code'].astype(str)

    merged = raw.merge(is_st[['date', 'stock_code', 'is_st']], on=['date', 'stock_code'], how='left')
    n_missing_st = merged['is_st'].isna().sum()
    if n_missing_st > 0:
        print(f'  [警告] {n_missing_st} 行在 is_st 表中找不到匹配，按非ST处理(保留)', flush=True)
    merged['is_st'] = merged['is_st'].fillna(0)

    n_before = len(merged)
    final = merged.loc[merged['is_st'] == 0].drop(columns=['is_st']).reset_index(drop=True)
    n_st_removed = n_before - len(final)
    print(f'剔除 ST: {n_st_removed} 行 ({n_st_removed / n_before * 100:.2f}%), '
          f'剩余 {len(final)} 行, {final["stock_code"].nunique()} 只股票', flush=True)

    final.to_pickle(FINAL_OUT_PATH)
    print(f'已保存最终因子面板(剔ST后) -> {FINAL_OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
