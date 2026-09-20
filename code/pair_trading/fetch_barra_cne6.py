#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抓取 Barra CNE6 (申万2021 行业) 因子数据
==========================================

数据源 (PostgreSQL, PG 字段名大小写敏感, 需双引号):
  dy1d_exposure_cne6_sw21   : 个股×日 的因子暴露 (52 因子: 20 风格 + 31 行业 + COUNTRY)
  dy1d_covariance_cne6_sw21 : 每日 因子×因子 协方差矩阵 (每行一个因子, 52 行/天)

用途: 对 output/0506_ou_pair 配对回测结果做 Barra 风险归因。

输出 (Data/all/):
  barra_exposure_cne6_sw21.pkl : 长表 [date, stock_code, <52 因子>]
  barra_cov_cne6_sw21.pkl      : 长表 [date, factor_id, factor_name, <52 因子>]

设计要点:
  - 52 个因子顺序由 covariance 表的 FACTOR_NAME (按 FACTOR_ID) 在运行时确定,
    暴露表只取这 52 列, 保证两表因子集严格对齐 (剔除 SW21 下已废弃的老行业空列)。
  - stock_code 统一 zfill(6), 与回测面板的股票代码格式一致。
  - 暴露表数据量大 (区间内约数百万行), 按年分块抓取后拼接, 避免单次请求超时/超内存。
"""
import os
import time
import numpy as np
import pandas as pd
from trading_rest_sdk import TradingRestClient

# 抓取区间 (覆盖 0506_ou_pair 所有实验的训练+测试范围; 可按需调整)
START_DATE = "20200101"
END_DATE = "20260610"

SAVE_DIR = "/root/quant/Data/all"
EXPOSURE_PATH = os.path.join(SAVE_DIR, "barra_exposure_cne6_sw21.pkl")
COV_PATH = os.path.join(SAVE_DIR, "barra_cov_cne6_sw21.pkl")

client = TradingRestClient(
    api_key='ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0',
    base_url="http://192.168.20.10:8080",
    timeout=6000,
)


def _q(col: str) -> str:
    """PG 列名加双引号 (大小写敏感)。"""
    return f'"{col}"'


def get_factor_list() -> list:
    """从协方差表取规范因子顺序 (按 FACTOR_ID 升序的 FACTOR_NAME), 共 52 个。"""
    q = '''SELECT DISTINCT "FACTOR_ID", "FACTOR_NAME"
           FROM dy1d_covariance_cne6_sw21
           WHERE "TRADE_DATE" = (SELECT MAX("TRADE_DATE") FROM dy1d_covariance_cne6_sw21)
           ORDER BY "FACTOR_ID"'''
    df = pd.DataFrame(client.execute_sql(sql=q, datasource="postgresql", timeout=600)['data'])
    factors = df['FACTOR_NAME'].tolist()
    print(f"因子集: {len(factors)} 个 (20 风格 + 31 行业 + COUNTRY)")
    return factors


def fetch_exposure(factors: list) -> pd.DataFrame:
    """按年分块抓取个股因子暴露, 拼接为长表。"""
    print(f"\n===== 抓取因子暴露 {START_DATE} ~ {END_DATE} =====")
    sel_cols = ', '.join([_q("TICKER_SYMBOL"), _q("TRADE_DATE")] + [_q(c) for c in factors])
    start_year = int(START_DATE[:4])
    end_year = int(END_DATE[:4])

    parts = []
    for yr in range(start_year, end_year + 1):
        lo = max(f"{yr}0101", START_DATE)
        hi = min(f"{yr}1231", END_DATE)
        q = f'''SELECT {sel_cols} FROM dy1d_exposure_cne6_sw21
                WHERE "TRADE_DATE" >= '{lo}' AND "TRADE_DATE" <= '{hi}'
                ORDER BY "TRADE_DATE", "TICKER_SYMBOL"'''
        t0 = time.time()
        chunk = pd.DataFrame(client.execute_sql(sql=q, datasource="postgresql", timeout=1800)['data'])
        parts.append(chunk)
        print(f"  {yr}: {len(chunk):>9,} 行, 耗时 {time.time()-t0:.0f}s")

    df = pd.concat(parts, ignore_index=True)
    df.rename(columns={'TICKER_SYMBOL': 'stock_code', 'TRADE_DATE': 'date'}, inplace=True)
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df[factors] = df[factors].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    return df


def fetch_covariance(factors: list) -> pd.DataFrame:
    """抓取每日因子协方差矩阵 (长表: 每行一个因子)。"""
    print(f"\n===== 抓取因子协方差 {START_DATE} ~ {END_DATE} =====")
    sel_cols = ', '.join([_q("TRADE_DATE"), _q("FACTOR_ID"), _q("FACTOR_NAME")] + [_q(c) for c in factors])
    q = f'''SELECT {sel_cols} FROM dy1d_covariance_cne6_sw21
            WHERE "TRADE_DATE" >= '{START_DATE}' AND "TRADE_DATE" <= '{END_DATE}'
            ORDER BY "TRADE_DATE", "FACTOR_ID"'''
    t0 = time.time()
    df = pd.DataFrame(client.execute_sql(sql=q, datasource="postgresql", timeout=1800)['data'])
    print(f"  协方差: {len(df):,} 行, 耗时 {time.time()-t0:.0f}s")

    df.rename(columns={'TRADE_DATE': 'date', 'FACTOR_ID': 'factor_id', 'FACTOR_NAME': 'factor_name'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['factor_id'] = pd.to_numeric(df['factor_id'], errors='coerce').astype(int)
    df[factors] = df[factors].apply(pd.to_numeric, errors='coerce').astype(np.float64)
    df = df.sort_values(['date', 'factor_id']).reset_index(drop=True)
    return df


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    factors = get_factor_list()

    # ── 因子暴露 ──
    exp = fetch_exposure(factors)
    exp.to_pickle(EXPOSURE_PATH)
    print(f"\n[暴露] 已保存: {EXPOSURE_PATH} ({os.path.getsize(EXPOSURE_PATH)/1024/1024:.1f} MB)")
    print(f"  行数={len(exp):,}, 股票={exp['stock_code'].nunique()}, "
          f"交易日={exp['date'].nunique()}, 范围={exp['date'].min().date()}~{exp['date'].max().date()}")
    na_ratio = exp[factors].isna().mean()
    print(f"  因子 NaN 比例: 最大 {na_ratio.max():.4f} ({na_ratio.idxmax()}), 均值 {na_ratio.mean():.4f}")

    # ── 因子协方差 ──
    cov = fetch_covariance(factors)
    cov.to_pickle(COV_PATH)
    print(f"\n[协方差] 已保存: {COV_PATH} ({os.path.getsize(COV_PATH)/1024/1024:.1f} MB)")
    print(f"  行数={len(cov):,}, 交易日={cov['date'].nunique()}, "
          f"每日因子数={cov.groupby('date').size().mode().iloc[0]}, "
          f"范围={cov['date'].min().date()}~{cov['date'].max().date()}")

    # 协方差对称性抽查 (最新一天)
    last = cov['date'].max()
    m = cov[cov['date'] == last].set_index('factor_name')[factors]
    m = m.reindex(index=factors, columns=factors)
    asym = np.nanmax(np.abs(m.values - m.values.T))
    print(f"  {last.date()} 协方差矩阵最大非对称度: {asym:.4f} (应接近 0)")
    print(f"  {last.date()} 对角线(方差)是否全为正: {bool((np.diag(m.values) > 0).all())}")


if __name__ == "__main__":
    main()
