"""从 zz_5001 拉取 2024/2025 全市场分钟频数据，按月分 pickle 落地。

- 按交易日串行 SQL 拉取，单日拉取后立即转 float32，避免 object 堆积。
- 同一月份所有日期拼成一个 DataFrame，保存到 /root/quant/Data/all/minute/YYYY-MM.pkl。
- stock_code / date 字段对齐 panel_origin.pkl 约定：
    * stock_code: 'SZ.300905' -> '300905' (6 位字符串)
    * date:       trade_date 转 datetime64[ns] 并 normalize
    * trade_time: 转 datetime64[ns] 保留到分钟
- 断点续跑：若 YYYY-MM.pkl 已存在则跳过整月。
"""

import os
import sys
import time
import gc
import warnings
import pandas as pd
import numpy as np
from trading_rest_sdk import TradingRestClient

warnings.filterwarnings('ignore')

OUT_DIR = '/root/quant/Data/all/minute'
os.makedirs(OUT_DIR, exist_ok=True)

API_KEY = 'ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0'
BASE_URL = 'http://192.168.20.10:8080'
START_DATE = '2024-01-01'
END_DATE = '2025-12-31'

# zz_5001 的数值字段（object -> float32）。除以下 ID/时间/计数字段外，其余都是数值。
NON_NUMERIC_COLS = {
    'stock_code', 'trade_date', 'trade_time',
    'update_date', 'update_time', 'zz_date',
}
INT_COLS = {
    'snapshot_count', 'total_ask_volume', 'total_bid_volume',
    'trade_count', 'volume',
}


def make_client():
    return TradingRestClient(api_key=API_KEY, base_url=BASE_URL, timeout=6000)


def list_trade_dates(client):
    """从 zz_500D 拿交易日历（panel_origin 日频数据也来自这张表），避免对 zz_5001 做 DISTINCT。"""
    sql = f"""
        SELECT DISTINCT trade_date
        FROM zz_500D
        WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
        ORDER BY trade_date ASC
    """
    res = client.execute_sql(sql=sql)
    df = pd.DataFrame(res['data'])
    df['trade_date'] = df['trade_date'].astype(str).str.split('T').str[0]
    return df['trade_date'].tolist()


def fetch_one_day(client, date_str):
    """拉取单个交易日分钟数据并转 float32/合适类型。"""
    sql = f"SELECT * FROM zz_5001 WHERE trade_date='{date_str}'"
    res = client.execute_sql(sql=sql)
    df = pd.DataFrame(res['data'])
    if df.empty:
        return df

    # stock_code: 'SZ.300905' / '300905.SZ' -> '300905' 6位
    sc = df['stock_code'].astype(str)
    # 兼容两种可能的前后缀写法，取最后一段非字母
    sc = sc.str.replace(r'^[A-Za-z]+\.', '', regex=True)   # 去前缀 'SZ.'
    sc = sc.str.split('.').str[0]                          # 去后缀 '.SZ'
    df['stock_code'] = sc.str.zfill(6)

    # date: trade_date -> datetime64[ns], normalize
    df['date'] = pd.to_datetime(df['trade_date'].astype(str).str.split('T').str[0]).values.astype('datetime64[ns]')

    # trade_time -> datetime64[ns]（保留到分钟）
    df['trade_time'] = pd.to_datetime(df['trade_time'].astype(str).str.replace(r'\+.*$', '', regex=True))

    # 丢弃原 trade_date 字符串列（用 date 替代）
    df.drop(columns=['trade_date'], inplace=True)

    # 数值列：object -> float32 / int64
    for col in df.columns:
        if col in NON_NUMERIC_COLS or col == 'date':
            continue
        if col in INT_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int64)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    # update_date / update_time / zz_date 保留为字符串即可，占空间不多
    return df


def process_month(client, year, month, dates_in_month):
    out_path = os.path.join(OUT_DIR, f'{year:04d}-{month:02d}.pkl')
    if os.path.exists(out_path):
        print(f'[skip] {out_path} already exists')
        return

    parts = []
    month_t0 = time.time()
    for d in dates_in_month:
        t0 = time.time()
        try:
            df_day = fetch_one_day(client, d)
        except Exception as e:
            print(f'[error] {d}: {e}; retrying once...')
            time.sleep(5)
            df_day = fetch_one_day(client, d)
        t1 = time.time()
        if df_day.empty:
            print(f'  {d}: empty (skip)')
            continue
        parts.append(df_day)
        print(f'  {d}: rows={len(df_day)}, {t1-t0:.1f}s')

    if not parts:
        print(f'[warn] {year}-{month:02d}: no data')
        return

    df_month = pd.concat(parts, ignore_index=True, copy=False)
    del parts
    gc.collect()

    df_month.sort_values(['stock_code', 'trade_time'], kind='mergesort', inplace=True)
    df_month.reset_index(drop=True, inplace=True)

    mem_mb = df_month.memory_usage(deep=True).sum() / 1024 / 1024
    df_month.to_pickle(out_path)
    elapsed = time.time() - month_t0
    print(f'[done] {out_path}  rows={len(df_month):,}  mem={mem_mb:.1f}MB  elapsed={elapsed/60:.1f}min')
    del df_month
    gc.collect()


def main():
    client = make_client()
    all_dates = list_trade_dates(client)
    print(f'共 {len(all_dates)} 个交易日: {all_dates[0]} ~ {all_dates[-1]}')

    # 按 (year, month) 分组
    buckets = {}
    for d in all_dates:
        y = int(d[:4])
        m = int(d[5:7])
        buckets.setdefault((y, m), []).append(d)

    for (y, m), dates_in_month in sorted(buckets.items()):
        print(f'\n=== {y}-{m:02d}: {len(dates_in_month)} 个交易日 ===')
        process_month(client, y, m, dates_in_month)


if __name__ == '__main__':
    main()
