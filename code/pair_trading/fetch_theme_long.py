"""抓取概念-个股关联数据, 落地为 Data/all/theme_long.parquet。

数据源:
    tkg_theme_sec_sc_his    (历史归档表, ≤ 2022-12-31)
    tkg_theme_sec_score     (在用表,    ≥ 2023-01-01)

字段映射 (列名标准化为 snake_case):
    STAT_DATE             → date            (datetime64, normalize)
    TICKER_SYMBOL         → stock_code      (str, 6 位, 已是)
    THEME_ID              → theme_id        (int)
    UNMARKET_NORM_SCORE   → unmarket_norm_score  (float)
    NORM_SCORE            → norm_score      (float)
    IS_STRONG_REL         → is_strong_rel   (int 0/1)
    IS_BELLWETHER         → is_bellwether   (int 0/1)

逻辑:
    1. 按月分批拉取 (单次 ~30 秒, 防止 SQL 超时)
    2. 历史段 < 2023-01-01 用 _his 表, ≥ 2023-01-01 用 _score 表
    3. inner join Data/all/price_non_st.pkl 的 (date, stock_code) → 过滤掉:
        - 周末/节假日 (price_non_st 只含交易日)
        - 已退市/未上市股票 (price_non_st 已过滤)
    4. 保存 parquet (snappy 压缩) 到 Data/all/theme_long.parquet
    5. 不动 price_non_st.pkl

运行:
    nohup python3 -u xgbcode/pair_trading/fetch_theme_long.py > /tmp/fetch_theme.log 2>&1 &
"""

import os
import sys
import time

import pandas as pd
from trading_rest_sdk import TradingRestClient

DATE_START = os.environ.get('FETCH_DATE_START', '2022-01-01')
DATE_END = os.environ.get('FETCH_DATE_END', '2026-01-01')      # 左闭右开
SPLIT_DATE = '2023-01-01'    # < 此日用 _his, >= 此日用 _score

PRICE_PATH = '/root/quant/Data/all/price_non_st.pkl'
OUTPUT_PATH = os.environ.get(
    'FETCH_OUTPUT_PATH', '/root/quant/Data/all/theme_long.parquet'
)

client = TradingRestClient(
    api_key='ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0',
    base_url='http://192.168.20.10:8080',
    timeout=6000,
)


def month_buckets(start: str, end: str):
    """生成 [start, end) 区间内所有月份的 (lo, hi) 左闭右开 (字符串日期)。"""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur < e:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt > e:
            nxt = e
        yield cur.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')
        cur = nxt


def fetch_one_month(table: str, lo: str, hi: str) -> pd.DataFrame:
    """从指定表拉一个月数据 (左闭右开)。"""
    q = f'''
        SELECT "STAT_DATE", "TICKER_SYMBOL", "THEME_ID",
               "UNMARKET_NORM_SCORE", "NORM_SCORE", "IS_STRONG_REL", "IS_BELLWETHER"
        FROM {table}
        WHERE "STAT_DATE" >= '{lo}' AND "STAT_DATE" < '{hi}'
    '''
    res = client.execute_sql(sql=q)
    return pd.DataFrame(res['data'])


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """字段标准化 + 类型转换。"""
    if len(df) == 0:
        return df
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(df['STAT_DATE']).dt.normalize()
    # 去时区 (parquet 不喜欢 tz-aware datetime 与 naive 混合)
    if out['date'].dt.tz is not None:
        out['date'] = out['date'].dt.tz_localize(None)
    out['stock_code'] = df['TICKER_SYMBOL'].astype(str).str.zfill(6)
    out['theme_id'] = df['THEME_ID'].astype('int32')
    out['unmarket_norm_score'] = pd.to_numeric(df['UNMARKET_NORM_SCORE'], errors='coerce').astype('float32')
    out['norm_score'] = pd.to_numeric(df['NORM_SCORE'], errors='coerce').astype('float32')
    out['is_strong_rel'] = pd.to_numeric(df['IS_STRONG_REL'], errors='coerce').fillna(0).astype('int8')
    out['is_bellwether'] = pd.to_numeric(df['IS_BELLWETHER'], errors='coerce').fillna(0).astype('int8')
    return out


def main():
    t_start = time.time()
    print(f'\n{"="*60}')
    print(f'fetch_theme_long  时间范围 {DATE_START} ~ {DATE_END}')
    print(f'  分批粒度: 按月')
    print(f'  切换日期: < {SPLIT_DATE} 用 tkg_theme_sec_sc_his, >= 用 tkg_theme_sec_score')
    print(f'  输出: {OUTPUT_PATH}')
    print(f'{"="*60}')

    # 加载 price_non_st 用于 inner join 过滤
    print(f'\n[1] 加载 price_non_st 用于过滤 (date, stock_code)...')
    t0 = time.time()
    price = pd.read_pickle(PRICE_PATH)
    price['date'] = pd.to_datetime(price['date']).dt.normalize()
    price['stock_code'] = price['stock_code'].astype(str).str.zfill(6)
    # 仅保留时间范围内
    price = price[(price['date'] >= pd.Timestamp(DATE_START)) &
                  (price['date'] < pd.Timestamp(DATE_END))]
    valid_keys = price[['date', 'stock_code']].drop_duplicates()
    print(f'  price_non_st 在 [{DATE_START}, {DATE_END}) 内: '
          f'{len(price):,} 行, '
          f'{price["stock_code"].nunique()} 股, '
          f'{price["date"].nunique()} 个交易日, '
          f'耗时 {time.time()-t0:.1f}s')
    del price

    # 按月分批拉取
    chunks = []
    n_months = sum(1 for _ in month_buckets(DATE_START, DATE_END))
    print(f'\n[2] 按月分批拉取 ({n_months} 个月)...')
    t_fetch = time.time()
    for i, (lo, hi) in enumerate(month_buckets(DATE_START, DATE_END), start=1):
        table = 'tkg_theme_sec_sc_his' if lo < SPLIT_DATE else 'tkg_theme_sec_score'
        t_m = time.time()
        df_raw = fetch_one_month(table, lo, hi)
        df_std = standardize(df_raw)
        # inner join valid_keys 过滤掉非交易日 + 不在 price_non_st 的股票
        n_before = len(df_std)
        df_filt = df_std.merge(valid_keys, on=['date', 'stock_code'], how='inner')
        n_after = len(df_filt)
        chunks.append(df_filt)
        elapsed = time.time() - t_m
        print(f'  [{i:>2d}/{n_months}] {lo} ({table[-9:]}): '
              f'{n_before:>8,} → {n_after:>8,} 行 (过滤 {(1-n_after/max(n_before,1))*100:.1f}%), '
              f'{elapsed:.1f}s')
    fetch_elapsed = time.time() - t_fetch
    print(f'  拉取完成, 总耗时 {fetch_elapsed:.0f}s ({fetch_elapsed/60:.1f}min)')

    # 拼接
    print(f'\n[3] 拼接 {len(chunks)} 个月数据...')
    t0 = time.time()
    df_all = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f'  总行数: {len(df_all):,}, '
          f'内存: {df_all.memory_usage(deep=True).sum()/1024/1024:.1f} MB, '
          f'耗时 {time.time()-t0:.1f}s')

    # 排序 (date, stock_code, theme_id)
    print(f'\n[4] 排序 (date, stock_code, theme_id)...')
    t0 = time.time()
    df_all = df_all.sort_values(['date', 'stock_code', 'theme_id'], kind='mergesort').reset_index(drop=True)
    print(f'  排序完成, 耗时 {time.time()-t0:.1f}s')

    # 落地 parquet
    print(f'\n[5] 写入 parquet: {OUTPUT_PATH}')
    t0 = time.time()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_all.to_parquet(OUTPUT_PATH, compression='snappy', index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f'  写入完成, 文件大小: {size_mb:.1f} MB, 耗时 {time.time()-t0:.1f}s')

    # 校验
    print(f'\n[6] 校验:')
    print(f'  date 范围: {df_all["date"].min().date()} ~ {df_all["date"].max().date()}')
    print(f'  stock_code 数: {df_all["stock_code"].nunique()}')
    print(f'  theme_id 数: {df_all["theme_id"].nunique()}')
    print(f'  unmarket_norm_score: 范围 [{df_all["unmarket_norm_score"].min():.4f}, '
          f'{df_all["unmarket_norm_score"].max():.4f}], 均值 {df_all["unmarket_norm_score"].mean():.4f}')
    print(f'  is_strong_rel=1 比例: {(df_all["is_strong_rel"]==1).mean()*100:.1f}%')
    print(f'  is_bellwether=1 比例: {(df_all["is_bellwether"]==1).mean()*100:.1f}%')
    # 抽查: 单股单日的概念数分布
    n_per_day = df_all.groupby(['date', 'stock_code']).size()
    print(f'  单股每日关联概念数: 中位 {n_per_day.median():.0f}, '
          f'min {n_per_day.min()}, max {n_per_day.max()}')

    total_elapsed = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'全部完成, 总耗时 {total_elapsed/60:.1f} min')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
