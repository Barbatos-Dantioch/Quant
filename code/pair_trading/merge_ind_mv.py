"""合并申万一级行业(zz_381) + 市值(zz_222) 到 price_non_st.pkl 和 s_ret_cne6_non_st.pkl。

参考 /root/quant/xgbcode/all/Dataloader.ipynb 中对 zz_222 (Cell 6) 的处理方式。
zz_381 是静态股票基础信息表(每只股票一行), 只取申万一级行业字段。
zz_222 是宽表(行: trade_date, 列: 股票代码, 值: 市值), 用 melt 转长表。

输出: 覆盖原文件, 并保留 .bak 备份。
"""

import os
import shutil

import pandas as pd
from trading_rest_sdk import TradingRestClient

PRICE_PATH = '/root/quant/Data/all/price_non_st.pkl'
SPRET_PATH = '/root/quant/Data/all/s_ret_cne6_non_st.pkl'

# 只保留 2026 年之前的数据, 对应所有时间过滤的上界
DATE_START = '2020-01-01'
DATE_END = '2026-01-01'  # 左闭右开: date < DATE_END

client = TradingRestClient(
    api_key='ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0',
    base_url='http://192.168.20.10:8080',
    timeout=6000,
)


def load_sw_industry_l1() -> pd.DataFrame:
    """从 zz_381 读取申万一级行业, 每只股票取 update_date 最新的一条。

    返回: [stock_code, sw_industry_l1_code, sw_industry_l1_name]
    """
    query = '''
        SELECT stock_code, sw_industry_l1_code, sw_industry_l1_name, update_date
        FROM zz_381
    '''
    result = client.execute_sql(sql=query)
    df = pd.DataFrame(result['data'])
    # stock_code 形如 'SZ.000001' 或 '000001.SZ', split 后取数字段
    parts = df['stock_code'].astype(str).str.split('.')
    df['stock_code'] = parts.apply(
        lambda ps: next((p for p in ps if p.isdigit()), ps[-1])
    ).str.zfill(6)
    df['update_date'] = pd.to_datetime(df['update_date'])
    # 同一股票多条记录时, 取 update_date 最大的那条
    df = df.sort_values(['stock_code', 'update_date'], kind='mergesort')
    df = df.drop_duplicates(subset=['stock_code'], keep='last').reset_index(drop=True)
    df = df[['stock_code', 'sw_industry_l1_code', 'sw_industry_l1_name']]
    print(f'[zz_381] 申万一级行业 (每股取最新一条): {len(df)} 只股票, '
          f'l1_code 非空: {df["sw_industry_l1_code"].notna().sum()}')
    return df


def load_market_value(start_date: str = DATE_START, end_date: str = DATE_END) -> pd.DataFrame:
    """从 zz_222 读取市值, 宽表 melt 为长表 [date, stock_code, market_value]。

    时间范围: [start_date, end_date) 左闭右开。
    """
    query = f'''
        SELECT DISTINCT *
        FROM zz_222
        WHERE trade_date >= '{start_date}' AND trade_date < '{end_date}'
        ORDER BY trade_date ASC
    '''
    result = client.execute_sql(sql=query)
    mv = pd.DataFrame(result['data'])
    mv['date'] = mv['trade_date'].astype(str).str.split('T').str[0]
    mv['date'] = pd.to_datetime(mv['date']).dt.normalize()
    # zz_222 除股票代码列外, 还有 trade_date / update_time 两个辅助列, 都要剔除
    drop_cols = [c for c in ('trade_date', 'update_time') if c in mv.columns]
    mv_long = mv.drop(columns=drop_cols).melt(
        id_vars=['date'], var_name='stock_code', value_name='market_value'
    )
    # 列名形如 'SH.600000' 或 'SZ.000001', 取其中的数字段, 补齐 6 位
    parts = mv_long['stock_code'].astype(str).str.split('.')
    mv_long['stock_code'] = parts.apply(
        lambda ps: next((p for p in ps if p.isdigit()), ps[-1])
    ).str.zfill(6)
    # 若 stock_code 非 6 位数字 (例如 update_time 这类辅助列名残留), 直接丢弃
    valid_mask = mv_long['stock_code'].str.fullmatch(r'\d{6}')
    if (~valid_mask).any():
        print(f'  [过滤] 丢弃非股票代码 melt 列: '
              f'{mv_long.loc[~valid_mask, "stock_code"].unique()[:5]} ...')
    mv_long = mv_long[valid_mask].reset_index(drop=True)
    mv_long['market_value'] = pd.to_numeric(mv_long['market_value'], errors='coerce')
    # 丢弃 market_value 为 NaN 的行, 减小体积 (合并时 left join 缺失仍会得到 NaN)
    mv_long = mv_long.dropna(subset=['market_value']).reset_index(drop=True)
    print(f'[zz_222] 市值长表: {len(mv_long)} 行, '
          f'日期 {mv_long["date"].min().date()} ~ {mv_long["date"].max().date()}, '
          f'股票 {mv_long["stock_code"].nunique()} 只')
    return mv_long


def merge_and_overwrite(pkl_path: str, ind_df: pd.DataFrame, mv_df: pd.DataFrame) -> None:
    """对 pkl_path 做合并并原地覆盖, 保留 .bak 备份。"""
    print(f'\n===== 处理 {pkl_path} =====')
    df = pd.read_pickle(pkl_path)
    n_raw = len(df)
    cols_before = df.columns.tolist()
    print(f'读取: {n_raw} 行, 列 {cols_before}')

    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # 截断到 2026 年之前 (date < DATE_END)
    df = df[df['date'] < pd.Timestamp(DATE_END)].reset_index(drop=True)
    n_before = len(df)
    print(f'截断 date < {DATE_END}: {n_raw} -> {n_before} 行')

    # 幂等性: 若之前写过这些列, 先去除, 避免 merge 产生 _x/_y 后缀冲突
    new_cols = ['sw_industry_l1_code', 'sw_industry_l1_name', 'market_value']
    existing_to_drop = [c for c in new_cols if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
        print(f'已存在的新列 {existing_to_drop} 先被剔除, 再重新合并')

    # 合并申万一级行业 (静态, 仅按 stock_code)
    df = df.merge(ind_df, on='stock_code', how='left')
    # 合并市值 (按 [stock_code, date])
    df = df.merge(mv_df, on=['stock_code', 'date'], how='left')

    assert len(df) == n_before, f'合并后行数不一致: {n_before} -> {len(df)}'

    df = df.sort_values(['stock_code', 'date'], kind='mergesort').reset_index(drop=True)

    miss_ind = df['sw_industry_l1_code'].isna().sum()
    miss_mv = df['market_value'].isna().sum()
    print(f'合并后: {len(df)} 行, 新增列 {new_cols}')
    print(f'  申万一级行业缺失: {miss_ind} ({miss_ind / len(df) * 100:.2f}%)')
    print(f'  市值缺失:       {miss_mv} ({miss_mv / len(df) * 100:.2f}%)')

    # 备份 + 覆盖写入
    bak_path = pkl_path + '.bak'
    if not os.path.exists(bak_path):
        shutil.copy2(pkl_path, bak_path)
        print(f'已备份: {bak_path}')
    else:
        print(f'备份已存在, 跳过备份: {bak_path}')
    df.to_pickle(pkl_path)
    size_mb = os.path.getsize(pkl_path) / 1024 / 1024
    print(f'已覆盖保存: {pkl_path} ({size_mb:.1f} MB)')


def main() -> None:
    ind_df = load_sw_industry_l1()
    mv_df = load_market_value(start_date=DATE_START, end_date=DATE_END)

    for pkl_path in (PRICE_PATH, SPRET_PATH):
        merge_and_overwrite(pkl_path, ind_df, mv_df)

    print('\n全部完成。')


if __name__ == '__main__':
    main()
