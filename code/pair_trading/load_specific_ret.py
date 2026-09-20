import os
import pandas as pd
from trading_rest_sdk import TradingRestClient

client = TradingRestClient(
    api_key='ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0',
    base_url="http://192.168.20.10:8080",
    timeout=6000
)

# PG 表字段名大小写敏感，需要加双引号
query = '''
    SELECT "TICKER_SYMBOL", "TRADE_DATE", "SPRET"
    FROM dy1d_specific_ret_cne6_sw21
    WHERE "TRADE_DATE" >= '20200101' AND "TRADE_DATE" <= '20251231'
    ORDER BY "TRADE_DATE", "TICKER_SYMBOL"
'''

result = client.execute_sql(sql=query, datasource="postgresql", timeout=600)
df = pd.DataFrame(result['data'])

df['SPRET'] = df['SPRET'].astype(float)
df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'], format='%Y%m%d')
df.rename(columns={'TICKER_SYMBOL': 'stock_code', 'TRADE_DATE': 'date', 'SPRET': 'spret'}, inplace=True)
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"数据量: {len(df)}")
print(f"股票数: {df['stock_code'].nunique()}")
print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"交易日数: {df['date'].nunique()}")
print(f"\n前5行:")
print(df.head())
print(f"\nspret 统计:")
print(df['spret'].describe())

# 数据质量检查
import numpy as np
print("\n===== 数据质量检查 =====")
print(f"NaN 数量:  stock_code={df['stock_code'].isna().sum()}, date={df['date'].isna().sum()}, spret={df['spret'].isna().sum()}")
print(f"Inf 数量:  spret={np.isinf(df['spret']).sum()}")
print(f"重复行数:  {df.duplicated(subset=['stock_code', 'date']).sum()}")
print(f"spret=0 数量: {(df['spret'] == 0).sum()}")
print(f"spret 极端值: min={df['spret'].min():.4f}, max={df['spret'].max():.4f}")
print(f"|spret|>50 数量: {(df['spret'].abs() > 50).sum()}")
print(f"|spret|>20 数量: {(df['spret'].abs() > 20).sum()}")

# 保存
save_path = '/root/quant/Data/all/specific_ret_cne6_sw21.pkl'
df.to_pickle(save_path)
print(f"\n已保存至 {save_path}，大小: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

# =====================================================================
# 剔除 ST 股票，生成 non_st 版本
# =====================================================================
print("\n===== 剔除 ST 股票 =====")

# 加载 ST 标记
query_st = '''
    SELECT DISTINCT stock_code, trade_date, is_st
    FROM zz_200
    WHERE trade_date >= '2020-01-01'
    ORDER BY stock_code, trade_date ASC
'''
result_st = client.execute_sql(sql=query_st)
is_st = pd.DataFrame(result_st['data'])
is_st.rename(columns={'trade_date': 'date'}, inplace=True)
is_st['date'] = is_st['date'].astype(str).str.split('T').str[0]
is_st['date'] = pd.to_datetime(is_st['date']).dt.normalize()
is_st['stock_code'] = is_st['stock_code'].str.split('.').str[1]
is_st['is_st'] = pd.to_numeric(is_st['is_st'], errors='coerce').fillna(1).astype(int)
is_st = is_st.sort_values(['stock_code', 'date']).reset_index(drop=True)
print(f"ST 标记: {len(is_st)} 行, ST=1 的记录: {(is_st['is_st'] == 1).sum()}")

# 只保留 is_st == 0 的 (stock_code, date) 对
non_st = is_st[is_st['is_st'] == 0][['stock_code', 'date']].copy()

# 剔除 panel_trade 中的 ST
pt = pd.read_pickle('/root/quant/Data/all/panel_trade.pkl')
pt['date'] = pd.to_datetime(pt['date'])
pt['stock_code'] = pt['stock_code'].astype(str).str.zfill(6)
n_before = len(pt)
pt_non_st = pt.merge(non_st, on=['stock_code', 'date'], how='inner')
pt_non_st = pt_non_st.sort_values(['stock_code', 'date']).reset_index(drop=True)
print(f"panel_trade: {n_before} → {len(pt_non_st)} (剔除 {n_before - len(pt_non_st)} 条)")

save_pt = '/root/quant/Data/all/price_non_st.pkl'
pt_non_st.to_pickle(save_pt)
print(f"已保存至 {save_pt}，大小: {os.path.getsize(save_pt) / 1024 / 1024:.1f} MB")
del pt, pt_non_st

# 剔除 specific_ret 中的 ST
n_before = len(df)
df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
df_non_st = df.merge(non_st, on=['stock_code', 'date'], how='inner')
df_non_st = df_non_st.sort_values(['stock_code', 'date']).reset_index(drop=True)
print(f"specific_ret: {n_before} → {len(df_non_st)} (剔除 {n_before - len(df_non_st)} 条)")

save_sr = '/root/quant/Data/all/s_ret_cne6_non_st.pkl'
df_non_st.to_pickle(save_sr)
print(f"已保存至 {save_sr}，大小: {os.path.getsize(save_sr) / 1024 / 1024:.1f} MB")
