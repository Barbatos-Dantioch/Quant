"""从 zz_200 拉取 2025 年 is_st 标记并缓存到本地。

参照 xgbcode/all/Dataloader.ipynb 前 20 行的逻辑：
- 通过 TradingRestClient 查询 zz_200 表
- 字段：stock_code, trade_date, is_st
- 规范化后落地为 /root/quant/Data/Tick/is_st_2025.pkl

用途：因子面板构建完成后，按 (stock_code, date) merge 剔除 is_st==1 的行。
"""
import os
import pandas as pd
from trading_rest_sdk import TradingRestClient

OUT_PATH = '/root/quant/Data/Tick/is_st_2025.pkl'
API_KEY = 'ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0'
BASE_URL = 'http://192.168.20.10:8080'


def fetch_is_st(start='2025-01-01', end='2025-12-31'):
    client = TradingRestClient(api_key=API_KEY, base_url=BASE_URL, timeout=6000)
    try:
        sql = f"""
            SELECT DISTINCT stock_code, trade_date, is_st
            FROM zz_200
            WHERE trade_date >= '{start}' AND trade_date <= '{end}'
            ORDER BY stock_code, trade_date ASC
        """
        res = client.execute_sql(sql=sql)
    finally:
        client.close()

    df = pd.DataFrame(res['data'])
    df.rename(columns={'trade_date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'].astype(str).str.split('T').str[0]).dt.normalize()
    df['stock_code'] = df['stock_code'].astype(str).str.split('.').str[-1]
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f'拉取 is_st 数据 (zz_200, 2025 年)...', flush=True)
    df = fetch_is_st()
    print(f'  行数: {len(df)}, 股票数: {df["stock_code"].nunique()}, '
          f'日期范围: {df["date"].min()} ~ {df["date"].max()}', flush=True)
    print(f'  is_st 分布: {df["is_st"].value_counts(dropna=False).to_dict()}', flush=True)
    df.to_pickle(OUT_PATH)
    print(f'  已保存 -> {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
