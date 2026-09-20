"""从交易所逐笔数据库表拉取全市场逐笔数据，按 表名/交易日.parquet 落地。

数据来源（中国两大交易所逐笔，字段为中文，均在 ClickHouse 库 clickhouse_data）：
- 上交所(SH)：zz_39_1 逐笔委托、zz_39_2 逐笔成交、zz_39_3 逐笔撤单
- 深交所(SZ)：zz_05 逐笔委托、zz_06_1 逐笔成交、zz_06_2 逐笔撤单

落地：/root/quant/Data/Tick/<表名>/<YYYY-MM-DD>.parquet

取数与性能（经实测确定）：
- 瓶颈是服务端单连接限速（~5.6MB/s），多连接可近线性聚合；8 并发约 38MB/s 为稳健甜点。
- 用 execute_sql(format="csv")：服务端直接返回 CSV，绕开 JSON 解析，传输更省、内存更低。
- 多线程并发（WORKERS，默认 10，可用命令行第 1 个参数覆盖）。
- 交易日历直接取自逐笔表 zz_39_3（同库，DISTINCT trade_date 仅 ~4s）。

健壮性：
- 断点续跑：目标 parquet 已存在则跳过。
- 原子落地：先写 .tmp 再 os.replace，避免中断产生半文件。
- 每任务独立临时下载目录，规避 SDK CSV 文件名（秒级时间戳）并发重名覆盖。
- 单任务失败自动重试一次，失败不影响其余任务。

字段：核心字段 + 接收时间，丢弃 kafka_*/zz_date/*_说明 等冗余列。
"""

import os
import sys
import time
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from trading_rest_sdk import TradingRestClient

warnings.filterwarnings('ignore')

OUT_ROOT = '/root/quant/Data/Tick'
TMP_ROOT = '/tmp/tick_dl'

API_KEY = 'ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0'
BASE_URL = 'http://192.168.20.10:8080'
START_DATE = '2025-01-01'
END_DATE = '2026-06-24'

# 逐笔 6 表所在 ClickHouse 库；交易日历直接取自其中的 zz_39_3
DS_TICK = 'clickhouse_data'
CALENDAR_TABLE = 'zz_39_3'

WORKERS = 10            # 并发连接数（实测：瓶颈在服务端CSV生成能力，提并发收益小且增内存/触发拒绝，10为稳定甜点）
TIMEOUT = 6000          # 单次请求超时（秒），大表 CSV 导出+下载较久

PARQUET_ENGINE = 'fastparquet'
PARQUET_COMPRESSION = 'zstd'

# 每张表保留列及类型分组：
#   str -> 原样字符串（证券代码、各类时间、方向/类型代码字符）
#   f32 -> 价格类，float32
#   f64 -> 金额类，float64
#   i64 -> 数量/各类 ID/序号/数值代码，int64
_SH_39 = {
    'str': ['证券代码', '订单或成交时间', '接收时间', '标识', '类型'],
    'f32': ['价格'],
    'f64': ['成交金额'],
    'i64': ['数量', '买方订单号', '卖方订单号', '序号', '逐笔序号'],
}
_SZ_06 = {
    'str': ['证券代码', '委托时间', '接收时间'],
    'f32': ['委托价格'],
    'f64': [],
    'i64': ['买方委托索引', '卖方委托索引', '委托数量', '成交类别',
            '序号', '消息记录号', '频道代码'],
}
TABLE_CONFIG = {
    'zz_39_1': _SH_39,
    'zz_39_2': _SH_39,
    'zz_39_3': _SH_39,
    'zz_05': {
        'str': ['证券代码', '委托时间', '接收时间'],
        'f32': ['委托价格'],
        'f64': [],
        'i64': ['买卖方向', '订单类别', '委托数量', '序号', '消息记录号', '频道代码'],
    },
    'zz_06_1': _SZ_06,
    'zz_06_2': _SZ_06,
}

# 大表单日数据量过大（zz_05 单日近 1 亿行），服务端生成整日 CSV 会超过 ~600s 超时。
# 故对大表按时间字段分段拉取（每段实测 ~250s，远低于超时），合并后写一个 parquet。
# 小表（zz_39_3/zz_06_2，单日 <2500万行，<350s）单请求即可，不分段。
SEGMENTED = {'zz_39_1', 'zz_39_2', 'zz_05', 'zz_06_1'}
TIME_COL = {
    'zz_39_1': '订单或成交时间', 'zz_39_2': '订单或成交时间', 'zz_39_3': '订单或成交时间',
    'zz_05': '委托时间', 'zz_06_1': '委托时间', 'zz_06_2': '委托时间',
}
# 时间分段边界（左闭右开，'HH:MM:SS' 字典序==时间序）：早盘密集处细分，
# 首段从 00:00:00 含集合竞价，末段到 99:99:99 兜住尾盘/盘后，保证无遗漏。
TIME_SEGMENTS = ['00:00:00', '09:31:00', '09:40:00', '09:55:00', '10:20:00',
                 '11:00:00', '11:30:00', '13:40:00', '14:30:00', '99:99:99']


def make_client(download_dir):
    return TradingRestClient(api_key=API_KEY, base_url=BASE_URL,
                             timeout=TIMEOUT, download_dir=download_dir)


def list_trade_dates(client):
    """交易日历取自逐笔表 zz_39_3（同库 clickhouse_data，DISTINCT trade_date 仅 ~4s）。"""
    sql = f"""
        SELECT DISTINCT trade_date
        FROM {CALENDAR_TABLE}
        WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
        ORDER BY trade_date ASC
    """
    res = client.execute_sql(sql=sql, datasource=DS_TICK)
    df = pd.DataFrame(res['data'])
    df['trade_date'] = df['trade_date'].astype(str).str.split('T').str[0]
    return df['trade_date'].tolist()


def _cast_types(df, cfg):
    """按列类型分组转换（段级调用，数值化后再 concat 可降低内存峰值）。"""
    for c in cfg['f32']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
    for c in cfg['f64']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float64)
    for c in cfg['i64']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.int64)
    return df


def _fetch_segment(client, table, cfg, date_str, col_sql, lo, hi):
    """拉取单表单日某时间段(可为全天) CSV，读入并转类型；读完即删临时 CSV。"""
    where = f"trade_date='{date_str}'"
    if lo is not None:
        tcol = TIME_COL[table]
        where += f" AND `{tcol}`>='{lo}' AND `{tcol}`<'{hi}'"
    sql = f"SELECT {col_sql} FROM {table} WHERE {where}"
    csv_path = client.execute_sql(sql=sql, format='csv',
                                  datasource=DS_TICK, timeout=TIMEOUT)
    try:
        # str 列显式指定，避免被推断成数值丢失前导零/格式
        df = pd.read_csv(csv_path, dtype={c: 'str' for c in cfg['str']})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    finally:
        try:
            os.remove(csv_path)
        except OSError:
            pass
    if df.empty:
        return df
    return _cast_types(df, cfg)


def _fetch_csv_to_df(client, table, cfg, date_str):
    """拉取单表单日全部数据：大表按时间分段拉取后合并，小表单请求。"""
    cols = cfg['str'] + cfg['f32'] + cfg['f64'] + cfg['i64']
    col_sql = ', '.join(f'`{c}`' for c in cols)

    if table in SEGMENTED:
        parts = []
        for i in range(len(TIME_SEGMENTS) - 1):
            d = _fetch_segment(client, table, cfg, date_str, col_sql,
                               TIME_SEGMENTS[i], TIME_SEGMENTS[i + 1])
            if not d.empty:
                parts.append(d)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        df = _fetch_segment(client, table, cfg, date_str, col_sql, None, None)

    if df.empty:
        return df
    df['date'] = np.datetime64(date_str)
    return df


def fetch_one(task):
    """处理单个 (表, 交易日) 任务：断点续跑 + 拉取 + 写 parquet。线程安全（独立 client/目录）。"""
    table, date_str = task
    cfg = TABLE_CONFIG[table]

    out_dir = os.path.join(OUT_ROOT, table)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{date_str}.parquet')
    if os.path.exists(out_path):
        return f'[skip] {table} {date_str}'

    tmp_dl = os.path.join(TMP_ROOT, f'{table}_{date_str}')
    os.makedirs(tmp_dl, exist_ok=True)
    client = make_client(tmp_dl)

    t0 = time.time()
    try:
        try:
            df = _fetch_csv_to_df(client, table, cfg, date_str)
        except Exception as e:
            time.sleep(5)
            df = _fetch_csv_to_df(client, table, cfg, date_str)
            _ = e

        if df.empty:
            return f'[empty] {table} {date_str}'

        tmp_pq = out_path + '.tmp'
        df.to_parquet(tmp_pq, engine=PARQUET_ENGINE,
                      compression=PARQUET_COMPRESSION, index=False)
        os.replace(tmp_pq, out_path)
        sz = os.path.getsize(out_path) / 1024 / 1024
        return f'[done] {table} {date_str} rows={len(df):,} {sz:.0f}MB {time.time()-t0:.0f}s'
    except Exception as e:
        return f'[ERROR] {table} {date_str}: {str(e)[:120]}'
    finally:
        client.close()
        shutil.rmtree(tmp_dl, ignore_errors=True)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else WORKERS
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)

    client = make_client(TMP_ROOT)
    dates = list_trade_dates(client)
    client.close()

    # 日期外层、表内层：并发批次自然混合大小表，内存更均衡
    tasks = [(t, d) for d in dates for t in TABLE_CONFIG]
    total = len(tasks)
    print(f'{len(dates)} 个交易日 × {len(TABLE_CONFIG)} 表 = {total} 个任务, '
          f'workers={workers}, {dates[0]} ~ {dates[-1]}', flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            print(f'[{done}/{total}] {fut.result()}', flush=True)


if __name__ == '__main__':
    main()
