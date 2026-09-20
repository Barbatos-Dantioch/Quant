#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抓取概念热度数据 tkg_theme_hot_rank (参照 fetch_barra_cne6.py)
============================================================

数据源 (同 theme_long 的 PG 库):
  tkg_theme_hot_rank : 概念热度 Top-20 排名 (日内多快照, 每快照 20 行 HOT_RANK 1~20)

字段 (原表 → 标准化 snake_case):
  EFFECTIVE_TIME       → effective_time   (datetime, 原 UTC; 晚间快照=北京 18-23 点)
  (派生)                → date             (effective_time 的归一化日期)
  THEME_ID             → theme_id          (int32, 与 theme_long 同 id 空间)
  HOT_RANK             → hot_rank          (int, 1=最热)
  THEME_CHG_PCT        → theme_chg_pct     (float, 概念涨幅%)
  ST_7_PCT_RISE_CNT    → st7_rise_cnt      (int, 成分涨>=7%家数)
  ST_9_PCT_RISE_CNT    → st9_rise_cnt      (int, 成分涨>=9%家数)
  TH_NEWS_REL_1D_CNT   → news_rel_1d       (int, 概念相关新闻数 1日)
  TH_NEWS_REL_2D_CNT   → news_rel_2d       (int, 概念相关新闻数 2日)

设计:
  - 单独落地 (不并入 theme_long), 与 fetch_barra_cne6 同思路。
  - 保留日内原始行 (不做日频聚合); 后续按需聚合 (如 (date,theme) 取 hot_rank 最小值)。
  - 全表约 20 万行, 一次查询即可 (timeout 充足)。

输出: Data/all/theme_hot_rank.parquet
运行: setsid nohup python3 -u xgbcode/pair_trading/fetch_theme_hot_rank.py \
        > /tmp/fetch_theme_hot_rank.log 2>&1 &
"""
import os
import time

import numpy as np
import pandas as pd
from trading_rest_sdk import TradingRestClient

SAVE_PATH = "/root/quant/Data/all/theme_hot_rank.parquet"

client = TradingRestClient(
    api_key="ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0",
    base_url="http://192.168.20.10:8080",
    timeout=6000,
)


def fetch() -> pd.DataFrame:
    q = '''SELECT "EFFECTIVE_TIME", "THEME_ID", "HOT_RANK", "THEME_CHG_PCT",
                  "ST_7_PCT_RISE_CNT", "ST_9_PCT_RISE_CNT",
                  "TH_NEWS_REL_1D_CNT", "TH_NEWS_REL_2D_CNT"
           FROM tkg_theme_hot_rank
           ORDER BY "EFFECTIVE_TIME", "HOT_RANK"'''
    t0 = time.time()
    res = client.execute_sql(sql=q)
    df = pd.DataFrame(res["data"])
    print(f"  原始: {len(df):,} 行, 耗时 {time.time()-t0:.0f}s")
    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    et = pd.to_datetime(df["EFFECTIVE_TIME"])
    if et.dt.tz is not None:
        et = et.dt.tz_localize(None)        # 去时区, 与 theme_long 一致 (原为 UTC 晚间)
    out["effective_time"] = et
    out["date"] = et.dt.normalize()
    out["theme_id"] = df["THEME_ID"].astype("int32")
    out["hot_rank"] = pd.to_numeric(df["HOT_RANK"], errors="coerce").astype("int16")
    out["theme_chg_pct"] = pd.to_numeric(df["THEME_CHG_PCT"], errors="coerce").astype("float32")
    out["st7_rise_cnt"] = pd.to_numeric(df["ST_7_PCT_RISE_CNT"], errors="coerce").fillna(0).astype("int32")
    out["st9_rise_cnt"] = pd.to_numeric(df["ST_9_PCT_RISE_CNT"], errors="coerce").fillna(0).astype("int32")
    out["news_rel_1d"] = pd.to_numeric(df["TH_NEWS_REL_1D_CNT"], errors="coerce").fillna(0).astype("int32")
    out["news_rel_2d"] = pd.to_numeric(df["TH_NEWS_REL_2D_CNT"], errors="coerce").fillna(0).astype("int32")
    out = out.sort_values(["effective_time", "hot_rank"]).reset_index(drop=True)
    return out


def main():
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"fetch_theme_hot_rank  概念热度 Top-20 (日内原始)")
    print(f"  输出: {SAVE_PATH}")
    print(f"{'='*60}\n")

    df_raw = fetch()
    df = standardize(df_raw)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_parquet(SAVE_PATH, compression="snappy", index=False)
    sz = os.path.getsize(SAVE_PATH) / 1024 / 1024

    print(f"\n[校验]")
    print(f"  总行数: {len(df):,}, 文件 {sz:.1f} MB")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  概念数: {df['theme_id'].nunique()}, hot_rank 范围 [{df['hot_rank'].min()}, {df['hot_rank'].max()}]")
    print(f"  日均快照行数: {len(df) / df['date'].nunique():.1f}")
    print(f"  theme_chg_pct 范围 [{df['theme_chg_pct'].min():.2f}, {df['theme_chg_pct'].max():.2f}]")
    print(f"  已写入: {SAVE_PATH}")
    print(f"\n全部完成, 总耗时 {(time.time()-t_start):.0f}s")


if __name__ == "__main__":
    main()
