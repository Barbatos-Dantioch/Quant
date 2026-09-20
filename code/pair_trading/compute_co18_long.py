"""
CO-18-long: 在更长区间 (2021-01 ~ 2025-12, 5 年) 上重算 CO-18 因子

配置完全复用 CO-18:
    factor_mode="predict_decay", n=60, lags=(1,), valid_window=20,
    topk1=100, coint_p=0.05, same_industry=True, leader_mv_pct=0.3,
    save_leader=True, pairing_mode="holdout", use_log=True

区别只有 BACKTEST_START: "2024-01-01" -> "2021-01-01"

实现:
    1. 沿用主脚本的数据加载逻辑 (log_panel / industry_map / mv_panel)
    2. import 主脚本并 monkey-patch BACKTEST_START 为 "2021-01-01"
       (build_cointegration_factor 函数内部读这个全局常量过滤输出日期)
    3. 调用 build_cointegration_factor, 把结果 pickle 到
       output/0416_pair_factor/CO-18-long/factor_df.pkl

不跑回测 (回测在分析阶段按行业拆分后分别做)。
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd

# 切到项目根目录 (主脚本 import 时会 os.chdir)
os.chdir("/root/quant")

sys.path.insert(0, "/root/quant/xgbcode/pair_trading")

import run_pair_factor as rpf  # noqa: E402

# ─── 1. 覆盖 BACKTEST_START ───
NEW_START = "2021-01-01"
NEW_END = "2025-12-31"
rpf.BACKTEST_START = NEW_START
rpf.BACKTEST_END = NEW_END
print(f"[monkey-patch] BACKTEST_START = {rpf.BACKTEST_START}, BACKTEST_END = {rpf.BACKTEST_END}")

OUT_DIR = "output/0416_pair_factor/CO-18-long"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "factor_df.pkl")

t0 = time.time()
print(f"\n========== CO-18-long 因子重算 {time.strftime('%Y-%m-%d %H:%M:%S')} ==========")


# ─── 2. 加载 log_panel (与 main 中 tag='non_st_price' 相同) ───
print(f"构造协整面板: log(close) from {rpf.PRICE_NON_ST_PATH}")
df = pd.read_pickle(rpf.PRICE_NON_ST_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[["date", "stock_code", "close_price"]].copy()
df = df[df["close_price"] > 0]
df["log_price"] = np.log(df["close_price"].astype(np.float64))
log_panel = df[["date", "stock_code", "log_price"]]
print(f"  log_panel: {len(log_panel)} 行, "
      f"{log_panel['date'].min().date()} ~ {log_panel['date'].max().date()}, "
      f"{log_panel['stock_code'].nunique()} 只股票")
del df; gc.collect()

# ─── 3. 加载 industry_map ───
print(f"加载行业映射 from {rpf.PRICE_NON_ST_PATH}")
ind_df = pd.read_pickle(rpf.PRICE_NON_ST_PATH)
ind_df = ind_df.dropna(subset=["sw_industry_l1_code"])
ind_map = (ind_df.groupby("stock_code")["sw_industry_l1_code"]
           .first().rename("industry"))
print(f"  industry_map: {len(ind_map)} 只股票")
del ind_df; gc.collect()

# ─── 4. 加载 mv_panel ───
print(f"加载市值面板 from {rpf.PRICE_NON_ST_PATH}")
mv_df = pd.read_pickle(rpf.PRICE_NON_ST_PATH)
mv_df["date"] = pd.to_datetime(mv_df["date"])
mv_df = mv_df[["date", "stock_code", "market_value"]].copy()
mv_df = mv_df[mv_df["market_value"] > 0]
mv_panel = mv_df
print(f"  mv_panel: {len(mv_panel)} 条, "
      f"{mv_panel['stock_code'].nunique()} 只股票, "
      f"{mv_panel['date'].nunique()} 个截面")
del mv_df; gc.collect()

# ─── 5. 调用 build_cointegration_factor (参数完全对齐 CO-18) ───
print(f"\n[开始] 因子计算 (BACKTEST {NEW_START} ~ {NEW_END}) ...")
factor_df = rpf.build_cointegration_factor(
    log_panel,
    n=60,
    lags=(1,),
    calc_every=1,
    topk1=100,
    coint_p=0.05,
    half_life_max=None,
    factor_mode="predict_decay",
    industry_map=ind_map,
    save_leader=True,
    use_log=True,
    pairing_mode="holdout",
    valid_window=20,
    leader_mv_pct=0.3,
    mv_panel=mv_panel,
)

elapsed = time.time() - t0
print(f"\n[完成] 因子 shape: {factor_df.shape}, "
      f"日期 {factor_df['date'].min()} ~ {factor_df['date'].max()}, "
      f"{factor_df['date'].nunique()} 天, 耗时 {elapsed/60:.1f} 分钟")

factor_df.to_pickle(OUT_PATH)
print(f"[保存] {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1024/1024:.1f} MB)")
