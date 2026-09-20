"""增量补丁: 拉取概念数据的新增段并 concat 到现有 theme_long.parquet.

策略 (与"完整重拉 60-90 分钟"对照, 增量预估 10-15 分钟):
    1. 加载现有 theme_long.parquet, 取出 date 最大值 (作为增量起点)
    2. 复用 fetch_theme_long.py 的拉取/标准化/过滤函数, 拉
       [last_date + 1 day, FETCH_DATE_END) 的新数据
    3. 对齐 dtype (与旧 schema 严格一致)
    4. concat 旧 + 新 → 排序 + 去重 (保险起见, drop_duplicates by [date, stock_code, theme_id])
    5. 备份旧文件到 .bak (若已存在则跳过备份)
    6. 写回 theme_long.parquet

环境变量:
    FETCH_DATE_END    增量终点 (左闭右开, 默认 2026-05-14)
    FETCH_FORCE_FROM  覆盖默认起点 (例如想从某日重拉, 慎用)
    FETCH_OUTPUT_PATH 默认 /root/quant/Data/all/theme_long.parquet
    FETCH_SMOKE       若为 "1", 只拉增量段第 1 个月, 不写盘

运行:
    setsid nohup python3 -u xgbcode/pair_trading/fetch_theme_long_inc.py \
        > /tmp/fetch_theme_long_inc.log 2>&1 &
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import pandas as pd

# 复用 fetch_theme_long.py 的函数
sys.path.insert(0, "/root/quant/xgbcode/pair_trading")
from fetch_theme_long import (  # noqa: E402
    SPLIT_DATE,
    PRICE_PATH,
    client,
    month_buckets,
    fetch_one_month,
    standardize,
)


DATE_END = os.environ.get("FETCH_DATE_END", "2026-05-14")  # 左闭右开
FORCE_FROM = os.environ.get("FETCH_FORCE_FROM")  # 例如 "2026-01-01"
SMOKE = os.environ.get("FETCH_SMOKE") == "1"
OUTPUT_PATH = os.environ.get(
    "FETCH_OUTPUT_PATH", "/root/quant/Data/all/theme_long.parquet"
)


# 旧 theme_long.parquet 的 dtype (增量必须对齐)
EXPECTED_DTYPES = {
    "date": "datetime64[ns]",
    "stock_code": "object",
    "theme_id": "int32",
    "unmarket_norm_score": "float32",
    "norm_score": "float32",
    "is_strong_rel": "int8",
    "is_bellwether": "int8",
}


def _backup_once(path: str) -> None:
    """首次写入时备份, 第二次及以后跳过."""
    bak = path + ".bak"
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copy2(path, bak)
        size_mb = os.path.getsize(bak) / 1024 / 1024
        print(f"  已备份: {bak} ({size_mb:.1f} MB)")
    elif os.path.exists(bak):
        print(f"  备份已存在, 跳过: {bak}")


def _verify_schema(df: pd.DataFrame, label: str) -> None:
    """严格 dtype + 列名校验."""
    missing = set(EXPECTED_DTYPES) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_DTYPES)
    if missing or extra:
        raise ValueError(f"[{label}] schema 不匹配: 缺 {missing}, 多 {extra}")
    for col, expected_dtype in EXPECTED_DTYPES.items():
        actual = str(df[col].dtype)
        if actual != expected_dtype:
            raise ValueError(f"[{label}] dtype 不一致: {col} 应是 {expected_dtype}, 实际 {actual}")
    print(f"  [{label}] schema 校验通过 (列序 + dtype 全部对齐)")


def main():
    t_start = time.time()
    print(f"\n{'='*64}")
    print(f"fetch_theme_long_inc  增量补丁")
    print(f"  目标终点: {DATE_END} (左闭右开)")
    print(f"  SMOKE 模式: {'是 (只拉 1 个月, 不写盘)' if SMOKE else '否'}")
    print(f"{'='*64}")

    # ─── 1. 读旧文件, 确定增量起点 ───
    print(f"\n[1] 加载旧文件 {OUTPUT_PATH}...")
    t0 = time.time()
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(f"旧文件不存在: {OUTPUT_PATH}, 请先做完整拉取 (fetch_theme_long.py)")
    old_df = pd.read_parquet(OUTPUT_PATH)
    last_date = old_df["date"].max()
    n_old = len(old_df)
    print(f"  旧数据: {n_old:,} 行, 日期范围 {old_df['date'].min().date()} ~ {last_date.date()}, "
          f"耗时 {time.time()-t0:.1f}s")
    _verify_schema(old_df, "旧数据")

    # 增量起点: 旧文件最后一天的下一个交易日 (用 price_non_st 的交易日历推, 简单点用日历日 + 1 天即可,
    # 因为 standardize 后会按 inner join valid_keys 过滤)
    if FORCE_FROM:
        inc_start = pd.Timestamp(FORCE_FROM)
        print(f"  [FORCE_FROM] 增量起点强制覆盖为: {inc_start.date()}")
    else:
        inc_start = last_date + pd.Timedelta(days=1)
        print(f"  增量起点 (last_date + 1d): {inc_start.date()}")

    inc_end = pd.Timestamp(DATE_END)
    if SMOKE:
        # SMOKE: 只拉第一个增量月
        inc_end_smoke = (inc_start + pd.offsets.MonthBegin(1)).normalize()
        if inc_end_smoke > inc_end:
            inc_end_smoke = inc_end
        inc_end = inc_end_smoke
        print(f"  [SMOKE] 终点缩到: {inc_end.date()}")

    if inc_start >= inc_end:
        print(f"\n[2] 增量范围为空 ({inc_start.date()} >= {inc_end.date()}), 无需拉取, 退出")
        return

    print(f"\n[2] 增量范围: [{inc_start.date()}, {inc_end.date()}), "
          f"约 {(inc_end - inc_start).days} 日历日")

    # ─── 3. 加载 price_non_st 用于 inner join 过滤 ───
    print(f"\n[3] 加载 price_non_st 用于 (date, stock) 过滤...")
    t0 = time.time()
    price = pd.read_pickle(PRICE_PATH)
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    price["stock_code"] = price["stock_code"].astype(str).str.zfill(6)
    price = price[(price["date"] >= inc_start) & (price["date"] < inc_end)]
    valid_keys = price[["date", "stock_code"]].drop_duplicates()
    print(f"  price_non_st 在增量段内: {len(price):,} 行, "
          f"{valid_keys['date'].nunique()} 个交易日, "
          f"耗时 {time.time()-t0:.1f}s")
    del price

    # ─── 4. 按月分批拉取增量 ───
    chunks = []
    inc_lo_str = inc_start.strftime("%Y-%m-%d")
    inc_hi_str = inc_end.strftime("%Y-%m-%d")
    n_months = sum(1 for _ in month_buckets(inc_lo_str, inc_hi_str))
    print(f"\n[4] 按月分批拉取 ({n_months} 个月)...")
    t_fetch = time.time()
    for i, (lo, hi) in enumerate(month_buckets(inc_lo_str, inc_hi_str), 1):
        # 增量段全部 >= SPLIT_DATE='2023-01-01', 全部走 _score 表
        table = "tkg_theme_sec_sc_his" if lo < SPLIT_DATE else "tkg_theme_sec_score"
        t_m = time.time()
        df_raw = fetch_one_month(table, lo, hi)
        df_std = standardize(df_raw)
        n_before = len(df_std)
        df_filt = df_std.merge(valid_keys, on=["date", "stock_code"], how="inner")
        n_after = len(df_filt)
        chunks.append(df_filt)
        elapsed = time.time() - t_m
        print(f"  [{i:>2d}/{n_months}] {lo} ({table[-9:]}): "
              f"{n_before:>8,} → {n_after:>8,} 行 "
              f"(过滤 {(1-n_after/max(n_before,1))*100:.1f}%), {elapsed:.1f}s")
    fetch_elapsed = time.time() - t_fetch
    print(f"  拉取完成, 总耗时 {fetch_elapsed:.0f}s ({fetch_elapsed/60:.1f}min)")

    # ─── 5. 拼接增量 ───
    print(f"\n[5] 拼接增量 {len(chunks)} 个月...")
    t0 = time.time()
    new_df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  增量行数: {len(new_df):,}, "
          f"日期 {new_df['date'].min().date()} ~ {new_df['date'].max().date()}, "
          f"耗时 {time.time()-t0:.1f}s")
    _verify_schema(new_df, "增量")

    # 增量与旧数据的边界检查 (不应有重叠日期, 否则就要 dedup)
    new_min = new_df["date"].min()
    if new_min <= last_date:
        print(f"  ⚠️  警告: 增量起点 {new_min.date()} <= 旧终点 {last_date.date()}, 后续会 dedup")

    # ─── 6. concat 旧 + 新, 去重 ───
    print(f"\n[6] concat 旧 ({n_old:,}) + 新 ({len(new_df):,})...")
    t0 = time.time()
    full_df = pd.concat([old_df, new_df], ignore_index=True)
    del old_df, new_df
    n_before_dedup = len(full_df)
    print(f"  concat 后: {n_before_dedup:,} 行, 耗时 {time.time()-t0:.1f}s")

    print(f"\n[7] 去重 (date, stock_code, theme_id) + 排序...")
    t0 = time.time()
    # 保留 last (新数据覆盖旧数据, 边界日期处更新)
    full_df = full_df.drop_duplicates(
        subset=["date", "stock_code", "theme_id"], keep="last"
    ).reset_index(drop=True)
    n_after_dedup = len(full_df)
    print(f"  去重: {n_before_dedup:,} -> {n_after_dedup:,} (重复 {n_before_dedup-n_after_dedup})")
    full_df = full_df.sort_values(
        ["date", "stock_code", "theme_id"], kind="mergesort"
    ).reset_index(drop=True)
    print(f"  排序完成, 耗时 {time.time()-t0:.1f}s")
    _verify_schema(full_df, "合并后")

    # ─── 8. SMOKE 校验 (不写盘) ───
    print(f"\n[8] 最终校验:")
    print(f"  总行数: {len(full_df):,}")
    print(f"  日期范围: {full_df['date'].min().date()} ~ {full_df['date'].max().date()}")
    print(f"  unique 日期数: {full_df['date'].nunique()}")
    print(f"  unique 股票数: {full_df['stock_code'].nunique()}")
    print(f"  unique 概念数: {full_df['theme_id'].nunique()}")

    if SMOKE:
        print(f"\n[SMOKE] 跳过写盘. 验证完成, 总耗时 {(time.time()-t_start)/60:.1f} min")
        return

    # ─── 9. 备份 + 写盘 ───
    print(f"\n[9] 备份 + 写盘 {OUTPUT_PATH}")
    _backup_once(OUTPUT_PATH)
    t0 = time.time()
    full_df.to_parquet(OUTPUT_PATH, compression="snappy", index=False)
    sz = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  ✓ 已写入 ({sz:.1f} MB), 耗时 {time.time()-t0:.1f}s")

    print(f"\n{'='*64}")
    print(f"全部完成, 总耗时 {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
