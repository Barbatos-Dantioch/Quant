"""一次性脚本: 完整重新拉取 panel_trade.pkl + price_non_st.pkl.

流程:
    1. zz_500D  → daily        (open_price, close_price, amount)
    2. zz_379   → status       (停牌/涨跌停标志, status==0 为可交易)
                                NOTE: zz_379 在 2026-02-12 后停止更新, 所以缺失日期
                                用 amount==0 代理停牌: status = 0 if amount>0 else 1
    3. zz_200   → is_st        (ST 标志, 用于过滤; is_st==0 表示非 ST)
    4. zz_381   → industry     (申万一级, 静态字段, 取 update_date 最新一条)
    5. zz_222   → market_value (宽表 melt 为 long)
    6. daily + status            → panel_trade   (含 amount 代理填充, 无 ST 过滤),
                                                    写到 Data/all/panel_trade.pkl
    7. panel_trade ⨝ is_st==0    → 非 ST 子集
    8. + industry + market_value → price_non_st, 写到 Data/all/price_non_st.pkl

status 代理规则 (在 zz_379 数据缺失日期生效):
    amount > 0  → status = 0  (近似可交易, 当日有成交)
    amount == 0 → status = 1  (近似停牌, 当日无成交)
    amount NaN  → status = 1  (无数据视为不可交易)

输出: 覆盖原文件, 保留 .bak 备份 (若 .bak 已存在则不重复备份). amount 字段不保留.

环境变量:
    FETCH_DATE_START   起始日期 (默认 2020-01-01)
    FETCH_DATE_END     终止日期 (左闭右开, 默认 2026-05-14, 实际拉取受数据库覆盖限制)
    FETCH_SMOKE        若设为 "1", 只拉 1 个月数据 + 不写盘, 用于流程冒烟

依据:
    - 数据源选择参考 xgbcode/all/Dataloader.ipynb (Cell 1/2/3/6)
    - stock_code 标准化方式参考 xgbcode/pair_trading/merge_ind_mv.py
    - 月分段 + 备份 + log 风格参考 xgbcode/pair_trading/fetch_theme_long.py

运行:
    setsid nohup python3 -u xgbcode/pair_trading/fetch_price_non_st.py \
        > /tmp/fetch_price_non_st.log 2>&1 &
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from trading_rest_sdk import TradingRestClient


# ──────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────

DATE_START = os.environ.get("FETCH_DATE_START", "2020-01-01")
DATE_END = os.environ.get("FETCH_DATE_END", "2026-05-14")  # 左闭右开
SMOKE = os.environ.get("FETCH_SMOKE") == "1"

DATA_DIR = "/root/quant/Data/all"
PANEL_TRADE_PATH = os.path.join(DATA_DIR, "panel_trade.pkl")
PRICE_NON_ST_PATH = os.path.join(DATA_DIR, "price_non_st.pkl")

client = TradingRestClient(
    api_key="ak_dd1746fcf2417d19cae64045fc1399b28a57f5f936b749b8e8f7a16ace8930c0",
    base_url="http://192.168.20.10:8080",
    timeout=6000,
)


# ──────────────────────────────────────────────────────────────────────
# 通用辅助
# ──────────────────────────────────────────────────────────────────────

def _normalize_stock_code(s: pd.Series) -> pd.Series:
    """把 'SH.600000' / '600000.SH' / '600000' 统一为 6 位字符串.

    与 xgbcode/pair_trading/merge_ind_mv.py 中的实现保持一致.
    """
    parts = s.astype(str).str.split(".")
    out = parts.apply(
        lambda ps: next((p for p in ps if p.isdigit()), ps[-1])
    ).astype(str).str.zfill(6)
    return out


def _normalize_date(s: pd.Series) -> pd.Series:
    """trade_date 兼容 'YYYY-MM-DDT00:00:00' 形式, 去时间部分."""
    s = s.astype(str).str.split("T").str[0]
    return pd.to_datetime(s).dt.normalize()


def month_buckets(start: str, end: str):
    """[start, end) 内按月切片, 生成 (lo, hi) 字符串元组."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur < e:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt > e:
            nxt = e
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _backup_once(pkl_path: str) -> None:
    """首次写入时备份, 第二次及以后跳过 (避免覆盖更早的备份)."""
    bak = pkl_path + ".bak"
    if os.path.exists(pkl_path) and not os.path.exists(bak):
        shutil.copy2(pkl_path, bak)
        size_mb = os.path.getsize(bak) / 1024 / 1024
        print(f"  已备份: {bak} ({size_mb:.1f} MB)")
    elif os.path.exists(bak):
        print(f"  备份已存在, 跳过: {bak}")


# ──────────────────────────────────────────────────────────────────────
# 拉取函数 (按表)
# ──────────────────────────────────────────────────────────────────────

def fetch_daily_price(start: str, end: str) -> pd.DataFrame:
    """zz_500D: 按月分段拉取 (open_price, close_price, amount).

    返回长表 [date, stock_code, open_price, close_price, amount].
    amount 用于 status 缺失日期的停牌代理 (amount==0 → 停牌).

    NOTE: zz_500D 在 ClickHouse 数据源, 必须显式 datasource='clickhouse'.
    """
    print(f"\n[zz_500D] 价格 {start} ~ {end} 按月分段拉取 (clickhouse)...")
    chunks = []
    n_months = sum(1 for _ in month_buckets(start, end))
    for i, (lo, hi) in enumerate(month_buckets(start, end), 1):
        t_m = time.time()
        q = f"""
            SELECT DISTINCT stock_code, trade_date, open_price, close_price, amount
            FROM zz_500D
            WHERE trade_date >= '{lo}' AND trade_date < '{hi}'
        """
        res = client.execute_sql(sql=q, datasource="clickhouse")
        df = pd.DataFrame(res["data"])
        if len(df) > 0:
            chunks.append(df)
        print(f"  [{i:>2d}/{n_months}] {lo}: {len(df):>7,} 行, {time.time()-t_m:.1f}s")
    if not chunks:
        return pd.DataFrame(columns=["date", "stock_code", "open_price", "close_price", "amount"])
    daily = pd.concat(chunks, ignore_index=True)
    del chunks
    daily["stock_code"] = _normalize_stock_code(daily["stock_code"])
    daily["date"] = _normalize_date(daily["trade_date"])
    daily["open_price"] = pd.to_numeric(daily["open_price"], errors="coerce")
    daily["close_price"] = pd.to_numeric(daily["close_price"], errors="coerce")
    daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce")
    daily = daily[["date", "stock_code", "open_price", "close_price", "amount"]]
    daily = daily.drop_duplicates(["date", "stock_code"]).sort_values(
        ["stock_code", "date"], kind="mergesort"
    ).reset_index(drop=True)
    print(f"  ✓ daily: {len(daily):,} 行, "
          f"{daily['date'].min().date()} ~ {daily['date'].max().date()}, "
          f"{daily['stock_code'].nunique()} 股, "
          f"amount==0 比例 {(daily['amount']==0).mean()*100:.2f}%")
    return daily


def fetch_status(start: str, end: str) -> pd.DataFrame:
    """zz_379: status (停牌/涨跌停标志). 按月分段."""
    print(f"\n[zz_379] status {start} ~ {end} 按月分段拉取...")
    chunks = []
    n_months = sum(1 for _ in month_buckets(start, end))
    for i, (lo, hi) in enumerate(month_buckets(start, end), 1):
        t_m = time.time()
        q = f"""
            SELECT DISTINCT stock_code, trade_date, status
            FROM zz_379
            WHERE trade_date >= '{lo}' AND trade_date < '{hi}'
        """
        res = client.execute_sql(sql=q)
        df = pd.DataFrame(res["data"])
        if len(df) > 0:
            chunks.append(df)
        print(f"  [{i:>2d}/{n_months}] {lo}: {len(df):>7,} 行, {time.time()-t_m:.1f}s")
    if not chunks:
        return pd.DataFrame(columns=["date", "stock_code", "status"])
    st = pd.concat(chunks, ignore_index=True)
    del chunks
    st["stock_code"] = _normalize_stock_code(st["stock_code"])
    st["date"] = _normalize_date(st["trade_date"])
    st["status"] = pd.to_numeric(st["status"], errors="coerce")
    st = st[["date", "stock_code", "status"]]
    st = st.drop_duplicates(["date", "stock_code"]).sort_values(
        ["stock_code", "date"], kind="mergesort"
    ).reset_index(drop=True)
    print(f"  ✓ status: {len(st):,} 行")
    return st


def fetch_is_st(start: str, end: str) -> pd.DataFrame:
    """zz_200: is_st (ST 标志, 1=ST, 0=非 ST). 按月分段."""
    print(f"\n[zz_200] is_st {start} ~ {end} 按月分段拉取...")
    chunks = []
    n_months = sum(1 for _ in month_buckets(start, end))
    for i, (lo, hi) in enumerate(month_buckets(start, end), 1):
        t_m = time.time()
        q = f"""
            SELECT DISTINCT stock_code, trade_date, is_st
            FROM zz_200
            WHERE trade_date >= '{lo}' AND trade_date < '{hi}'
        """
        res = client.execute_sql(sql=q)
        df = pd.DataFrame(res["data"])
        if len(df) > 0:
            chunks.append(df)
        print(f"  [{i:>2d}/{n_months}] {lo}: {len(df):>7,} 行, {time.time()-t_m:.1f}s")
    if not chunks:
        return pd.DataFrame(columns=["date", "stock_code", "is_st"])
    s = pd.concat(chunks, ignore_index=True)
    del chunks
    s["stock_code"] = _normalize_stock_code(s["stock_code"])
    s["date"] = _normalize_date(s["trade_date"])
    s["is_st"] = pd.to_numeric(s["is_st"], errors="coerce").fillna(1).astype(np.int8)
    s = s[["date", "stock_code", "is_st"]]
    s = s.drop_duplicates(["date", "stock_code"]).sort_values(
        ["stock_code", "date"], kind="mergesort"
    ).reset_index(drop=True)
    print(f"  ✓ is_st: {len(s):,} 行, ST 比例 {(s['is_st']==1).mean()*100:.2f}%")
    return s


def fetch_industry() -> pd.DataFrame:
    """zz_381: 静态行业表, 每股取 update_date 最新一条.

    返回 [stock_code, sw_industry_l1_code, sw_industry_l1_name].
    """
    print("\n[zz_381] 申万一级行业 (静态全表, 每股取最新)...")
    t0 = time.time()
    q = """
        SELECT stock_code, sw_industry_l1_code, sw_industry_l1_name, update_date
        FROM zz_381
    """
    res = client.execute_sql(sql=q)
    df = pd.DataFrame(res["data"])
    df["stock_code"] = _normalize_stock_code(df["stock_code"])
    df["update_date"] = pd.to_datetime(df["update_date"])
    df = df.sort_values(["stock_code", "update_date"], kind="mergesort")
    df = df.drop_duplicates(subset=["stock_code"], keep="last").reset_index(drop=True)
    df = df[["stock_code", "sw_industry_l1_code", "sw_industry_l1_name"]]
    n_with_ind = df["sw_industry_l1_code"].notna().sum()
    print(f"  ✓ industry: {len(df):,} 只股票, {n_with_ind} 只有行业, "
          f"耗时 {time.time()-t0:.1f}s")
    return df


def fetch_market_value(start: str, end: str) -> pd.DataFrame:
    """zz_222: 宽表 (按 trade_date 一行 × 各股票为列), melt 为长表.

    按月分段以避免大查询超时. 返回 [date, stock_code, market_value].
    """
    print(f"\n[zz_222] market_value {start} ~ {end} 按月分段拉取...")
    chunks = []
    n_months = sum(1 for _ in month_buckets(start, end))
    for i, (lo, hi) in enumerate(month_buckets(start, end), 1):
        t_m = time.time()
        q = f"""
            SELECT DISTINCT *
            FROM zz_222
            WHERE trade_date >= '{lo}' AND trade_date < '{hi}'
        """
        res = client.execute_sql(sql=q)
        df = pd.DataFrame(res["data"])
        if len(df) == 0:
            print(f"  [{i:>2d}/{n_months}] {lo}: 0 行 (跳过), {time.time()-t_m:.1f}s")
            continue
        df["date"] = _normalize_date(df["trade_date"])
        drop_cols = [c for c in ("trade_date", "update_time") if c in df.columns]
        mv_long = df.drop(columns=drop_cols).melt(
            id_vars=["date"], var_name="stock_code", value_name="market_value"
        )
        mv_long["stock_code"] = _normalize_stock_code(mv_long["stock_code"])
        # 过滤非股票代码 (例如 update_time 这类辅助列残留)
        valid = mv_long["stock_code"].str.fullmatch(r"\d{6}")
        mv_long = mv_long[valid]
        mv_long["market_value"] = pd.to_numeric(mv_long["market_value"], errors="coerce")
        mv_long = mv_long.dropna(subset=["market_value"]).reset_index(drop=True)
        chunks.append(mv_long[["date", "stock_code", "market_value"]])
        print(f"  [{i:>2d}/{n_months}] {lo}: {len(mv_long):>7,} 行, {time.time()-t_m:.1f}s")
    if not chunks:
        return pd.DataFrame(columns=["date", "stock_code", "market_value"])
    mv = pd.concat(chunks, ignore_index=True)
    del chunks
    mv = mv.drop_duplicates(["date", "stock_code"]).sort_values(
        ["stock_code", "date"], kind="mergesort"
    ).reset_index(drop=True)
    print(f"  ✓ market_value: {len(mv):,} 行")
    return mv


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print(f"\n{'='*64}")
    print(f"fetch_price_non_st  时间范围 [{DATE_START}, {DATE_END})")
    print(f"  SMOKE 模式: {'是 (只拉 1 个月, 不写盘)' if SMOKE else '否'}")
    print(f"{'='*64}")

    if SMOKE:
        # 冒烟: 只拉 [DATE_START, DATE_START + 1 month) 验证流程
        smoke_end = (pd.Timestamp(DATE_START) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
        ds, de = DATE_START, smoke_end
        print(f"  SMOKE 实际范围: [{ds}, {de})")
    else:
        ds, de = DATE_START, DATE_END

    # ─── 1. 拉取 5 个数据源 ───
    daily = fetch_daily_price(ds, de)
    status = fetch_status(ds, de)
    is_st = fetch_is_st(ds, de)
    industry = fetch_industry()
    market_value = fetch_market_value(ds, de)

    # ─── 2. 构造 panel_trade = daily ⨝ status, 缺失用 amount 代理 ───
    print(f"\n[合并] panel_trade = daily ⨝ status (left join, 无 ST 过滤)...")
    t0 = time.time()
    panel_trade = daily.merge(status, on=["stock_code", "date"], how="left")
    panel_trade = panel_trade.sort_values(
        ["stock_code", "date"], kind="mergesort"
    ).reset_index(drop=True)
    n_status_miss = int(panel_trade["status"].isna().sum())
    pct_miss = n_status_miss / len(panel_trade) * 100
    print(f"  原始 status 缺失: {n_status_miss:,} ({pct_miss:.2f}%), 耗时 {time.time()-t0:.1f}s")

    # ─── 2b. status 代理填充 (amount > 0 → 0; amount == 0/NaN → 1) ───
    if n_status_miss > 0:
        print(f"\n[代理] status 缺失日期用 amount 代理 (amount==0 → 1, amount>0 → 0)...")
        t0 = time.time()
        miss_mask = panel_trade["status"].isna()
        amount_pos = (panel_trade["amount"] > 0).fillna(False)
        # 缺失行: amount > 0 → 0 (可交易); 否则 → 1 (停牌)
        proxy = np.where(amount_pos, 0.0, 1.0)
        panel_trade.loc[miss_mask, "status"] = proxy[miss_mask]
        n_proxy_zero = int((miss_mask & amount_pos).sum())
        n_proxy_one = int((miss_mask & ~amount_pos).sum())
        print(f"  代理填充: {n_status_miss:,} 行 → "
              f"status=0 (amount>0) {n_proxy_zero:,} 行, "
              f"status=1 (amount==0/NaN) {n_proxy_one:,} 行, "
              f"耗时 {time.time()-t0:.1f}s")
        # 代理后应已无缺失
        assert panel_trade["status"].isna().sum() == 0, "代理填充后仍有 status NaN, 检查 amount 列"

    # 移除 amount: 不进 panel_trade.pkl, 保持现有字段结构
    panel_trade = panel_trade.drop(columns=["amount"])

    # ─── 3. ST 过滤 → price_non_st 中间体 ───
    print(f"\n[过滤] panel_trade ⨝ is_st (inner) 后保留 is_st==0 ...")
    t0 = time.time()
    n_before = len(panel_trade)
    pn = panel_trade.merge(is_st, on=["stock_code", "date"], how="inner")
    pn = pn[pn["is_st"] == 0].drop(columns=["is_st"]).reset_index(drop=True)
    n_after = len(pn)
    print(f"  非 ST 子集: {n_before:,} -> {n_after:,} 行 "
          f"(过滤 {(1-n_after/n_before)*100:.2f}%), 耗时 {time.time()-t0:.1f}s")

    # ─── 4. 合并行业 (静态) ───
    print(f"\n[合并] price_non_st ⨝ industry (left join, 静态)...")
    t0 = time.time()
    pn = pn.merge(industry, on="stock_code", how="left")
    n_ind_miss = pn["sw_industry_l1_code"].isna().sum()
    print(f"  行业: 缺失 {n_ind_miss} ({n_ind_miss/len(pn)*100:.2f}%), "
          f"耗时 {time.time()-t0:.1f}s")

    # ─── 5. 合并市值 + 缺失值用 (按股票时序的) 前后均值填充 ───
    # 步骤:
    #   a. left join 保留所有 (date, stock) 行
    #   b. 按 stock_code 分组对 market_value 做线性插值: 单点缺失 = 前后均值;
    #      多点连续缺失 = 线性递增. 不做 ffill/bfill (不外推首尾, 避免假设
    #      股票上市前 / 退市后的市值).
    #   c. 经过 b 仍 NaN 的行 = 该股票首/尾边界无 mv 历史, dropna 剔除
    print(f"\n[合并] price_non_st ⨝ market_value (left join, by [stock_code, date])...")
    t0 = time.time()
    n_before = len(pn)
    pn = pn.merge(market_value, on=["stock_code", "date"], how="left")
    n_mv_miss_raw = int(pn["market_value"].isna().sum())
    print(f"  market_value: left join 后缺失 {n_mv_miss_raw} 行")

    # 必须按 (stock_code, date) 排序才能正确插值
    pn = pn.sort_values(["stock_code", "date"], kind="mergesort").reset_index(drop=True)
    # interpolate(method='linear') 默认只填内部 NaN, 不填首尾 NaN (limit_direction='forward')
    # → 单点缺失 = 前后均值; 多点连续缺失 = 线性递增; 首尾仍保留 NaN
    pn["market_value"] = pn.groupby("stock_code")["market_value"].transform(
        lambda s: s.interpolate(method="linear")
    )
    n_after_interp = int(pn["market_value"].isna().sum())
    n_filled = n_mv_miss_raw - n_after_interp
    print(f"  线性插值 (前后均值) 填充: {n_filled} 行, 仍 NaN: {n_after_interp} 行")

    # 仍 NaN 的行 = 首/尾边界无前后值, 直接 dropna
    if n_after_interp > 0:
        pn = pn.dropna(subset=["market_value"]).reset_index(drop=True)
    print(f"  最终: {n_before:,} -> {len(pn):,} 行 (插值 {n_filled} 行 + dropna {n_after_interp} 行), "
          f"耗时 {time.time()-t0:.1f}s")

    # 字段顺序: 与现有 price_non_st.pkl 对齐
    pn = pn[[
        "close_price", "open_price", "stock_code", "date", "status",
        "sw_industry_l1_code", "sw_industry_l1_name", "market_value",
    ]]
    pn = pn.sort_values(["stock_code", "date"], kind="mergesort").reset_index(drop=True)

    # ─── 6. 校验 ───
    print(f"\n[校验]")
    print(f"  panel_trade: {len(panel_trade):,} 行, "
          f"日期 {panel_trade['date'].min().date()} ~ {panel_trade['date'].max().date()}, "
          f"{panel_trade['stock_code'].nunique()} 股")
    print(f"  price_non_st: {len(pn):,} 行, "
          f"日期 {pn['date'].min().date()} ~ {pn['date'].max().date()}, "
          f"{pn['stock_code'].nunique()} 股, "
          f"{pn['date'].nunique()} 个交易日")
    for c in pn.columns:
        nn = pn[c].notna().sum()
        print(f"    {c:<22s}: 非 NaN {nn:>9,}/{len(pn):,} ({nn/len(pn)*100:.2f}%)")

    # ─── 7. 写盘 (SMOKE 不写) ───
    if SMOKE:
        print(f"\n[SMOKE] 跳过写盘. 验证完成, 总耗时 {(time.time()-t_total)/60:.1f} min")
        return

    print(f"\n[写盘 1/2] {PANEL_TRADE_PATH}")
    _backup_once(PANEL_TRADE_PATH)
    panel_trade.to_pickle(PANEL_TRADE_PATH)
    sz = os.path.getsize(PANEL_TRADE_PATH) / 1024 / 1024
    print(f"  ✓ 已写入 ({sz:.1f} MB)")

    print(f"\n[写盘 2/2] {PRICE_NON_ST_PATH}")
    _backup_once(PRICE_NON_ST_PATH)
    pn.to_pickle(PRICE_NON_ST_PATH)
    sz = os.path.getsize(PRICE_NON_ST_PATH) / 1024 / 1024
    print(f"  ✓ 已写入 ({sz:.1f} MB)")

    print(f"\n{'='*64}")
    print(f"全部完成, 总耗时 {(time.time()-t_total)/60:.1f} min")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
