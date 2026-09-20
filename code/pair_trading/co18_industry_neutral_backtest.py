"""
CO-18 行业中性化对比回测

三种口径对比:
  baseline     : 原始因子 + 全市场等权基准 (校验与主回测 metrics 是否一致)
  ind_bench    : 原始因子 + 行业基准超额 (每只股票减去"当天所在行业等权均值")
  ind_neutral  : 行业中性化因子 (行业内 zscore) + 全市场等权基准

输入:
    output/0416_pair_factor/CO-18/factor_df.pkl
    Data/all/price_non_st.pkl

输出 (output/0416_pair_factor/_analysis/CO18_industry_neutral/):
    compare_metrics.csv       三种口径指标对比
    nav_compare.png           三条头组超额 NAV 同屏对比
    head_industry_share.png   baseline vs ind_neutral 头组行业占比柱状对比
    summary.json              摘要

口径:
    - 时间: 2024-01-01 ~ 2025-12-31
    - n_groups = 10
    - fwd_days = 1
    - 最小行业样本: 每日行业内股票数 >= 5 (否则 zscore / 均值不稳)
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr

FACTOR_PATH = "output/0416_pair_factor/CO-18/factor_df.pkl"
PRICE_PATH = "Data/all/price_non_st.pkl"
OUT_DIR = "output/0416_pair_factor/_analysis/CO18_industry_neutral"

DATE_START = "2024-01-01"
DATE_END = "2025-12-31"
N_GROUPS = 10
FWD_DAYS = 1
MIN_IND_SIZE = 5

rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_merged():
    """读因子 + 价格 + 行业 map, 返回已过滤可交易 + 带 fwd_ret/ind_code 的 merged df。"""
    print(f"[读] factor: {FACTOR_PATH}")
    fdf = pd.read_pickle(FACTOR_PATH)
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf[(fdf["date"] >= DATE_START) & (fdf["date"] <= DATE_END)].copy()
    print(f"   因子: {len(fdf)} 条, {fdf['date'].nunique()} 天, {fdf['stock_code'].nunique()} 只")

    print(f"[读] price: {PRICE_PATH}")
    price = pd.read_pickle(PRICE_PATH)
    price["date"] = pd.to_datetime(price["date"])
    price = price[(price["date"] >= DATE_START) & (price["date"] <= DATE_END)].copy()
    price["tradable"] = (
        (price["status"] == 0)
        & (price["open_price"] > 0)
        & price["open_price"].notna()
        & price["close_price"].notna()
    )
    price = price.sort_values(["stock_code", "date"])
    price["fwd_close"] = price.groupby("stock_code")["close_price"].shift(-FWD_DAYS)
    price["fwd_tradable"] = price.groupby("stock_code")["tradable"].shift(-1)
    price["fwd_ret"] = (price["fwd_close"] - price["close_price"]) / price["close_price"]

    ind_src = price.dropna(subset=["sw_industry_l1_code"])
    ind_map = ind_src.groupby("stock_code").agg(
        ind_code=("sw_industry_l1_code", "first"),
        ind_name=("sw_industry_l1_name", "first"),
    )
    print(f"   行业 map: {len(ind_map)} 只, {ind_map['ind_code'].nunique()} 个一级行业")

    merged = fdf.merge(
        price[["date", "stock_code", "tradable", "fwd_ret", "fwd_tradable"]],
        on=["date", "stock_code"], how="inner",
    )
    merged = merged[merged["tradable"] & merged["fwd_tradable"].fillna(False)].copy()
    merged = merged.dropna(subset=["factor", "fwd_ret"])
    merged = merged.merge(ind_map, left_on="stock_code", right_index=True, how="left")
    merged = merged.dropna(subset=["ind_code"]).copy()
    print(f"   合并后: {len(merged)} 条")
    return merged


def compute_industry_stats(merged: pd.DataFrame):
    """在每天的行业内预计算:
        - ind_mean_ret    : 每天 每行业的 fwd_ret 等权均值 (用于 ind_bench)
        - factor 的行业 zscore (用于 ind_neutral): factor_z[i] = (f[i]-mu_ind)/sigma_ind
    返回新列:  fwd_ret_ind_excess, factor_neutral
    """
    grp = merged.groupby(["date", "ind_code"])
    ind_stats = grp.agg(
        ind_mean_ret=("fwd_ret", "mean"),
        ind_mean_factor=("factor", "mean"),
        ind_std_factor=("factor", "std"),
        ind_size=("factor", "size"),
    ).reset_index()
    # 对小行业 (size<MIN_IND_SIZE) 丢弃: 把 ind_mean_ret / std 置 NaN, 后续会过滤
    ind_stats.loc[ind_stats["ind_size"] < MIN_IND_SIZE, ["ind_mean_ret", "ind_mean_factor", "ind_std_factor"]] = np.nan

    merged = merged.merge(ind_stats, on=["date", "ind_code"], how="left")
    merged["fwd_ret_ind_excess"] = merged["fwd_ret"] - merged["ind_mean_ret"]
    merged["factor_neutral"] = (merged["factor"] - merged["ind_mean_factor"]) / \
                                merged["ind_std_factor"].replace(0, np.nan)
    return merged


def run_one_variant(merged: pd.DataFrame, factor_col: str, excess_col: str, n_groups: int,
                    return_head_members: bool = False):
    """通用分组回测:
        factor_col : 选股用的因子列 (factor 或 factor_neutral)
        excess_col : 算超额用的单股超额收益列 (fwd_ret - 全市场均值 / 行业均值)
                     注意: 这里传入的是"已经预先算好的单股超额", 每日头组取平均即可
    返回:
        dict(ic_mean, ic_ir, head_excess_ret_series, head_excess_nav_series,
             head_ann_excess, head_sharpe, head_max_dd, n_dates,
             head_ind_share (DataFrame, per-date head 中各行业占比))
    """
    ic_list = []
    head_excess_daily = []
    dates_list = []
    head_ind_rows = []   # (date, ind_code, share)

    for dt, grp in merged.groupby("date"):
        g = grp.dropna(subset=[factor_col, excess_col]).copy()
        if len(g) < n_groups * 2:
            continue
        g["group"] = pd.qcut(
            g[factor_col].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        head_mask = (g["group"] == n_groups - 1)
        head = g[head_mask]
        if len(head) == 0:
            continue

        dates_list.append(dt)
        head_excess_daily.append(head[excess_col].mean())

        # 头组超额与"因子 vs fwd_ret"的 IC (IC 用原始 fwd_ret - 全市场均值后的排序是一样的,
        # 直接用 spearman(factor, fwd_ret) 即可)
        ic, _ = spearmanr(g[factor_col], g["fwd_ret"])
        if not np.isnan(ic):
            ic_list.append(ic)

        if return_head_members:
            ind_counts = head["ind_code"].value_counts(normalize=True)
            for ic_code, share in ind_counts.items():
                head_ind_rows.append((dt, ic_code, float(share)))

    if not dates_list:
        raise RuntimeError("无有效交易日")

    dates_arr = pd.DatetimeIndex(dates_list)
    head_ret = pd.Series(head_excess_daily, index=dates_arr)
    head_nav = (1 + head_ret).cumprod()

    ann = 250.0
    head_ann_excess = head_ret.mean() * ann
    head_ann_vol = head_ret.std() * np.sqrt(ann)
    head_sharpe = head_ann_excess / head_ann_vol if head_ann_vol > 0 else 0.0
    dd = ((head_nav - head_nav.cummax()) / head_nav.cummax()).min()

    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list))
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

    head_ind_df = None
    if return_head_members:
        head_ind_df = pd.DataFrame(head_ind_rows, columns=["date", "ind_code", "share"])

    return {
        "ic_mean": ic_mean, "ic_std": ic_std, "ic_ir": ic_ir,
        "head_excess_ret": head_ret, "head_excess_nav": head_nav,
        "head_ann_excess": float(head_ann_excess),
        "head_ann_excess_vol": float(head_ann_vol),
        "head_sharpe": float(head_sharpe),
        "head_max_dd": float(dd),
        "n_dates": len(dates_list),
        "head_ind_share": head_ind_df,
    }


def plot_nav_compare(results: dict, outpath: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {"baseline": "#616161", "ind_bench": "#1565c0", "ind_neutral": "#2e7d32"}
    for name, res in results.items():
        nav = res["head_excess_nav"]
        label = (f"{name}  "
                 f"IC={res['ic_mean']:+.4f}, ICIR={res['ic_ir']:+.2f}, "
                 f"ann={res['head_ann_excess']*100:+.2f}%, "
                 f"SR={res['head_sharpe']:+.2f}, DD={res['head_max_dd']*100:+.1f}%")
        ax.plot(nav.index, nav.values, lw=1.6, color=colors.get(name, "black"), label=label)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title("CO-18 Head Excess NAV: baseline vs industry bench vs industry neutral (2024-2025)",
                 fontsize=11)
    ax.set_ylabel("Head Group Excess NAV")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {outpath}")


def plot_head_industry_share(base_df: pd.DataFrame, neutral_df: pd.DataFrame,
                              ind_name_map: dict, outpath: str):
    """baseline vs ind_neutral 的头组行业占比对比"""
    base_avg = base_df.groupby("ind_code")["share"].mean()
    neutral_avg = neutral_df.groupby("ind_code")["share"].mean()
    all_inds = sorted(set(base_avg.index) | set(neutral_avg.index),
                      key=lambda c: -(base_avg.get(c, 0) + neutral_avg.get(c, 0)))
    base_vals = [base_avg.get(c, 0.0) for c in all_inds]
    neutral_vals = [neutral_avg.get(c, 0.0) for c in all_inds]
    labels = [ind_name_map.get(c, c) for c in all_inds]

    x = np.arange(len(all_inds))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(14, len(all_inds) * 0.5), 6))
    ax.bar(x - w/2, np.array(base_vals) * 100, width=w, label="baseline", color="#616161", alpha=0.8)
    ax.bar(x + w/2, np.array(neutral_vals) * 100, width=w, label="ind_neutral", color="#2e7d32", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Avg share of head group (%)")
    ax.set_title("Head Group Industry Share: baseline vs ind_neutral (2024-2025 avg)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {outpath}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    merged = load_merged()
    merged = compute_industry_stats(merged)

    # 全市场等权基准 (用于 baseline 和 ind_neutral)
    bm_mkt = merged.groupby("date")["fwd_ret"].transform("mean")
    merged["fwd_ret_mkt_excess"] = merged["fwd_ret"] - bm_mkt

    print(f"[回测] baseline (original factor + market bench) ...")
    res_base = run_one_variant(merged, factor_col="factor",
                                excess_col="fwd_ret_mkt_excess",
                                n_groups=N_GROUPS, return_head_members=True)
    print(f"  IC={res_base['ic_mean']:+.4f}, ICIR={res_base['ic_ir']:+.2f}, "
          f"ann={res_base['head_ann_excess']*100:+.2f}%, "
          f"SR={res_base['head_sharpe']:+.2f}, DD={res_base['head_max_dd']*100:+.1f}%, "
          f"n_days={res_base['n_dates']}")

    print(f"[回测] ind_bench (original factor + industry bench) ...")
    # 对 ind_bench, 需要丢弃 fwd_ret_ind_excess 为 NaN 的股票(行业 size<5)
    merged_b = merged.dropna(subset=["fwd_ret_ind_excess"])
    res_ib = run_one_variant(merged_b, factor_col="factor",
                              excess_col="fwd_ret_ind_excess",
                              n_groups=N_GROUPS, return_head_members=False)
    print(f"  IC={res_ib['ic_mean']:+.4f}, ICIR={res_ib['ic_ir']:+.2f}, "
          f"ann={res_ib['head_ann_excess']*100:+.2f}%, "
          f"SR={res_ib['head_sharpe']:+.2f}, DD={res_ib['head_max_dd']*100:+.1f}%, "
          f"n_days={res_ib['n_dates']}")

    print(f"[回测] ind_neutral (industry-neutralized factor + market bench) ...")
    merged_n = merged.dropna(subset=["factor_neutral"])
    res_in = run_one_variant(merged_n, factor_col="factor_neutral",
                              excess_col="fwd_ret_mkt_excess",
                              n_groups=N_GROUPS, return_head_members=True)
    print(f"  IC={res_in['ic_mean']:+.4f}, ICIR={res_in['ic_ir']:+.2f}, "
          f"ann={res_in['head_ann_excess']*100:+.2f}%, "
          f"SR={res_in['head_sharpe']:+.2f}, DD={res_in['head_max_dd']*100:+.1f}%, "
          f"n_days={res_in['n_dates']}")

    # 指标汇总
    rows = []
    for name, res in [("baseline", res_base), ("ind_bench", res_ib), ("ind_neutral", res_in)]:
        rows.append({
            "variant": name,
            "n_dates": res["n_dates"],
            "ic_mean": round(res["ic_mean"], 6),
            "ic_std": round(res["ic_std"], 6),
            "ic_ir": round(res["ic_ir"], 6),
            "head_ann_excess": round(res["head_ann_excess"], 6),
            "head_ann_excess_vol": round(res["head_ann_excess_vol"], 6),
            "head_sharpe": round(res["head_sharpe"], 6),
            "head_max_dd": round(res["head_max_dd"], 6),
        })
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(OUT_DIR, "compare_metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[CSV] compare_metrics.csv")
    print(mdf.to_string(index=False))

    # 图: NAV 对比
    plot_nav_compare(
        {"baseline": res_base, "ind_bench": res_ib, "ind_neutral": res_in},
        os.path.join(OUT_DIR, "nav_compare.png"),
    )

    # 图: baseline vs ind_neutral 头组行业占比
    ind_name_map = merged[["ind_code", "ind_name"]].drop_duplicates().set_index("ind_code")["ind_name"].to_dict()
    plot_head_industry_share(
        res_base["head_ind_share"], res_in["head_ind_share"],
        ind_name_map,
        os.path.join(OUT_DIR, "head_industry_share.png"),
    )

    # 摘要
    summary = {
        "baseline": {
            "ic_mean": res_base["ic_mean"],
            "ic_ir": res_base["ic_ir"],
            "head_ann_excess": res_base["head_ann_excess"],
            "head_sharpe": res_base["head_sharpe"],
            "head_max_dd": res_base["head_max_dd"],
        },
        "ind_bench": {
            "ic_mean": res_ib["ic_mean"],
            "ic_ir": res_ib["ic_ir"],
            "head_ann_excess": res_ib["head_ann_excess"],
            "head_sharpe": res_ib["head_sharpe"],
            "head_max_dd": res_ib["head_max_dd"],
        },
        "ind_neutral": {
            "ic_mean": res_in["ic_mean"],
            "ic_ir": res_in["ic_ir"],
            "head_ann_excess": res_in["head_ann_excess"],
            "head_sharpe": res_in["head_sharpe"],
            "head_max_dd": res_in["head_max_dd"],
        },
        "note": {
            "DATE_RANGE": f"{DATE_START} ~ {DATE_END}",
            "N_GROUPS": N_GROUPS,
            "FWD_DAYS": FWD_DAYS,
            "MIN_IND_SIZE": MIN_IND_SIZE,
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"[JSON] summary.json")

    print(f"\n[完成] 产物在 {OUT_DIR}")


if __name__ == "__main__":
    main()
