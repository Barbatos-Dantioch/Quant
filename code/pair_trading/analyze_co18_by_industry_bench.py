"""
CO-18 分行业头组超额: 全市场基准 vs 行业基准 对比

目的: 在 analyze_co18_by_industry.py 的基础上,
对每个行业同时计算两种头组超额:
    head_ann_excess_mkt = 头组收益 - 全市场等权均值 (原有口径)
    head_ann_excess_ind = 头组收益 - 行业等权均值 (新口径)
        注意: "行业等权" = 行业内除头组外的股票等权? 还是所有股票 (含头组自己) 等权?
              采用常规口径 = "行业内所有当日可交易股票等权" (含头组自己)
              这样 diff = head_ann_excess_mkt - head_ann_excess_ind = 行业相对全市场的 beta 贡献

输入:
    output/0416_pair_factor/CO-18/factor_df.pkl
    Data/all/price_non_st.pkl

输出 (output/0416_pair_factor/_analysis/CO18_by_industry_bench/):
    industry_bench_compare.csv   两种口径对比表
    industry_bench_compare.png   柱状图对比 (两组并排)
    summary.json
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
OUT_DIR = "output/0416_pair_factor/_analysis/CO18_by_industry_bench"

DATE_START = "2024-01-01"
DATE_END = "2025-12-31"
N_GROUPS = 5
MIN_AVG_PER_DATE = 20
FWD_DAYS = 1

rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_merged():
    print(f"[读] factor: {FACTOR_PATH}")
    fdf = pd.read_pickle(FACTOR_PATH)
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf[(fdf["date"] >= DATE_START) & (fdf["date"] <= DATE_END)].copy()
    print(f"   因子: {len(fdf)} 条, {fdf['date'].nunique()} 天")

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


def compute_benchmarks(merged: pd.DataFrame):
    """给每条样本加两列:
        bm_mkt: 当天全市场等权 fwd_ret
        bm_ind: 当天所在行业等权 fwd_ret
    """
    merged = merged.copy()
    merged["bm_mkt"] = merged.groupby("date")["fwd_ret"].transform("mean")
    merged["bm_ind"] = merged.groupby(["date", "ind_code"])["fwd_ret"].transform("mean")
    return merged


def industry_backtest_dual(ind_df: pd.DataFrame, n_groups: int) -> dict | None:
    """单行业分层回测, 同时算 mkt / ind 两种头组超额。"""
    ic_list = []
    head_excess_mkt = []
    head_excess_ind = []
    dates_list = []

    for dt, grp in ind_df.groupby("date"):
        if len(grp) < n_groups * 2:
            continue
        g = grp.copy()
        g["group"] = pd.qcut(
            g["factor"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        head = g[g["group"] == n_groups - 1]
        if len(head) == 0:
            continue

        dates_list.append(dt)
        head_ret = head["fwd_ret"].mean()
        # bm_mkt 和 bm_ind 都是整列一样 (按 date 或 date+ind 做 transform, 所有行一致),
        # 随便取一个头组股票的值即可
        bm_mkt_dt = head["bm_mkt"].iloc[0]
        bm_ind_dt = head["bm_ind"].iloc[0]
        head_excess_mkt.append(head_ret - bm_mkt_dt)
        head_excess_ind.append(head_ret - bm_ind_dt)

        ic, _ = spearmanr(g["factor"], g["fwd_ret"])
        if not np.isnan(ic):
            ic_list.append(ic)

    if len(dates_list) < 100:
        return None

    dates_arr = pd.DatetimeIndex(dates_list)
    he_mkt = pd.Series(head_excess_mkt, index=dates_arr)
    he_ind = pd.Series(head_excess_ind, index=dates_arr)

    ann = 250.0
    out = {
        "n_dates": len(dates_list),
        "ic_mean": float(np.mean(ic_list)),
        "ic_ir": float(np.mean(ic_list) / (np.std(ic_list) + 1e-12)),
        "head_ann_excess_mkt": float(he_mkt.mean() * ann),
        "head_ann_excess_ind": float(he_ind.mean() * ann),
        "head_sharpe_mkt": float(he_mkt.mean() * ann / (he_mkt.std() * np.sqrt(ann) + 1e-12)),
        "head_sharpe_ind": float(he_ind.mean() * ann / (he_ind.std() * np.sqrt(ann) + 1e-12)),
    }
    return out


def plot_compare(metrics_df: pd.DataFrame, outpath: str):
    """每个行业: 左柱 head_ann_excess_mkt, 右柱 head_ann_excess_ind, 下方线显示 beta_contrib。
       按 head_ann_excess_mkt 降序排列 (与之前 industry_ranking 同步)。"""
    df = metrics_df.sort_values("head_ann_excess_mkt", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    w = 0.38
    labels = df["ind_name"].tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(df) * 0.55), 9),
                                    gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)

    ax1.bar(x - w/2, df["head_ann_excess_mkt"] * 100, width=w,
            color="#616161", alpha=0.85, label="vs Market eq-weighted")
    ax1.bar(x + w/2, df["head_ann_excess_ind"] * 100, width=w,
            color="#2e7d32", alpha=0.85, label="vs Industry eq-weighted")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_ylabel("Head Ann Excess (%)")
    ax1.set_title(
        f"CO-18 Head Ann Excess: Market bench vs Industry bench  "
        f"(2024-2025, n_groups={N_GROUPS})",
        fontsize=11,
    )
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right")

    for i, row in df.iterrows():
        ax1.text(i - w/2, row["head_ann_excess_mkt"] * 100 + (0.3 if row["head_ann_excess_mkt"] > 0 else -0.7),
                 f"{row['head_ann_excess_mkt']*100:.1f}", ha="center", fontsize=7, color="#424242")
        ax1.text(i + w/2, row["head_ann_excess_ind"] * 100 + (0.3 if row["head_ann_excess_ind"] > 0 else -0.7),
                 f"{row['head_ann_excess_ind']*100:.1f}", ha="center", fontsize=7, color="#1b5e20")

    beta = df["head_ann_excess_mkt"] - df["head_ann_excess_ind"]
    colors_beta = ["#1565c0" if v > 0 else "#c62828" for v in beta]
    ax2.bar(x, beta * 100, color=colors_beta, alpha=0.85)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("beta contribution (%)\n= mkt - ind")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(beta):
        ax2.text(i, v * 100 + (0.2 if v > 0 else -0.5),
                 f"{v*100:+.1f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {outpath}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    merged = load_merged()
    merged = compute_benchmarks(merged)

    # 按行业样本量过滤
    ind_size = merged.groupby(["ind_code", "ind_name"]).agg(
        n_rows=("factor", "count"), n_dates=("date", "nunique"),
    )
    ind_size["avg_per_date"] = ind_size["n_rows"] / ind_size["n_dates"]
    keep_inds = ind_size[ind_size["avg_per_date"] >= MIN_AVG_PER_DATE].index.get_level_values("ind_code").tolist()
    print(f"[过滤] 日均 >= {MIN_AVG_PER_DATE}: 保留 {len(keep_inds)} / {len(ind_size)} 个行业")

    rows = []
    for ind_code in keep_inds:
        ind_name = ind_size.xs(ind_code, level="ind_code").index[0]
        ind_df = merged[merged["ind_code"] == ind_code]
        res = industry_backtest_dual(ind_df, N_GROUPS)
        if res is None:
            continue

        rows.append({
            "ind_code": ind_code,
            "ind_name": ind_name,
            "n_dates": res["n_dates"],
            "ic_mean": round(res["ic_mean"], 6),
            "ic_ir": round(res["ic_ir"], 6),
            "head_ann_excess_mkt": round(res["head_ann_excess_mkt"], 6),
            "head_ann_excess_ind": round(res["head_ann_excess_ind"], 6),
            "beta_contrib": round(res["head_ann_excess_mkt"] - res["head_ann_excess_ind"], 6),
            "head_sharpe_mkt": round(res["head_sharpe_mkt"], 6),
            "head_sharpe_ind": round(res["head_sharpe_ind"], 6),
        })

        print(f"  [{ind_code} {ind_name}]"
              f"  mkt={res['head_ann_excess_mkt']*100:+6.2f}%"
              f"  ind={res['head_ann_excess_ind']*100:+6.2f}%"
              f"  diff={100*(res['head_ann_excess_mkt']-res['head_ann_excess_ind']):+6.2f}%"
              f"  IC={res['ic_mean']:+.4f}")

    mdf = pd.DataFrame(rows).sort_values("head_ann_excess_mkt", ascending=False)
    mdf.to_csv(os.path.join(OUT_DIR, "industry_bench_compare.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n[CSV] industry_bench_compare.csv")
    print(mdf.to_string(index=False))

    plot_compare(mdf, os.path.join(OUT_DIR, "industry_bench_compare.png"))

    # 摘要统计
    summary = {
        "n_industries": int(len(mdf)),
        "mean_mkt_excess": float(mdf["head_ann_excess_mkt"].mean()),
        "mean_ind_excess": float(mdf["head_ann_excess_ind"].mean()),
        "mean_beta_contrib": float(mdf["beta_contrib"].mean()),
        "n_sign_flip": int(((mdf["head_ann_excess_mkt"] > 0) !=
                             (mdf["head_ann_excess_ind"] > 0)).sum()),
        "max_abs_diff": float(mdf["beta_contrib"].abs().max()),
        "top3_diff_positive": mdf.sort_values("beta_contrib", ascending=False).head(3)[
            ["ind_name", "head_ann_excess_mkt", "head_ann_excess_ind", "beta_contrib"]
        ].to_dict("records"),
        "top3_diff_negative": mdf.sort_values("beta_contrib", ascending=True).head(3)[
            ["ind_name", "head_ann_excess_mkt", "head_ann_excess_ind", "beta_contrib"]
        ].to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"[JSON] summary.json")
    print(f"\n[完成] 产物在 {OUT_DIR}")


if __name__ == "__main__":
    main()
