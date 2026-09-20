"""
CO-18 分行业分析

目的: 把 CO-18 (holdout + predict_decay) 的因子按 SW 一级行业拆开,
看哪些行业预测效果好、稳定性如何 (跨年度一致性)。

输入:
    output/0416_pair_factor/CO-18/factor_df.pkl
    Data/all/price_non_st.pkl

输出 (output/0416_pair_factor/_analysis/CO18_by_industry/):
    industry_metrics.csv      每行业 IC / ICIR / 头组年化超额 / Sharpe / 最大回撤
    industry_yearly_ic.csv    行业 × 年度 的 IC / 头组超额, 看稳定性
    industry_ranking.png      行业排序柱状图 (IC + 头组超额)
    industry_yearly_heatmap.png 行业 × 年度 IC 热力图
    top_industries_nav.png    按头组超额年化 Top 5 行业的累计超额净值曲线

关键口径:
    - 时间范围: 2024-01-01 ~ 2025-12-31 (与 factor_df 覆盖范围一致)
    - n_groups = 5
    - 基准 = 全市场可交易股票等权 (不是行业内等权, 方便看行业本身 beta)
    - 头组 = factor 最大的 1/5 股票
    - fwd_days = 1 (与主回测一致)
    - 最小日均截面股票数 = 20 (<20 的行业跳过)
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
OUT_DIR = "output/0416_pair_factor/_analysis/CO18_by_industry"

DATE_START = "2024-01-01"
DATE_END = "2025-12-31"
N_GROUPS = 5
MIN_AVG_PER_DATE = 20
FWD_DAYS = 1

# 中文字体 (服务器上常见的 SimHei / DejaVu Sans 兜底)
rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_inputs():
    """读 factor_df、price、行业 map, 并合并出 fwd_ret、tradable、行业。"""
    print(f"[读] factor: {FACTOR_PATH}")
    fdf = pd.read_pickle(FACTOR_PATH)
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf[(fdf["date"] >= DATE_START) & (fdf["date"] <= DATE_END)].copy()
    print(f"   因子: {len(fdf)} 条, {fdf['date'].nunique()} 天, {fdf['stock_code'].nunique()} 只")

    print(f"[读] price: {PRICE_PATH}")
    price = pd.read_pickle(PRICE_PATH)
    price["date"] = pd.to_datetime(price["date"])
    price = price[(price["date"] >= DATE_START) & (price["date"] <= DATE_END)].copy()
    # 可交易标记 (同 build_tradable_mask)
    price["tradable"] = (
        (price["status"] == 0)
        & (price["open_price"] > 0)
        & price["open_price"].notna()
        & price["close_price"].notna()
    )
    # fwd_ret (下一日 close-to-close 收益) + fwd 日可交易
    price = price.sort_values(["stock_code", "date"])
    price["fwd_close"] = price.groupby("stock_code")["close_price"].shift(-FWD_DAYS)
    price["fwd_tradable"] = price.groupby("stock_code")["tradable"].shift(-1)
    price["fwd_ret"] = (price["fwd_close"] - price["close_price"]) / price["close_price"]

    # 行业 map (每只股票取第一个非空 sw_industry_l1_code / name)
    ind_src = price.dropna(subset=["sw_industry_l1_code"])
    ind_map = ind_src.groupby("stock_code").agg(
        ind_code=("sw_industry_l1_code", "first"),
        ind_name=("sw_industry_l1_name", "first"),
    )
    print(f"   行业 map: {len(ind_map)} 只, {ind_map['ind_code'].nunique()} 个一级行业")

    merged = fdf.merge(
        price[["date", "stock_code", "tradable", "fwd_ret", "fwd_tradable"]],
        on=["date", "stock_code"],
        how="inner",
    )
    merged = merged[merged["tradable"] & merged["fwd_tradable"].fillna(False)].copy()
    merged = merged.dropna(subset=["factor", "fwd_ret"])
    merged = merged.merge(ind_map, left_on="stock_code", right_index=True, how="left")
    merged = merged.dropna(subset=["ind_code"]).copy()
    print(f"   合并后: {len(merged)} 条 (过滤不可交易 + 缺 fwd_ret)")

    return merged


def compute_market_benchmark(merged: pd.DataFrame) -> pd.Series:
    """全市场等权基准: 每天所有可交易股票的 fwd_ret 均值 (不看行业, 不看是否在因子内)。

    注意: merged 已经按因子覆盖 + 可交易过滤过, 再按 date 做均值得到"全市场等权 fwd_ret"。
    """
    bm = merged.groupby("date")["fwd_ret"].mean()
    bm.name = "bm_fwd_ret"
    return bm


def industry_backtest(
    ind_df: pd.DataFrame,
    bm_fwd_ret: pd.Series,
    n_groups: int,
) -> dict:
    """单行业分层回测 + 头组 vs 全市场基准。

    返回 dict: ic_list, ic_mean, ic_ir, head_excess_ret(Series), head_excess_nav(Series),
              head_ann_excess, head_ann_excess_vol, head_sharpe, head_max_dd, n_dates, avg_per_date
    """
    ic_list = []
    head_ret_list = []
    dates_list = []

    for dt, grp in ind_df.groupby("date"):
        if len(grp) < n_groups * 2:
            continue
        grp = grp.copy()
        grp["group"] = pd.qcut(
            grp["factor"].rank(method="first"),
            q=n_groups, labels=False, duplicates="drop",
        )
        dates_list.append(dt)
        head_ret = grp.loc[grp["group"] == n_groups - 1, "fwd_ret"].mean()
        head_ret_list.append(head_ret if not np.isnan(head_ret) else 0.0)
        ic, _ = spearmanr(grp["factor"], grp["fwd_ret"])
        if not np.isnan(ic):
            ic_list.append(ic)

    if len(dates_list) < 100:
        return None

    dates_arr = pd.DatetimeIndex(dates_list)
    head_ret_arr = pd.Series(head_ret_list, index=dates_arr)
    bm_aligned = bm_fwd_ret.reindex(dates_arr).fillna(0.0)
    head_excess_ret = head_ret_arr - bm_aligned
    head_excess_nav = (1 + head_excess_ret).cumprod()

    ann = 250.0
    head_ann_excess = head_excess_ret.mean() * ann
    head_ann_excess_vol = head_excess_ret.std() * np.sqrt(ann)
    head_sharpe = head_ann_excess / head_ann_excess_vol if head_ann_excess_vol > 0 else 0.0
    dd = ((head_excess_nav - head_excess_nav.cummax()) / head_excess_nav.cummax()).min()

    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list))
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

    return {
        "ic_list": ic_list,
        "ic_dates": dates_arr,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "head_excess_ret": head_excess_ret,
        "head_excess_nav": head_excess_nav,
        "head_ann_excess": float(head_ann_excess),
        "head_ann_excess_vol": float(head_ann_excess_vol),
        "head_sharpe": float(head_sharpe),
        "head_max_dd": float(dd),
        "n_dates": len(dates_list),
        "avg_per_date": ind_df.groupby("date").size().mean(),
    }


def yearly_breakdown(
    ind_df: pd.DataFrame,
    bm_fwd_ret: pd.Series,
    n_groups: int,
) -> pd.DataFrame:
    """行业 × 年度 的 IC 和头组超额, 用来做热力图。"""
    ind_df = ind_df.copy()
    ind_df["year"] = ind_df["date"].dt.year

    rows = []
    for year, yrdf in ind_df.groupby("year"):
        ic_list = []
        head_excess_list = []
        for dt, grp in yrdf.groupby("date"):
            if len(grp) < n_groups * 2:
                continue
            grp = grp.copy()
            grp["group"] = pd.qcut(
                grp["factor"].rank(method="first"),
                q=n_groups, labels=False, duplicates="drop",
            )
            head_ret = grp.loc[grp["group"] == n_groups - 1, "fwd_ret"].mean()
            bm_dt = bm_fwd_ret.get(dt, np.nan)
            if not np.isnan(head_ret) and not np.isnan(bm_dt):
                head_excess_list.append(head_ret - bm_dt)
            ic, _ = spearmanr(grp["factor"], grp["fwd_ret"])
            if not np.isnan(ic):
                ic_list.append(ic)
        if len(ic_list) < 30:
            continue
        rows.append({
            "year": int(year),
            "n_dates": len(ic_list),
            "ic_mean": float(np.mean(ic_list)),
            "ic_ir": float(np.mean(ic_list) / (np.std(ic_list) + 1e-12)),
            "head_ann_excess": float(np.mean(head_excess_list) * 250.0),
        })
    return pd.DataFrame(rows)


def plot_ranking(metrics_df: pd.DataFrame, outpath: str):
    """行业排序柱状图: 左轴 head_ann_excess, 右轴 Rank IC, 按 head_ann_excess 降序。"""
    df = metrics_df.sort_values("head_ann_excess", ascending=False).reset_index(drop=True)
    labels = df["ind_label"].tolist()
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(max(14, len(df) * 0.45), 6))
    colors = ["#2e7d32" if v > 0 else "#c62828" for v in df["head_ann_excess"]]
    bars = ax1.bar(x, df["head_ann_excess"] * 100, color=colors, alpha=0.75,
                   label="Head Group Ann Excess (%)")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_ylabel("Head Group Ann Excess (%)", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, df["ic_mean"], "o-", color="#1565c0", lw=1.5, ms=6,
             label="Rank IC")
    ax2.set_ylabel("Rank IC", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax2.axhline(0, color="#1565c0", lw=0.5, alpha=0.3)

    for i, (exc, ic, icir) in enumerate(zip(df["head_ann_excess"], df["ic_mean"], df["ic_ir"])):
        ax1.text(i, exc * 100 + (0.3 if exc > 0 else -0.6), f"{exc*100:.1f}%",
                 ha="center", fontsize=7)
        ax2.text(i, ic + 0.003, f"IR={icir:.2f}", ha="center", fontsize=6, color="#1565c0")

    ax1.set_title(f"CO-18 Industry Breakdown (2024-2025, n_groups={N_GROUPS}, "
                  f"benchmark=market equal-weighted)\n"
                  f"Bars=Head Ann Excess (sorted desc), Line=Rank IC, Text=ICIR",
                  fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {outpath}")


def plot_yearly_heatmap(yearly_df: pd.DataFrame, outpath: str):
    """行业 × 年度 的 IC 热力图。"""
    piv = yearly_df.pivot(index="ind_label", columns="year", values="ic_mean")
    # 行按全期 IC 均值降序 (更好看)
    row_order = piv.mean(axis=1).sort_values(ascending=False).index
    piv = piv.reindex(row_order)

    fig, ax = plt.subplots(figsize=(max(6, piv.shape[1] * 1.6 + 2), max(6, piv.shape[0] * 0.3)))
    vmax = max(abs(piv.min().min()), abs(piv.max().max()))
    im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=10)
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > vmax * 0.55 else "black")

    ax.set_title(f"CO-18 Rank IC: industry × year (sorted by mean IC)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Rank IC")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {outpath}")


def plot_top_industries_nav(metrics_df: pd.DataFrame,
                             industry_result: dict,
                             outpath: str,
                             top_n: int = 5):
    """Top N 行业的头组超额净值曲线 (按 head_ann_excess 排序)。"""
    top = metrics_df.sort_values("head_ann_excess", ascending=False).head(top_n)
    bot = metrics_df.sort_values("head_ann_excess", ascending=True).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    ax = axes[0]
    for _, row in top.iterrows():
        nav = industry_result[row["ind_code"]]["head_excess_nav"]
        ax.plot(nav.index, nav.values, lw=1.5,
                label=f"{row['ind_label']} (ann={row['head_ann_excess']*100:.1f}%)")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title(f"Top {top_n} Industries (by Head Ann Excess)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylabel("Head Excess NAV (vs market eq-weighted)")

    ax = axes[1]
    for _, row in bot.iterrows():
        nav = industry_result[row["ind_code"]]["head_excess_nav"]
        ax.plot(nav.index, nav.values, lw=1.5,
                label=f"{row['ind_label']} (ann={row['head_ann_excess']*100:.1f}%)")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title(f"Bottom {top_n} Industries (by Head Ann Excess)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(f"CO-18 Head Group Excess NAV by Industry (2024-2025)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {outpath}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    merged = load_inputs()

    # 行业样本统计 + 过滤
    ind_size = merged.groupby(["ind_code", "ind_name"]).agg(
        n_rows=("factor", "count"),
        n_dates=("date", "nunique"),
        n_stocks=("stock_code", "nunique"),
    )
    ind_size["avg_per_date"] = ind_size["n_rows"] / ind_size["n_dates"]
    keep_inds = ind_size[ind_size["avg_per_date"] >= MIN_AVG_PER_DATE].index.get_level_values("ind_code").tolist()
    print(f"[过滤] 日均 >= {MIN_AVG_PER_DATE}: 保留 {len(keep_inds)} / {len(ind_size)} 个行业")
    dropped = ind_size[ind_size["avg_per_date"] < MIN_AVG_PER_DATE]
    if len(dropped) > 0:
        print(f"[过滤] 跳过: {dropped.reset_index()[['ind_name','avg_per_date']].to_dict('records')}")

    bm_fwd_ret = compute_market_benchmark(merged)
    print(f"[基准] 全市场等权 fwd_ret: {len(bm_fwd_ret)} 天, 均值 {bm_fwd_ret.mean()*10000:.1f} bps/天")

    # 全域回测 (每行业一次)
    print(f"\n[回测] 各行业分层回测 (n_groups={N_GROUPS}) ...")
    metrics_rows = []
    yearly_rows = []
    industry_result = {}

    for ind_code in keep_inds:
        ind_name = ind_size.loc[ind_code, :].index[0] if isinstance(ind_size.loc[ind_code, :], pd.DataFrame) else \
                    ind_size.xs(ind_code, level="ind_code").index[0]
        ind_df = merged[merged["ind_code"] == ind_code]

        res = industry_backtest(ind_df, bm_fwd_ret, N_GROUPS)
        if res is None:
            print(f"  [{ind_code} {ind_name}] 有效日期不足, 跳过")
            continue
        industry_result[ind_code] = res

        metrics_rows.append({
            "ind_code": ind_code,
            "ind_name": ind_name,
            "ind_label": f"{ind_name}",
            "n_dates": res["n_dates"],
            "avg_per_date": round(float(res["avg_per_date"]), 1),
            "ic_mean": round(res["ic_mean"], 6),
            "ic_std": round(res["ic_std"], 6),
            "ic_ir": round(res["ic_ir"], 6),
            "head_ann_excess": round(res["head_ann_excess"], 6),
            "head_ann_excess_vol": round(res["head_ann_excess_vol"], 6),
            "head_sharpe": round(res["head_sharpe"], 6),
            "head_max_dd": round(res["head_max_dd"], 6),
        })

        # 年度分解
        yr_df = yearly_breakdown(ind_df, bm_fwd_ret, N_GROUPS)
        if len(yr_df) > 0:
            yr_df.insert(0, "ind_code", ind_code)
            yr_df.insert(1, "ind_label", ind_name)
            yearly_rows.append(yr_df)

        print(f"  [{ind_code} {ind_name}] IC={res['ic_mean']:+.4f}, "
              f"ICIR={res['ic_ir']:+.3f}, "
              f"HeadExcAnn={res['head_ann_excess']*100:+.2f}%, "
              f"Sharpe={res['head_sharpe']:+.2f}, "
              f"DD={res['head_max_dd']*100:+.2f}%, "
              f"n_days={res['n_dates']}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("head_ann_excess", ascending=False)
    metrics_df.to_csv(os.path.join(OUT_DIR, "industry_metrics.csv"),
                      index=False, encoding="utf-8-sig")
    print(f"\n[CSV] industry_metrics.csv  (Top10 by HeadExcAnn):")
    print(metrics_df.head(10).to_string(index=False))

    yearly_df = pd.concat(yearly_rows, ignore_index=True)
    yearly_df.to_csv(os.path.join(OUT_DIR, "industry_yearly_ic.csv"),
                     index=False, encoding="utf-8-sig")
    print(f"\n[CSV] industry_yearly_ic.csv")

    # 图
    plot_ranking(metrics_df, os.path.join(OUT_DIR, "industry_ranking.png"))
    plot_yearly_heatmap(yearly_df, os.path.join(OUT_DIR, "industry_yearly_heatmap.png"))
    plot_top_industries_nav(metrics_df, industry_result,
                             os.path.join(OUT_DIR, "top_industries_nav.png"), top_n=5)

    # 摘要
    summary = {
        "n_industries_kept": int(len(metrics_df)),
        "n_industries_dropped": int(len(dropped)),
        "mean_ic_across_industries": float(metrics_df["ic_mean"].mean()),
        "mean_head_excess_across_industries": float(metrics_df["head_ann_excess"].mean()),
        "n_industries_positive_head_excess": int((metrics_df["head_ann_excess"] > 0).sum()),
        "n_industries_positive_ic": int((metrics_df["ic_mean"] > 0).sum()),
        "top5_head_excess": metrics_df.head(5)[["ind_name", "ic_mean", "ic_ir",
                                                  "head_ann_excess", "head_sharpe"]].to_dict("records"),
        "bot5_head_excess": metrics_df.tail(5)[["ind_name", "ic_mean", "ic_ir",
                                                  "head_ann_excess", "head_sharpe"]].to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[JSON] summary.json")

    print(f"\n[完成] 产物在 {OUT_DIR}")


if __name__ == "__main__":
    main()
