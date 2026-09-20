#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OU 配对组合 Barra (CNE6/SW21) 因子暴露分析
============================================

对某个 OU 配对实验的每日多/空持仓, 计算组合在 Barra 52 因子
(20 风格 + 31 申万2021 行业 + COUNTRY) 上的暴露时序与汇总。

口径:
  - 等权 + 美元中性: 多头每只权重 +1/N_long, 空头每只权重 +1/N_short。
  - 单腿因子暴露(某日) = 该腿持仓股票在该因子暴露的等权均值。
      风格因子: 已截面标准化, 暴露单位 = 标准差; 缺失值按股票跳过 (skipna)。
      行业因子: 0/1 哑变量, 缺失填 0, 暴露 = 组合在该行业的持仓占比。
      COUNTRY: 恒为 1, 单腿暴露 ≈ 1。
  - 多空净暴露 = 多头腿暴露 − 空头腿暴露 (dollar-neutral 组合的净因子敞口)。

输入:
  HOLDINGS_PATH : stock_pred.parquet (列 date, stock_code, side[long/short], ...)
  EXPOSURE_PATH : Data/all/barra_exposure_cne6_sw21.pkl (date, stock_code, <52 因子>)

输出 (OUT_DIR):
  exposure_long.csv / exposure_short.csv / exposure_net.csv  各腿因子暴露时序
  exposure_summary.csv      每因子每腿: 均值/标准差/|均值|/t值 + 因子分类
  coverage.csv              每日持仓被 Barra 覆盖的比例
  barra_exposure_summary.md  关键暴露汇总表
  style_exposure_ts.png      风格因子净暴露时序 (|均值| top8)
  avg_exposure_style.png     风格因子平均暴露条形 (多/空/净)
  avg_exposure_industry.png  行业因子平均净暴露条形
"""
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ── 路径 ──
EXP_TAG = "OU-P00-zs-cv5-h5"
HOLDINGS_PATH = f"/root/quant/output/0506_ou_pair/barra/{EXP_TAG}/stock_pred.parquet"
EXPOSURE_PATH = "/root/quant/Data/all/barra_exposure_cne6_sw21.pkl"
OUT_DIR = f"/root/quant/output/0506_ou_pair/barra/{EXP_TAG}/barra_exposure"

# ── 因子分类 (FACTOR_ID 0-19 风格, 20-50 行业, 51 COUNTRY) ──
STYLE_FACTORS = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP',
                 'LEVERAGE', 'LIQUIDTY', 'MIDCAP', 'DIVYILD', 'EARNQLTY', 'EARNVAR',
                 'INVSQLTY', 'LTREVRSL', 'PROFIT', 'ANALSENTI', 'INDMOM', 'SEASON', 'STREVRSL']
COUNTRY_FACTOR = 'COUNTRY'


def load_inputs():
    print(f"读取持仓: {HOLDINGS_PATH}")
    hold = pd.read_parquet(HOLDINGS_PATH, columns=['date', 'stock_code', 'side'])
    hold['date'] = pd.to_datetime(hold['date'])
    hold['stock_code'] = hold['stock_code'].astype(str).str.zfill(6)
    dates = set(hold['date'].unique())
    print(f"  持仓: {len(hold):,} 行, {len(dates)} 个交易日, "
          f"side 分布 {hold['side'].value_counts().to_dict()}")

    print(f"读取因子暴露: {EXPOSURE_PATH}")
    exp = pd.read_pickle(EXPOSURE_PATH)
    exp['stock_code'] = exp['stock_code'].astype(str).str.zfill(6)
    # 只保留持仓涉及的交易日, 降内存
    exp = exp[exp['date'].isin(dates)].reset_index(drop=True)
    factors = [c for c in exp.columns if c not in ('date', 'stock_code')]
    industry_factors = [c for c in factors if c not in STYLE_FACTORS and c != COUNTRY_FACTOR]
    print(f"  暴露(过滤到持仓日): {len(exp):,} 行; 因子 {len(factors)} "
          f"(风格 {len(STYLE_FACTORS)} + 行业 {len(industry_factors)} + COUNTRY)")
    return hold, exp, factors, industry_factors


def compute_leg_exposure(hold, exp, factors, industry_factors):
    """合并持仓与暴露, 返回多头/空头每日因子暴露 (等权均值) 及覆盖率。"""
    merged = hold.merge(exp, on=['date', 'stock_code'], how='left')
    # 覆盖率: 命中 Barra 暴露的持仓占比 (用 COUNTRY 非空判断该行是否有 Barra 数据)
    merged['_covered'] = merged[COUNTRY_FACTOR].notna()
    coverage = (merged.groupby('date')['_covered'].mean()
                .rename('coverage').reset_index())
    n_missing = int((~merged['_covered']).sum())
    print(f"  合并: {len(merged):,} 行, 未命中 Barra 暴露 {n_missing:,} "
          f"({n_missing/len(merged)*100:.2f}%), 日均覆盖率 {coverage['coverage'].mean()*100:.2f}%")

    # 行业 + COUNTRY 缺失填 0 (哑变量语义: 不属于该行业/无数据); 风格保留 NaN 由 mean skipna 跳过
    merged[industry_factors + [COUNTRY_FACTOR]] = \
        merged[industry_factors + [COUNTRY_FACTOR]].fillna(0.0)

    # 仅对命中 Barra 的持仓做暴露统计 (未命中行风格全 NaN, 不影响 skipna 均值)
    grp = merged.groupby(['date', 'side'])[factors].mean()
    long_exp = grp.xs('long', level='side').sort_index()
    short_exp = grp.xs('short', level='side').sort_index()
    # 对齐两腿日期 (理论一致), 取交集
    common = long_exp.index.intersection(short_exp.index)
    long_exp = long_exp.loc[common]
    short_exp = short_exp.loc[common]
    net_exp = long_exp - short_exp
    return long_exp, short_exp, net_exp, coverage


def summarize(long_exp, short_exp, net_exp, factors, industry_factors):
    """每因子每腿的均值/标准差/|均值|/t值 汇总。"""
    def _stats(df, leg):
        mean = df.mean()
        std = df.std()
        n = len(df)
        tval = mean / (std / np.sqrt(n)).replace(0, np.nan)
        return pd.DataFrame({'leg': leg, 'mean': mean, 'std': std,
                             'abs_mean': mean.abs(), 'tstat': tval})

    out = pd.concat([_stats(long_exp, 'long'),
                     _stats(short_exp, 'short'),
                     _stats(net_exp, 'net')]).reset_index().rename(columns={'index': 'factor'})

    def _ftype(f):
        if f in STYLE_FACTORS:
            return 'style'
        if f == COUNTRY_FACTOR:
            return 'country'
        return 'industry'
    out['factor_type'] = out['factor'].map(_ftype)
    return out


def write_markdown(summary, out_path):
    """关键暴露汇总 Markdown: 净暴露按 |均值| 排序的风格/行业 top。"""
    net = summary[summary['leg'] == 'net'].copy()
    style = net[net['factor_type'] == 'style'].sort_values('abs_mean', ascending=False)
    indu = net[net['factor_type'] == 'industry'].sort_values('abs_mean', ascending=False)

    def _tbl(df, k=None):
        d = df.head(k) if k else df
        lines = ["| 因子 | 净暴露均值 | 标准差 | t值 |", "|---|---:|---:|---:|"]
        for _, r in d.iterrows():
            lines.append(f"| {r['factor']} | {r['mean']:+.4f} | {r['std']:.4f} | {r['tstat']:+.2f} |")
        return "\n".join(lines)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# {EXP_TAG} 组合 Barra 因子暴露汇总\n\n")
        f.write("口径: 等权美元中性多空组合, 净暴露 = 多头腿均值 − 空头腿均值。"
                "风格因子单位为标准差, 行业为持仓占比之差。\n\n")
        f.write("## 风格因子净暴露 (按 |均值| 排序)\n\n")
        f.write(_tbl(style) + "\n\n")
        f.write("## 行业因子净暴露 Top 15 (按 |均值| 排序)\n\n")
        f.write(_tbl(indu, 15) + "\n")
    print(f"  已写: {out_path}")


def plot_style_ts(net_exp, summary, out_path):
    """风格因子净暴露时序 (|净均值| top8)。"""
    net_style = summary[(summary['leg'] == 'net') & (summary['factor_type'] == 'style')]
    top = net_style.sort_values('abs_mean', ascending=False).head(8)['factor'].tolist()
    plt.figure(figsize=(14, 7))
    for f in top:
        plt.plot(net_exp.index, net_exp[f], lw=1.3, alpha=0.85, label=f)
    plt.axhline(0, color='gray', ls='--', alpha=0.6)
    plt.title(f"{EXP_TAG} 多空净暴露·风格因子时序 (|均值| Top8)")
    plt.xlabel("日期"); plt.ylabel("净暴露 (标准差)")
    plt.legend(ncol=4, loc='upper center', fontsize=9)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=140); plt.close('all')
    print(f"  已存图: {out_path}")


def plot_style_bar(summary, out_path):
    """风格因子平均暴露条形 (多/空/净)。"""
    piv = summary[summary['factor_type'] == 'style'].pivot(
        index='factor', columns='leg', values='mean').reindex(STYLE_FACTORS)
    x = np.arange(len(STYLE_FACTORS)); w = 0.27
    plt.figure(figsize=(15, 6))
    plt.bar(x - w, piv['long'], w, label='多头腿', color='red', alpha=0.8)
    plt.bar(x, piv['short'], w, label='空头腿', color='green', alpha=0.8)
    plt.bar(x + w, piv['net'], w, label='多空净', color='blue', alpha=0.85)
    plt.axhline(0, color='gray', ls='--', alpha=0.6)
    plt.xticks(x, STYLE_FACTORS, rotation=60, ha='right', fontsize=9)
    plt.title(f"{EXP_TAG} 风格因子平均暴露 (标准差)")
    plt.ylabel("平均暴露"); plt.legend(); plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close('all')
    print(f"  已存图: {out_path}")


def plot_industry_bar(summary, industry_factors, out_path):
    """行业因子平均净暴露条形 (按净均值排序)。"""
    net = summary[(summary['leg'] == 'net') &
                  (summary['factor'].isin(industry_factors))].sort_values('mean')
    plt.figure(figsize=(15, 6))
    colors = ['green' if v < 0 else 'red' for v in net['mean']]
    plt.bar(net['factor'], net['mean'], color=colors, alpha=0.8)
    plt.axhline(0, color='gray', ls='--', alpha=0.6)
    plt.xticks(rotation=60, ha='right', fontsize=9)
    plt.title(f"{EXP_TAG} 行业因子平均净暴露 (多−空, 持仓占比之差)")
    plt.ylabel("平均净暴露"); plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close('all')
    print(f"  已存图: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hold, exp, factors, industry_factors = load_inputs()
    long_exp, short_exp, net_exp, coverage = compute_leg_exposure(
        hold, exp, factors, industry_factors)

    # ── 落地时序 CSV ──
    long_exp.to_csv(os.path.join(OUT_DIR, "exposure_long.csv"))
    short_exp.to_csv(os.path.join(OUT_DIR, "exposure_short.csv"))
    net_exp.to_csv(os.path.join(OUT_DIR, "exposure_net.csv"))
    coverage.to_csv(os.path.join(OUT_DIR, "coverage.csv"), index=False)

    summary = summarize(long_exp, short_exp, net_exp, factors, industry_factors)
    summary.to_csv(os.path.join(OUT_DIR, "exposure_summary.csv"), index=False)

    write_markdown(summary, os.path.join(OUT_DIR, "barra_exposure_summary.md"))
    plot_style_ts(net_exp, summary, os.path.join(OUT_DIR, "style_exposure_ts.png"))
    plot_style_bar(summary, os.path.join(OUT_DIR, "avg_exposure_style.png"))
    plot_industry_bar(summary, industry_factors, os.path.join(OUT_DIR, "avg_exposure_industry.png"))

    # ── 控制台关键结论 ──
    net = summary[summary['leg'] == 'net']
    print("\n===== 多空净暴露 |均值| Top10 =====")
    print(net.sort_values('abs_mean', ascending=False)
          [['factor', 'factor_type', 'mean', 'std', 'tstat']].head(10).to_string(index=False))
    print(f"\n全部产出已保存至: {OUT_DIR}")


if __name__ == "__main__":
    main()
