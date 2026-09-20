#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""非线性因子变换：从基础因子截面排名构造衍生因子。"""
import pandas as pd


def _cross_section_rank_pct(panel, base_col):
    """当日截面百分位排名，取值 [0, 1]。"""
    return panel.groupby('date', sort=False)[base_col].rank(pct=True, method='average')


def compute_mid_quantile_dev(panel, base_col, out_col=None):
    """中间最优（倒 U 型）：值越大越接近截面中位。

    mid_dev = 0.5 - |rank_pct - 0.5|，取值 [0, 0.5]。
    适用于 deep_cancel_ratio 等「中间组收益最高、两端均差」的因子。
    """
    out_col = out_col or f'{base_col}_mid_dev'
    out = panel.copy()
    rank_pct = _cross_section_rank_pct(out, base_col)
    out[out_col] = 0.5 - (rank_pct - 0.5).abs()
    return out


def compute_extreme_quantile_dev(panel, base_col, out_col=None):
    """两端最优（U 型 / 微笑曲线）：值越大越偏离截面中位。

    extreme = 2 * |rank_pct - 0.5|，取值 [0, 1]。
    适用于 realized_skew 等「头尾组收益最高、中间组最差」的因子。
    """
    out_col = out_col or f'{base_col}_extreme'
    out = panel.copy()
    rank_pct = _cross_section_rank_pct(out, base_col)
    out[out_col] = 2.0 * (rank_pct - 0.5).abs()
    return out
