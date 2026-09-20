#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归概率加权实验
================

在 h64_32(NN-mse, [64,32])选股基础上, 用一个"价差是否回归"的二分类器输出 P(revert),
归一化后作为仓位权重(从等权改为按回归概率加权), 再做加权多空回测。

- 选股(选哪些 pair + 多空腿): 复用 h64_32 回归模型 (reg.run_nn_rolling) → |预测ΔX| top20%, 方向 sign(μ-X)
- 分类器(预测 P(revert)): 环境变量 CLF_MODEL = nn / lgb
    标签 A: y = 1[ sign(μ-X)·ΔX > 0 ] (未来 5 日价差朝 μ 方向移动=回归)
    纯监督, 不加 WS 截面加权; 其余(22 特征/ROLL504/5段/单种子42/train-val)与 h64_32 一致
- 权重: 每日在选出持仓内, w = p / Σp (组内和=1, 高回归概率仓位更大)
- 回测: _pair_runner.run_longshort_backtest_weighted

输出: output/0506_ou_pair/nn/tuning/OU-P-RPL-NN-WS-noMktInd-ROLL504-h64_32-revw{NN|LGB}-cv5-h5/
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir("/root/quant")
# 固定为 h64_32 配置(noMktInd + ROLL504 + [64,32]); 这些 env 须在 import reg 前设好
os.environ.setdefault("DROP_GROUPS", "mktind")
os.environ.setdefault("ROLL_DAYS", "504")
os.environ.setdefault("NN_HIDDEN", "64,32")

_PT_DIR = "/root/quant/xgbcode/pair_trading"
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

import run_ou_pair_nn_ws as reg          # 复用: run_nn_rolling(选股) / MLP / 常量 / 预处理超参
import run_ou_pair_lgb_exp_ws as base
from _pair_runner import run_longshort_backtest_weighted, save_summary_row

import lightgbm as lgb
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CLF_MODEL = os.environ.get("CLF_MODEL", "nn").strip().lower()
assert CLF_MODEL in ("lgb", "nn"), f"CLF_MODEL 仅支持 lgb/nn, 实际 {CLF_MODEL!r}"
CLF_SEED = 42

FEATURES = reg.FEATURES
OUTPUT_DIR = base.OUTPUT_DIR
NN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "nn", "tuning")
EXP_NAME = f"OU-P-RPL-NN-WS-noMktInd-ROLL504-h64_32-revw{CLF_MODEL.upper()}-cv5-h5"
EXP_DIR = os.path.join(NN_OUTPUT_DIR, EXP_NAME)
_EPS = 1e-12


# =====================================================================
# 分类器(二分类: 价差是否回归)滚动训练 + 预测 P(revert)
# =====================================================================

def _fit_predict_lgb(X_tr, y_tr, X_va, y_va, X_te) -> np.ndarray:
    params = dict(base.LGB_PARAMS, objective="binary", metric="binary_logloss", random_state=CLF_SEED)
    dtr = lgb.Dataset(X_tr.astype(np.float32), label=y_tr, feature_name=FEATURES)
    valid, names, cb = [dtr], ["train"], [lgb.log_evaluation(period=0)]
    if X_va is not None and len(X_va) > 0:
        dva = lgb.Dataset(X_va.astype(np.float32), label=y_va, reference=dtr, feature_name=FEATURES)
        valid.append(dva); names.append("val")
        cb = [lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)]
    bst = lgb.train(params, dtr, num_boost_round=500, valid_sets=valid, valid_names=names, callbacks=cb)
    bi = bst.best_iteration if (X_va is not None and len(X_va) > 0) else bst.current_iteration()
    return bst.predict(X_te.astype(np.float32), num_iteration=bi)


def _fit_predict_nn(X_tr, y_tr, X_va, y_va, X_te) -> np.ndarray:
    # 预处理: train 丢 NaN → 中位数填补 → 缩尾 → 标准化 (与 h64_32 一致, 仅在 train fit)
    ok = ~np.isnan(X_tr).any(axis=1)
    Xc, yc = X_tr[ok], y_tr[ok]
    imp = SimpleImputer(strategy="median").fit(Xc)
    lo = np.percentile(Xc, reg.WINSOR_Q[0], axis=0); hi = np.percentile(Xc, reg.WINSOR_Q[1], axis=0)
    prep = lambda X: np.clip(imp.transform(X), lo, hi)
    Xcw = np.clip(Xc, lo, hi)
    sc = StandardScaler().fit(Xcw)
    Xtr_in = sc.transform(Xcw).astype(np.float32)
    has_va = X_va is not None and len(X_va) > 0
    Xva_in = sc.transform(prep(X_va)).astype(np.float32) if has_va else None
    Xte_in = sc.transform(prep(X_te)).astype(np.float32)

    torch.manual_seed(CLF_SEED); np.random.seed(CLF_SEED)
    g = torch.Generator().manual_seed(CLF_SEED)
    model = reg.MLP(Xtr_in.shape[1], reg.HIDDEN, reg.DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=reg.LR, weight_decay=reg.WEIGHT_DECAY)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr_in); yt = torch.from_numpy(yc.astype(np.float32))
    if has_va:
        Xv = torch.from_numpy(Xva_in); yv = torch.from_numpy(y_va.astype(np.float32))
    n = len(Xt); best = float("inf"); best_state = None; bad = 0
    max_ep = reg.MAX_EPOCHS if has_va else reg.NO_VAL_EPOCHS
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, reg.BATCH):
            idx = perm[i:i + reg.BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossfn(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        if has_va:
            model.eval()
            with torch.no_grad():
                vl = float(lossfn(model(Xv), yv))
            if vl < best - 1e-6:
                best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= reg.PATIENCE:
                    break
    if has_va and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(Xte_in)).numpy()
    return 1.0 / (1.0 + np.exp(-logit))


def train_clf_rolling(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n分类器({CLF_MODEL})滚动训练 P(revert) ...")
    df = df.copy()
    df["prob"] = np.nan
    df_legal = df[df["is_legal"]]
    pool = df_legal[~df_legal["delta_X"].isna()].copy()
    pool["_y"] = (np.sign(pool["mu"] - pool["X_t"]) * pool["delta_X"] > 0).astype(np.float32)
    pool_dates = np.array(sorted(pool["date"].unique()))

    for seg in reg.ROLLING_SEGMENTS:
        name, ts0, train_hi, test_lo, test_hi = seg
        train_hi = pd.Timestamp(train_hi); test_lo = pd.Timestamp(test_lo); test_hi = pd.Timestamp(test_hi)
        if reg._ROLL_DAYS > 0:
            elig = pool_dates[pool_dates <= train_hi]
            train_lo = pd.Timestamp(elig[-reg._ROLL_DAYS] if len(elig) >= reg._ROLL_DAYS else elig[0])
        else:
            train_lo = pd.Timestamp(ts0)
        tr = pool[(pool["date"] >= train_lo) & (pool["date"] <= train_hi)].sort_values("date")
        if len(tr) == 0:
            continue
        ud = sorted(tr["date"].unique()); nd = len(ud); ntr = int(np.floor(nd * reg.TRAIN_FRAC))
        ted = ud[ntr - 1]; gp = min(ntr + reg.TRAIN_VAL_GAP_DAYS, nd)
        if gp >= nd:
            m_tr = tr["date"] <= ted; m_va = pd.Series(False, index=tr.index)
        else:
            m_tr = tr["date"] <= ted; m_va = tr["date"] >= ud[gp]
        test = df_legal[(df_legal["date"] >= test_lo) & (df_legal["date"] <= test_hi)]
        if int(m_tr.sum()) < 100 or len(test) == 0:
            continue
        X_tr = tr.loc[m_tr, FEATURES].to_numpy(np.float64); y_tr = tr.loc[m_tr, "_y"].to_numpy(np.float32)
        nva = int(m_va.sum())
        X_va = tr.loc[m_va, FEATURES].to_numpy(np.float64) if nva > 0 else None
        y_va = tr.loc[m_va, "_y"].to_numpy(np.float32) if nva > 0 else None
        X_te = test[FEATURES].to_numpy(np.float64)
        t0 = time.time()
        if CLF_MODEL == "lgb":
            prob = _fit_predict_lgb(X_tr, y_tr, X_va, y_va, X_te)
        else:
            prob = _fit_predict_nn(X_tr, y_tr, X_va, y_va, X_te)
        df.loc[test.index, "prob"] = prob
        print(f"  [{name}] train {int(m_tr.sum()):,}/val {nva:,}/test {len(test):,}; "
              f"正样本率 {y_tr.mean():.3f}; prob 均值 {np.nanmean(prob):.3f}; {time.time()-t0:.1f}s")
    return df


# =====================================================================
# main
# =====================================================================

def _add_calmar(s):
    ann = s.get("annual_return", 0.0); mdd = s.get("max_drawdown", 0.0)
    s["calmar_ratio"] = round(ann / abs(mdd), 3) if mdd != 0 else 0.0
    return s


def main():
    print(f"\n{'='*70}\n回归概率加权 [{EXP_NAME}]  CLF={CLF_MODEL}  PID {os.getpid()}\n{'='*70}\n")
    metrics_path = os.path.join(EXP_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        print("已有 metrics.json, 跳过"); return
    os.makedirs(EXP_DIR, exist_ok=True)

    t_start = time.time()
    px = base.load_price_wides()
    df = reg._load_or_build_features(px)
    df["date"] = pd.to_datetime(df["date"])
    need = list(dict.fromkeys(FEATURES + ["date", "stock_i", "stock_j", "is_legal", "delta_X", "mu", "X_t"]))
    df = df[[c for c in need if c in df.columns]]

    # 1) h64_32 选股信号
    print("\n[1/2] h64_32 回归选股信号 ...")
    df_sel, _ = reg.run_nn_rolling(df.copy())
    # 2) 分类器 P(revert)
    print("\n[2/2] 回归概率分类器 ...")
    df_clf = train_clf_rolling(df.copy())
    df_sel = df_sel.copy()
    df_sel["prob"] = df_clf["prob"]            # 同索引对齐

    # 3) 合并: h64_32 选出持仓 + 归一化 P(revert) 加权
    held = df_sel[df_sel["is_legal"] & df_sel["pred_delta_X"].notna() & df_sel["prob"].notna()]
    long_w: Dict = {}; short_w: Dict = {}; daily: List[Dict] = []
    for d, sub in held.groupby("date"):
        n = len(sub)
        n_top = min(n, max(reg.TOP_PCT_MIN_N, int(math.ceil(n * reg.TOP_PCT))))
        top = sub.assign(_ap=sub["pred_delta_X"].abs()).nlargest(n_top, "_ap")
        p = top["prob"].clip(lower=0.0).to_numpy(np.float64)
        w = (np.full(len(top), 1.0 / len(top)) if p.sum() <= _EPS else p / p.sum())
        sgn = np.sign(top["mu"] - top["X_t"]).to_numpy()
        ii = top["stock_i"].to_numpy(); jj = top["stock_j"].to_numpy()
        lw: Dict = {}; sw: Dict = {}
        for k in range(len(top)):
            if sgn[k] > 0:
                lw[ii[k]] = lw.get(ii[k], 0.0) + w[k]; sw[jj[k]] = sw.get(jj[k], 0.0) + w[k]
            elif sgn[k] < 0:
                lw[jj[k]] = lw.get(jj[k], 0.0) + w[k]; sw[ii[k]] = sw.get(ii[k], 0.0) + w[k]
        if lw and sw:
            long_w[d] = lw; short_w[d] = sw; daily.append({"date": d, "n_top": len(top)})
    long_h = pd.Series(long_w).sort_index(); short_h = pd.Series(short_w).sort_index()
    print(f"\n加权持仓: {len(long_h)} 个截面")

    bt = run_longshort_backtest_weighted(
        long_holdings_w=long_h, short_holdings_w=short_h,
        open_prices=px["open_wide"], close_prices=px["close_wide"], status_data=px["status_wide"],
        output_dir=EXP_DIR, commission_rate=reg.COMMISSION, holding_period=reg.HOLDING_PERIOD)
    stats = bt["statistics"]
    for k in ("head", "tail", "tail_raw", "longshort", "benchmark", "head_excess", "tail_excess", "ls_excess"):
        if k in stats:
            stats[k] = _add_calmar(stats[k])

    stats_df = pd.DataFrame(daily)
    metrics = {
        "experiment": EXP_NAME, "clf_model": CLF_MODEL,
        "selection": "h64_32 (NN-mse [64,32]) |pred ΔX| top20%/min50, dir sign(μ-X)",
        "weight": "normalized P(revert) (prop), label=sign(μ-X)·ΔX>0, no WS",
        "n_features": len(FEATURES), "features": FEATURES,
        "holding_period": reg.HOLDING_PERIOD, "commission_rate": reg.COMMISSION,
        "top_pct": reg.TOP_PCT, "top_pct_min_n": reg.TOP_PCT_MIN_N,
        "n_signal_days": int(len(stats_df)),
        "long": stats["head"], "short": stats["tail"], "longshort": stats["longshort"],
        "benchmark": stats["benchmark"], "long_excess": stats["head_excess"], "ls_excess": stats["ls_excess"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    ls = stats["longshort"]
    print(f"\n[{EXP_NAME}] 多空年化 {ls['annual_return']:.2f} 夏普 {ls['sharpe_ratio']:.3f} "
          f"Calmar {ls.get('calmar_ratio',0):.3f} MDD {ls['max_drawdown']:.2f}")
    print(f"总耗时 {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
