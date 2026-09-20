#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验引擎：核心训练/预测/回测/报告 功能
所有阶段的实验都调用此模块
"""
import sys, os, json, time, pickle, shutil, math, random, gc, warnings, traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import xgboost as xgb

warnings.filterwarnings('ignore')

RANDOM_SEED = 33
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

WORK_DIR = r'E:\zizhou\因子回测'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output', 'experiments')

# ============ 参数类 ============
class Params:
    EPS = 1e-8
    TRAIN_RATIO = 0.8
    START_DATE = '2020-01-01'
    END_DATE = '2026-01-01'
    N_GROUPS = 10
    EARLY_STOPPING_ROUNDS = 30
    MIN_IMPROVE = 0.001
    NUM_BOOST_ROUND = 500
    FEATURE_COLS = []

# 当前最优模型参数 v1
BEST_PARAMS_V1 = {
    'objective': 'reg:squarederror',
    'max_depth': 6, 'min_child_weight': 1,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'gamma': 5.0, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'learning_rate': 0.05, 'tree_method': 'hist', 'seed': RANDOM_SEED
}

params = Params()
panel = None
avg_return = None
feature_cols = []
dfs = []

# ============ 数据加载 ============
NON_FEATURE_COLS = {'date','stock_code','c_pct_5','is_st',
                    'close_price','close_price_1','close_price_6',
                    'open_price','open_price_1','open_price_6',
                    'change_pct','status','weight',
                    'sw_industry_l1_code','sw_industry_l2_code',
                    'sw_industry_l3_code','update_time','market_value'}

def load_data():
    global panel, avg_return, feature_cols, dfs
    os.chdir(WORK_DIR)
    print("加载数据...")
    panel = pd.read_pickle(os.path.join('Data', 'panel.pkl'))
    # 替换 inf 为 nan，然后 dropna（原地操作减少内存）
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.dropna(inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # 安全检查：确保标签列存在
    assert 'c_pct_5' in panel.columns, "错误: panel 中缺少 c_pct_5 列！请检查 Dataloader.ipynb"

    # 识别特征列（排除非特征列）
    all_cols = set(panel.columns)
    excluded = all_cols & NON_FEATURE_COLS
    unknown = all_cols - NON_FEATURE_COLS - {c for c in all_cols if c.startswith('f')}
    if unknown:
        print(f"[警告] 发现未知列（非因子、非排除列）: {unknown}")
        print(f"  这些列将被作为特征使用！如果包含前瞻数据，请加入 NON_FEATURE_COLS")

    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    # 原地转换类型，避免额外内存分配
    for col in feature_cols:
        if panel[col].dtype != np.float64:
            panel[col] = panel[col].astype(np.float32)  # 用 float32 节省一半内存
    params.FEATURE_COLS = feature_cols

    avg_return = pd.read_pickle(os.path.join('Data', 'avg_return.pkl'))

    dfs = half_year(panel, params.START_DATE, params.END_DATE)
    print(f"数据加载完成, 特征数: {len(feature_cols)}, 半年区间: {len(dfs)}")
    print(f"  排除的非特征列: {sorted(excluded)}")
    gc.collect()

# ============ 工具函数 ============
def calculate_rank_ic(f, r):
    mask = ~(pd.isna(f) | pd.isna(r))
    if mask.sum() < 10: return np.nan
    ic, _ = spearmanr(f[mask], r[mask])
    return ic

def split_tv(df, ratio=None):
    if ratio is None: ratio = params.TRAIN_RATIO
    cal = sorted(df['date'].unique().tolist())
    k = int(math.floor(len(cal) * ratio))
    return df[df['date'].isin(cal[:k])], df[df['date'].isin(cal[k:])]

def half_year(df, start, end):
    result = []
    t = max(pd.to_datetime(start).normalize(), df['date'].min())
    end_dt = pd.to_datetime(end).normalize()
    while t < end_dt:
        tlast = t
        t = min(end_dt, df['date'].iloc[-1], t + pd.DateOffset(months=6))
        result.append(df[((df['date'] >= tlast) & (df['date'] < t))])
        if t == df['date'].iloc[-1]: break
    return result

def prep_xy(df, fcols=None, lcol='c_pct_5'):
    if fcols is None: fcols = params.FEATURE_COLS
    return df[fcols].fillna(0).values.astype(np.float32), df[lcol].values.astype(np.float32)

# ============ 标签处理函数 ============
def neutralize_label_mv(df, label_col='c_pct_5'):
    """标签截面市值中性化+zscore（默认方式）- 内存优化版"""
    # 只提取需要的列，大幅减少内存
    cols_needed = ['date', 'stock_code', label_col]
    if 'market_value' in df.columns:
        cols_needed.append('market_value')
        sub_df = df[cols_needed].copy()
    else:
        sub_df = df[cols_needed].copy()
        sub_df = sub_df.merge(panel[['date','stock_code','market_value']], on=['date','stock_code'], how='left')

    new_label = sub_df[label_col].values.copy()
    for dt, idx_arr in sub_df.groupby('date').groups.items():
        try:
            grp = sub_df.loc[idx_arr]
            mv = np.log(grp['market_value'].astype(float).values + 1e-10)
            X = np.column_stack([np.ones(len(mv)), mv])
            y = grp[label_col].values
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r = y - (X @ b)
            s = r.std()
            if s > params.EPS:
                r = (r - r.mean()) / s
            new_label[idx_arr] = r
        except:
            pass
    # 在原 df 上替换标签列（避免复制整个大 DataFrame）
    result = df.copy()
    result[label_col] = new_label
    del sub_df, new_label; gc.collect()
    return result

# ============ 特征预处理函数库（向量化版）============
def transform_none(df, fcols=None):
    """无变换"""
    return df

def transform_zscore(df, fcols=None):
    """截面zscore — 向量化实现，只复制特征列"""
    if fcols is None: fcols = params.FEATURE_COLS
    # 只提取特征列做变换，避免复制整个 DataFrame
    feat = df[fcols]
    grouped = feat.groupby(df['date'])
    means = grouped.transform('mean')
    stds = grouped.transform('std')
    stds = stds.replace(0, np.nan)
    result = ((feat - means) / stds).fillna(0)
    # 赋值回去（pandas CoW 模式下会自动处理）
    df = df.copy()
    df[fcols] = result
    del feat, means, stds, result; gc.collect()
    return df

def transform_rank(df, fcols=None):
    """截面rank归一化到[0,1] — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    df[fcols] = df[fcols].fillna(0)
    df[fcols] = df.groupby('date')[fcols].rank(method='average', pct=True)
    return df

def transform_mad(df, fcols=None):
    """截面MAD标准化 — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    grouped = df.groupby('date')[fcols]
    medians = grouped.transform('median')
    mad = (df[fcols] - medians).abs().groupby(df['date']).transform('median')
    mad = mad.replace(0, np.nan)
    df[fcols] = (df[fcols] - medians) / (1.4826 * mad)
    df[fcols] = df[fcols].fillna(0)
    return df

def _winsorize_series(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)

def transform_winsorize_zscore(df, fcols=None, lower=0.01, upper=0.99):
    """截面winsorize+zscore — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    # 先 winsorize
    for c in fcols:
        df[c] = df.groupby('date')[c].transform(lambda x: x.clip(x.quantile(lower), x.quantile(upper)))
    # 再 zscore
    grouped = df.groupby('date')[fcols]
    means = grouped.transform('mean')
    stds = grouped.transform('std')
    stds = stds.replace(0, np.nan)
    df[fcols] = (df[fcols] - means) / stds
    df[fcols] = df[fcols].fillna(0)
    return df

def transform_winsorize3_zscore(df, fcols=None):
    return transform_winsorize_zscore(df, fcols, 0.03, 0.97)

def transform_quantile(df, fcols=None):
    """截面分位数变换(映射到正态分布) — 需逐组逐列"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    for c in fcols:
        df[c] = df.groupby('date')[c].transform(
            lambda x: pd.Series(
                QuantileTransformer(output_distribution='normal', random_state=RANDOM_SEED)
                .fit_transform(x.fillna(0).values.reshape(-1,1)).flatten(),
                index=x.index
            )
        )
    return df

def transform_minmax(df, fcols=None):
    """截面[0,1]归一化 — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    grouped = df.groupby('date')[fcols]
    mins = grouped.transform('min')
    maxs = grouped.transform('max')
    rng = maxs - mins
    rng = rng.replace(0, np.nan)
    df[fcols] = (df[fcols] - mins) / rng
    df[fcols] = df[fcols].fillna(0)
    return df

def transform_robust(df, fcols=None):
    """截面IQR标准化 — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    grouped = df.groupby('date')[fcols]
    medians = grouped.transform('median')
    q25 = grouped.transform(lambda x: x.quantile(0.25))
    q75 = grouped.transform(lambda x: x.quantile(0.75))
    iqr = q75 - q25
    iqr = iqr.replace(0, np.nan)
    df[fcols] = (df[fcols] - medians) / iqr
    df[fcols] = df[fcols].fillna(0)
    return df

def transform_percentile_rank(df, fcols=None):
    """截面百分位rank — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    df[fcols] = df[fcols].fillna(0)
    df[fcols] = df.groupby('date')[fcols].rank(method='average', pct=True)
    return df

def transform_log_rank(df, fcols=None):
    """log(1+rank) — 向量化实现"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    df[fcols] = df[fcols].fillna(0)
    df[fcols] = np.log1p(df.groupby('date')[fcols].rank(method='average'))
    return df

def transform_power(df, fcols=None):
    """Yeo-Johnson power变换 — 需逐组逐列"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    for c in fcols:
        df[c] = df.groupby('date')[c].transform(
            lambda x: pd.Series(
                PowerTransformer(method='yeo-johnson')
                .fit_transform(x.fillna(0).values.reshape(-1,1)).flatten(),
                index=x.index
            )
        )
    return df

# ---- 中性化函数（向量化版）----
def _neutralize_features_generic(df, fcols, build_X_fn, extra_cols):
    """通用特征中性化框架：用矩阵运算批量处理所有特征列"""
    if fcols is None: fcols = params.FEATURE_COLS
    df = df.copy()
    for col_name in extra_cols:
        if col_name not in df.columns:
            df = df.merge(panel[['date','stock_code',col_name]], on=['date','stock_code'], how='left')

    # 提取特征矩阵为 numpy 数组，批量处理
    feat_arr = np.nan_to_num(df[fcols].values.astype(np.float64), nan=0.0)
    groups = df.groupby('date').groups
    for dt, idx in groups.items():
        try:
            sub = df.loc[idx]
            X = build_X_fn(sub)
            if X is None: continue
            idx_pos = np.array([df.index.get_loc(i) for i in idx])
            # 批量回归：Y 矩阵 (n_samples, n_features)
            Y = feat_arr[idx_pos]
            # 最小二乘法批量求解：X @ B = Y → B = (X'X)^-1 X'Y
            B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            R = Y - X @ B
            # 批量 zscore
            means = R.mean(axis=0, keepdims=True)
            stds = R.std(axis=0, keepdims=True)
            stds[stds < params.EPS] = 1.0  # 避免除以0
            feat_arr[idx_pos] = (R - means) / stds
        except:
            pass
    df[fcols] = feat_arr
    del feat_arr; gc.collect()
    return df

def neutralize_features_mv(df, fcols=None):
    """特征截面市值中性化+zscore"""
    def _build_X(sub):
        mv = np.log(sub['market_value'].astype(float).values + 1e-10)
        return np.column_stack([np.ones(len(mv)), mv])
    return _neutralize_features_generic(df, fcols, _build_X, ['market_value'])

def neutralize_features_industry(df, fcols=None):
    """特征截面行业中性化"""
    def _build_X(sub):
        ind = pd.get_dummies(sub['sw_industry_l1_code'], drop_first=True).values
        if ind.shape[1] == 0: return None
        return np.column_stack([np.ones(len(ind)), ind])
    return _neutralize_features_generic(df, fcols, _build_X, ['sw_industry_l1_code'])

def neutralize_features_mv_industry(df, fcols=None):
    """特征截面市值+行业中性化"""
    def _build_X(sub):
        mv = np.log(sub['market_value'].astype(float).values + 1e-10)
        ind = pd.get_dummies(sub['sw_industry_l1_code'], drop_first=True).values
        return np.column_stack([np.ones(len(mv)), mv, ind])
    return _neutralize_features_generic(df, fcols, _build_X, ['market_value', 'sw_industry_l1_code'])

# ============ 训练/预测/回测 ============
def train_silent(tr, va, mp, fcols=None, label_fn=None,
                 early_stop_rounds=None, min_improve=None, num_boost_round=None):
    if label_fn is None: label_fn = neutralize_label_mv
    if early_stop_rounds is None: early_stop_rounds = params.EARLY_STOPPING_ROUNDS
    if min_improve is None: min_improve = params.MIN_IMPROVE
    if num_boost_round is None: num_boost_round = params.NUM_BOOST_ROUND

    tr = label_fn(tr); va = label_fn(va)
    Xt, yt = prep_xy(tr, fcols); Xv, yv = prep_xy(va, fcols)
    def ric(yp, d):
        ic = calculate_rank_ic(pd.Series(yp), pd.Series(d.get_label()))
        return ("rank_ic", float(ic if not np.isnan(ic) else -1.0))
    dt = xgb.DMatrix(Xt, label=yt); dv = xgb.DMatrix(Xv, label=yv)

    class ES(xgb.callback.TrainingCallback):
        def __init__(s, r, mi): s.r=r; s.mi=mi; s.best=-np.inf; s.w=0
        def after_iteration(s, m, ep, el):
            if 'eval' in el and 'rank_ic' in el['eval']:
                v = el['eval']['rank_ic'][-1]
                if v > s.best + s.mi: s.best=v; s.w=0
                else:
                    s.w += 1
                    if s.w >= s.r: return True
            return False

    er = {}
    callbacks = [ES(early_stop_rounds, min_improve)] if early_stop_rounds > 0 else []
    model = xgb.train(params=mp, dtrain=dt, num_boost_round=num_boost_round,
        evals=[(dt,'train'),(dv,'eval')], evals_result=er, feval=ric,
        callbacks=callbacks, verbose_eval=False)
    tic = er.get("train",{}).get("rank_ic",[]); vic = er.get("eval",{}).get("rank_ic",[])
    return model, {
        'best_iteration': (np.argmax(vic)+1) if vic else 0,
        'best_val_ic': max(vic) if vic else 0,
        'final_train_ic': tic[-1] if tic else 0,
        'num_iterations': len(vic),
        'train_ic_curve': tic,
        'val_ic_curve': vic,
    }

def predict_factors(model, df, fcols=None, label_fn=None):
    if label_fn is None: label_fn = neutralize_label_mv
    orig = df['c_pct_5'].values.copy()
    dn = label_fn(df)
    X, yn = prep_xy(dn, fcols)
    yp = model.predict(xgb.DMatrix(X))
    del X; gc.collect()  # 释放特征矩阵
    r = df[['date','stock_code']].copy()
    r['factor']=yp; r['return_neutral']=yn; r['return']=orig
    del dn, yp, yn, orig; gc.collect()
    return r.sort_values('date').reset_index(drop=True)

def backtest(df, n=10, save_path=None, rebalance_days=5):
    bl, di = [], []
    head_adv_list, head_win_list = [], []
    head_turnover_list = []
    prev_head_codes = None
    rc = 'return_neutral' if 'return_neutral' in df.columns else 'return'
    avg_series = avg_return.set_index('date')['avg_return'] if isinstance(avg_return, pd.DataFrame) else avg_return
    all_dates = sorted(df['date'].unique())
    rebalance_dates = all_dates[::rebalance_days]  # 每隔rebalance_days天调仓
    for dt in rebalance_dates:
        rd = df[df['date']==dt].copy()
        if len(rd) < n*2: continue
        rd['group'] = pd.qcut(rd['factor'].rank(method='first'), q=n, labels=False, duplicates='drop')
        head_codes = set(rd.loc[rd['group'] == (n - 1), 'stock_code'].dropna().astype(str))
        if head_codes:
            if prev_head_codes is not None and len(prev_head_codes) > 0:
                overlap = len(prev_head_codes & head_codes)
                head_turnover_list.append(float(1.0 - overlap / len(prev_head_codes)))
            prev_head_codes = head_codes
        ic = calculate_rank_ic(rd['factor'], rd[rc])
        if not np.isnan(ic): di.append(ic)
        pr = rd.groupby('group')['return'].mean().reset_index(); pr['date']=dt; bl.append(pr)

        try:
            avg_ret_dt = avg_series.loc[dt]
        except Exception:
            avg_ret_dt = np.nan
        if pd.notna(avg_ret_dt) and not np.isinf(avg_ret_dt):
            ex = pr.set_index('group')['return'] - float(avg_ret_dt)
            if (n - 1) in ex.index:
                top_ex = ex.loc[n - 1]
                other_ex = ex.drop(index=n - 1, errors='ignore')
                if len(other_ex) > 0:
                    best_other = other_ex.max()
                    head_adv_list.append(float(top_ex - best_other))
                    head_win_list.append(float(top_ex > best_other))
    if not bl: return {}
    bt = pd.concat(bl).pivot(index='date',columns='group',values='return').sort_index()
    avs = avg_series
    avs = avs.reindex(bt.index); bad = avs.isna()|np.isinf(avs)
    if bad.any(): bt=bt.drop(index=avs[bad].index); avs=avs[~bad].reindex(bt.index)
    excess = bt.sub(avs, axis=0); nav = (1+excess).cumprod()
    if save_path:
        plot_nav = nav.copy()
        if not isinstance(plot_nav.index, pd.DatetimeIndex):
            plot_nav.index = pd.to_datetime(plot_nav.index)
        start_idx = plot_nav.index[0] - pd.Timedelta(days=1)
        start_row = pd.DataFrame(
            data=np.ones((1, len(plot_nav.columns)), dtype=np.float64),
            index=[start_idx],
            columns=plot_nav.columns,
        )
        plot_nav = pd.concat([start_row, plot_nav], axis=0)
        plt.figure(figsize=(12,6))
        for g in range(n):
            if g in plot_nav.columns: plot_nav[g].plot(label=f'Group {g}', alpha=0.8)
        plt.title('Group Backtest Excess Return'); plt.xlabel('Date'); plt.ylabel('Cumulative Excess Return')
        plt.legend(loc='best'); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(save_path); plt.close('all')
    res = {}
    if di:
        res['rank_ic_mean']=np.mean(di); res['rank_ic_std']=np.std(di)
        res['rank_ic_ir']=res['rank_ic_mean']/res['rank_ic_std'] if res['rank_ic_std']>0 else 0
    if head_adv_list:
        res['head_advantage'] = float(np.mean(head_adv_list))
        res['head_win_rate'] = float(np.mean(head_win_list))
    if head_turnover_list:
        res['head_turnover_mean'] = float(np.mean(head_turnover_list))
        res['head_turnover_std'] = float(np.std(head_turnover_list))
        res['head_turnover_last'] = float(head_turnover_list[-1])
        res['head_turnover_count'] = int(len(head_turnover_list))
    if n-1 in nav.columns:
        top_excess_series = excess[n - 1].dropna()
        if len(top_excess_series) > 0:
            # 这里按调仓频率将头组单期超额收益年化，便于同时观察收益水平与波动水平。
            periods_per_year = 252.0 / float(rebalance_days)
            mean_excess = float(top_excess_series.mean())
            vol_excess = float(np.std(top_excess_series.to_numpy(dtype=np.float64)))
            ann_excess_return = mean_excess * periods_per_year
            ann_excess_vol = vol_excess * math.sqrt(periods_per_year)
            res['head_excess_return_annualized'] = ann_excess_return
            res['head_excess_volatility_annualized'] = ann_excess_vol
            res['head_excess_ir'] = ann_excess_return / ann_excess_vol if ann_excess_vol > 0 else 0.0
        top_nav = nav[n - 1]
        running_peak = top_nav.cummax()
        drawdown = 1.0 - top_nav / running_peak.replace(0, np.nan)
        res['head_excess_max_drawdown'] = float(drawdown.max()) if len(drawdown) > 0 else np.nan
        res['top_group_excess_return']=top_nav.iloc[-1]
        other_cols = [c for c in nav.columns if c != n - 1]
        if other_cols:
            other_best_nav = nav[other_cols].max(axis=1)
            res['head_cum_excess_advantage'] = float((top_nav - other_best_nav).mean())
            res['head_cum_excess_win_rate'] = float((top_nav > other_best_nav).mean())
    return res

# ============ 统一的实验运行函数 ============
def run_one(model_params=None, transform_fn=None, fcols=None, label_fn=None,
            save_path=None, window_size=6, expanding=False,
            early_stop_rounds=None, min_improve=None, num_boost_round=None,
            train_ratio=None):
    """
    运行一次完整滚动训练+回测。
    transform_fn: 特征变换函数(df)->df
    fcols: 特征列（可指定子集）
    label_fn: 标签处理函数
    window_size: 滚动窗口半年数
    expanding: True=扩展窗口
    """
    if model_params is None: model_params = BEST_PARAMS_V1
    tdl, fdf, infos = [], None, []
    min_windows = window_size if not expanding else max(4, window_size)

    for i in range(1, len(dfs)):
        tdl.append(dfs[i-1]); tdf = dfs[i]
        if not expanding and len(tdl) > window_size: tdl.pop(0)
        if len(tdl) < min_windows: continue

        dtr = pd.concat(tdl, ignore_index=True)
        tr, va = split_tv(dtr, train_ratio)
        if transform_fn:
            tr = transform_fn(tr); va = transform_fn(va)
            tdf_t = transform_fn(tdf.copy())
        else:
            tdf_t = tdf
        model, info = train_silent(tr, va, model_params, fcols, label_fn,
                                    early_stop_rounds, min_improve, num_boost_round)
        infos.append(info)
        tf = predict_factors(model, tdf_t, fcols, label_fn)
        fdf = tf if fdf is None else pd.concat([fdf, tf], ignore_index=True)
        del model, tr, va, dtr, tf, tdf_t; gc.collect()

    if fdf is None or len(fdf) == 0: return None, {'rank_ic_mean': -999}
    res = backtest(fdf, save_path=save_path)
    res['training_info'] = infos
    return fdf, res

def _save_training_curves(gdir, exp_name, training_info):
    """为每个实验保存训练曲线图（Train IC vs Val IC）"""
    n_windows = len(training_info)
    if n_windows == 0: return

    fig, axes = plt.subplots(1, n_windows, figsize=(5*n_windows, 4), squeeze=False)
    for wi, info in enumerate(training_info):
        ax = axes[0][wi]
        tic = info.get('train_ic_curve', [])
        vic = info.get('val_ic_curve', [])
        if tic:
            ax.plot(range(1, len(tic)+1), tic, label='Train IC', alpha=0.8)
        if vic:
            ax.plot(range(1, len(vic)+1), vic, label='Val IC', alpha=0.8)
            best_iter = info.get('best_iteration', 0)
            if 0 < best_iter <= len(vic):
                ax.axvline(x=best_iter, color='r', linestyle='--', alpha=0.5, label=f'Best@{best_iter}')
        ax.set_title(f'W{wi}', fontsize=10)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rank IC')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{exp_name} - Training Curves', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(gdir, f'{exp_name}_training_curves.png'), dpi=100)
    plt.close('all')

# ============ 实验组运行器 ============
def run_experiment_group(group_name, experiments, output_dir=None, timeout_minutes=60):
    """
    运行一组实验。
    experiments: list of dict, 每个dict有:
        name, desc, 以及run_one的参数（model_params, transform_fn, fcols, ...）
    返回: (best_name, best_ic, all_results)
    """
    if output_dir is None: output_dir = OUTPUT_DIR
    gdir = os.path.join(output_dir, group_name)
    os.makedirs(gdir, exist_ok=True)

    results = []
    best_ic = -999; best_name = None; best_fdf = None

    print(f"\n{'='*60}")
    print(f"实验组: {group_name} ({len(experiments)}个实验)")
    print(f"{'='*60}")

    for exp in experiments:
        name = exp['name']; desc = exp.get('desc', name)
        print(f"\n--- [{name}] {desc} ---")
        t0 = time.time()
        try:
            sp = os.path.join(gdir, f'{name}_backtest.png')
            fdf, res = run_one(
                model_params=exp.get('model_params'),
                transform_fn=exp.get('transform_fn'),
                fcols=exp.get('fcols'),
                label_fn=exp.get('label_fn'),
                save_path=sp,
                window_size=exp.get('window_size', 6),
                expanding=exp.get('expanding', False),
                early_stop_rounds=exp.get('early_stop_rounds'),
                min_improve=exp.get('min_improve'),
                num_boost_round=exp.get('num_boost_round'),
                train_ratio=exp.get('train_ratio'),
            )
            elapsed = time.time() - t0
            res['name'] = name; res['desc'] = desc; res['elapsed'] = elapsed
            ic = res.get('rank_ic_mean', -999)
            ir = res.get('rank_ic_ir', 0)
            te = res.get('top_group_excess_return', 0)

            for wi, info in enumerate(res.get('training_info', [])):
                print(f"  W{wi}: iter={info['num_iterations']}, best={info['best_iteration']}, "
                      f"TrainIC={info['final_train_ic']:.4f}, ValIC={info['best_val_ic']:.4f}")
            ha = res.get('head_advantage', np.nan)
            hw = res.get('head_win_rate', np.nan)
            hca = res.get('head_cum_excess_advantage', np.nan)
            hcw = res.get('head_cum_excess_win_rate', np.nan)
            print(
                f"  >> IC={ic:.4f}, IC_IR={ir:.4f}, TopExcess={te:.4f}, "
                f"HeadAdv={ha:.4f}, HeadWin={hw:.4f}, "
                f"HeadCumAdv={hca:.4f}, HeadCumWin={hcw:.4f} ({elapsed:.0f}s)"
            )

            # 保存每个子实验的因子值
            if fdf is not None:
                fdf.to_pickle(os.path.join(gdir, f'{name}_factor_df.pkl'))

            # 保存训练曲线图
            _save_training_curves(gdir, name, res.get('training_info', []))

            if ic > best_ic:
                best_ic = ic; best_name = name; best_fdf = fdf
            results.append(res)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  !! ERROR: {e} ({elapsed:.0f}s)")
            traceback.print_exc()
            results.append({'name': name, 'desc': desc, 'elapsed': elapsed,
                            'rank_ic_mean': -999, 'error': str(e)})
        gc.collect()

    # 保存最佳结果
    if best_fdf is not None:
        best_fdf.to_pickle(os.path.join(gdir, 'best_factor_df.pkl'))
        src = os.path.join(gdir, f'{best_name}_backtest.png')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(gdir, 'best_backtest.png'))

    _save_report(gdir, group_name, results, best_name)

    print(f"\n>> {group_name} 完成! 最优: {best_name} (IC={best_ic:.4f})")
    return best_name, best_ic, results

def _save_report(gdir, group_name, results, best_name):
    md = [f"# {group_name} - 回测报告\n\n"]
    md.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 最佳实验摘要
    best_res = None
    for r in results:
        if r.get('name') == best_name: best_res = r; break

    md.append(f"## 最佳实验: {best_name}\n\n")
    if best_res and 'error' not in best_res:
        md.append(f"| 指标 | 值 |\n|------|------|\n")
        md.append(f"| 平均 Rank IC | {best_res.get('rank_ic_mean',0):.4f} |\n")
        md.append(f"| Rank IC 标准差 | {best_res.get('rank_ic_std',0):.4f} |\n")
        md.append(f"| IC_IR (IC/IC_std) | {best_res.get('rank_ic_ir',0):.4f} |\n")
        md.append(f"| 头组累计超额收益(净值) | {best_res.get('top_group_excess_return',0):.4f} |\n")
        md.append(f"| 年化头组超额收益 | {best_res.get('head_excess_return_annualized',float('nan')):.4f} |\n")
        md.append(f"| 头组超额波动 | {best_res.get('head_excess_volatility_annualized',float('nan')):.4f} |\n")
        md.append(f"| 头组超额收益IR | {best_res.get('head_excess_ir',float('nan')):.4f} |\n")
        md.append(f"| 最大回撤 | {best_res.get('head_excess_max_drawdown',float('nan')):.4f} |\n")
        md.append(f"| 头组优势 | {best_res.get('head_advantage',float('nan')):.4f} |\n")
        md.append(f"| 头组胜率 | {best_res.get('head_win_rate',float('nan')):.4f} |\n")
        md.append(f"| 头组累计超额优势 | {best_res.get('head_cum_excess_advantage',float('nan')):.4f} |\n")
        md.append(f"| 头组累计超额胜率 | {best_res.get('head_cum_excess_win_rate',float('nan')):.4f} |\n")
        md.append(f"| 耗时 | {best_res.get('elapsed',0):.0f}s |\n\n")

    # 全部实验对比表
    md.append(f"## 全部实验对比\n\n")
    md.append(f"| 编号 | 描述 | Rank IC | IC Std | IC_IR | 头组超额(净值) | 年化头组超额收益 | 头组超额波动 | 头组超额收益IR | 最大回撤 | 头组优势 | 头组胜率 | 头组累计超额优势 | 头组累计超额胜率 | 耗时 |\n")
    md.append(f"|------|------|---------|--------|-------|----------------|------------------|--------------|----------------|----------|----------|----------|------------------|------------------|------|\n")
    for r in results:
        star = " **" if r.get('name') == best_name else ""
        err = " (ERROR)" if 'error' in r else ""
        md.append(f"| {r.get('name','')}{star} | {r.get('desc','')}{err} | "
                  f"{r.get('rank_ic_mean',0):.4f} | {r.get('rank_ic_std',0):.4f} | "
                  f"{r.get('rank_ic_ir',0):.4f} | "
                  f"{r.get('top_group_excess_return',0):.4f} | "
                  f"{r.get('head_excess_return_annualized',float('nan')):.4f} | "
                  f"{r.get('head_excess_volatility_annualized',float('nan')):.4f} | "
                  f"{r.get('head_excess_ir',float('nan')):.4f} | "
                  f"{r.get('head_excess_max_drawdown',float('nan')):.4f} | "
                  f"{r.get('head_advantage',float('nan')):.4f} | "
                  f"{r.get('head_win_rate',float('nan')):.4f} | "
                  f"{r.get('head_cum_excess_advantage',float('nan')):.4f} | "
                  f"{r.get('head_cum_excess_win_rate',float('nan')):.4f} | "
                  f"{r.get('elapsed',0):.0f}s |\n")

    # 输出文件清单
    md.append(f"\n## 输出文件\n\n")
    md.append(f"| 文件 | 说明 |\n|------|------|\n")
    md.append(f"| `best_factor_df.pkl` | 最佳实验({best_name})的完整因子值 |\n")
    md.append(f"| `best_backtest.png` | 最佳实验的分组回测图 |\n")
    for r in results:
        n = r.get('name','')
        if 'error' not in r:
            md.append(f"| `{n}_backtest.png` | {r.get('desc','')} 的回测图 |\n")
            md.append(f"| `{n}_factor_df.pkl` | {r.get('desc','')} 的因子值 |\n")

    # 训练过程详情
    md.append(f"\n## 训练过程详情\n\n")
    for r in results:
        if 'error' in r:
            md.append(f"### {r.get('name','')} - {r.get('desc','')}\n\n")
            md.append(f"**错误**: {r.get('error','')}\n\n")
            continue
        md.append(f"### {r.get('name','')} - {r.get('desc','')}\n\n")
        md.append(f"- Rank IC: {r.get('rank_ic_mean',0):.4f}\n")
        md.append(f"- IC Std: {r.get('rank_ic_std',0):.4f}\n")
        md.append(f"- IC_IR: {r.get('rank_ic_ir',0):.4f}\n")
        md.append(f"- 头组累计超额(净值): {r.get('top_group_excess_return',0):.4f}\n")
        md.append(f"- 年化头组超额收益: {r.get('head_excess_return_annualized',float('nan')):.4f}\n")
        md.append(f"- 头组超额波动: {r.get('head_excess_volatility_annualized',float('nan')):.4f}\n")
        md.append(f"- 头组超额收益IR: {r.get('head_excess_ir',float('nan')):.4f}\n")
        md.append(f"- 最大回撤: {r.get('head_excess_max_drawdown',float('nan')):.4f}\n")
        md.append(f"- 头组优势: {r.get('head_advantage',float('nan')):.4f}\n")
        md.append(f"- 头组胜率: {r.get('head_win_rate',float('nan')):.4f}\n")
        md.append(f"- 头组累计超额优势: {r.get('head_cum_excess_advantage',float('nan')):.4f}\n")
        md.append(f"- 头组累计超额胜率: {r.get('head_cum_excess_win_rate',float('nan')):.4f}\n\n")
        md.append(f"| 窗口 | 迭代轮数 | 最佳轮次 | Train IC | Val IC |\n")
        md.append(f"|------|---------|---------|----------|--------|\n")
        for wi, info in enumerate(r.get('training_info',[])):
            md.append(f"| W{wi} | {info['num_iterations']} | {info['best_iteration']} | "
                      f"{info['final_train_ic']:.4f} | {info['best_val_ic']:.4f} |\n")
        md.append("\n")

    with open(os.path.join(gdir, 'results.md'), 'w', encoding='utf-8') as f: f.writelines(md)
