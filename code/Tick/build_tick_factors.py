"""基于 /root/quant/Data/Tick/ 逐笔数据构建 Tick.md 17.1+17.2 共 19 个日频因子。

架构（关键性能设计）：
- 数据规模：单日 6 表约 300M 行、52GB 内存，直接 groupby('stock_code') 会超时/OOM。
- 优化路径：
  1. 加载每张 parquet 后，立即按 (证券代码, 时间) 排序 → 得到"分股票连续块"；
  2. 用 np.unique(return_index=True) 得到每只股票的 [lo, hi) 区间，
     因子计算函数直接接受 numpy 切片，避免 pandas groupby 开销；
  3. SH / SZ 分别独立处理（一只股票只属于一个市场），处理完丢弃对应大表；
  4. 单进程内内存峰值≈单表峰值（SZ zz_05 单日 14GB）+ 输出因子 (≈几MB)。

- 单机 236GB 内存，4-6 workers 即可安全（单进程峰值 ~20GB × 6 = 120GB）。

因子输出：/root/quant/Data/Tick/factors/daily/{date}.pkl
最终合并面板：/root/quant/Data/Tick/factors/tick_factor_panel.pkl (剔ST)
"""
import os
import sys
import time
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import fastparquet

warnings.filterwarnings('ignore')

# ============================================================
# 路径与常量
# ============================================================
TICK_ROOT = '/root/quant/Data/Tick'
DAILY_OUT_DIR = os.path.join(TICK_ROOT, 'factors', 'daily')
FINAL_PATH = os.path.join(TICK_ROOT, 'factors', 'tick_factor_panel.pkl')

# 因子参数（见 plan 文档）
LARGE_QUANTILE = 0.90
TAIL_START = '14:50:00'
TAIL_END   = '15:00:00'
AUCTION_START = '09:15:00'
AUCTION_END   = '09:25:00'
AUCTION_FOLLOW_START = '09:30:00'
AUCTION_FOLLOW_END   = '09:35:00'
IMPACT_DELTA_MS = 60_000
CANCEL_DRIFT_MS = 300_000
BURST_WINDOW_MS = 10_000
ACCEL_STEP_MS = 300_000
FAST_CANCEL_MS = 3_000
MINUTE_MS = 60_000

# 撤单深挖因子（§16）参数
CANCEL_PRE_TRADE_MS = 5_000     # 16.3 抢先撤退判定窗口
CANCEL_REFILL_MS = 5_000        # 16.9 撤后再挂窗口
DEEP_CANCEL_THR = 0.005         # 16.6 深档撤单价格偏离阈值 0.5%
LARGE_CANCEL_TOP = 0.05         # 16.10 大额撤单 top 分位

FACTOR_COLS = [
    'active_net_ratio', 'large_active_net_ratio', 'order_imbalance_amt',
    'cancel_imbalance', 'fast_cancel_ratio', 'fill_amt_ratio',
    'aggressor_ratio', 'tail_active_net_ratio', 'buy_impact_1m', 'realized_skew',
    'minute_buy_absorption', 'static_absorption_buy', 'absorption_asymmetry',
    'side_switch_rate', 'p_sellcancel_to_activebuy',
    'cancel_burstiness', 'drift_after_buy_cancel', 'order_arrival_accel',
    'auction_follow_through',
    # §16 撤单深挖
    'cancel_life_median', 'cancel_life_dispersion', 'pre_trade_cancel_ratio',
    'cancel_to_fill_ratio', 'partial_fill_cancel_ratio', 'deep_cancel_ratio',
    'aggressive_cancel_ratio', 'cancel_flow_toxicity', 'refill_after_cancel_ratio',
    'large_cancel_concentration',
]


# ============================================================
# 时间字符串 -> 毫秒（当日相对） — numpy 定长切片实现（对50M行加速~40倍）
# ============================================================
def time_str_to_ms(series):
    """向量化 'HH:MM:SS.mmm'（长度固定=12）-> int32 ms。缺失/异常返回 -1。

    原理：UCS-4 编码下每字符 4 字节，将字符串数组视为 (N,12) 的 uint32，
    直接用位置切片取每位数字（codepoint - 48）。
    """
    arr = series.values
    if len(arr) == 0:
        return np.array([], dtype=np.int32)
    # 强制转为定长 U12，非 12 位（含 NaN）用空串填充 => 得到全零切片，最终识别为异常
    arr = arr.astype('U12')
    n = len(arr)
    try:
        view = arr.view(np.uint32).reshape(n, 12)
    except ValueError:
        # 长度不足 12 时 view 失败，退回逐条解析
        out = np.full(n, -1, dtype=np.int32)
        for i, t in enumerate(arr):
            try:
                hms, dot, msp = t.partition('.')
                hh, mm, ss = hms.split(':')
                out[i] = (int(hh)*3600 + int(mm)*60 + int(ss))*1000 + (int(msp) if msp else 0)
            except Exception:
                pass
        return out
    d = view.astype(np.int32) - 48  # '0' = 48
    # HH:MM:SS.mmm  index: 0,1  3,4  6,7  9,10,11
    h = d[:, 0] * 10 + d[:, 1]
    m = d[:, 3] * 10 + d[:, 4]
    s = d[:, 6] * 10 + d[:, 7]
    ms = d[:, 9] * 100 + d[:, 10] * 10 + d[:, 11]
    out = ((h * 3600 + m * 60 + s) * 1000 + ms).astype(np.int32)
    # 异常值检查：只检查数字位是否 0-9，且分隔符位置正确
    digit_cols = [0, 1, 3, 4, 6, 7, 9, 10, 11]
    bad = (d[:, digit_cols] > 9).any(axis=1) | \
          (view[:, 2] != ord(':')) | (view[:, 5] != ord(':')) | (view[:, 8] != ord('.'))
    if bad.any():
        out[bad] = -1
    return out


def stock_code_last6(series):
    """从 'SH.600000' / 'SZ.000001'（长度固定=9）取后 6 位。"""
    arr = series.values
    if len(arr) == 0:
        return np.array([], dtype='U6')
    arr = arr.astype('U9')
    view = arr.view(np.uint32).reshape(-1, 9)
    # 取列 3-8 (共6个字符)，转回 U6
    back = view[:, 3:9].astype(np.uint32).tobytes()
    return np.frombuffer(back, dtype='U6').copy()


def _single_time_ms(t):
    hms, dot, ms = t.partition('.')
    h, m, sec = hms.split(':')
    return (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000 + (int(ms) if ms else 0)


TAIL_START_MS = _single_time_ms(TAIL_START)
TAIL_END_MS = _single_time_ms(TAIL_END)
AUCTION_START_MS = _single_time_ms(AUCTION_START)
AUCTION_END_MS = _single_time_ms(AUCTION_END)
FOLLOW_START_MS = _single_time_ms(AUCTION_FOLLOW_START)
FOLLOW_END_MS = _single_time_ms(AUCTION_FOLLOW_END)


# ============================================================
# 数据加载：按 stock+time 排序，返回 numpy 数组集合和股票起止索引
# ============================================================
def _read_and_sort_sh_order(date_str, is_trade=False, is_cancel=False):
    """SH 委托/成交/撤单：读取、按 (stock, ts) 排序、返回 numpy 数组。
    委托表：需要 amt=price×qty；成交表：amt 用 `成交金额`；撤单：amt=price×qty。
    """
    if is_trade:
        table = 'zz_39_2'
        cols = ['证券代码', '订单或成交时间', '标识', '价格', '成交金额', '数量',
                '买方订单号', '卖方订单号']
    elif is_cancel:
        table = 'zz_39_3'
        cols = ['证券代码', '订单或成交时间', '标识', '价格', '数量',
                '买方订单号', '卖方订单号']
    else:
        table = 'zz_39_1'
        cols = ['证券代码', '订单或成交时间', '标识', '价格', '数量',
                '买方订单号', '卖方订单号']

    path = f'{TICK_ROOT}/{table}/{date_str}.parquet'
    if not os.path.exists(path):
        return None
    pf = fastparquet.ParquetFile(path)
    df = pf.to_pandas(columns=cols)
    if df.empty:
        return None

    # 直接对原始 证券代码 做 factorize（'SH.600854' 等），比先切成 6 位再 factorize 快 ~6x：
    # pandas hashmap 对 object 数组有优化，先切成 U6 反而多一次拷贝。
    codes_id, uniq_full = pd.factorize(df['证券代码'].values, sort=True)
    # 最后对 uniq_full 去前缀（只 5000 条，几乎零成本）
    uniq_codes = np.array([s.split('.')[-1] for s in uniq_full], dtype='U6')

    ts_ms = time_str_to_ms(df['订单或成交时间'])
    side = df['标识'].astype(str).values
    price = df['价格'].astype(np.float32).values
    qty = df['数量'].astype(np.int32).values
    if is_trade:
        amt = df['成交金额'].astype(np.float32).values
    else:
        amt = (price.astype(np.float64) * qty.astype(np.float64)).astype(np.float32)
    buy_id = df['买方订单号'].astype(np.int64).values
    sell_id = df['卖方订单号'].astype(np.int64).values
    del df

    # 过滤 ts_ms<0
    mask = ts_ms >= 0
    if not mask.all():
        codes_id = codes_id[mask]
        ts_ms = ts_ms[mask]
        side = side[mask]
        price = price[mask]
        qty = qty[mask]
        amt = amt[mask]
        buy_id = buy_id[mask]
        sell_id = sell_id[mask]

    order_idx = np.argsort(codes_id, kind='stable')

    stock_code_sorted = uniq_codes  # 已排序的唯一 code 数组
    ts_ms = ts_ms[order_idx]
    side = side[order_idx]
    price = price[order_idx]
    qty = qty[order_idx]
    amt = amt[order_idx]
    buy_id = buy_id[order_idx]
    sell_id = sell_id[order_idx]

    # 找每 code 的起止边界（sorted codes_id[order_idx] 是非降的 int）
    sorted_ids = codes_id[order_idx]
    _, first_idx = np.unique(sorted_ids, return_index=True)
    boundaries = np.append(first_idx, len(sorted_ids)).astype(np.int64)

    out = {
        'codes': np.asarray(stock_code_sorted),
        'bounds': boundaries,
        'ts_ms': ts_ms,
        'side': side,
        'price': price,
        'qty': qty,
        'amt': amt,
        'buy_id': buy_id,
        'sell_id': sell_id,
    }
    return out


def _read_and_sort_sz(date_str, table):
    """SZ 三表统一读取。返回类似结构，但 SZ 用 (channel, order_id) 关联。"""
    if table == 'zz_05':
        cols = ['证券代码', '委托时间', '委托价格', '买卖方向', '订单类别',
                '委托数量', '消息记录号', '频道代码']
    else:
        cols = ['证券代码', '委托时间', '委托价格', '买方委托索引', '卖方委托索引',
                '委托数量', '成交类别', '频道代码']

    path = f'{TICK_ROOT}/{table}/{date_str}.parquet'
    if not os.path.exists(path):
        return None
    pf = fastparquet.ParquetFile(path)
    df = pf.to_pandas(columns=cols)
    if df.empty:
        return None

    codes_id, uniq_full = pd.factorize(df['证券代码'].values, sort=True)
    uniq_codes = np.array([s.split('.')[-1] for s in uniq_full], dtype='U6')

    ts_ms = time_str_to_ms(df['委托时间'])
    price = df['委托价格'].astype(np.float32).values
    qty = df['委托数量'].astype(np.int32).values
    amt = (price.astype(np.float64) * qty.astype(np.float64)).astype(np.float32)
    channel = df['频道代码'].astype(np.int32).values

    if table == 'zz_05':
        # 委托：side by 买卖方向，order_id = 消息记录号
        # 注：Tick.md 描述有误——实测 zz_06_1/2 的买方委托索引/卖方委托索引 引用的是委托表的
        # `消息记录号`，而非 `序号`。校验：委托 000001 序号=92 时消息记录号=89，
        # 撤单 000001 买方委托索引=89 恰好对上消息记录号。
        bs = df['买卖方向'].astype(np.int16).values
        side = np.where(bs == 49, 'B', np.where(bs == 50, 'S', 'N'))
        order_id = df['消息记录号'].astype(np.int64).values
        buy_id = np.zeros(0, dtype=np.int64)
        sell_id = np.zeros(0, dtype=np.int64)
    else:
        buy_idx = df['买方委托索引'].astype(np.int64).values
        sell_idx = df['卖方委托索引'].astype(np.int64).values
        if table == 'zz_06_1':
            # 成交：主动方向按 买方委托索引 > 卖方委托索引
            side = np.where(buy_idx > sell_idx, 'B', 'S')
            buy_id = buy_idx
            sell_id = sell_idx
            order_id = np.zeros(0, dtype=np.int64)
        else:  # zz_06_2 撤单
            side = np.where(buy_idx > 0, 'B', 'S')
            order_id = np.where(buy_idx > 0, buy_idx, sell_idx)
            buy_id = np.zeros(0, dtype=np.int64)
            sell_id = np.zeros(0, dtype=np.int64)
    del df

    mask = ts_ms >= 0
    if not mask.all():
        codes_id = codes_id[mask]
        ts_ms = ts_ms[mask]
        side = side[mask]
        price = price[mask]
        qty = qty[mask]
        amt = amt[mask]
        channel = channel[mask]
        if len(order_id):
            order_id = order_id[mask]
        if len(buy_id):
            buy_id = buy_id[mask]
            sell_id = sell_id[mask]

    order_idx = np.argsort(codes_id, kind='stable')
    ts_ms = ts_ms[order_idx]
    side = side[order_idx]
    price = price[order_idx]
    qty = qty[order_idx]
    amt = amt[order_idx]
    channel = channel[order_idx]
    if len(order_id):
        order_id = order_id[order_idx]
    if len(buy_id):
        buy_id = buy_id[order_idx]
        sell_id = sell_id[order_idx]

    sorted_ids = codes_id[order_idx]
    _, first_idx = np.unique(sorted_ids, return_index=True)
    boundaries = np.append(first_idx, len(sorted_ids)).astype(np.int64)

    out = {
        'codes': np.asarray(uniq_codes),
        'bounds': boundaries,
        'ts_ms': ts_ms,
        'side': side,
        'price': price,
        'qty': qty,
        'amt': amt,
        'channel': channel,
        'order_id': order_id,
        'buy_id': buy_id,
        'sell_id': sell_id,
    }
    return out


# ============================================================
# 因子计算（分市场，接受 numpy 切片）
# ============================================================
def _safe_div(a, b):
    return a / b if b else np.nan


def _slice_from_dict(dct, lo, hi):
    """取一只股票的所有数组切片（视图，无拷贝）。"""
    out = {}
    for k, v in dct.items():
        if k in ('codes', 'bounds'):
            continue
        if isinstance(v, np.ndarray) and len(v) > 0:
            out[k] = v[lo:hi]
        elif isinstance(v, np.ndarray):
            out[k] = v
    return out


def compute_factors_one_stock(o, t, c, market):
    """输入：某只股票当日的委托/成交/撤单 numpy 数组字典。
    market: 'SH' or 'SZ'，用于选择关联键。
    返回 dict of 19 factors。缺失分母时对应因子为 NaN。
    """
    out = {col: np.nan for col in FACTOR_COLS}

    # 主动方向过滤：成交表只保留 B/S（去掉集合竞价 N）
    if t is not None and 'side' in t and len(t['side']) > 0:
        dir_mask = np.isin(t['side'], np.array(['B', 'S']))
    else:
        dir_mask = None

    # ---------- 派生总量 ----------
    total_trade_amt = float(t['amt'].sum()) if (t and len(t['amt'])) else 0.0
    if dir_mask is not None and dir_mask.any():
        t_dir_side = t['side'][dir_mask]
        t_dir_amt = t['amt'][dir_mask]
        t_dir_ts = t['ts_ms'][dir_mask]
        t_dir_price = t['price'][dir_mask]
        active_buy_mask = t_dir_side == 'B'
        active_sell_mask = t_dir_side == 'S'
        active_buy_amt = float(t_dir_amt[active_buy_mask].sum())
        active_sell_amt = float(t_dir_amt[active_sell_mask].sum())
        active_trade_amt = active_buy_amt + active_sell_amt
    else:
        t_dir_side = t_dir_amt = t_dir_ts = t_dir_price = np.array([])
        active_buy_mask = active_sell_mask = np.array([], dtype=bool)
        active_buy_amt = active_sell_amt = active_trade_amt = 0.0

    if o and len(o['amt']):
        o_buy_mask = o['side'] == 'B'
        o_sell_mask = o['side'] == 'S'
        buy_order_amt = float(o['amt'][o_buy_mask].sum())
        sell_order_amt = float(o['amt'][o_sell_mask].sum())
    else:
        buy_order_amt = sell_order_amt = 0.0
        o_buy_mask = o_sell_mask = np.array([], dtype=bool)
    order_amt = buy_order_amt + sell_order_amt

    if c and len(c['amt']):
        c_buy_mask = c['side'] == 'B'
        c_sell_mask = c['side'] == 'S'
        buy_cancel_amt = float(c['amt'][c_buy_mask].sum())
        sell_cancel_amt = float(c['amt'][c_sell_mask].sum())
    else:
        buy_cancel_amt = sell_cancel_amt = 0.0

    # ---------- 17.1 因子 ----------
    if total_trade_amt > 0:
        out['active_net_ratio'] = (active_buy_amt - active_sell_amt) / total_trade_amt

    # 大单主动买卖净额比：用 t_dir_amt 90 分位数
    if len(t_dir_amt) > 0 and total_trade_amt > 0:
        thr = np.quantile(t_dir_amt, LARGE_QUANTILE)
        large_mask = t_dir_amt >= thr
        lb = float(t_dir_amt[large_mask & active_buy_mask].sum())
        ls = float(t_dir_amt[large_mask & active_sell_mask].sum())
        out['large_active_net_ratio'] = (lb - ls) / total_trade_amt

    if order_amt > 0:
        out['order_imbalance_amt'] = (buy_order_amt - sell_order_amt) / order_amt

    # 买卖撤单率差
    bcr = _safe_div(buy_cancel_amt, buy_order_amt)
    scr = _safe_div(sell_cancel_amt, sell_order_amt)
    if not (np.isnan(bcr) or np.isnan(scr)):
        out['cancel_imbalance'] = bcr - scr

    # aggressor_ratio
    if total_trade_amt > 0:
        out['aggressor_ratio'] = active_trade_amt / total_trade_amt

    # 尾盘主动买入净额比
    if len(t_dir_ts) > 0:
        tail_m = (t_dir_ts >= TAIL_START_MS) & (t_dir_ts < TAIL_END_MS)
        tail_amt = float(t_dir_amt[tail_m].sum())
        if tail_amt > 0:
            tb = float(t_dir_amt[tail_m & active_buy_mask].sum())
            ts_amt = float(t_dir_amt[tail_m & active_sell_mask].sum())
            out['tail_active_net_ratio'] = (tb - ts_amt) / tail_amt

    # 快速撤单比例 + 订单成交率（委托-成交-撤单关联）
    _fill, _fast = _lifecycle_factors_arrays(o, t, c, market)
    if pd.notna(_fill):
        out['fill_amt_ratio'] = _fill
    if pd.notna(_fast):
        out['fast_cancel_ratio'] = _fast

    # 买入冲击 1min
    if len(t_dir_ts) > 5:
        out['buy_impact_1m'] = _impact_1m(t_dir_ts, t_dir_price, active_buy_mask, IMPACT_DELTA_MS)

    # 已实现偏度
    if len(t_dir_price) >= 20:
        px = t_dir_price.astype(np.float64)
        pos = px > 0
        if pos.sum() >= 20:
            lp = np.log(px[pos])
            r = np.diff(lp)
            if len(r) >= 10 and r.std() > 1e-10:
                out['realized_skew'] = float(pd.Series(r).skew())

    # ---------- 17.2 因子 ----------
    if len(t_dir_ts) > 0:
        buy_abs, sell_abs, static_buy_abs = _absorption(
            t_dir_ts, t_dir_side, t_dir_price, t_dir_amt,
            active_buy_mask, active_sell_mask, active_buy_amt, active_sell_amt
        )
        out['minute_buy_absorption'] = buy_abs
        out['static_absorption_buy'] = static_buy_abs
        if pd.notna(buy_abs) and pd.notna(sell_abs):
            out['absorption_asymmetry'] = buy_abs - sell_abs

    # side_switch_rate
    if len(t_dir_side) >= 2:
        sides = t_dir_side  # 已按 ts 排序
        out['side_switch_rate'] = float(np.sum(sides[1:] != sides[:-1])) / (len(sides) - 1)

    # P(卖撤单 -> 主动买)
    if len(t_dir_ts) > 0 and c and len(c['ts_ms']):
        out['p_sellcancel_to_activebuy'] = _p_sellcancel_to_activebuy(
            t_dir_ts, active_buy_mask, active_sell_mask,
            c['ts_ms'], c['side']
        )

    # 撤单爆发度
    if c and len(c['ts_ms']) >= 10:
        cnts = _bin_count(c['ts_ms'], BURST_WINDOW_MS)
        if cnts.size > 0 and cnts.mean() > 0:
            out['cancel_burstiness'] = float(cnts.max()) / float(cnts.mean())

    # 大额买撤后价格漂移
    if c and len(c['ts_ms']) > 0 and len(t_dir_ts) > 0:
        out['drift_after_buy_cancel'] = _drift_after_buy_cancel(
            c['ts_ms'], c['side'], c['amt'],
            t_dir_ts, t_dir_price
        )

    # 订单到达加速度
    if o and len(o['ts_ms']) >= 20:
        cnts = _bin_count(o['ts_ms'], ACCEL_STEP_MS)
        if len(cnts) >= 2:
            prev = cnts[:-1].astype(np.float64)
            curr = cnts[1:].astype(np.float64)
            m = prev > 0
            if m.any():
                out['order_arrival_accel'] = float(np.max((curr[m] - prev[m]) / prev[m]))

    # 竞价信号兑现
    if o and len(o['ts_ms']) > 0 and len(t_dir_ts) > 0:
        out['auction_follow_through'] = _auction_follow(
            o['ts_ms'], o['side'], o['amt'],
            t_dir_ts, t_dir_price
        )

    # §16 撤单深挖 10 因子（联合委托/成交/撤单，一次性计算）
    out.update(_cancel_deep_factors(o, t, c, market))

    return out


def _lifecycle_factors_arrays(o, t, c, market):
    """返回 (fill_amt_ratio, fast_cancel_ratio)。全 numpy 矢量化实现。

    SH: 关联键 = order_id（委托方向决定用买/卖订单号，另一侧为0）
    SZ: 关联键 = channel << 40 | order_id
    实现要点：
    - 先算 o_keys（委托表订单键 -> ts, amt）；用 pd.factorize 得到密集编码
    - 成交表拆买/卖两侧订单号，用 np.searchsorted 找是否在 o_keys 里（O(N log N)）
    - 撤单同理，且计算 cancel_ts - order_ts <= 3s 的金额
    """
    if not o or len(o.get('ts_ms', [])) == 0:
        return np.nan, np.nan

    # ---- 构造委托键 ----
    if market == 'SH':
        buy_side = o['side'] == 'B'
        o_keys = np.where(buy_side, o['buy_id'], o['sell_id']).astype(np.int64)
    else:
        o_keys = (o['channel'].astype(np.int64) << 40) | o['order_id'].astype(np.int64)

    if len(o_keys) == 0:
        return np.nan, np.nan

    # 委托金额 by key，委托时间 min by key
    # 通过 pandas groupby 一次搞定（键在同一只股票内数量有限，通常几万条）
    o_key_sorted_idx = np.argsort(o_keys, kind='stable')
    ks = o_keys[o_key_sorted_idx]
    amts_s = o['amt'][o_key_sorted_idx]
    ts_s = o['ts_ms'][o_key_sorted_idx]
    uniq_keys, first_idx = np.unique(ks, return_index=True)
    # 求 key -> total amt / min ts（组内 ts_ms 未必递增，用 reduceat 求组内最小）
    boundaries = np.append(first_idx, len(ks))
    n_key = len(uniq_keys)
    key_amt = np.add.reduceat(amts_s.astype(np.float64), first_idx) if n_key > 0 else np.array([])
    # 组内 min ts
    key_ts_min = np.minimum.reduceat(ts_s, first_idx) if n_key > 0 else np.array([])
    total_order_amt = float(key_amt.sum())

    # ---- 计算 filled_amt ----
    filled_amt = 0.0
    if t and len(t.get('ts_ms', [])) > 0:
        if market == 'SH':
            t_key_buy = t['buy_id'].astype(np.int64)
            t_key_sell = t['sell_id'].astype(np.int64)
        else:
            t_ch = t['channel'].astype(np.int64) << 40
            t_key_buy = t_ch | t['buy_id'].astype(np.int64)
            t_key_sell = t_ch | t['sell_id'].astype(np.int64)
        # 用 searchsorted 判定是否在 uniq_keys 中
        idx_b = np.searchsorted(uniq_keys, t_key_buy)
        idx_s = np.searchsorted(uniq_keys, t_key_sell)
        in_b = (idx_b < n_key) & (uniq_keys[np.clip(idx_b, 0, n_key-1)] == t_key_buy) & (t_key_buy > 0)
        in_s = (idx_s < n_key) & (uniq_keys[np.clip(idx_s, 0, n_key-1)] == t_key_sell) & (t_key_sell > 0)
        filled_amt = float(t['amt'][in_b].sum()) + float(t['amt'][in_s].sum())

    fill_ratio = _safe_div(filled_amt, total_order_amt)

    # ---- 计算 fast_cancel_ratio ----
    fast_ratio = np.nan
    if c and len(c.get('ts_ms', [])) > 0:
        if market == 'SH':
            c_buy_side = c['side'] == 'B'
            c_keys = np.where(c_buy_side, c['buy_id'], c['sell_id']).astype(np.int64)
        else:
            c_keys = (c['channel'].astype(np.int64) << 40) | c['order_id'].astype(np.int64)
        c_ts = c['ts_ms']
        c_amt = c['amt']
        total_cancel_amt = float(c_amt.sum())
        # 对每个 c_keys，找 uniq_keys 中的位置，取 key_ts_min
        idx = np.searchsorted(uniq_keys, c_keys)
        matched = (idx < n_key) & (uniq_keys[np.clip(idx, 0, n_key-1)] == c_keys)
        order_ts_arr = np.where(matched, key_ts_min[np.clip(idx, 0, n_key-1)], -1)
        fast_mask = matched & ((c_ts - order_ts_arr) <= FAST_CANCEL_MS)
        fast_amt = float(c_amt[fast_mask].sum())
        fast_ratio = _safe_div(fast_amt, total_cancel_amt)

    return fill_ratio, fast_ratio


def _order_keys(d, market, kind):
    """构造关联键（int64）。kind ∈ {'order','cancel','trade'}。
    - order/cancel：按 side 选同侧订单号（SH）或 channel<<40|order_id（SZ），返回单键数组。
    - trade：返回 (买侧键, 卖侧键) 二元组。
    """
    if market == 'SH':
        if kind == 'trade':
            return d['buy_id'].astype(np.int64), d['sell_id'].astype(np.int64)
        buy_side = d['side'] == 'B'
        return np.where(buy_side, d['buy_id'], d['sell_id']).astype(np.int64)
    else:
        ch = d['channel'].astype(np.int64) << 40
        if kind == 'trade':
            return ch | d['buy_id'].astype(np.int64), ch | d['sell_id'].astype(np.int64)
        return ch | d['order_id'].astype(np.int64)


def _cancel_deep_factors(o, t, c, market):
    """一次性计算 §16 的 10 个撤单深挖因子，返回 dict。
    复用委托键映射，避免重复构建。缺失分母/样本不足记 NaN。
    """
    res = {
        'cancel_life_median': np.nan, 'cancel_life_dispersion': np.nan,
        'pre_trade_cancel_ratio': np.nan, 'cancel_to_fill_ratio': np.nan,
        'partial_fill_cancel_ratio': np.nan, 'deep_cancel_ratio': np.nan,
        'aggressive_cancel_ratio': np.nan, 'cancel_flow_toxicity': np.nan,
        'refill_after_cancel_ratio': np.nan, 'large_cancel_concentration': np.nan,
    }
    if not c or len(c.get('ts_ms', [])) == 0:
        return res

    c_ts = c['ts_ms']
    c_amt = c['amt'].astype(np.float64)
    c_side = c['side']
    c_price = c['price'].astype(np.float64)
    total_cancel_amt = float(c_amt.sum())
    n_cancel = len(c_ts)
    c_keys = _order_keys(c, market, 'cancel')

    # ---- 委托键映射：key -> 首个委托 ts（min）----
    have_orders = o and len(o.get('ts_ms', [])) > 0
    if have_orders:
        o_keys = _order_keys(o, market, 'order')
        o_sort = np.argsort(o_keys, kind='stable')
        ks = o_keys[o_sort]
        o_ts_s = o['ts_ms'][o_sort]
        o_amt_s = o['amt'][o_sort].astype(np.float64)
        uniq_keys, first_idx = np.unique(ks, return_index=True)
        n_key = len(uniq_keys)
        key_ts_min = np.minimum.reduceat(o_ts_s, first_idx) if n_key > 0 else np.array([])
        key_amt = np.add.reduceat(o_amt_s, first_idx) if n_key > 0 else np.array([])

    # ---- 16.1/16.2 撤单存活时间 ----
    if have_orders and n_key > 0:
        idx = np.searchsorted(uniq_keys, c_keys)
        matched = (idx < n_key) & (uniq_keys[np.clip(idx, 0, n_key - 1)] == c_keys)
        if matched.sum() >= 10:
            order_ts = key_ts_min[np.clip(idx, 0, n_key - 1)]
            life = (c_ts[matched] - order_ts[matched]).astype(np.float64) / 1000.0  # 秒
            life = life[life >= 0]
            if len(life) >= 10:
                med = float(np.median(life))
                res['cancel_life_median'] = med
                if med > 0:
                    q75, q25 = np.percentile(life, [75, 25])
                    res['cancel_life_dispersion'] = float((q75 - q25) / med)

    # ---- 16.4/16.5 撤单/成交额比、部分成交后撤单 ----
    if have_orders and n_key > 0 and t and len(t.get('ts_ms', [])) > 0:
        tb_key, ts_key = _order_keys(t, market, 'trade')
        t_amt = t['amt'].astype(np.float64)
        # 每个委托键累计成交量、成交额
        def _accum(key_arr, val):
            ii = np.searchsorted(uniq_keys, key_arr)
            inb = (ii < n_key) & (uniq_keys[np.clip(ii, 0, n_key - 1)] == key_arr) & (key_arr > 0)
            out = np.zeros(n_key, dtype=np.float64)
            if inb.any():
                np.add.at(out, ii[inb], val[inb])
            return out
        key_fill_amt = _accum(tb_key, t_amt) + _accum(ts_key, t_amt)
        t_qty = t['qty'].astype(np.float64)
        key_fill_qty = _accum(tb_key, t_qty) + _accum(ts_key, t_qty)
        filled_amt = float(key_fill_amt.sum())
        if filled_amt > 0:
            res['cancel_to_fill_ratio'] = total_cancel_amt / filled_amt
        # 部分成交后撤单：key 既部分成交(0<fill_qty<order_qty)又被撤
        key_order_qty = np.add.reduceat(
            (o['qty'][o_sort]).astype(np.float64), first_idx) if n_key > 0 else np.array([])
        cancelled_key = np.zeros(n_key, dtype=bool)
        ci = np.searchsorted(uniq_keys, c_keys)
        cin = (ci < n_key) & (uniq_keys[np.clip(ci, 0, n_key - 1)] == c_keys)
        cancelled_key[ci[cin]] = True
        partial = (key_fill_qty > 0) & (key_fill_qty < key_order_qty) & cancelled_key
        if n_key >= 20:
            res['partial_fill_cancel_ratio'] = float(partial.sum()) / n_key

    # ---- 参考价：撤单时点前最近成交价（16.3/16.6/16.7 共用）----
    have_trades = t and len(t.get('ts_ms', [])) > 0
    if have_trades:
        t_ts_all = t['ts_ms']
        t_px_all = t['price'].astype(np.float64)
        # 成交已按 ts 排序（读取时保证组内时间递增）
        ref_idx = np.searchsorted(t_ts_all, c_ts, side='right') - 1
        # 无前值的用首笔成交价
        ref_idx_filled = np.where(ref_idx >= 0, ref_idx, 0)
        ref_price = t_px_all[ref_idx_filled]
        ref_valid = (ref_price > 0)
    else:
        ref_price = None
        ref_valid = None

    # ---- 16.6 深档撤单占比 ----
    if ref_price is not None and total_cancel_amt > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.abs(c_price / ref_price - 1.0)
        deep_mask = ref_valid & (c_price > 0) & (dev > DEEP_CANCEL_THR)
        res['deep_cancel_ratio'] = float(c_amt[deep_mask].sum()) / total_cancel_amt

    # ---- 16.7 激进侧撤单占比（买撤价>=ref 或 卖撤价<=ref）----
    if ref_price is not None and total_cancel_amt > 0:
        is_buy = c_side == 'B'
        aggr = ref_valid & (c_price > 0) & (
            (is_buy & (c_price >= ref_price)) | (~is_buy & (c_price <= ref_price)))
        res['aggressive_cancel_ratio'] = float(c_amt[aggr].sum()) / total_cancel_amt

    # ---- 16.3 成交前防御性撤退：撤单前 5s 内无"不利成交" ----
    if have_trades and total_cancel_amt > 0:
        t_ts_all = t['ts_ms']
        t_side_all = t['side']
        # 主动买成交序列 ts、主动卖成交序列 ts
        buy_t_ts = t_ts_all[t_side_all == 'B']
        sell_t_ts = t_ts_all[t_side_all == 'S']
        is_buy = c_side == 'B'
        # 买撤的"不利成交"= 之前有主动卖（价格下压）；卖撤的"不利"= 之前有主动买
        # 判定窗口内是否存在不利成交
        def _has_recent(evt_ts, q_ts):
            if len(evt_ts) == 0:
                return np.zeros(len(q_ts), dtype=bool)
            pos = np.searchsorted(evt_ts, q_ts, side='right') - 1
            ok = pos >= 0
            recent = np.zeros(len(q_ts), dtype=bool)
            recent[ok] = (q_ts[ok] - evt_ts[pos[ok]]) <= CANCEL_PRE_TRADE_MS
            return recent
        adverse = np.where(is_buy, _has_recent(sell_t_ts, c_ts), _has_recent(buy_t_ts, c_ts))
        preempt = ~adverse  # 抢先撤退：无不利成交
        res['pre_trade_cancel_ratio'] = float(c_amt[preempt].sum()) / total_cancel_amt

    # ---- 16.8 撤单流单边毒性（分钟撤单额加权净撤方向）----
    if total_cancel_amt > 0:
        bucket = (c_ts // MINUTE_MS).astype(np.int64)
        _, bfirst = np.unique(bucket, return_index=True)
        row_b = np.searchsorted(bucket[bfirst], bucket)
        nb = len(bfirst)
        is_buy = c_side == 'B'
        buy_b = np.bincount(row_b[is_buy], weights=c_amt[is_buy], minlength=nb) if is_buy.any() else np.zeros(nb)
        sell_b = np.bincount(row_b[~is_buy], weights=c_amt[~is_buy], minlength=nb) if (~is_buy).any() else np.zeros(nb)
        tot_b = buy_b + sell_b
        with np.errstate(divide='ignore', invalid='ignore'):
            imb = np.where(tot_b > 0, (buy_b - sell_b) / tot_b, 0.0)
            w = tot_b / total_cancel_amt
        res['cancel_flow_toxicity'] = float(np.sum(w * imb))

    # ---- 16.9 撤后再挂率（撤单后 5s 内同侧新委托）----
    if have_orders and total_cancel_amt > 0:
        o_ts = o['ts_ms']
        o_side = o['side']
        # 分买/卖侧，各自委托 ts 已排序（读取时组内时间递增）
        buy_o_ts = np.sort(o_ts[o_side == 'B'])
        sell_o_ts = np.sort(o_ts[o_side == 'S'])
        is_buy = c_side == 'B'

        def _has_future(evt_ts, q_ts):
            if len(evt_ts) == 0:
                return np.zeros(len(q_ts), dtype=bool)
            pos = np.searchsorted(evt_ts, q_ts, side='left')
            ok = pos < len(evt_ts)
            hit = np.zeros(len(q_ts), dtype=bool)
            hit[ok] = (evt_ts[pos[ok]] - q_ts[ok]) <= CANCEL_REFILL_MS
            return hit
        refill = np.where(is_buy, _has_future(buy_o_ts, c_ts), _has_future(sell_o_ts, c_ts))
        res['refill_after_cancel_ratio'] = float(c_amt[refill].sum()) / total_cancel_amt

    # ---- 16.10 大额撤单集中度（top5% 撤单额占比）----
    if n_cancel >= 20 and total_cancel_amt > 0:
        k = max(1, int(np.ceil(n_cancel * LARGE_CANCEL_TOP)))
        top_sum = float(np.sort(c_amt)[-k:].sum())
        res['large_cancel_concentration'] = top_sum / total_cancel_amt

    return res


def _impact_1m(ts_ms, price, side_mask, delta_ms):
    """side_mask 是 B 的 mask。对每笔主动买，找 t + delta 后的成交价。"""
    if not side_mask.any():
        return np.nan
    sub_ts = ts_ms[side_mask]
    sub_px = price[side_mask].astype(np.float64)
    all_ts = ts_ms  # 全体已按 ts 排序
    all_px = price.astype(np.float64)
    end_ts = sub_ts + delta_ms
    idx = np.searchsorted(all_ts, end_ts, side='right') - 1
    valid = (idx >= 0) & (sub_px > 0)
    if not valid.any():
        return np.nan
    future_px = all_px[idx[valid]]
    origin_px = sub_px[valid]
    rets = (future_px - origin_px) / origin_px
    return float(np.nanmean(rets))


def _absorption(ts, side, price, amt, buy_mask, sell_mask, total_buy, total_sell):
    """分钟买入吸收、卖出吸收、静止买入吸收。ts/price/amt 已按 ts 排序。全 numpy 实现。"""
    if len(ts) == 0:
        return np.nan, np.nan, np.nan

    bucket = (ts // MINUTE_MS).astype(np.int64)
    # bucket 排序（因 ts 已排序，bucket 也是单调非降）
    _, first_idx = np.unique(bucket, return_index=True)
    last_idx = np.append(first_idx[1:], len(bucket)) - 1
    b_price_first = price[first_idx].astype(np.float64)
    b_price_last = price[last_idx].astype(np.float64)
    b_ret = np.where(b_price_first > 0, (b_price_last - b_price_first) / b_price_first, 0.0)
    # 生成 dense bucket idx，同 first_idx 对齐
    n_buckets = len(first_idx)
    # 用 searchsorted 把每笔成交映射到 dense bucket index
    # bucket 数组值 -> 对应 first_idx 中的位置
    row_bucket_idx = np.searchsorted(bucket[first_idx], bucket)  # dense 桶编号

    # 计算每桶买/卖累积金额（用 bincount）
    buy_bucket_amt = np.bincount(row_bucket_idx[buy_mask], weights=amt[buy_mask].astype(np.float64),
                                  minlength=n_buckets) if buy_mask.any() else np.zeros(n_buckets)
    sell_bucket_amt = np.bincount(row_bucket_idx[sell_mask], weights=amt[sell_mask].astype(np.float64),
                                   minlength=n_buckets) if sell_mask.any() else np.zeros(n_buckets)

    buy_abs = np.nan
    sell_abs = np.nan
    if total_buy > 0:
        buy_abs = float(buy_bucket_amt[b_ret <= 0].sum()) / total_buy
    if total_sell > 0:
        sell_abs = float(sell_bucket_amt[b_ret >= 0].sum()) / total_sell

    static_buy_abs = np.nan
    if total_buy > 0:
        prev_price = np.empty_like(price)
        prev_price[0] = -1  # 首笔无前值，用不可能相等的哨兵值
        prev_price[1:] = price[:-1]
        static_mask = buy_mask & (price == prev_price)
        static_buy_abs = float(amt[static_mask].sum()) / total_buy
    return buy_abs, sell_abs, static_buy_abs


def _p_sellcancel_to_activebuy(t_ts, buy_mask, sell_mask, c_ts, c_side):
    """P(卖撤单 -> 紧邻下一个事件为主动买)。合并事件序列后按 ts 排序。"""
    # 事件类型编码：sell_cancel=0, active_buy=1, active_sell=2, buy_cancel=3
    sc_mask = c_side == 'S'
    bc_mask = c_side == 'B'
    if not sc_mask.any():
        return np.nan
    if not buy_mask.any():
        return 0.0

    ev_ts = np.concatenate([c_ts[sc_mask], t_ts[buy_mask], t_ts[sell_mask], c_ts[bc_mask]])
    ev_type = np.concatenate([
        np.zeros(sc_mask.sum(), dtype=np.int8),
        np.ones(buy_mask.sum(), dtype=np.int8),
        np.full(sell_mask.sum(), 2, dtype=np.int8),
        np.full(bc_mask.sum(), 3, dtype=np.int8),
    ])
    order = np.argsort(ev_ts, kind='stable')
    et = ev_type[order]

    # 找 et==0 的位置，看后一位是否 ==1
    sc_pos = np.where(et == 0)[0]
    # 排除最后一位
    sc_pos = sc_pos[sc_pos < len(et) - 1]
    if len(sc_pos) == 0:
        return np.nan
    next_ev = et[sc_pos + 1]
    return float(np.mean(next_ev == 1))


def _bin_count(ts, window_ms):
    if len(ts) == 0:
        return np.array([], dtype=np.int64)
    bucket = (ts // window_ms).astype(np.int64)
    lo, hi = bucket.min(), bucket.max()
    return np.bincount(bucket - lo, minlength=hi - lo + 1)


def _drift_after_buy_cancel(c_ts, c_side, c_amt, t_ts, t_price):
    buy_mask = c_side == 'B'
    if not buy_mask.any() or len(t_ts) == 0:
        return np.nan
    bc_amt = c_amt[buy_mask]
    bc_ts = c_ts[buy_mask]
    thr = np.quantile(bc_amt, LARGE_QUANTILE)
    events = bc_ts[bc_amt >= thr]
    if len(events) == 0:
        return np.nan
    idx0 = np.searchsorted(t_ts, events, side='right') - 1
    idx1 = np.searchsorted(t_ts, events + CANCEL_DRIFT_MS, side='right') - 1
    valid = (idx0 >= 0) & (idx1 >= 0)
    if not valid.any():
        return np.nan
    p0 = t_price[idx0[valid]].astype(np.float64)
    p1 = t_price[idx1[valid]].astype(np.float64)
    ok = p0 > 0
    if not ok.any():
        return np.nan
    return float(np.nanmean((p1[ok] - p0[ok]) / p0[ok]))


def _auction_follow(o_ts, o_side, o_amt, t_ts, t_price):
    a_m = (o_ts >= AUCTION_START_MS) & (o_ts < AUCTION_END_MS)
    if not a_m.any():
        return np.nan
    a_side = o_side[a_m]
    a_amt = o_amt[a_m]
    b = float(a_amt[a_side == 'B'].sum())
    s = float(a_amt[a_side == 'S'].sum())
    if b + s <= 0:
        return np.nan
    imb = (b - s) / (b + s)

    m0 = (t_ts >= FOLLOW_START_MS) & (t_ts < FOLLOW_START_MS + 60_000)
    m1 = (t_ts >= FOLLOW_END_MS) & (t_ts < FOLLOW_END_MS + 60_000)
    if not m0.any() or not m1.any():
        return np.nan
    p0 = float(t_price[m0][0])
    p1 = float(t_price[m1][0])
    if p0 <= 0:
        return np.nan
    return float((p1 - p0) / p0 * np.sign(imb))


# ============================================================
# 单日处理
# ============================================================
def _fix_sz_cancel_amt(orders, cancels):
    """SZ 撤单表 `委托价格` 恒为 0，需从委托表按 (channel, order_id) 关联补齐价格，
    再 amt = price × qty。就地修改 cancels['price'] 与 cancels['amt']。

    实现要点：委托表每个 (channel, order_id) 唯一，直接对 o_keys 排序后
    用 np.searchsorted 做 O(N log M) 查找，避免 pandas groupby（对 1 亿行极慢）。
    注：同时回填 price，价格位置类因子（deep_cancel/aggressive_cancel）依赖它。
    """
    if orders is None or cancels is None:
        return
    if len(orders.get('order_id', [])) == 0 or len(cancels.get('order_id', [])) == 0:
        return
    o_keys = (orders['channel'].astype(np.int64) << 40) | orders['order_id'].astype(np.int64)
    c_keys = (cancels['channel'].astype(np.int64) << 40) | cancels['order_id'].astype(np.int64)
    # 委托 key 已经是唯一的（每条委托一个新 order_id），排序后 searchsorted
    o_sort = np.argsort(o_keys, kind='stable')
    o_keys_s = o_keys[o_sort]
    o_price_s = orders['price'][o_sort]
    # 对每个撤单键查找委托键
    idx = np.searchsorted(o_keys_s, c_keys)
    matched = (idx < len(o_keys_s)) & (o_keys_s[np.clip(idx, 0, len(o_keys_s) - 1)] == c_keys)
    matched_price = np.where(matched, o_price_s[np.clip(idx, 0, len(o_keys_s) - 1)], 0.0).astype(np.float32)
    cancels['price'] = matched_price
    cancels['amt'] = (matched_price.astype(np.float64) * cancels['qty'].astype(np.float64)).astype(np.float32)


def process_market(market_data, market):
    """遍历该市场所有股票，返回 rows 列表。
    market_data: dict[table_type -> read_dict]，key='orders'/'trades'/'cancels'
    """
    orders = market_data.get('orders')
    trades = market_data.get('trades')
    cancels = market_data.get('cancels')

    # SZ 撤单表价格全 0，需借助委托表补齐（SH 撤单价格有值，不需要）
    if market == 'SZ':
        _fix_sz_cancel_amt(orders, cancels)

    all_codes = set()
    for d in (orders, trades, cancels):
        if d is not None:
            all_codes.update(d['codes'].tolist())
    all_codes = sorted(all_codes)

    def _find_range(d, code):
        if d is None:
            return None
        idx = np.searchsorted(d['codes'], code)
        if idx >= len(d['codes']) or d['codes'][idx] != code:
            return None
        lo, hi = d['bounds'][idx], d['bounds'][idx + 1]
        return _slice_from_dict(d, lo, hi)

    rows = []
    for code in all_codes:
        o = _find_range(orders, code)
        t = _find_range(trades, code)
        c = _find_range(cancels, code)
        try:
            f = compute_factors_one_stock(o, t, c, market)
        except Exception as e:
            f = {col: np.nan for col in FACTOR_COLS}
            f['_error'] = str(e)[:80]
        f['stock_code'] = code
        rows.append(f)
    return rows


def process_one_day(date_str):
    out_path = os.path.join(DAILY_OUT_DIR, f'{date_str}.pkl')
    if os.path.exists(out_path):
        return f'[skip] {date_str}'
    os.makedirs(DAILY_OUT_DIR, exist_ok=True)

    t0 = time.time()

    # SH
    sh_o = _read_and_sort_sh_order(date_str)
    sh_t = _read_and_sort_sh_order(date_str, is_trade=True)
    sh_c = _read_and_sort_sh_order(date_str, is_cancel=True)
    t_sh_read = time.time() - t0
    rows_sh = process_market({'orders': sh_o, 'trades': sh_t, 'cancels': sh_c}, 'SH')
    t_sh_all = time.time() - t0
    # 释放
    del sh_o, sh_t, sh_c
    gc.collect()

    t1 = time.time()
    sz_o = _read_and_sort_sz(date_str, 'zz_05')
    sz_t = _read_and_sort_sz(date_str, 'zz_06_1')
    sz_c = _read_and_sort_sz(date_str, 'zz_06_2')
    t_sz_read = time.time() - t1
    rows_sz = process_market({'orders': sz_o, 'trades': sz_t, 'cancels': sz_c}, 'SZ')
    t_sz_all = time.time() - t1
    del sz_o, sz_t, sz_c
    gc.collect()

    rows = rows_sh + rows_sz
    if not rows:
        return f'[empty] {date_str}'

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(date_str)
    cols = ['date', 'stock_code'] + FACTOR_COLS
    if '_error' in df.columns:
        cols = cols + ['_error']
    df = df[cols]

    tmp = out_path + '.tmp'
    df.to_pickle(tmp)
    os.replace(tmp, out_path)

    del rows, df
    gc.collect()

    return (f'[done] {date_str} sh({t_sh_read:.0f}s+{t_sh_all-t_sh_read:.0f}s) '
            f'sz({t_sz_read:.0f}s+{t_sz_all-t_sz_read:.0f}s) '
            f'total={time.time()-t0:.0f}s')


def list_dates():
    files = sorted(os.listdir(os.path.join(TICK_ROOT, 'zz_39_3')))
    return [f.replace('.parquet', '') for f in files if f.endswith('.parquet')]


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dates = list_dates()
    only = sys.argv[2] if len(sys.argv) > 2 else None
    if only:
        dates = only.split(',')
    print(f'{len(dates)} 个交易日, workers={workers}, first={dates[0]}, last={dates[-1]}', flush=True)

    done = 0
    total = len(dates)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one_day, d): d for d in dates}
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                res = f'[ERROR] {futs[fut]}: {str(e)[:120]}'
            print(f'[{done}/{total}] {res}', flush=True)

    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
