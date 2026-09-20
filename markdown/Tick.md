# 逐笔委托 / 成交 / 撤单数据日频因子设计

## 1. 概述

基于逐笔委托、逐笔成交、逐笔撤单数据，构造股票-日期维度的日频因子，重点刻画：

- 资金主动买卖方向
- 委托簿供需压力
- 撤单行为与虚假流动性（第 5 节统一整合）
- 订单生命周期与成交效率
- 微观波动与流动性承接能力
- 日内分段交易行为
- 主动流动性吸收与盘口承接韧性（低相关补充）
- 订单流事件序列与状态转移（低相关补充）
- 交易节奏与订单到达加速度（低相关补充）

输出格式：

```text
trade_date, stock_code, factor_1, factor_2, ...
```

数据为沪深两市逐笔，字段为中文且口径不同，需先按第 2 节统一为派生基础变量再计算因子。主动买卖方向无需 tick rule / Lee-Ready 推断：上交所成交表自带方向标识，深交所用委托索引先后顺序判定（见 2.2）。撤单与生命周期类因子按订单编号 / 委托索引关联（见 2.4）。

## 2. 数据字段映射与计算口径

### 2.1 数据表与字段

| 市场 | 逐笔委托 | 逐笔成交 | 逐笔撤单 |
| --- | --- | --- | --- |
| 上交所 SH | `zz_39_1` | `zz_39_2` | `zz_39_3` |
| 深交所 SZ | `zz_05` | `zz_06_1` | `zz_06_2` |

上交所三表共用字段：

- `证券代码`、`订单或成交时间`（`HH:MM:SS.mmm`）、`标识`、`类型`、`价格`、`成交金额`、`数量`、`买方订单号`、`卖方订单号`、`序号`、`逐笔序号`
- `标识`：`B` 买 / `S` 卖 / `N` 集合竞价（仅成交表出现 `N`）
- `类型`：`A` 委托（zz_39_1）/ `T` 成交（zz_39_2）/ `D` 撤单（zz_39_3），同表恒定
- 委托、撤单：仅对应方向的订单号非零（买单填 `买方订单号`、卖单填 `卖方订单号`，另一侧为 0）；成交：买卖双方订单号均非零
- `逐笔序号`：全市场逐笔统一序号，用于跨委托/成交/撤单的事件排序

深交所委托 `zz_05` 字段：

- `证券代码`、`委托时间`、`买卖方向`、`订单类别`、`委托价格`、`委托数量`、`序号`、`频道代码`
- `买卖方向`：`49` 买 / `50` 卖（ASCII 码存为整数）
- `订单类别`：`49` 市价 / `50` 限价 / `85` 本方最优
- `序号`：委托索引，频道内唯一

深交所成交 `zz_06_1` / 撤单 `zz_06_2` 字段：

- `证券代码`、`委托时间`、`买方委托索引`、`卖方委托索引`、`委托价格`、`委托数量`、`成交类别`、`序号`、`频道代码`
- `成交类别`：`70` 成交（zz_06_1）/ `52` 撤单（zz_06_2）
- 撤单 zz_06_2：仅被撤一侧的委托索引非零

### 2.2 主动买卖方向判定

- 上交所成交（zz_39_2）：直接取 `标识`，`B` = 主动买，`S` = 主动卖，`N` = 集合竞价（不计入主动方向）。
- 深交所成交（zz_06_1）：比较 `买方委托索引` 与 `卖方委托索引`，索引大者为后到的主动方。`买方委托索引 > 卖方委托索引` → 主动买，否则主动卖。
- 委托方向：SH 取 `标识`，SZ 取 `买卖方向`。

### 2.3 成交金额与基础量口径

- 成交金额：SH 直接用 `成交金额`；SZ 无此列，用 `委托价格 × 委托数量`。
- 成交价 / 成交量：SH `价格` / `数量`；SZ `委托价格` / `委托数量`。
- 委托金额：委托量 × 委托价（SH `数量 × 价格`，SZ `委托数量 × 委托价格`）。
- 撤单金额：撤单记录的价 × 量（SH 撤单 `成交金额` 恒为 0，用 `价格 × 数量`；SZ 用 `委托价格 × 委托数量`）。

### 2.4 订单关联键（生命周期 / 撤单关联）

- 上交所：以订单号关联。委托 zz_39_1 的订单号（买单 `买方订单号`、卖单 `卖方订单号`）↔ 成交 zz_39_2 的 `买方订单号` / `卖方订单号` ↔ 撤单 zz_39_3 的同侧订单号。
- 深交所：以 (`频道代码`, 委托索引) 关联。委托 zz_05 的 `序号` ↔ 成交 / 撤单的 `买方委托索引` / `卖方委托索引`。委托索引仅在频道内唯一，必须同 `频道代码` 才能关联。

### 2.5 派生基础变量命名

后续因子统一使用以下派生变量（分市场按 2.1–2.4 口径计算后合并，不再区分市场）：

| 派生变量 | 含义 | 来源 |
| --- | --- | --- |
| `active_buy_amt` / `active_sell_amt` | 主动买 / 卖成交额 | 成交表，方向见 2.2 |
| `active_buy_qty` / `active_sell_qty` | 主动买 / 卖成交量 | 成交表 |
| `total_trade_amt` | 总成交额 | 成交表 |
| `buy_order_amt` / `sell_order_amt` | 买 / 卖委托额 | 委托表 |
| `buy_order_qty` / `sell_order_qty` | 买 / 卖委托量 | 委托表 |
| `order_count` | 委托笔数 | 委托表 |
| `buy_cancel_amt` / `sell_cancel_amt` | 买 / 卖撤单额 | 撤单表 |
| `cancel_count` | 撤单笔数 | 撤单表 |
| `mid_price` | 盘口中间价 | 由委托 / 成交 / 撤单增量重构盘口得到 |

`mid_price` 与盘口最优价需用委托、成交、撤单逐笔增量维护各价位挂单量重构，开销较高，仅在需要价差 / 近端挂单 / 韧性类因子时计算。

### 2.6 全局落地口径（所有因子统一遵守）

以下参数与规则为全市场、全因子统一约定，各因子章节不再重复：

| 项 | 规定值 |
| --- | --- |
| 时间戳单位 | 当日相对毫秒 `ms`（`HH:MM:SS.mmm` → `((H*3600+M*60+S)*1000+mmm`），解析失败记 `-1` 并剔除 |
| 交易时段 | 连续竞价 09:30:00–11:30:00、13:00:00–15:00:00；集合竞价 09:15:00–09:25:00（开盘）、14:57:00–15:00:00（收盘），非交易时段记录一律剔除 |
| 大单阈值 | 每股票每日、按对应金额序列的 **90 分位数**（`quantile=0.90`）为界，`amt ≥ thr` 记为大单；委托类大单用委托金额序列分位，成交类用成交金额序列分位，撤单类用撤单金额序列分位 |
| 小额阈值 | 每股票每日成交金额序列的 **10 分位数**，`amt ≤ thr` 记为小额 |
| 分钟桶 | `bucket = ts_ms // 60000`，桶收益 `ret_bucket = (末tick价 - 首tick价) / 首tick价` |
| 主动方向 | SH 取成交表 `标识`（`B`/`S`，`N` 集合竞价不计主动）；SZ 取 `买方委托索引 > 卖方委托索引 → 主动买`，否则主动卖 |
| 关联键 | SH = 同侧订单号；SZ = `(频道代码 << 40) \| 委托索引`，委托索引取 zz_05 `序号`（实现中对应字段为 `消息记录号`，见 2.4 注） |
| SH 撤单额 | 撤单表 `价格 × 数量`（`成交金额` 恒 0，不用） |
| SZ 撤单额 | 撤单表 `委托价格` 恒 0，须按关联键从委托表取原委托 `委托价格`，再 `× 撤单委托数量` |
| 缺失处理 | 任一因子分母为 0 或有效样本不足其最小样本数时，该因子当日记 `NaN`，不填 0 |
| 最小样本 | 逐笔收益类 ≥ 20 笔有效成交；事件序列类 ≥ 2 事件；转移概率类前项事件 ≥ 1；分桶回归类 ≥ 10 桶；否则记 `NaN` |
| 价格有效性 | 计算收益 / 冲击 / 漂移时，基准价 `≤ 0` 的样本剔除 |
| 未来价定位 | "`t+Δ` 时的价" = 时间 `≤ t+Δ` 的最后一笔成交价（`searchsorted(side='right')-1`），无则该样本剔除 |
| 时长归一 | 窗口 / 爆发 / 加速类因子按当日实际连续竞价时长归一，半日市 / 停牌按实际时长折算 |

> 注：`买方委托索引 / 卖方委托索引` 实证对应 zz_05 的 `消息记录号`（而非文档字面 `序号`），关联时以 `消息记录号` 为准。

## 3. 委托簿供需类因子

### 3.1 委托买卖不平衡

刻画当日买卖委托压力。

```text
order_imbalance_qty = (buy_order_qty - sell_order_qty) / (buy_order_qty + sell_order_qty)
order_imbalance_amt = (buy_order_amt - sell_order_amt) / (buy_order_amt + sell_order_amt)
```

### 3.2 大单委托占比

刻画大额挂单压力。

```text
large_buy_order_ratio  = large_buy_order_amt  / buy_order_amt
large_sell_order_ratio = large_sell_order_amt / sell_order_amt
large_order_imbalance  = (large_buy_order_amt - large_sell_order_amt) / large_order_amt
```

落地规定：
- 大单阈值 `thr = quantile(委托金额序列, 0.90)`（当日该股全部委托），`amt ≥ thr` 记大单。
- `large_order_amt = large_buy_order_amt + large_sell_order_amt`。
- 分母 `buy_order_amt` / `sell_order_amt` / `large_order_amt` 为 0 时对应因子记 `NaN`。

### 3.3 委托价格激进度

刻画委托价格相对成交价的激进程度。

```text
buy_aggressive  = (buy_order_price  - ref_price) / ref_price
sell_aggressive = (ref_price - sell_order_price) / ref_price
```

落地规定：
- `ref_price` 取**该笔委托时点前最近一笔成交价**（无成交则用当日首笔成交价；仍无则该笔剔除）。
- 聚合：对买 / 卖分别取**委托金额加权均值**，得 `buy_aggressive` / `sell_aggressive` 两个因子。
- 另输出 `aggressive_order_ratio = 激进委托金额 / 委托总额`，激进定义：买委托价 ≥ ref_price 或卖委托价 ≤ ref_price；SZ 额外把 `订单类别 ∈ {49 市价, 85 本方最优}` 直接计入激进。

### 3.4 委托规模分布

刻画单笔委托量的分布形态，用于识别机构拆单与散户行为。

```text
order_size_mean = buy_order_qty + sell_order_qty) / order_count   # = 全部委托量 / 委托笔数
order_size_skew = skew(单笔委托量序列)      # scipy.stats.skew，样本偏度
order_size_kurt = kurtosis(单笔委托量序列)  # scipy.stats.kurtosis，超额峰度（正态=0）
order_size_cv   = std(单笔委托量序列) / mean(单笔委托量序列)
```

落地规定：单笔委托量序列取当日该股全部委托的 `委托数量`；委托笔数 `< 20` 时 skew/kurt/cv 记 `NaN`。

## 4. 成交主动性类因子

### 4.1 主动买卖净额

刻画主动交易资金方向。

```text
active_net_amt   = active_buy_amt - active_sell_amt
active_net_ratio = (active_buy_amt - active_sell_amt) / total_trade_amt
active_net_mv    = active_net_amt / float_market_cap
```

### 4.2 大单主动买卖净额

刻画大额资金主动方向。

```text
large_active_net_amt   = large_active_buy_amt - large_active_sell_amt
large_active_net_ratio = large_active_net_amt / total_trade_amt
```

大单可按固定金额阈值或股票自身分位数定义，横截面可比性以分位数为佳。

### 4.3 主动买卖持续性

刻画主动方向是否连续，适合识别持续扫货或持续抛压。落地为 3 个因子（成交按 ts 排序后取主动方向序列 `dir ∈ {+1 买, -1 卖}`）：

```text
max_active_buy_run  = 最长连续主动买入笔数
max_active_sell_run = 最长连续主动卖出笔数
active_dir_autocorr = corr(dir[t], dir[t-1])   # 一阶自相关
```

落地规定：主动成交笔数 `< 20` 时三者记 `NaN`；自相关分母（方差）为 0 时记 `NaN`。

### 4.4 主动 / 被动成交占比

区分"吃单"与"挂单被动成交"，刻画交易急迫程度。

```text
aggressor_ratio     = active_trade_amt / total_trade_amt
aggressor_imbalance = (active_buy_amt - active_sell_amt) / active_trade_amt
```

落地规定：
- `active_trade_amt = active_buy_amt + active_sell_amt`（`标识 ∈ {B,S}` 的成交额，剔除 `N`）。
- SH 集合竞价成交（`标识=N`）计入 `total_trade_amt` 但不计入主动额，故 `aggressor_ratio ≤ 1`；SZ 无 `N`，`aggressor_ratio` 恒为 1（无被动概念）——因此该因子仅对 SH 有区分度，SZ 记 1.0。

### 4.5 交易笔数与单笔均额

刻画参与者结构。

```text
trade_count       = 当日成交笔数（含 N）
trade_avg_amt     = total_trade_amt / trade_count
small_trade_ratio = small_trade_amt / total_trade_amt
```

落地规定：小额阈值 `thr = quantile(成交金额序列, 0.10)`，`amt ≤ thr` 记小额，`small_trade_amt` 为这些成交额之和。

## 5. 撤单因子

按 2.4 关联键（SH 订单号、SZ 频道+委托索引）串联委托、成交、撤单三表，统一刻画撤单的**水平、存活时序、爆发集中度、成交因果、价格位置与策略性幌骗**六个维度。全部为股票-日频。

### 5.0 统一约定（撤单类因子补充 2.6）

| 项 | 规定值 |
| --- | --- |
| `cancel_life` | `cancel_ts - order_ts`（关联到原委托，单位 ms）；关联不到原委托的撤单在时长类因子中剔除，但仍计入撤单额类分母 |
| 撤单额 | SH `价格 × 数量`；SZ 关联委托价 × 撤单数量（见 2.6） |
| 价格参考价 `ref_price` | 撤单时点**前最近一笔成交价**（无则用当日首笔成交价；仍无则该撤单剔除） |
| 大单阈值 | 撤单金额序列 90 分位 |
| 快速撤单阈值 | 默认 3000 ms（§5.2.1）；10000 ms 为可选补充 |

### 5.1 撤单率水平

#### 5.1.1 总撤单率

```text
cancel_qty_ratio   = cancel_qty / order_qty
cancel_amt_ratio   = cancel_amt / order_amt
cancel_count_ratio = cancel_count / order_count
```

#### 5.1.2 买卖盘撤单率差

刻画买卖盘流动性稳定性差异。

```text
buy_cancel_ratio  = buy_cancel_amt / buy_order_amt
sell_cancel_ratio = sell_cancel_amt / sell_order_amt
cancel_imbalance  = buy_cancel_ratio - sell_cancel_ratio
```

买盘撤单率高表示托单偏虚；卖盘撤单率高表示抛压撤退、潜在上行动力增强。**默认落地 `cancel_imbalance`**。

#### 5.1.3 大单撤单占比

```text
large_cancel_ratio      = large_cancel_amt      / cancel_amt
large_buy_cancel_ratio  = large_buy_cancel_amt  / buy_cancel_amt
large_sell_cancel_ratio = large_sell_cancel_amt / sell_cancel_amt
```

落地规定：大单阈值 `thr = quantile(撤单金额序列, 0.90)`（当日该股全部撤单），`amt ≥ thr` 记大单；分母为 0 时对应因子记 `NaN`。

#### 5.1.4 集合竞价虚挂差

```text
auction_cancel_gap = (auction_order_amt_peak - auction_final_matched_amt) / auction_order_amt_peak
```

落地规定：`auction_order_amt_peak` = 09:15:00–09:25:00 累计委托额；`auction_final_matched_amt` = 开盘成交额（SH 取 `标识=N` 成交，SZ 取 09:25:00–09:30:00 首笔集中成交）；归一化到 `[0,1]`，peak 为 0 记 `NaN`。详见 §13.1。

### 5.2 撤单存活时间与快速撤单

按 2.4 关联委托与撤单时间，识别短时挂撤与虚假流动性。

#### 5.2.1 快速撤单

`cancel_life = cancel_ts - order_ts`（单位 ms）。`cancel_life ≤ 3000 ms` 记快速撤单；关联不到原委托的不计分子，撤单额口径下仍计入 `total_cancel_amt` 分母。

**撤单额口径**（快速撤单占全部撤单的比例）：

```text
fast_cancel_ratio      = cancel_amt_life_le_3s  / total_cancel_amt
fast_cancel_ratio_10s  = cancel_amt_life_le_10s / total_cancel_amt   # 可选，阈值 10s
```

**委托笔数口径**（快速撤单委托占全部委托的比例，分子按关联键去重）：

```text
frequent_cancel_ratio      = fast_cancel_order_count      / order_count
frequent_buy_cancel_ratio  = fast_buy_cancel_order_count  / buy_order_count
frequent_sell_cancel_ratio = fast_sell_cancel_order_count / sell_order_count
```

落地规定：
- 阈值默认 3000 ms；10s 口径为可选补充。
- **默认落地 `fast_cancel_ratio`**（即 3s 撤单额占比）。
- 委托笔数口径三个因子为可选补充；分母为 0 记 `NaN`。

#### 5.2.2 撤单存活时间中位数 `cancel_life_median`

```text
cancel_life_median = median( cancel_life )   # 单位：秒
```

- 测度目的：区分「闪撤（高频/幌骗）」与「耐心撤（策略性调整）」。
- 经济含义：委托从挂出到撤销的典型存活时长；越短越偏幌骗/抢跑，越长越偏正常挂单调整。
- 落地：关联撤单数 `< 10` 记 `NaN`。
- 区分：`fast_cancel_ratio` 只统计 ≤3s 金额占比（二值阈值），本因子刻画整体时长分布中心。

#### 5.2.3 撤单存活时间离散度 `cancel_life_dispersion`

```text
cancel_life_dispersion = (Q75(cancel_life) - Q25(cancel_life)) / median(cancel_life)
```

- 测度目的：识别撤单是否「程序化整齐」。
- 经济含义：程序化幌骗撤单时长高度一致 → 离散度低；人工/零散撤单 → 离散度高。
- 落地：`median = 0` 或关联撤单数 `< 10` 记 `NaN`。与 5.2.2 正交（中心 vs 宽度）。

#### 5.2.4 撤单存活时间分布形态

```text
cancel_life_skew = skew(cancel_life 序列)       # scipy.stats.skew
cancel_life_kurt = kurtosis(cancel_life 序列)   # scipy.stats.kurtosis，超额峰度（正态=0）

# 时长桶（秒）：[0,3]、(3,10]、(10,60]、(60,+∞)
cancel_life_hhi  = sum_k ( bucket_count_k / n )^2
```

落地规定：`cancel_life` 仅计入可关联原委托的撤单（单位秒）；关联撤单数 `< 20` 记 `NaN`；为可选补充因子。

### 5.3 撤单时序与集中度

#### 5.3.1 撤单爆发度 `cancel_burstiness`

```text
cancel_burstiness = max_window(cancel_count) / mean_window(cancel_count)
```

落地规定：窗口固定 `10000 ms`，只统计有撤单的时间跨度内的桶；撤单笔数 `< 10` 记 `NaN`；`mean_window = 0` 记 `NaN`。取值无上界，横截面标准化前建议截尾。亦作为交易节奏类因子的撤单侧度量（§12.2）。

#### 5.3.2 撤单流单边毒性 `cancel_flow_toxicity`

```text
分钟桶 j: imb_j = (buy_cancel_amt_j - sell_cancel_amt_j) / (buy_cancel_amt_j + sell_cancel_amt_j)
cancel_flow_toxicity = Σ_j w_j * imb_j ,  w_j = 桶撤单额 / 全天撤单额
```

- 测度目的：识别「单边流动性突然抽离」——一侧集中撤退常预示该侧后续行情。
- 经济含义：撤单额加权的分钟净撤方向（>0 买侧撤退为主，<0 卖侧撤退为主）。
- 落地：全天撤单额为 0 记 `NaN`。
- 区分：`cancel_imbalance` 是全天买卖撤单率差（水平量）；本因子是分钟加权的结构量，突出集中单边撤。

#### 5.3.3 大额撤单集中度 `large_cancel_concentration`

```text
large_cancel_concentration = sum(top5%_cancel_amt) / total_cancel_amt
```

- 测度目的：区分「散撤（大量小单撤）」与「巨单抽离（少数大撤主导）」。
- 经济含义：撤单额前 5% 的撤单占全天撤单额比例，高值表示少数巨单主导撤离。
- 落地：撤单笔数 `< 20` 记 `NaN`。
- 区分：`cancel_burstiness` 看**时间**集中，本因子看**金额**集中，正交。

### 5.4 撤单-成交因果

#### 5.4.1 部分成交后撤单占比 `partial_fill_cancel_ratio`

```text
partial_fill_cancel_ratio = partial_fill_then_cancel_order_count / order_count
```

- 测度目的：捕捉「试探性执行」——成交一部分探到冲击后撤掉剩余。
- 落地：`partial_fill_then_cancel` = 同一关联键既在成交表（`0 < 累计成交量 < 委托量`）又在撤单表的委托数；委托数 `< 20` 记 `NaN`。

#### 5.4.2 成交前防御性撤退比例 `pre_trade_cancel_ratio`

```text
pre_trade_cancel_ratio = 抢先撤退撤单额 / total_cancel_amt
```

- 测度目的：度量挂单方在价格朝其不利方向移动**之前**主动缩手。
- 经济含义：撤单前 5s 内**没有**推动价格朝其不利方向的主动成交，即视为「抢先撤退」，属防御性/知情撤单。
- 落地：判定窗口 5000 ms；`total_cancel_amt = 0` 记 `NaN`。
- 区分：`p_sellcancel_to_activebuy` 看「卖撤 → 随后主动买」的转移；本因子看「撤在不利成交之前」，因果方向不同。

#### 5.4.3 订单流撤单状态转移

将当日委托 / 成交 / 撤单按事件时间合并为单一事件流（SH 用 `逐笔序号`，SZ 用 `频道代码` 内 `序号` 配合 `委托时间` 排序）。状态集合：`{主动买, 主动卖, 买委托, 卖委托, 买撤单, 卖撤单}`。

```text
P(a -> b) = count( 序列中 a 紧跟 b ) / count(a)

p_sellcancel_to_activebuy = P(卖撤单 -> 主动买)   # 上方阻力撤离后扫货
p_buycancel_to_activesell = P(买撤单 -> 主动卖)   # 下方托单撤离后抛售
p_activebuy_to_buyorder   = P(主动买 -> 买委托)    # 主动买后挂单跟随（可选）
```

落地规定：
- 合并委托 / 成交 / 撤单为单一事件流，按 ts 稳定排序（ts 相同按 SH `逐笔序号` / SZ 频道内 `序号`）。
- `P(a->b) = count(序列中 a 紧跟 b) / count(a)`；前项事件数 `= 0` 记 `NaN`。
- **默认落地 `p_sellcancel_to_activebuy`**；另两项为可选补充。

#### 5.4.4 成交-撤单交替强度 `trade_cancel_alternation`

```text
trade_cancel_alternation = count( 成交 -> 撤单 -> 成交 三连模式 ) / event_count
```

落地规定：在合并事件流上滑动窗口统计 `成交,撤单,成交` 三连出现次数，除以事件总数；事件数 `< 3` 记 `NaN`。为可选补充因子。

### 5.5 撤单价格位置

#### 5.5.1 深档撤单占比 `deep_cancel_ratio`

```text
deep_cancel_ratio = 深档撤单额 / total_cancel_amt
```

- 测度目的：区分「盘口附近真实报价撤单」与「远离盘口的幌骗单撤单」。
- 经济含义：撤单委托价偏离参考价 `|cancel_price/ref_price - 1| > 0.5%` 记深档。
- 落地：阈值 **0.5%**；`total_cancel_amt = 0` 或无有效 ref 记 `NaN`。

#### 5.5.2 激进侧撤单占比 `aggressive_cancel_ratio`

```text
aggressive_cancel_ratio = 激进侧撤单额 / total_cancel_amt
```

- 测度目的：度量「最优报价撤退」——盘口最前排撤单最伤流动性。
- 经济含义：买撤价 `≥ ref_price` 或卖撤价 `≤ ref_price` 记激进侧（撤走近端流动性）。
- 落地：`total_cancel_amt = 0` 或无有效 ref 记 `NaN`。与 5.5.1 互补（远档幌骗 vs 近端抽单）。

#### 5.5.3 最优价附近撤补比（暂不落地）

```text
near_touch_cancel_add_ratio = near_touch_cancel_amt / near_touch_add_order_amt
```

`near_touch` 需重构同侧最优价，属**盘口重构类，暂不落地**。

### 5.6 策略性幌骗行为

#### 5.6.1 撤后再挂率 `refill_after_cancel_ratio`

```text
refill_after_cancel_ratio = 命中撤单额 / total_cancel_amt
```

- 测度目的：捕捉幌骗典型模式——撤单后短时在**同侧**再次挂单（挂-撤-再挂循环）。
- 经济含义：每笔撤单后 5s 内，同股**同侧**出现新委托，则该撤单计入命中。
- 落地：窗口 5000 ms；`total_cancel_amt = 0` 记 `NaN`。

### 5.7 撤单方向领先收益

刻画撤单行为对未来价格的方向性预测力。

```text
drift_after_buy_cancel(Δ)  = mean( (price(t+Δ) - price(t)) / price(t) | t 为撤买单时刻 )
drift_after_sell_cancel(Δ) = mean( (price(t+Δ) - price(t)) / price(t) | t 为撤卖单时刻 )

cancel_leads_price_1m  = drift_after_buy_cancel(60s)  - drift_after_sell_cancel(60s)
cancel_leads_price_5m  = drift_after_buy_cancel(300s) - drift_after_sell_cancel(300s)
cancel_leads_price_15m = drift_after_buy_cancel(900s) - drift_after_sell_cancel(900s)
```

- 测度目的：捕捉知情撤单——知情者提前得知信息后提前撤销不看好方向的挂单，撤单方向领先后续价格变动。
- 经济含义：撤买单后价格跌、撤卖单后价格涨 → 撤单含知情信息。差分项抵消当日公共市场漂移、放大方向性信号：大幅为负 → 知情撤单偏空；大幅为正 → 偏多；接近 0 → 无方向性预测力。
- 窗口含义：Δ 衡量知情撤单的预测时效。输出 1m / 5m / 15m 三档，横向比较可知预测力何时衰减；默认主推 5m。
- 落地：撤单方向按 §2.4（SH `标识`、SZ 非零委托索引侧）；基准价 / 未来价按 §2.6 未来价定位；仅连续竞价时段；撤买单、撤卖单各需 ≥ 20 笔，不足记 `NaN`。
- 区分：`pre_trade_cancel_ratio`（5.4.2）看撤在不利成交**之前**（因果反向、二值阈值）；本因子看撤单**之后**的价格漂移（连续量、方向性）。`buy_impact_1m`（7.2）是成交冲击；本因子是撤单后的价格实现，因果链不同。

### 5.8 幌骗识别类因子

基于幌骗的完整行为链（挂虚假挂单 → 诱导对手方主动成交 → 撤掉虚假挂单），结合撤单表与盘口快照识别幌骗并反推真实意图方向。

#### 5.8.1 幌骗得手反向成交 `spoofing_fruit`

```text
spoofing_fruit_buy  = Σ( 撤买单前 Δs 内主动卖成交额 ) / Σ( 撤买单额 )
spoofing_fruit_sell = Σ( 撤大卖单前 Δs 内主动买成交额 ) / Σ( 撤大卖单额 )
spoofing_fruit      = spoofing_fruit_buy - spoofing_fruit_sell
```

- 测度目的：捕捉幌骗"得手"的时序证据——幌骗单**自身未成交**，但对手方被诱导主动成交后幌骗单随即撤掉。
- 经济含义：挂大买单托价 → 对手方在高位主动卖出（被诱导）→ 幌骗者达成出货目的后撤掉托单。`spoofing_fruit_buy` 高 → 出货型幌骗活跃 → 后续偏空；差分刻画幌骗得手方向。
- 关键时序：成交在撤单**之前**（查 `[t-Δ, t]` 窗口），不是之后。
- 落地：Δ = 5000 ms；大单阈值 = 撤单额 90 分位（注：拆单幌骗会漏检，需盘口快照辅助，见 5.8.2）；幌骗单需在成交表无记录（未成交）；`total_cancel_amt = 0` 记 `NaN`。
- 区分：`p_sellcancel_to_activebuy`（5.4.3）看紧邻事件转移概率（笔数、无窗口、无大单限定）；本因子看金额强度 + 时间窗口 + 大单未成交限定，更针对幌骗。

#### 5.8.2 撤单驱动虚假盘口压力 `fake_pressure_imbalance`

需 5 档盘口快照（1 秒频率）。

```text
对相邻快照 (t, t+1)，各档挂单量减少分解为:
  成交消耗 = 该价位档对应成交量
  撤单驱动 = 挂单量减少 - 成交消耗（短存活且未成交的撤单）

fake_pressure_buy  = Σ( 买盘撤单驱动消退量 ) / Σ( 买盘总挂单量 )
fake_pressure_sell = Σ( 卖盘撤单驱动消退量 ) / Σ( 卖盘总挂单量 )
fake_pressure_imbalance = fake_pressure_buy - fake_pressure_sell
```

- 测度目的：盘口压力的"短暂出现 + 消失"中，由撤单（非成交）驱动的比例 → 直接度量虚假流动性。
- 经济含义：盘口压力消失若是成交消耗 = 真实流动性兑现；若是撤单驱动 = 虚假压力抽离（幌骗特征）。买盘虚假压力高 → 托单虚 → 后续跌。
- 三维联合识别：盘口关键档位（压力在哪里）× 撤单驱动（消退原因）× 短存活（挂撤间隔 ≤ 3s）。
- 落地：短存活阈值 3000 ms（快照间隔内）；仅统计撤单能解释的挂单减少；分母（总挂单量）为 0 记 `NaN`。
- 区分：`fast_cancel_ratio`（5.2.1）只看撤单存活时长，不知撤单对盘口压力影响；`deep_cancel_ratio`（5.5.1）只看撤单价格位置，不知存活时间；本因子联合盘口 + 撤单 + 存活时间。
- 限制：1 秒快照会漏掉存活 < 1s 的幌骗；5 档覆盖近端，深档幌骗不可见。

#### 5.8.3 幌骗倾向指数 `spoofing_imbalance`

```text
幌骗嫌疑撤单 = 满足 {大单 ∧ 深档 ∧ 短存活 ∧ 未成交} 的撤单
spoofing_buy_tendency  = 幌骗嫌疑撤买单额 / 买撤单总额
spoofing_sell_tendency = 幌骗嫌疑撤卖单额 / 卖撤单总额
spoofing_imbalance     = spoofing_buy_tendency - spoofing_sell_tendency
```

- 测度目的：从行为特征**事前**筛选幌骗嫌疑撤单（多特征联合），不依赖事后价格或反向成交。
- 经济含义：同时满足四特征的撤单幌骗嫌疑极高。`spoofing_imbalance > 0`（买侧幌骗嫌疑多）→ 托单型幌骗 → 真实意图出货 → 后续跌。
- 落地：大单 = 撤单额 90 分位；深档 = 偏离参考价 > 0.5%（§5.5.1）；短存活 = `cancel_life ≤ 3000 ms`（§5.2.1）；未成交 = 关联委托在成交表无记录。可放宽为"满足其中 ≥ 3 项"以增加样本。
- 区分：`spoofing_fruit`（5.8.1）看"得手证据"（事后反向成交）；本因子看"行为特征"（事前多特征筛选），两者互补。
- 限制：大单限定会漏拆单幌骗，需盘口快照辅助（见 5.8.2）。

#### 5.8.4 挂撤循环深度 `cycle_depth_imbalance`

```text
cycle_depth_buy  = mean( 买侧连续挂-撤-再挂循环的次数 )
cycle_depth_sell = mean( 卖侧连续挂-撤-再挂循环的次数 )
cycle_depth_imbalance = cycle_depth_buy - cycle_depth_sell
```

- 测度目的：单次挂-撤-再挂可能是正常调整，连续多次循环是典型幌骗操作（反复制造压力-撤-再制造）。
- 经济含义：循环次数越多，幌骗嫌疑越大。买侧循环深 → 反复托价 → 出货意图 → 后续跌；差分反推方向。
- 落地：同侧撤单后 5s 内出现新委托记 1 次，该新委托若又撤且 5s 内再出现新委托记 2 次，依此类推；取当日所有循环链的最大深度或平均深度。循环链数 `< 3` 记 `NaN`。
- 区分：`refill_after_cancel_ratio`（5.6.1）只看"撤后 5s 同侧再挂"的命中占比（二值）；本因子看**循环次数**（深度），刻画反复性。
- 限制：需追踪循环链，实现成本较高。

#### 5.8.5 撤单价格改善再挂率 `price_improve_refill_ratio`

```text
price_improve_refill_ratio = 撤后 Δs 内同侧以更激进价格再挂的撤单额 / total_cancel_amt
```

- 测度目的：区分"被动再挂"与"急迫追价再挂"——撤单不是因为放弃，而是价格不满意，立刻以更激进价格重新挂。
- 经济含义：撤买单后以更高价再挂买 → 真实急买；撤卖单后以更低价再挂卖 → 真实急卖。是"主动型"行为，属幌骗的反面信号（真实交易急迫性强）。
- 落地：Δ = 5000 ms；更激进 = 再挂买价 > 撤单价 或 再挂卖价 < 撤单价；`total_cancel_amt = 0` 记 `NaN`。
- 区分：`refill_after_cancel_ratio`（5.6.1）只看同侧再挂（不区分价格）；本因子要求**价格更激进**，把被动再挂与急迫追价再挂分开。

### 5.9 时段撤单类因子

不同交易时段参与者结构与信息环境差异显著，撤单的动机与信息含量也不同。本节按时段切片统计撤单行为，提取时段特异性的撤单信号。

#### 5.9.1 早盘虚挂识别 `morning_fake_hang`

```text
morning_fake_hang = 09:25-09:30 撤单额 / 09:25-09:30 (撤单额 + 成交额)
```

- 测度目的：开盘前 5 分钟（集合竞价尾声到连续竞价开始）的撤单占比。
- 经济含义：集合竞价期间挂单试探性强（不需要真实成交），开盘前撤掉避免被连续竞价吃到 → 虚挂识别。占比高 → 开盘价虚高/虚低，后续回归。
- 落地：时段窗口 [09:25:00, 09:30:00)；分母=该时段撤单额+成交额；分母为 0 记 `NaN`。

#### 5.9.2 开盘撤单方向 `open_cancel_imbalance`

```text
open_cancel_imbalance = (Σ早盘买撤单额 - Σ早盘卖撤单额) / Σ早盘撤单额
```

- 测度目的：早盘（09:30-10:00）撤单的买卖方向。
- 经济含义：开盘是信息集中释放期，知情者快速根据开盘情况调整挂单。买撤单多 → 开盘后买方不满意当前价位（认为高了）→ 看空；卖撤单多 → 卖方认为低了 → 看多。
- 落地：时段窗口 [09:30:00, 10:00:00)；`Σ早盘撤单额 = 0` 记 `NaN`。

#### 5.9.3 午前撤单避险 `pre_close_cancel_ratio`

```text
pre_close_cancel_ratio = Σ午前撤单额 / Σ全天撤单额
```

- 测度目的：上午收盘前 10 分钟（11:20-11:30）撤单占全天撤单的比例。
- 经济含义：午休前不确定性高（午间可能出现消息），知情者倾向于撤掉挂单规避午间风险。占比高 → 隔夜/午休风险厌恶情绪强 → 可能暗示有未公开信息待释放。
- 落地：时段窗口 [11:20:00, 11:30:00)；归一到全天撤单额，便于横截面比较；`Σ全天撤单额 = 0` 记 `NaN`。

#### 5.9.4 午后开盘撤单激进度 `reopen_cancel_aggression`

```text
reopen_cancel_aggression = (Σ午后开盘撤单额 - Σ上午尾盘撤单额) / Σ全天撤单额
```

- 测度目的：13:00 重开后的撤单相对 11:20-11:30 的变化。
- 经济含义：午休期间信息消化后，午后开盘撤单激增 → 午间信息改变了预期，知情者密集调整挂单。撤单额激增方向反映午间信息方向。
- 落地：午后开盘窗口 [13:00:00, 13:10:00)，上午尾盘窗口 [11:20:00, 11:30:00)；可进一步拆买卖方向；`Σ全天撤单额 = 0` 记 `NaN`。

#### 5.9.5 尾盘撤单压力 `late_cancel_imbalance`

```text
late_cancel_imbalance = (Σ尾盘买撤单额 - Σ尾盘卖撤单额) / Σ尾盘撤单额
```

- 测度目的：尾盘（14:45-15:00）撤单的买卖方向。
- 经济含义：尾盘是机构调仓、做价的最后窗口。尾盘买撤单激增 → 买方放弃买入（看跌次日）或在尾盘撤掉托单让价回落；卖撤单激增 → 卖方放弃卖出（看涨次日）。方向信号针对次日。
- 落地：时段窗口 [14:45:00, 15:00:00]（含集合竞价收盘）；`Σ尾盘撤单额 = 0` 记 `NaN`。

### 5.10 已落地撤单因子一览

| 输出列名 | 小节 | 主要含义 |
| --- | --- | --- |
| `cancel_imbalance` | 5.1.2 | 买卖盘撤单稳定性差异 |
| `large_cancel_ratio` | 5.1.3 | 大单撤单占比 |
| `large_buy_cancel_ratio` | 5.1.3 | 买单大单撤单占比 |
| `large_sell_cancel_ratio` | 5.1.3 | 卖单大单撤单占比 |
| `fast_cancel_ratio` | 5.2.1 | 3s 内快速撤单额占比 |
| `frequent_cancel_ratio` | 5.2.1 | 3s 内快速撤单委托占比（可选） |
| `frequent_buy_cancel_ratio` | 5.2.1 | 买单快速撤单委托占比（可选） |
| `frequent_sell_cancel_ratio` | 5.2.1 | 卖单快速撤单委托占比（可选） |
| `cancel_burstiness` | 5.3.1 | 撤单时间爆发 |
| `p_sellcancel_to_activebuy` | 5.4.3 | 卖撤 → 主动买转移概率 |
| `cancel_life_median` | 5.2.2 | 撤单存活时间中位数 |
| `cancel_life_dispersion` | 5.2.3 | 撤单存活时间离散度 |
| `cancel_life_skew` | 5.2.4 | 撤单存活时间偏度（可选） |
| `cancel_life_kurt` | 5.2.4 | 撤单存活时间峰度（可选） |
| `cancel_life_hhi` | 5.2.4 | 撤单存活时间分桶 HHI（可选） |
| `pre_trade_cancel_ratio` | 5.4.2 | 防御性/知情撤退 |
| `partial_fill_cancel_ratio` | 5.4.1 | 试探性执行 |
| `deep_cancel_ratio` | 5.5.1 | 远档幌骗撤单 |
| `aggressive_cancel_ratio` | 5.5.2 | 近端流动性抽离 |
| `cancel_flow_toxicity` | 5.3.2 | 单边集中撤退 |
| `refill_after_cancel_ratio` | 5.6.1 | 挂-撤-再挂幌骗循环 |
| `large_cancel_concentration` | 5.3.3 | 巨单抽离集中度 |
| `cancel_leads_price_1m` | 5.7 | 撤单方向领先收益（60s） |
| `cancel_leads_price_5m` | 5.7 | 撤单方向领先收益（300s，主推） |
| `cancel_leads_price_15m` | 5.7 | 撤单方向领先收益（900s） |
| `spoofing_fruit` | 5.8.1 | 幌骗得手反向成交差分 |
| `fake_pressure_imbalance` | 5.8.2 | 撤单驱动虚假盘口压力差分（需盘口快照） |
| `spoofing_imbalance` | 5.8.3 | 多特征联合幌骗嫌疑差分 |
| `cycle_depth_imbalance` | 5.8.4 | 挂-撤-再挂循环深度差分 |
| `price_improve_refill_ratio` | 5.8.5 | 撤后更激进价格再挂率（幌骗反面信号） |
| `morning_fake_hang` | 5.9.1 | 开盘前 5 分钟虚挂程度 |
| `open_cancel_imbalance` | 5.9.2 | 早盘撤单买卖方向 |
| `pre_close_cancel_ratio` | 5.9.3 | 午前避险撤单占比 |
| `reopen_cancel_aggression` | 5.9.4 | 午后开盘撤单激增 |
| `late_cancel_imbalance` | 5.9.5 | 尾盘撤单买卖方向 |

按 2.4 将委托、成交、撤单关联。

### 6.1 订单成交率

```text
fill_qty_ratio        = filled_qty / order_qty
fill_amt_ratio        = filled_amt / order_amt
full_fill_order_ratio = full_fill_order_count / order_count
```

落地规定：
- `filled_amt`：成交表中，`买方订单号` 或 `卖方订单号` 命中委托关联键的成交额之和（买卖两侧都统计）。
- `order_amt`：当日该股全部委托额。
- `full_fill_order_count`：`该委托累计成交量 ≥ 委托量` 的委托数（需按关联键聚合成交量）；`< 20` 委托时记 `NaN`。
- **默认落地 `fill_amt_ratio`**；其余两项为可选补充。

### 6.2 部分成交后撤单比例

见 §5.4.1 `partial_fill_cancel_ratio`。

### 6.3 订单存活时间

```text
order_life = end_time - order_time
```

落地规定：
- `end_time` = 该委托的**撤单时间**（若被撤）或**最后一笔成交时间**（若成交未撤）；两者皆无则取当日连续竞价结束时间 15:00:00。
- 落地聚合：`order_life_mean`（全部委托均值，单位秒）、`order_life_amt_weighted`（委托额加权均值）；委托数 `< 20` 记 `NaN`。

## 7. 流动性与价格冲击类因子

### 7.1 高频 Amihud

```text
tick_amihud = mean_over_minutes( |minute_return| / minute_amt ) * 1e9
```

落地规定：按分钟桶计算，`minute_return = (末tick价-首tick价)/首tick价`，`minute_amt` 为桶内成交额；对有成交的分钟桶取均值，乘 `1e9` 缩放；有效桶 `< 10` 记 `NaN`。

### 7.2 主动买卖冲击与不对称

```text
buy_impact_1m    = mean( (price(t+60s) - price(t)) / price(t) | 主动买 at t )
sell_impact_1m   = mean( (price(t+60s) - price(t)) / price(t) | 主动卖 at t )
impact_asymmetry = buy_impact_1m - |sell_impact_1m|
```

落地规定：
- `Δ = 60000 ms`；`price(t+60s)` 按 2.6"未来价定位"取时间 ≤ t+60s 的最后一笔成交价。
- 主动买 / 卖笔数各需 `> 5`，否则对应因子记 `NaN`。
- **默认落地 `buy_impact_1m`**；`sell_impact_1m` / `impact_asymmetry` 为可选补充。

### 7.3 Kyle's λ

刻画单位净订单流引起的价格变动，度量市场深度。

```text
按分钟分桶: Δmid_j = λ * net_order_flow_j + ε
kyle_lambda = OLS 斜率
```

落地规定：桶取 1 分钟；`Δmid_j` 用桶末与桶首成交价之差（无盘口时以成交价代理 mid）；`net_order_flow_j = 主动买量 - 主动卖量`（桶内）；有效桶 `< 10` 记 `NaN`；`net_order_flow` 方差为 0 记 `NaN`。属分桶回归类。

### 7.4 有效价差

用成交价相对中间价的偏离刻画实际交易成本。

```text
effective_spread = 成交额加权均值( 2 * |trade_price - mid_price| / mid_price )
```

`mid_price` 需重构盘口（见 2.5），属**盘口重构类，暂不落地**。

### 7.5 VPIN

刻画知情交易概率，按成交量分桶统计买卖不平衡。

```text
VPIN = mean_over_buckets( |bucket_buy_vol - bucket_sell_vol| / bucket_vol )
```

落地规定：等成交量分桶，桶容量 = 当日成交总量 / 50（即 50 桶）；买 / 卖量按主动方向归类；不足 50 桶（低流动性）记 `NaN`。属分桶类。

## 8. 微观波动结构类因子

### 8.1 高频已实现矩

用**分钟对数收益**构造日内波动结构（逐笔噪声过大，统一用分钟）。

```text
r_j           = ln(minute_close_j) - ln(minute_close_{j-1})   # 相邻分钟对数收益
realized_vol  = sqrt( sum(r_j^2) )
realized_skew = skew(r_j 序列)       # scipy.stats.skew，样本偏度
realized_kurt = kurtosis(r_j 序列)   # scipy.stats.kurtosis，超额峰度
```

落地规定：`minute_close_j` 取该分钟桶末笔成交价（价 ≤ 0 剔除）；有效分钟收益 `< 10` 时 skew/kurt 记 `NaN`。**默认落地 `realized_skew`**；vol/kurt 为可选补充。

### 8.2 成交集中度

刻画成交在时间或价格上的集中程度（HHI）。

```text
time_hhi  = sum( (segment_amt / total_amt)^2 )       # 按日内固定时段
price_hhi = sum( (price_level_amt / total_amt)^2 )   # 按成交价档
```

落地规定：
- `time_hhi` 时段划分固定为 §14 的 7 段（集合竞价 / 09:30-10:00 / 10:00-11:30 / 13:00-14:00 / 14:00-14:30 / 14:30-15:00 / 尾盘最后 10 分钟）。
- `price_hhi` 价格档 = 成交价按最小报价单位（沪深主板 0.01 元）取整分组。
- `total_amt = 0` 记 `NaN`。

## 9. 主动流动性吸收与承接类因子（低相关补充）

刻画主动成交是否真正推动价格，与第 7 节"冲击大小"互补：这里捕捉"打不动 / 砸不动"的吸收承接，方向含义与冲击因子不同。

### 9.1 主动买入吸收率 / 主动卖出承接率

逐笔实现：将成交按时间排序，对每笔主动成交记录其后 `Δ`（如 30s）成交价变化方向。

```text
buy_absorption  = Σ active_buy_amt  * 1{ price(t+Δ) <= price(t) } / active_buy_amt_total
sell_absorption = Σ active_sell_amt * 1{ price(t+Δ) >= price(t) } / active_sell_amt_total
```

含义：主动买入后价格未上行的金额占比越高 → 上方吸收越强（偏弱）；主动卖出后价格未下行占比越高 → 下方承接越强（偏强）。

分钟分桶简化实现（避免逐笔对齐未来价），**为默认落地口径**：

```text
minute_buy_absorption  = Σ_bucket active_buy_amt_bucket  * 1{ ret_bucket <= 0 } / active_buy_amt_total
minute_sell_absorption = Σ_bucket active_sell_amt_bucket * 1{ ret_bucket >= 0 } / active_sell_amt_total
```

落地规定：`ret_bucket = (桶末tick价 - 桶首tick价)/桶首tick价`；分母（当日主动买 / 卖总额）为 0 记 `NaN`。`buy_absorption` 采用此分钟口径。

### 9.2 价格静止吸收

捕捉瞬时"打不动"：成交价等于上一笔成交价（tick 不变）时的主动成交占比。

```text
static_absorption_buy  = Σ active_buy_amt  * 1{ price(t) == price(t-1) } / active_buy_amt_total
static_absorption_sell = Σ active_sell_amt * 1{ price(t) == price(t-1) } / active_sell_amt_total
```

落地规定：`price(t-1)` 为该股成交序列（按 ts 排序）前一笔成交价，首笔无前值不计静止；分母为 0 记 `NaN`。**默认落地 `static_absorption_buy`**。

### 9.3 承接非对称

```text
absorption_asymmetry = minute_buy_absorption - minute_sell_absorption
```

落地规定：用 9.1 的分钟吸收口径；任一侧为 `NaN` 时结果 `NaN`。`> 0` 上方吸收强于下方承接（偏弱），`< 0` 偏强。

## 10. 盘口韧性与补单类因子（低相关补充）

需按 2.4 关联订单并用委托 / 成交 / 撤单增量近似维护各价位挂单量，刻画流动性被消耗后的恢复能力。

> **已移除** `same_price_replenish_ratio`（原 10.1 同价位补单率）：与主动吸收类因子（§9）信息重叠且经济含义不清晰，不再纳入推荐因子集。

### 10.1 流动性恢复速度

```text
replenish_speed = mean( 同侧某价位挂单额恢复到冲击前水平所需秒数 )
```

需增量维护各价位挂单量（盘口重构），属**盘口重构类，暂不落地**。

### 10.2 最优价附近撤补比

见 §5.5.3（盘口重构类，暂不落地）。

## 11. 订单流事件序列类因子（低相关补充）

将当日委托 / 成交 / 撤单按事件时间合并为单一事件流，刻画事件因果顺序而非全天总量。撤单相关状态转移见 §5.4.3–5.4.4。

### 11.1 买卖事件切换频率

```text
side_switch_rate = count( side(t) != side(t-1) ) / (active_trade_count - 1)
```

落地规定：`side` 取主动成交方向序列（按 ts 排序，仅 `B`/`S`）；主动成交笔数 `< 2` 记 `NaN`。

### 11.2 大额成交后委托响应

```text
order_response_after_large_buy = future_60s_buy_order_amt / future_60s_sell_order_amt
```

落地规定：对每笔大额主动买（成交额 90 分位以上），统计其后 60s 内买 / 卖新增委托额，全日累加后取比值；分母为 0 记 `NaN`。为可选补充因子。

## 12. 交易节奏与到达加速度类因子（低相关补充）

不看绝对活跃度，而看活跃度的变化与时间分布，与第 8 节集中度（按金额占比）口径不同。

### 12.1 订单到达加速度

原定义"尾盘最后 5m / 前 5m"与"全天最大升温"含义差异较大，落地**拆为两个独立因子**：

```text
order_arrival_accel_max   = max_j( order_count_bucket[j] / order_count_bucket[j-1] - 1 )   # 全天相邻 5m 桶最大升温
order_arrival_accel_close = order_count(14:55-15:00) / order_count(14:50-14:55) - 1        # 尾盘最后 5m 相对前 5m
```

落地规定：桶宽 `300000 ms`（5 分钟）；`order_arrival_accel_max` 只在前一桶委托数 `> 0` 的相邻对上取最大；委托数 `< 20` 记 `NaN`。分母为 0 记 `NaN`。

### 12.2 撤单爆发度

见 §5.3.1 `cancel_burstiness`，作为节奏类因子的撤单侧度量。

### 12.3 成交间隔变异系数

```text
trade_interarrival_cv = std(Δt between trades) / mean(Δt between trades)
```

落地规定：`Δt` 为相邻成交时间差（ms），只用连续竞价时段；成交笔数 `< 20` 记 `NaN`；`mean = 0` 记 `NaN`。

## 13. 集合竞价与尾盘兑现类因子（低相关补充）

利用竞价时段大量挂撤与试探行为，以及尾盘加速，刻画信号真实性。

### 13.1 集合竞价虚挂差

见 §5.1.4 `auction_cancel_gap`。

### 13.2 集合竞价信号兑现

```text
auction_order_imbalance = (auction_buy_order_amt - auction_sell_order_amt) / (auction_buy_order_amt + auction_sell_order_amt)
auction_follow_through  = ret(09:30 -> 09:35) * sign(auction_order_imbalance)
```

落地规定：
- `auction_order_imbalance` 用 09:15:00–09:25:00 委托额买卖不平衡。
- `ret(09:30->09:35)` = `(p1 - p0)/p0`，`p0` 取 09:30:00 后首笔成交价、`p1` 取 09:35:00 后首笔成交价；任一时点无成交或 `p0 ≤ 0` 记 `NaN`。

### 13.3 尾盘加速强度

```text
pre_close_accel = active_net_amt(14:55-15:00) / |active_net_amt(14:30-14:55)|
```

落地规定：`active_net_amt = active_buy_amt - active_sell_amt`（分段内）；分母绝对值为 0 记 `NaN`。

## 14. 日内分段因子

将上述因子按固定时段分别构造。推荐分段：

```text
集合竞价 / 09:30-10:00 / 10:00-11:30 / 13:00-14:00 / 14:00-14:30 / 14:30-15:00 / 尾盘最后 10 分钟
```

重点因子：早盘主动买入净额、尾盘主动买入净额、尾盘委托不平衡、尾盘买 / 卖盘撤单率、集合竞价委托不平衡、分段吸收率与撤单爆发度（§5.3.1）。同样的全天净买入在早盘与尾盘含义不同，分段通常比全天聚合更有信息。

## 15. 异常交易行为类因子

### 15.1 频繁挂撤单

见 §5.2.1 委托笔数口径。

### 15.2 扫单行为

- 连续主动买入成交笔数 / 金额
- 主动买入成交价格连续上移次数
- 连续主动卖出打穿买盘次数

### 15.3 涨跌停附近行为（A 股）

- 涨停价附近买入委托占比 / 买单撤单率 / 涨停前主动买入强度
- 跌停价附近卖出委托占比 / 卖单撤单率

## 16. 推荐因子集合

本节 **30 个因子**为当前设计落地口径集合（`build_tick_factors.py` 已实现 28 个；§5.1.3 三个大单撤单因子待实现；`cancel_to_fill_ratio`、`drift_after_buy_cancel` 已移除），公式均按第 2–15 节落地口径，无歧义。因子名即输出列名。

### 16.1 第一版（含义清晰、实现成本低）

| 输出列名 | 公式（落地口径） | 主要含义 |
| --- | --- | --- |
| `active_net_ratio` | `(active_buy_amt - active_sell_amt) / total_trade_amt` | 主动资金方向 |
| `large_active_net_ratio` | `(large_active_buy_amt - large_active_sell_amt) / total_trade_amt`，大单=成交额 90 分位 | 大额资金方向 |
| `order_imbalance_amt` | `(buy_order_amt - sell_order_amt) / order_amt` | 挂单供需压力 |
| `cancel_imbalance` | `buy_cancel_amt/buy_order_amt - sell_cancel_amt/sell_order_amt`（5.1.2） | 买卖盘稳定性差异 |
| `large_cancel_ratio` | `large_cancel_amt / cancel_amt`（5.1.3） | 大单撤单占比 |
| `large_buy_cancel_ratio` | `large_buy_cancel_amt / buy_cancel_amt`（5.1.3） | 买单大单撤单占比 |
| `large_sell_cancel_ratio` | `large_sell_cancel_amt / sell_cancel_amt`（5.1.3） | 卖单大单撤单占比 |
| `fast_cancel_ratio` | `cancel_amt_life_le_3s / total_cancel_amt`（5.2.1 撤单额口径） | 虚假流动性 |
| `fill_amt_ratio` | `filled_amt / order_amt` | 委托真实性与成交效率 |
| `aggressor_ratio` | `active_trade_amt / total_trade_amt`（SZ 恒 1） | 交易急迫程度 |
| `tail_active_net_ratio` | `(tail_buy_amt - tail_sell_amt) / tail_trade_amt`，尾盘 14:50–15:00 | 尾盘资金行为 |
| `buy_impact_1m` | `mean((price(t+60s)-price(t))/price(t) \| 主动买)` | 买入推动价格能力 |
| `realized_skew` | 分钟对数收益样本偏度（≥20 笔） | 短期反转 / 动量 |

### 16.2 第二版（与第一版相关性低的补充）

| 输出列名 | 公式（落地口径） | 主要含义 |
| --- | --- | --- |
| `minute_buy_absorption` | 分钟买入吸收率（9.1 分钟口径） | 上方供给 / 吸收强度 |
| `static_absorption_buy` | 价格静止时主动买占比（9.2） | 逐笔"打不动"强度 |
| `absorption_asymmetry` | `minute_buy_absorption - minute_sell_absorption`（9.3） | 多空吸收差 |
| `side_switch_rate` | 主动方向切换率（11.1） | 订单流分歧 / 趋势 |
| `p_sellcancel_to_activebuy` | `P(卖撤单 -> 主动买)`（5.4.3） | 阻力撤离后扫货 |
| `cancel_burstiness` | `max_10s(cancel_count) / mean_10s(cancel_count)`（5.3.1） | 流动性突撤 |
| `order_arrival_accel` | 全天相邻 5m 桶最大委托升温 `order_arrival_accel_max`（12.1） | 活跃度突变 |
| `auction_follow_through` | `ret(09:30->09:35) * sign(竞价委托不平衡)`（13.2） | 竞价信号真实性 |

> 说明：`order_arrival_accel` 列当前对应 12.1 的 `order_arrival_accel_max`（全天最大升温）；12.1 的尾盘口径 `order_arrival_accel_close` 为可选补充因子，暂未纳入因子集。

### 16.3 撤单因子（§5 已落地 16 个）

完整定义见 §5，汇总如下：

| 输出列名 | 公式（落地口径） | 主要含义 |
| --- | --- | --- |
| `cancel_life_median` | 撤单存活时间中位数（秒，5.2.2） | 闪撤 vs 耐心撤 |
| `cancel_life_dispersion` | `IQR(cancel_life)/median`（5.2.3） | 撤单时长相对离散度 |
| `cancel_life_skew` | `skew(cancel_life)`（5.2.4，可选） | 撤单时长偏态 |
| `cancel_life_kurt` | `kurtosis(cancel_life)`（5.2.4，可选） | 撤单时长峰度 |
| `cancel_life_hhi` | 时长分桶 HHI（5.2.4，可选） | 撤单时长档位集中度 |
| `pre_trade_cancel_ratio` | 不利成交前 5s 抢先撤退额 / 撤单额（5.4.2） | 防御性/知情撤退 |
| `partial_fill_cancel_ratio` | 部分成交后被撤委托数 / 委托数（5.4.1） | 试探性执行 |
| `deep_cancel_ratio` | 偏离参考价 >0.5% 撤单额 / 撤单额（5.5.1） | 远档幌骗撤单 |
| `aggressive_cancel_ratio` | 激进侧撤单额 / 撤单额（5.5.2） | 近端流动性抽离 |
| `cancel_flow_toxicity` | 分钟撤单额加权净撤方向（5.3.2） | 单边集中撤退 |
| `refill_after_cancel_ratio` | 撤后 5s 同侧再挂命中额 / 撤单额（5.6.1） | 挂-撤-再挂幌骗循环 |
| `large_cancel_concentration` | top5% 撤单额 / 撤单额（5.3.3） | 巨单抽离集中度 |

> 注：第一版另有 5 个撤单因子（含 §5.1.3 三个大单撤单占比），第二版另有 2 个，与上表 9 个合计 16 个已落地；§5.2.4 另可选 3 个分布形态因子，详见 §5.8。

## 17. 标准化说明

- 金额类因子用成交额、委托额、流通市值或过去均值归一化，消除规模偏差。
- 大单阈值优先用分位数，保证大小盘可比。
- 横截面统一做去极值 + 标准化，按需行业 / 市值中性化。
- 全天逐笔因子只能用于收盘后调仓或下一交易日预测，收益检验用 `T+1` 或未来 `N` 日收益。
- 序列 / 状态转移概率类（第 5.4.3 节、第 11 节）取值已在 `[0, 1]`，无需额外归一化，但需过滤事件数过少的低流动性股票，或按事件数加权以抑制噪声。
- 窗口类（加速度、爆发度，第 5.3、12 节）对停牌、半日市需按实际交易时长归一，避免时长差异污染横截面。
