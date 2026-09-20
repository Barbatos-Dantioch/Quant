#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 第 1 行 (#!/usr/bin/env python): "shebang", 让脚本可直接 ./xxx.py 执行时用 env 找到 python。
# 第 2 行 (coding: utf-8): 声明源码文件编码为 UTF-8, 这样文件里的中文注释不会乱码。
"""
生成 NZC 去重候选池 pair_log_full_NZCdedup.parquet
=================================================

与基线 OU-P00-zs-cv5-h5 候选池的唯一区别:
  行业内贪心去重的排序键由 cv_mean_ic(OU cv5)改为 NZC(训练窗零穿越次数, 降序);
  cv_mean_ic 仍照常计算并写入 pair_log(供下游 NN 当特征用), 仅不再用于去重排序。
  NZC 同值时以 cv_mean_ic 降序破平局(由 process_one_section dedup_rank="nzc" 实现)。

其余完全一致: train=252, valid=20, signal_type=zscore, cv_folds=5, cv_label_h=5, cap=1。
覆盖区间 2021-07-01 ~ 2026-05-13(与现有 pair_log_full.parquet 对齐), 一次性生成,
直接输出 output/0506_ou_pair/pair_log_full_NZCdedup.parquet, 供
run_ou_pair_lgb_exp_ws.py / run_ou_pair_nn_ws.py 经 PAIR_POOL=NZCdedup 读取。

只跑信号生成阶段, 不做回测。预计单进程 ~2 小时。
"""
# 上面这段被三引号 """...""" 包起来的文字叫 "模块文档字符串(docstring)", 是写给"人"看的说明,
# 程序不会执行它; 它的作用相当于本文件的"使用说明书"。

# from __future__ import annotations: 让本文件里的"类型注解"延迟求值(当作字符串处理)。
# 初学阶段只需知道: 这是新版 Python 的兼容写法, 写在所有 import 最前面即可, 无副作用。
from __future__ import annotations

# import 语句: 把别人写好的"工具库"加载进来, 之后才能使用其中的函数。
import os    # os: 操作系统相关 (路径拼接、改工作目录、读环境变量、makedirs 等)
import sys   # sys: 解释器相关 (这里用 sys.path 来告诉 python 去哪里找我们自己的模块)
import time  # time: 计时 (time.time() 取当前时间戳) 与格式化当前时间字符串

import pandas as pd  # pandas: 处理"表格型数据"的核心库; "as pd" 给它取个简称, 后面用 pd 代替 pandas

# os.chdir: 把"当前工作目录"切到项目根目录。
# 为什么需要: 下游 run_ou_pair 里很多路径是相对路径(如 "Data/all/...", "output/..."),
# 只有当工作目录是 /root/quant 时这些相对路径才指向正确的文件。
os.chdir("/root/quant")

# 变量名以下划线 _ 开头(如 _FACTOR1_DIR)是 Python 的一种约定: 表示"内部/私有", 不希望被外部引用。
_FACTOR1_DIR = "/root/quant/xgbcode/pair_trading"
# sys.path 是一个列表, python 按里面的目录顺序去搜索可被 import 的模块。
# 我们要 import 的 run_ou_pair.py 就在 _FACTOR1_DIR 里, 默认不一定在搜索路径中, 所以手动加进去。
# "if ... not in ...": 仅当该目录还没在 sys.path 时才插入, 避免重复添加。
if _FACTOR1_DIR not in sys.path:
    # insert(0, x): 把 x 插到列表"最前面"(下标 0), 让我们的目录被"优先"搜索到。
    sys.path.insert(0, _FACTOR1_DIR)

# Monkey-patch(猴子补丁): 在"运行时"直接修改别的模块里的变量值, 而不去改它的源码文件。
# 这里把 run_ou_pair 模块的 3 个全局配置改掉, 使它按我们要的时间范围加载数据。
import run_ou_pair as ROP  # 把 run_ou_pair 模块整体导入, 简称 ROP; 之后用 ROP.xxx 访问它的函数/变量
ROP.BACKTEST_START = "2021-07-01"  # 信号(回测)起始日
ROP.BACKTEST_END   = "2026-05-13"  # 信号(回测)结束日
# 2021-07-01 这天要算 OU 模型, 需要往前约 272 个交易日的历史价格; 把数据加载起点设到 2020-01-02,
# 多留一段缓冲(buffer), 保证最早的信号日也有足够历史可用。
ROP.DATA_START     = "2020-01-02"

# ── 下面是本脚本的配置常量(全大写是"常量"的命名约定, 表示运行期不再改动) ──
OUTPUT_DIR   = "output/0506_ou_pair"
# os.path.join: 用系统分隔符把目录和文件名拼成完整路径, 比手写 "a/b" 更安全(跨平台)。
OUT_PATH     = os.path.join(OUTPUT_DIR, "pair_log_full_NZCdedup.parquet")
TRAIN_WINDOW = 252      # OU 训练窗长度(交易日), 与基线 OU-P00-zs-cv5-h5 一致
VALID_WINDOW = 20       # 验证窗长度
SIGNAL_TYPE  = "zscore" # 信号公式类型
CV_FOLDS     = 5        # 交叉验证折数
CV_LABEL_H   = 5        # CV 中每折标签的未来收益周期(天)
DEDUP_RANK   = "nzc"    # 本实验唯一变量: 行业内去重的排序键改为 NZC(其余全部沿用基线)


# def: 定义一个函数。函数是"把一段逻辑打包起来, 起个名字, 方便调用"。这里把主流程放进 main()。
def main():
    # print: 把信息打印到屏幕(或被重定向到日志文件)。
    # f"...": "f-string(格式化字符串)", 字符串前加 f 后, 里面 {表达式} 会被替换成它的值。
    #   例如 {'='*60}: 字符串乘法, "=" 重复 60 次, 打印一条分隔线。
    print(f"\n{'='*60}")  # "\n" 是换行符
    print(f"生成 NZC 去重候选池  {time.strftime('%Y-%m-%d %H:%M:%S')}")  # strftime: 把当前时间格式化成字符串
    print(f"  PID: {os.getpid()}")          # getpid: 取本进程号, 方便之后用 ps/kill 找到它
    print(f"  回测范围: {ROP.BACKTEST_START} ~ {ROP.BACKTEST_END}")
    print(f"  输出文件: {OUT_PATH}")
    print(f"  参数: train={TRAIN_WINDOW}, valid={VALID_WINDOW}, signal_type={SIGNAL_TYPE}, "
          f"cv_folds={CV_FOLDS}, cv_label_h={CV_LABEL_H}, dedup_rank={DEDUP_RANK}")
    print(f"{'='*60}\n")

    # makedirs: 创建目录; exist_ok=True 表示"目录已存在也不报错"。
    # 在"写文件之前"先确保目录存在, 是防止 FileNotFoundError 的好习惯。
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()  # 记录开始时间戳(秒), 用于最后计算总耗时
    # prepare_data(): 调用 run_ou_pair 里的数据准备函数, 返回一个"字典(dict)"。
    # 字典是"键->值"的集合, 用 data["键名"] 取出对应的值。
    data = ROP.prepare_data()
    cal = data["cal"]  # cal: 全市场交易日序列(日历)

    # 下面是"字典推导式(dict comprehension)", 一行生成一个字典:
    #   {键: 值 for 变量 in 可迭代对象}
    # enumerate(序列): 在遍历时同时给出"下标 i"和"元素 dt"。
    # 这里把"每个交易日 dt"映射到"它在 cal 中的位置下标 i", 之后能用 O(1) 速度按日期查下标。
    date_to_idx = {dt: i for i, dt in enumerate(pd.DatetimeIndex(cal))}
    backtest_dates = data["backtest_dates"]  # 需要生成信号的所有交易日列表
    print(f"\n回测截面数: {len(backtest_dates)}")  # len(x): 取序列长度(元素个数)

    # 准备两个空列表(list), 用来在循环里逐步收集结果, 循环结束后再统一合并。
    pair_log_chunks = []  # 收集每个交易日产出的 pair_log 小表(DataFrame)
    daily_stats = []      # 收集每个交易日的统计行(字典), 便于事后诊断
    n_dates = len(backtest_dates)

    # 主循环: 逐个交易日处理。enumerate 让我们同时拿到序号 k(从 0 开始)和日期 sig_date。
    for k, sig_date in enumerate(backtest_dates):
        # 防御性检查: 若这个日期不在日历映射里, 用 continue 跳过本次循环, 直接进入下一天。
        if sig_date not in date_to_idx:
            continue
        t_idx = date_to_idx[sig_date]  # 取该信号日在 cal 中的下标, 传给下游函数
        # dict.get(键, 默认值): 取字典里该键的值; 若键不存在, 返回"默认值"(这里是空集合 set())。
        # 这样即使某天没有空头池数据, 也不会报错(KeyError)。
        sp_set = data["short_pool_by_date"].get(sig_date, set())
        # verbose 决定本次是否打印详细日志。为避免刷屏, 只在前 3 天、每隔 10 天、最后一天打印。
        #   (k < 3): 前 3 天      (k % 10 == 0): k 能被 10 整除(%是取余数)   (k == n_dates-1): 最后一天
        # 这三个条件用 or 连接, 任意一个成立就为 True。
        verbose = (k < 3) or (k % 10 == 0) or (k == n_dates - 1)

        # 调用核心函数处理这一个截面(这一天)。这里用"关键字参数(参数名=值)"传参,
        # 好处是: 可读性强, 且不依赖参数顺序。
        # 关键: dedup_rank=DEDUP_RANK 就是本实验的唯一改动——让行业内去重按 NZC 排序。
        out = ROP.process_one_section(
            section_idx=t_idx, cal=cal,
            log_price_wide=data["log_price_wide"], ret_wide=data["ret_wide"],
            industry_codes=data["industry_codes"], stock_codes=data["stock_codes"],
            short_pool_set=sp_set,
            train_window=TRAIN_WINDOW, valid_window=VALID_WINDOW,
            signal_type=SIGNAL_TYPE, cv_folds=CV_FOLDS, cv_label_h=CV_LABEL_H,
            dedup_rank=DEDUP_RANK,
            verbose=verbose,
        )
        # out 是 process_one_section 返回的字典。若该天没有可用 pair, 它会带 "skip": True。
        if out.get("skip"):
            # list.append(x): 往列表末尾追加一个元素。这里追加一行"被跳过"的统计字典。
            # 这些 out.get("键", 0) 都给了默认值 0, 即使某键缺失也安全。
            daily_stats.append({
                "date": sig_date, "skip": True, "reason": out.get("reason", ""),
                "n_pairs_total": out.get("n_pairs_total", 0),
                "n_pairs_ou_pass": out.get("n_pairs_ou_pass", 0),
                "n_pairs_dedup": out.get("n_pairs_dedup", 0),
                "n_pairs_legal": out.get("n_pairs_legal", 0),
                "n_pairs_top20": out.get("n_pairs_top20", 0),
            })
            continue  # 跳过的天不收集 pair_log, 直接进入下一天
        # 正常的天: 把该天的 pair_log(一个 DataFrame)收集起来。
        pair_log_chunks.append(out["pair_log"])
        daily_stats.append({
            "date": sig_date, "skip": False, "reason": "",
            "n_pairs_total": out["n_pairs_total"],
            "n_pairs_ou_pass": out["n_pairs_ou_pass"],
            "n_pairs_dedup": out["n_pairs_dedup"],
            "n_pairs_legal": out["n_pairs_legal"],
            "n_pairs_top20": out["n_pairs_top20"],
        })

    # 循环结束。"if not 列表": 当列表为空时 not [] 为 True, 说明一天都没产出, 报错并 return 提前结束函数。
    if not pair_log_chunks:
        print("\n[ERROR] 无任何有效截面, 退出")
        return

    # pd.concat(列表, ignore_index=True): 把"很多张小表"沿行方向拼成"一张大表";
    #   ignore_index=True 表示丢弃各小表原来的行号, 重新从 0 连续编号。
    pair_log_full = pd.concat(pair_log_chunks, ignore_index=True)
    # 下面三行是"数据规整", 保证列的类型统一(下游读取时不会因类型不一致出错):
    # 用 df["列名"] = ... 的写法可"新增列"或"覆盖原列"。
    pair_log_full["date"] = pd.to_datetime(pair_log_full["date"])  # 把 date 列统一转成 datetime 时间类型
    # 链式调用: .astype(str) 先转成字符串, 再 .str.zfill(6) 左侧补 0 到 6 位。
    #   股票代码如 600000、000001 必须是 6 位字符串; 若被当成数字会丢掉前导 0(1 而非 000001)。
    pair_log_full["stock_i"] = pair_log_full["stock_i"].astype(str).str.zfill(6)
    pair_log_full["stock_j"] = pair_log_full["stock_j"].astype(str).str.zfill(6)
    # sort_values(列名列表): 按多列排序(先 date, 再 stock_i, 再 stock_j), 让输出有稳定、可读的顺序。
    # reset_index(drop=True): 排序后行号会乱, 重置成 0,1,2,...; drop=True 表示不把旧行号留成新列。
    pair_log_full = pair_log_full.sort_values(
        ["date", "stock_i", "stock_j"]).reset_index(drop=True)
    # pd.DataFrame(字典列表): 把"由字典组成的列表"转成一张表, 每个字典是一行, 字典的键成为列名。
    stats_df = pd.DataFrame(daily_stats)

    out_stats = os.path.join(OUTPUT_DIR, "daily_stats_NZCdedup.parquet")
    # to_parquet(路径, index=False): 把表保存成 parquet 文件(列式存储, 读写快、体积小);
    #   index=False 表示不把行号也写进文件。
    pair_log_full.to_parquet(OUT_PATH, index=False)
    stats_df.to_parquet(out_stats, index=False)

    # 收尾: 打印汇总信息, 方便一眼确认结果是否正常。
    print(f"\n{'='*60}")
    # time.time()-t0: 现在时间减去开始时间 = 总耗时(秒); ":.0f" 表示按 0 位小数(取整)显示。
    print(f"完成, 总耗时 {time.time()-t0:.0f}s")
    # ":," 是千分位分隔(如 1,234,567), 让大数字更易读; .nunique() 统计某列的"不同值个数"。
    print(f"  pair_log: {len(pair_log_full):,} 行, "
          f"{pair_log_full['date'].nunique()} 个截面, "
          f"{pair_log_full['date'].min().date()} ~ {pair_log_full['date'].max().date()}")
    # is_legal 是布尔列(True/False)。对布尔列求 .sum() = True 的个数; 求 .mean() = True 的占比。
    # int(...) 把结果转成整数; *100 与 ":.1f" 把占比显示成 1 位小数的百分数。
    print(f"  is_legal=True: {int(pair_log_full['is_legal'].sum()):,} 行 "
          f"({pair_log_full['is_legal'].mean()*100:.1f}%)")
    print(f"  is_top20=True: {int(pair_log_full['is_top20'].sum()):,} 行")
    print(f"  已写入: {OUT_PATH}")
    print(f"  已写入: {out_stats}")
    print(f"{'='*60}")


# 这是 Python 的经典入口写法:
#   当本文件被"直接运行"时, 特殊变量 __name__ 的值是 "__main__", 条件成立 → 执行 main()。
#   当本文件被别的脚本"import 导入"时, __name__ 是模块名(非 "__main__"), 不会自动执行 main()。
# 好处: 同一个文件既能当脚本跑, 又能当库被别处复用, 互不干扰。
if __name__ == "__main__":
    main()
