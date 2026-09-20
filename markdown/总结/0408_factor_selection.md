
# 0408_factor_selection 实验系列总结

> 输出目录：`output/0408_factor_selection/`
> 共 34 组实验，验证不同因子筛选方法对多空回测效果的影响

## 共同框架

- **数据**：`all` 库（全市场，训练头组，1088 特征）+ `short` 库（融券池，训练尾组，1088 特征），回测 2024-2025，尾组约束仅从市值前 75% 中选
- **因子分组**：A 组 565 个因子，B 组 523 个因子，FULL 组 1088 个（全部）因子
- **模型**：`fusion` 组，XGBoost，top_k=10%，α=0.5，半年滚动窗口，3 种子集成（与 0407 系列同一套 fusion 基准配置）
- **筛选方法**：
  - `filter_corr`：按相关性阈值剔除冗余因子（threshold 越低剔除越多）
  - `filter_rankic`：按 Rank IC 排序保留 top keep_ratio 比例因子
  - `filter_mv_exposure`：剔除市值暴露超过阈值的因子

## 实验设计与结果（多空指标）

| 编号 | 因子池 | 筛选方法 | 参数 | 多空年化 | 多空夏普 | 多空Calmar |
|---|---|---|---|---|---|---|
| FA-BL | A(565) | 无 | — | 60.4% | 2.247 | 2.26 |
| FA-CORR-95 | A(565) | filter_corr | threshold=0.95 | 59.0% | 2.188 | 2.21 |
| FA-CORR-90 | A(565) | filter_corr | threshold=0.90 | 54.6% | 2.026 | 2.02 |
| FA-CORR-80 | A(565) | filter_corr | threshold=0.80 | 56.0% | 2.075 | 2.11 |
| FA-IC-90 | A(565) | filter_rankic | keep_ratio=0.9 | 56.3% | 2.122 | 2.16 |
| FA-IC-80 | A(565) | filter_rankic | keep_ratio=0.8 | 58.2% | 2.139 | 2.15 |
| FA-IC-70 | A(565) | filter_rankic | keep_ratio=0.7 | 53.2% | 2.017 | 2.14 |
| FA-IC-60 | A(565) | filter_rankic | keep_ratio=0.6 | 54.0% | 1.984 | 2.02 |
| FB-BL | B(523) | 无 | — | 58.0% | 2.348 | 2.76 |
| FB-CORR-95 | B(523) | filter_corr | threshold=0.95 | 58.4% | 2.337 | 2.69 |
| FB-CORR-90 | B(523) | filter_corr | threshold=0.90 | 59.2% | 2.361 | 2.76 |
| FB-CORR-80 | B(523) | filter_corr | threshold=0.80 | 49.8% | 1.950 | 2.32 |
| FB-IC-90 | B(523) | filter_rankic | keep_ratio=0.9 | 63.4% | 2.555 | 3.12 |
| FB-IC-80 | B(523) | filter_rankic | keep_ratio=0.8 | 60.0% | 2.387 | 3.02 |
| FB-IC-70 | B(523) | filter_rankic | keep_ratio=0.7 | 58.4% | 2.253 | 2.54 |
| FB-IC-60 | B(523) | filter_rankic | keep_ratio=0.6 | 60.4% | 2.379 | 2.77 |
| FULL-BL | FULL(1088) | 无 | — | 67.1% | 2.616 | 3.01 |
| FULL-CORR-95 | FULL(1088) | filter_corr | threshold=0.95 | 71.0% | 2.839 | 3.23 |
| FULL-CORR-90 | FULL(1088) | filter_corr | threshold=0.90 | 73.7% | 2.890 | 3.35 |
| FULL-CORR-80 | FULL(1088) | filter_corr | threshold=0.80 | 69.3% | 2.762 | 3.11 |
| FULL-IC-90 | FULL(1088) | filter_rankic | keep_ratio=0.9 | 75.2% | 2.982 | 3.51 |
| FULL-IC-80 | FULL(1088) | filter_rankic | keep_ratio=0.8 | 69.9% | 2.764 | 3.18 |
| FULL-IC-70 | FULL(1088) | filter_rankic | keep_ratio=0.7 | 无回测结果（未跑完） | — | — |
| FULL-MV-10 | FULL(1088) | filter_mv_exposure | threshold=0.1 | 55.1% | 2.253 | 2.95 |
| FULL-TAIL-MV75 | FULL(1088) | 无 | （尾组市值前75%对照组，同FULL-BL配置） | 70.3% | 2.819 | 3.24 |

## 结论

- FULL 组（全部 1088 因子不筛选/轻筛选）整体优于 A 组、B 组子集，说明因子筛选未带来明显增益，反而全量因子表现更稳健
- FULL 组内以 `filter_rankic keep_ratio=0.9`（FULL-IC-90）最优，多空夏普 2.982、Calmar 3.51，为全系列最佳
- `filter_mv_exposure`（剔除市值暴露因子）反而降低表现（FULL-MV-10 夏普仅 2.253），说明市值暴露因子对多空收益有正贡献
- A 组（565因子）整体弱于 B 组（523因子）和 FULL 组，相关性/IC 筛选对 A 组提升有限

