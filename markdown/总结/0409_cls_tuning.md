
# 0409_cls_tuning 实验系列总结

> 输出目录：`output/0409_cls_tuning/`
> 主题：分类模型（cls）超参数调优，回归模型固定

## 共同框架

- **数据/回测**：同 0407/0408 系列（融券池尾组 + 全市场头组，2024-2025 回测）
- **回归模型固定覆盖参数**：`max_depth=6, colsample_bytree=0.5, reg_alpha=2.0`（其余为 baseline_config 默认值）
- **融合公式**：`score = (1-α)×reg + (α/2)×top − (α/2)×bot`，α=0.5
- **Baseline**：CBL-01，`top_k=0.1`，分类模型无覆盖参数（使用默认 XGB 分类配置），`cls_boost_round=200, cls_early_stop=20`
- 各实验仅调整分类模型的 `top_k`（分类样本头尾比例）、分类模型超参数（depth/colsample/gamma/mcw/reg_alpha）以及 boost_round/early_stop

## 实验设计与结果

| 编号 | top_k | 分类模型参数差异（相对默认） | boost_round/early_stop | 多空年化 | 多空夏普 | 多空Calmar |
|---|---|---|---|---|---|---|
| CBL-01 | 0.10 | 无（baseline） | 200/20 | 75.0% | 3.042 | 3.43 |
| CA-01 | 0.05 | depth=3, gamma=8, reg_alpha=2 | 150/15 | 74.0% | 2.888 | 3.33 |
| CA-02 | 0.05 | depth=4, gamma=8 | 200/20 | 76.4% | 3.005 | 3.40 |
| CB-01 | 0.15 | depth=6, colsample=0.5, gamma=3 | 400/30 | 72.6% | 3.004 | 3.41 |
| CB-02 | 0.15 | colsample=0.5, gamma=3 | 300/20 | 73.3% | 3.016 | 3.51 |
| CC-01 | 0.10 | depth=7, colsample=0.5 | 400/30 | 73.6% | 3.034 | 3.37 |
| CC-02 | 0.10 | depth=6, colsample=0.5 | 400/30 | 75.8% | 3.117 | 3.53 |
| CD-01 | 0.10 | depth=4, mcw=2, colsample=0.5 | 300/20 | 73.7% | 3.014 | 3.54 |
| CD-02 | 0.10 | mcw=3, colsample=0.4, reg_alpha=2 | 200/20 | 73.8% | 3.012 | 3.45 |
| CE-01 | 0.10 | depth=6, colsample=0.5, gamma=3 | 400/30 | 76.1% | 3.119 | 3.46 |
| CE-02 | 0.15 | depth=3, colsample=0.5 | 400/20 | 74.4% | 3.027 | 3.58 |

## 结论

- 各变体夏普集中在 2.89~3.12，相对 baseline（CBL-01 夏普 3.042）提升有限
- 最优为 **CE-01**（top_k=0.1, depth=6, colsample=0.5, gamma=3, boost_round=400）：夏普 3.119，年化 76.1%
- 次优 CC-02（同结构，boost_round 相同但 early_stop=30 而非 CE-01 的30，两者几乎同配置）夏普 3.117
- top_k=0.05（收紧分类阈值，CA 系列）与 top_k=0.15（放宽，CB/CE-02 系列）均未展现出明显优于 top_k=0.10 的效果，分类模型超参数（depth/colsample/gamma）的影响大于 top_k 本身

