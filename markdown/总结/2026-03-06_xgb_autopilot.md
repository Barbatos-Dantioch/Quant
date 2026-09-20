
# 2026-03-06_xgb_autopilot 实验系列总结

> 输出目录：`output/2026-03-06_xgb_autopilot/`
> 主题：XGBoost 多阶段自动调参流水线（Stage 0~2 实际产出结果），均为**头组持仓（多头）实验**
> 基础配置详见 `markdown/baseline_config.md`（Stage0 baseline 与该文档一致）

## 共同框架

- **Backbone（骨架）**：
  - `ControlBackbone`：413 特征，`max_depth=5, colsample=0.3, gamma=5, reg_alpha=1`，固定窗口 window_size=4（非expanding），early_stop=30
  - `HeadBackbone`：325 特征，`reg_alpha=3`，window_size=6（expanding扩展窗），early_stop=30，min_improve=0.0005
  - 两者均为 3 种子（33/42/101）集成
- **持仓回测**：头组（第10组）5日调仓，手续费0.07%，基准 `avg_return.pkl`（5日开盘到开盘收益）

## Stage 0：baseline 确认

| 编号 | 说明 | Rank IC | 头组超额年化 | 头组超额IR | 头组超额最大回撤 |
|---|---|---|---|---|---|
| BL-0 | ControlBackbone baseline | 0.0557 | 10.16% | 1.101 | -14.30% |

持仓回测口径（考虑调仓成本）：超额年化 7.76%，超额夏普 0.53，超额最大回撤 -13.25%

## Stage 1：Control / Head Backbone 对照

| 编号 | Backbone | Rank IC | 头组超额年化 | 头组超额IR | 头组超额最大回撤 |
|---|---|---|---|---|---|
| B0 | ControlBackbone（同BL-0） | 0.0557 | 10.16% | 1.101 | -14.30% |
| B1 | HeadBackbone | 0.0565 | 9.48% | 1.127 | -7.07% |

**结论**：HeadBackbone（扩展窗+更强正则）Rank IC 略高，超额回撤显著更小（-7.07% vs -14.30%），但超额年化收益略低

## Stage 2：单点筛选/标签/权重消融（基于 ControlBackbone）

| 编号 | 改动点 | Rank IC | 头组超额年化 | 头组超额IR | 备注 |
|---|---|---|---|---|---|
| S2-00 | baseline（同BL-0） | 0.0557 | 10.16% | 1.101 | — |
| S2-01 | rank_label（排名标签） | 0.0671 | 4.80% | 0.390 | IC 更高但超额年化和IR明显下降 |
| S2-02 | sample_weight（样本加权） | -0.0326 | 8.78% | 0.904 | Rank IC 转负，效果变差 |
| S2-03 | lambda_rank（排序损失） | — | — | — | 运行报错（标签需为非负整数，XGBoost ranking 检查失败） |
| S2-04 | label_stretch（标签拉伸） | -0.0083 | 9.45% | 1.047 | 与baseline接近，略弱 |

**结论**：Stage2 所有改动均未超越 baseline（S2-00），rank_label 虽提升 Rank IC 但超额表现反而下降，说明 Rank IC 与超额收益指标并不总是一致

## 最终结果

流水线最终选定的最优模型仍为 **BL-0**（stage0_baseline），后续 Stage 3（单设计）、Stage 4（复杂设计）、Stage 5（交互设计）在 `pipeline_state.json` 中记录为 `None`，即未产出优于 baseline 的结果或未完整执行。

| 指标 | BL-0（最终最优） |
|---|---|
| Rank IC | 0.0557 |
| 头组超额年化收益 | 10.16% |
| 头组超额IR | 1.101 |
| 头组超额最大回撤 | -14.30% |
| 持仓回测超额年化 | 7.76% |
| 持仓回测超额夏普 | 0.53 |
| 持仓回测超额最大回撤 | -13.25% |

