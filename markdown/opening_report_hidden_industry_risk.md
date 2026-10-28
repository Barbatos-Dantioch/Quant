# 申万行业约束下 A 股量化投资组合的隐性关联风险与风控改进

## ——基于概念共属和分析师共同覆盖的研究

## 一、研究背景与问题提出

### （一）研究背景

行业中性化是 A 股量化投资组合管理中的常见风险控制方式。对于指数增强类产品，组合通常以基准指数的申万行业权重为目标，限制组合相对基准在各行业上的主动偏离，以避免收益主要来自行业配置，而更多来自个股选择。

申万行业分类具有单一归属特征：每只股票通常只被分配至一个行业。该方式便于组合管理，但不能完整反映企业的多业务经营、产品多元化和产业链关联。例如，某家公司在申万分类中属于建筑材料行业，但其产品同时应用于电子、半导体或新能源产业链。若优化器只根据申万行业矩阵进行约束，该股票在申万电子行业上的名义暴露可能为零，但组合仍可能通过该股票间接承担电子产业链风险。

因此，可能出现以下情况：

```text
申万行业权重合规，但组合在跨行业业务关联或产业链关联上的风险仍然集中。
```

国际研究表明，企业之间的经济关联不完全等同于传统行业分类。Hoberg and Phillips（2016）基于公司披露文本构建动态产品市场网络，允许企业与多个产品市场和竞争者形成连续关联。MSCI Barra 风险模型也曾引入多行业暴露，以改善单一行业归属下的风险预测和收益归因。

本文不试图在缺少完整分部收入和主营业务数据的情况下精确估计企业真实多行业收入权重，而是使用同花顺概念共属和分析师共同覆盖数据，构造申万行业之外的隐性关联风险代理，并检验将其加入组合优化器后能否改善组合风险特征。

### （二）研究问题

1. 申万同行业股票与跨申万行业股票之间是否存在显著不同的概念关联和分析师关联？
2. 在满足申万行业约束的组合中，是否仍然存在稳定的隐性关联主动暴露？
3. 概念关联和共同分析师覆盖是否能够解释传统申万行业约束之外的组合波动或尾部风险？
4. 在原有申万行业约束基础上加入隐性关联约束，是否能够降低波动率、最大回撤和 CVaR？
5. 隐性关联约束带来的风险改善是否足以覆盖收益损失、换手率上升和交易成本增加？

## 二、研究意义

### （一）理论意义

第一，将传统行业分类的讨论从“分类质量”推进到“组合风控后果”。本文考察传统单一行业归属是否会造成组合层面的风险遗漏。

第二，将概念关联和分析师共同覆盖从收益预测维度转化为组合风险诊断维度。共同分析师覆盖研究通常关注信息传播、收益溢出和动量预测，本文则关注这些关联是否导致组合在传统行业约束之外形成共同风险。

第三，建立“单一行业分类—隐性关联暴露—组合风险—风控改进”的完整研究链条。

### （二）实践意义

本文为量化机构提供一种不依赖完整主营业务拆分数据的辅助风控工具：使用同花顺概念识别跨行业主题或业务关联，使用共同分析师覆盖识别信息网络关联，并将其转化为组合层面的风险约束或软惩罚。

## 三、文献综述

### （一）量化选股与机器学习资产定价

传统资产定价研究形成了规模、价值、动量、盈利能力和投资等经典因子体系。Fama and French（1993, 2015）构建了市场、规模、价值、盈利和投资因子框架；Jegadeesh and Titman（1993）发现中期价格动量；Moskowitz and Grinblatt（1999）讨论了行业动量及其与个股动量之间的关系。

Gu, Kelly and Xiu（2020）比较了正则化回归、树模型、随机森林、梯度提升树和神经网络等方法，发现机器学习能够通过捕捉预测变量的非线性交互改善横截面收益预测。Kelly, Pruitt and Su（2019）说明股票特征也可以被视为潜在风险暴露的代理；Gu, Kelly and Xiu（2021）进一步使用自编码器提取非线性潜在资产定价因子。

本文使用 LightGBM 作为基础收益预测模型，但不将其作为研究贡献的核心；收益预测与关联风险识别分别建模。

### （二）行业分类与多行业暴露

Hoberg and Phillips（2016）使用公司披露文本构建产品市场相似度网络，说明企业可以同时与多个产品市场形成关联，而不应被固定在单一行业标签内。Hoberg and Phillips（2017）进一步研究多行业企业的行业选择与产品语言重叠。

MSCI Barra USE3 风险模型允许股票具有多个非零行业暴露，并将多行业配置用于改善收益来源识别和风险预测。但该类模型通常依赖更丰富的企业业务信息，与本文利用概念共属构造代理暴露的方法不同。

因此，本文不将概念矩阵等同于真实主营业务矩阵，而将其定义为：

```text
申万单一行业分类之外的隐性业务、主题和产业链关联代理。
```

### （三）股票关联网络与收益溢出

Cohen and Frazzini（2008）研究客户—供应商关系与可预测收益；Menzly and Ozbas（2010）研究供应链中的信息传导；Cohen and Lou（2012）讨论产品市场相似度与股票收益联动。

Ali and Hirshleifer（2020）基于共同分析师覆盖构建股票关联网络，发现分析师关联股票之间存在显著的动量溢出。共同分析师覆盖更可能反映信息传播、投资者关注和共同估值框架，因此本文将其定义为“信息网络关联风险”，而不是主营业务暴露。

### （四）组合风险、行业中性化与优化

Grinold and Kahn（2000）系统讨论主动组合构建、跟踪误差和风险预算；Clarke, de Silva and Thorley（2006）讨论组合风险与最小方差管理；Jagannathan and Ma（2003）说明组合约束可能通过限制极端权重改善组合稳定性；Ledoit and Wolf（2004）提出高维协方差矩阵收缩方法。

DeMiguel, Garlappi and Uppal（2009）提醒复杂优化模型可能受到估计误差影响，因此本文采用滚动样本外检验。Harvey and Liu（2015）、White（2000）以及 Bailey et al.（2014）分别从多重检验、数据挖掘和回测过拟合角度说明量化研究需要严格控制样本外偏差。

### （五）文献述评

现有文献存在三方面不足：

1. 行业分类和文本网络研究较少考察传统行业分类在组合行业中性化中的风险遗漏；
2. 概念关联和分析师共同覆盖研究主要用于收益预测，较少用于组合层面的风险诊断；
3. 较少形成“申万行业约束—隐性关联暴露—组合风险—优化器改进”的完整链条。

本文的定位是：不直接估计真实多行业主营业务收入，而是使用概念共属和共同分析师覆盖构造隐性关联风险代理，检验其在传统申万行业风控中的增量价值。

## 四、研究假设

### 假设一：跨申万行业股票之间存在显著的概念关联

在控制市值、流动性、波动率和市场 beta 后，跨申万行业股票之间仍存在显著的概念共属关系。

### 假设二：跨申万行业股票之间存在显著的分析师网络关联

共同分析师覆盖不仅发生在同一申万行业内部，也会形成跨行业的信息关联。

### 假设三：申万行业约束组合仍存在隐性关联主动暴露

满足申万行业约束的组合，在概念隐性维度和分析师网络维度上仍会出现相对于基准的主动暴露。

### 假设四：隐性关联暴露能够预测未来组合风险

隐性关联主动暴露越高，未来组合波动率、最大回撤和 CVaR 越高。

### 假设五：增补隐性关联约束能够改善风控效果

在申万行业约束基础上增加概念或分析师关联约束，可以降低组合尾部风险和极端回撤，但不必然提高绝对收益。

## 五、研究设计

### （一）数据来源与样本范围

样本区间为 2018 年 1 月至 2026 年 9 月，主要数据包括：

- A 股股票日线收益、成交量、成交额、换手率和市值；
- 申万一级、二级和三级行业数据；
- 同花顺股票—概念归属数据；
- 分析师—股票覆盖数据。

所有变量遵循时点可得原则：在 t 日形成的变量最早用于 t+1 日或下一调仓日交易。概念成分和分析师覆盖必须使用历史时点可见版本。

### （二）LightGBM 基础选股模型

预测未来 20 个交易日的行业中性超额收益：

```text
y[i,t] = 个股 i 的未来收益 - 所属申万行业的未来收益
```

特征包括：

- 价格与动量：1、3、6、12 个月收益、12-1 动量、短期反转、相对行业动量；
- 波动与流动性：20/60 日波动率、下行波动率、最大回撤、换手率、成交额变化、Amihud 非流动性；
- 估值：市盈率、市净率、市销率、盈利收益率；
- 基本面：ROE、ROA、毛利率、资产负债率、经营现金流、盈利增长和盈利稳定性。

采用滚动样本外训练：

```text
训练窗口：36 或 60 个月
验证窗口：12 个月
测试窗口：12 个月
调仓频率：月度
```

每次选择模型得分前 10% 的股票进入候选组合，并设置 Elastic Net 和简单因子排序作为基准模型。

### （三）概念关联矩阵

定义股票—概念矩阵：

```text
X_t[i,c] = 1：股票 i 在 t 时点属于概念 c
X_t[i,c] = 0：否则
```

不使用“概念成分股越少，概念权重越高”的逆频率方法，因为概念规模不能直接代表概念影响力。

对概念数据进行以下过滤：删除成分股过少、覆盖过广、存续时间过短以及明显非业务性质的标签。

### （四）NMF 隐性关联维度

对股票—概念矩阵进行非负矩阵分解：

```text
X_t ≈ Z_t × H_t
```

其中：

```text
Z_t：股票 × K 个隐性关联维度
H_t：K 个隐性关联维度 × 概念
```

含义为：

```text
Z_t[i,k]：股票 i 对隐性维度 k 的暴露
H_t[k,c]：概念 c 对隐性维度 k 的贡献
```

NMF 要求：

```text
Z_t[i,k] ≥ 0
H_t[k,c] ≥ 0
```

因此一只股票可以同时暴露于多个隐性关联维度。

测试：

```text
K = 5、10、15、20、30
```

根据重构误差、维度可解释性、随机种子稳定性、滚动窗口稳定性和样本外风控效果选择 K。NMF 不要求不同维度正交；对高度重复的维度通过载荷相似度、股票暴露相关性和收益相关性进行合并或删除。

组合相对基准的主动权重为：

```text
w_A,t = w_P,t - w_B,t
```

第 k 个概念隐性维度的主动暴露为：

```text
E_concept,k,t = Σ_i w_A,t[i] × Z_t[i,k]
```

向量形式为：

```text
E_concept,t = Z_t' × w_A,t
```

### （五）概念收益和动态风险

对个股收益进行横截面回归：

```text
R[i,t] = α_t + β_m,t × Market[i,t]
               + β_style,t × Style[i,t]
               + β_SW,t × SWIndustry[i,t]
               + u[i,t]
```

其中 `u[i,t]` 为控制市场、风格、申万行业和流动性因素后的残差收益。

根据 NMF 股票暴露构造隐性维度收益：

```text
F[k,t] = Σ_i Z_t[i,k] × u[i,t] / Σ_i Z_t[i,k]
```

使用过去 60、120 或 252 个交易日估计隐性维度收益协方差矩阵：

```text
Σ_concept,t = Cov(F[t-L+1:t])
```

并使用协方差收缩提高高维估计稳定性。概念关联风险为：

```text
Risk_concept,t
= E_concept,t' × Σ_concept,t × E_concept,t
```

### （六）分析师共同覆盖网络

定义：

```text
A_t[i,j] = 股票 i 与股票 j 的共同分析师覆盖强度
```

可采用：

```text
A_t[i,j] = 共同覆盖股票 i 和 j 的分析师数量
```

也可对覆盖范围广的分析师降权：

```text
A_t[i,j]
= Σ_a I(a 同时覆盖 i 和 j) / log(1 + m_a)
```

其中 `m_a` 为分析师 a 覆盖的股票数量。

分析师网络通过谱分解、低秩分解或 NMF 得到：

```text
Z_analyst,t
```

并计算：

```text
E_analyst,t = Z_analyst,t' × (w_P,t - w_B,t)
```

分析师网络风险为：

```text
Risk_analyst,t
= E_analyst,t' × Σ_analyst,t × E_analyst,t
```

### （七）组合优化模型

组合优化目标为：

```text
最大化：

预测收益
- λ_TE × 传统主动风险
- λ_C × 概念关联风险
- λ_A × 分析师网络风险
- λ_TC × 交易成本
```

基本约束：

```text
Σ_i w_i = 1
w_i ≥ 0
w_i ≤ w_i,max
```

申万行业约束：

```text
|S_t' × (w - w_B)| ≤ δ_SW
```

概念隐性维度约束：

```text
|Z_concept,t' × (w - w_B)| ≤ η_C
```

分析师网络约束：

```text
|Z_analyst,t' × (w - w_B)| ≤ η_A
```

### （八）对照组合

1. 无行业约束组合：LightGBM 选股加基本组合约束；
2. 申万行业约束组合；
3. 申万行业加概念关联约束组合；
4. 申万行业加分析师网络约束组合；
5. 申万行业、概念关联和分析师网络联合约束组合。

主要比较：

```text
申万约束 vs 申万 + 概念约束
申万约束 vs 申万 + 分析师约束
申万约束 vs 联合约束
```

## 六、评价指标

### （一）收益和风险

- 年化收益率和年化超额收益；
- 年化波动率；
- 最大回撤；
- 跟踪误差；
- CVaR；
- 夏普比率；
- 信息比率；
- 成本后 Alpha。

### （二）交易特征

- 年化换手率；
- 交易成本；
- 组合可行性比例；
- 权重集中度。

### （三）风控改善效率

```text
RiskImprovement
= (Risk_SW - Risk_SW+Concept) / Risk_SW
```

约束成本为：

```text
ConstraintCost
= Alpha_SW - Alpha_SW+Concept
```

重点考察风险改善幅度是否超过收益和交易成本代价。

## 七、统计检验

### （一）概念关联是否跨越申万行业

```text
ConceptSimilarity[i,j,t]
= α + β × CrossSW[i,j,t]
+ γ' × Controls[i,j,t]
+ μ_t + ε[i,j,t]
```

控制变量包括市值差异、流动性差异、波动率差异、beta 差异和机构覆盖差异。

### （二）隐性暴露是否预测未来风险

```text
FutureRisk_P,t+1:t+h
= α + β1 × HED_concept,t
+ β2 × HEC_concept,t
+ β3 × Controls_t + ε_t
```

未来风险可以使用未来波动率、最大回撤、CVaR 和极端负收益。

### （三）增量解释力

比较以下模型：

```text
模型一：主动收益 = 申万行业因子 + 风格因子 + 残差
模型二：主动收益 = 申万行业因子 + 风格因子 + 概念隐性因子 + 残差
模型三：主动收益 = 申万行业因子 + 风格因子 + 分析师网络因子 + 残差
```

比较调整后 R²、残差波动率、因子显著性和极端损失解释能力。

## 八、稳健性检验

1. 使用 NMF、PCA/SVD 和谱分解比较隐性维度；
2. 使用 60、120 和 252 日风险估计窗口；
3. 比较 0/1 概念矩阵和行归一化概念矩阵；
4. 比较 ±0.5%、±1%、±2% 和完全行业中性；
5. 比较周度、双周和月度调仓；
6. 比较 LightGBM、Elastic Net 和简单因子排序；
7. 分析牛市、熊市、高波动和低波动时期；
8. 使用不同随机种子和滚动窗口检查 NMF 稳定性；
9. 使用 Bootstrap、SPA 或 Reality Check 控制多重检验和数据挖掘偏差。

## 九、预期创新点

### 创新点一：从行业分类质量转向组合风控后果

研究申万单一行业归属如何影响组合行业风险控制，而不只是比较分类标准本身。

### 创新点二：构造申万行业之外的隐性关联风险代理

在缺少完整主营业务拆分数据的条件下，使用同花顺概念共属和共同分析师覆盖构造两类风险代理。

### 创新点三：将关联网络嵌入主动组合优化

将概念隐性维度和分析师网络维度转化为主动暴露，并与申万行业约束共同进入优化器。

### 创新点四：从收益预测拓展至风控增量价值

重点考察概念和分析师关联能否降低组合波动、CVaR 和极端回撤。

## 十、研究局限

1. 同花顺概念不是完整主营业务数据，可能包含主题、政策和市场叙事标签；
2. 共同分析师覆盖主要反映信息网络，不等同于业务或产业链关系；
3. 概念标签存在主观性和更新滞后；
4. NMF 分解具有一定非唯一性，需要进行稳定性检验；
5. 新增约束可能降低收益并提高换手率，因此应评价风险收益权衡，而不是只追求风险最低。

## 十一、研究计划

### 第一阶段：数据整理与变量定义

完成收益、申万行业、同花顺概念和分析师覆盖数据整理，处理历史时点可得性、缺失值和停牌问题。

### 第二阶段：关联结构分析

构造股票—概念矩阵、概念共属矩阵和分析师共同覆盖矩阵，比较 NMF、PCA 和谱分解结果。

### 第三阶段：基础选股模型

完成因子构造、LightGBM 训练、样本外打分和简单组合回测。

### 第四阶段：风控优化与回测

完成申万约束、概念约束、分析师约束和联合约束组合的滚动样本外回测。

### 第五阶段：稳健性分析与论文撰写

完成参数检验、市场状态分析、多重检验和论文修改。

## 十二、预期结论

本文可能得到以下结果之一：

1. 概念关联具有显著增量风控价值；
2. 概念关联有效，而分析师关联价值有限；
3. 分析师网络有效，而概念标签噪声较大；
4. 隐性关联风险存在，但硬约束的收益代价过高，适合使用软惩罚或风险监控指标。

无论最终结果如何，本文的核心结论边界应当明确：

```text
本文不声称精确测量了上市公司的真实多行业主营业务暴露，
而是检验基于概念共属和分析师共同覆盖构造的隐性关联风险代理，
是否具有解释和改善组合风险的增量价值。
```

## 参考文献

1. Ali, U., & Hirshleifer, D. (2020). Shared Analyst Coverage: Unifying Momentum Spillover Effects. *Journal of Financial Economics*, 136(3), 649–675.
2. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39–69.
3. Clarke, R., de Silva, H., & Thorley, S. (2006). Minimum-Variance Portfolio Composition. *Journal of Portfolio Management*, 33(1), 31–45.
4. Cohen, L., & Frazzini, A. (2008). Economic Links and Predictable Returns. *Journal of Finance*, 63(4), 1977–2011.
5. Cohen, L., & Lou, D. (2012). Complicated Firms. *Journal of Financial Economics*, 104(2), 383–400.
6. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915–1953.
7. Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1), 3–56.
8. Fama, E. F., & French, K. R. (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1–22.
9. Freyberger, J., Neuhierl, A., & Weber, M. (2020). Dissecting Characteristics Nonparametrically. *Review of Financial Studies*, 33(5), 2326–2377.
10. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill.
11. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223–2273.
12. Gu, S., Kelly, B., & Xiu, D. (2021). Autoencoder Asset Pricing Models. *Journal of Econometrics*, 222(1), 429–450.
13. Harvey, C. R., & Liu, Y. (2015). Lucky Factors. *Journal of Financial Economics*, 116(1), 71–94.
14. Hoberg, G., & Phillips, G. (2016). Text-Based Network Industries and Endogenous Product Differentiation. *Journal of Political Economy*, 124(5), 1423–1465.
15. Hoberg, G., & Phillips, G. (2017). Conglomerate Industry Choice and Product Language. *Management Science*, 64(8), 3735–3754.
16. Jagannathan, R., & Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. *Journal of Finance*, 58(4), 1651–1684.
17. Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*, 48(1), 65–91.
18. Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics Are Covariances: A Unified Model of Risk and Return. *Journal of Financial Economics*, 134(3), 501–524.
19. Ledoit, O., & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
20. Menzly, L., & Ozbas, O. (2010). Market Segmentation and Cross-Predictability of Returns. *Journal of Finance*, 65(4), 1555–1580.
21. Moskowitz, T. J., & Grinblatt, M. (1999). Do Industries Explain Momentum? *Journal of Finance*, 54(4), 1249–1290.
22. White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126.
23. MSCI. (2011). *Multiple Industry Allocations in the Barra US Equity Model USE3*. MSCI Research.
24. MSCI. (2012). *Barra US Equity Model USE4*. MSCI Research.
25. 熊和平等（2023）。基于分析师共同覆盖的 A 股关联公司动量溢出研究。正式提交前核对完整作者、刊物及页码信息。
26. 高雅（2024）。基于同花顺概念板块的主题关联与资本市场反应研究。正式提交前核对完整作者、刊物及页码信息。
27. 申万宏源金融工程团队（2021）。申万行业分类调整及其对量化投资的影响研究报告。正式提交前核对报告标题和发布日期。
