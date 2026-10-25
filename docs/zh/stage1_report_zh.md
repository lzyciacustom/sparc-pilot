> 说明：第一阶段（HEST-Bench）中文报告，保留原样；英文版方法见 `docs/stage1_hestbench/`，汇总见 `docs/RESULTS.md`。

# SPARC 改进方案 pilot 报告（Aim 2）

数据：HEST-Bench 57 张 Visium 全转录组切片（16 个患者组；CCRCC 24、PRAD 23、HCC 2、LYMPH_IDC 4、READ 4）。4 折患者组交叉验证，患者组从不跨训练/测试。编码器 H-optimus-0（H-optimus-1 申请未获批）。程序：SPARC 代码列出的 43 个程序中可从 MSigDB 2026.2.Hs 取得的 36 个。所有准确率均同时报告原始 Pearson 与去深度（对 log 总 counts 残差化）Pearson。

## 0. 基线复现

| 指标 | 全部 57 张 | READ（CRC，4 张） |
|---|---|---|
| 按切片平均 Pearson | 0.342 | 0.456 |
| 中心化 Pearson | 0.246 | 0.323 |
| 去深度 Pearson | 0.147 | 0.248 |
| 跨 patch 合并 Pearson（论文图 2a 口径） | 0.528（论文 0.74） | — |

与论文差距主要来自训练规模（每折约 40 张 vs HEST-1k）与编码器。表：[baseline_sparcmap_ssgsea_cv.csv]({{artifact:fb0c3023-b6c2-4cf7-b3f3-28a46eb1adbc}})。

## 1. 程序表示 / 训练目标（Aim 2 第 i 条）

![程序目标 pilot]({{artifact:5ce81dc7-e181-40c1-9ec3-c1fb932e4aed}})

- ssGSEA 目标本身的共享成分：切片内 PC1 中位 0.598、程序间平均 r 0.537；PC1 与 log 深度 |r| 中位 0.939。MLM 为 0.225 / 0.078，ULM 0.351 / 0.255。
- **随机基因集对照**（大小匹配，3 次抽样）：ssGSEA 随机集 Pearson 0.316 vs 真实程序 0.342；16 个患者组层面差 +0.025，p = 0.21。中心化后随机集仅 0.044（真实 0.246），去深度后 0.083（真实 0.147）。原始 Pearson 主要反映共享深度/细胞密度轴。
- H&E 可预测性（去深度 Pearson）：ULM 0.233、ssGSEA 去深度 0.233、MLM 0.177、ssGSEA 0.147。MLM − ssGSEA +0.030（组 p = 0.009）；ULM − MLM +0.056（组 p = 3.1e-5）。
- 拆半特异性（同程序 − 跨程序拆半 r）：ULM 0.188、ssGSEA 中心化 0.191、ssGSEA 0.162、MLM 0.105。
- READ：ULM 目标下 23/36 个程序去深度 Pearson > 0.3（E2F 0.69、MYC 0.67、G2M 0.65、ECM 0.64），随机基因集 0 个。仅 4 张切片、2 个患者组。

**结论**：去除共享成分 → 支持；"联合估计（MLM）本身更好" → 部分支持（收益来自去除共享成分，ULM 或去深度 ssGSEA 不差于 MLM）；Spectra → 无定论。

偏差：1 个训练 seed；MLM/ULM 随机对照各 1 次抽样；Spectra 仅 8 张切片、2,000 epochs、精简基因面板、无 H&E 预测测试。详见 [methods_targets.md]({{artifact:04590f73-9cbe-4bdf-a1f6-b4b7dbc1d6dd}})、[A34_cv_summary.csv]({{artifact:87e85238-91a8-4c67-95d6-b8b062e8a297}})、[A34_paired_tests.csv]({{artifact:2940fd6d-e6a5-4281-8d83-f62aa34e4d29}})。

## 2. CONCERT 式空间建模与预测头（Aim 2 第 ii 条）

![空间上下文消融]({{artifact:bcc66955-ae24-4a31-a5c3-a48ffb7d5178}})

- CONCERT 式注意力偏置（距离 + H&E 相似度 + 区室）vs 仅距离（SPARC-Map，k=16），2 seeds：ssGSEA raw 0.372 vs 0.362（+0.010，切片 p = 0.016，组 p = 0.13）；去深度 +0.005（组 p = 0.029）。MLM：+0.006 / +0.003，p > 0.3。
- 同一基线不同 seed 的去深度差异为 −0.020（p = 0.0013），大于任何架构效应；CONCERT 的去深度效应在两个 seed 中符号不一致（+0.012 / −0.002）。
- 非空间 MLP：ssGSEA 上与仅距离模型无差异（去深度 −0.006，p = 0.75）；MLM 上比仅距离低约 0.01（组 p = 0.003）。
- k=48 无收益（仅 1 seed）。READ 无集中收益。

![预测头比较]({{artifact:fa22b074-3dfe-45a3-b91a-7c707777d828}})

- 低秩高斯头：准确率基本不变；CRPS 0.567→0.439（ssGSEA）、0.513→0.377（MLM），p < 1e-9；单次采样恢复程序间相关结构（‖ΔC‖_F 4.96 vs 点预测 11.43）。
- Flow matching 头：raw Pearson 比确定性头低 0.038（ssGSEA）/ 0.043（MLM），边界 spot 准确率更低，采样协方差也不如高斯头。
- 校准：所有 16 个患者组都覆盖不足；高斯头 90% 区间覆盖 0.36–0.90（ssGSEA，均值 0.695）；spread-skill 仅 0.02–0.05。
- 点预测会过度相关程序（ssGSEA 程序间平均 r：预测 0.738 vs 真实 0.528）。

**结论**：CONCERT 式空间建模 → 不支持（效应小于 seed 噪声）；低秩高斯头 → 支持；flow matching 优于可解析概率头 → 不支持；按患者评估校准 → 部分支持（必要，但两种头都未校准）。

偏差：batch 1024 / lr 2e-4 / bf16（所有变体一致，含重跑基线）；k=48 仅 1 seed；CONCERT 未原样复现，是自行设计的加性注意力偏置；高斯与 flow 头都加了基线损失；无多重比较校正。详见 [methods_spatial.md]({{artifact:0f172bf1-b0d2-410e-882b-023a0a22a5ca}})、[B1_paired_tests.csv]({{artifact:be52ad73-9c33-4b5b-a4ee-023f1182e22e}})、[B2_head_summary.csv]({{artifact:928af93c-7558-4646-834d-b7bd341e73d3}})。

## 3. EPIG 主动选片（Aim 2 第 iii 条）

![主动选片学习曲线]({{artifact:822e5117-1de3-41e8-9493-38af94442a8f}})

- 设计：4 折 × 3 个随机初始集 = 12 次配对运行；2 张起步，每轮 +2 张，共 6 轮至 14 张；5 成员集成；ssGSEA 目标。
- 学习曲线 AUC（raw / 去深度）：随机 0.287 / 0.117；最大不确定性 0.293 / 0.114；EPIG 0.345 / 0.135；H&E 相似度（无模型对照）0.354 / 0.138。
- EPIG − 随机：raw +0.058（12/12，p = 0.0005；组 p = 0.0002）；去深度 +0.018（组 p = 0.008）。
- EPIG − 最大不确定性：raw +0.053（16/16 组）；去深度 +0.021（组 p = 0.16，不显著）。最大不确定性与随机无差异（p = 0.97）。
- **EPIG − H&E 相似度**：raw −0.009（p = 0.15），去深度 −0.003（p = 0.73）。EPIG 选片中 75% 与目标同组织类型（随机 50%）。
- 去深度后所有策略收益缩小约 3 倍。44–56% 的选择落在已覆盖的患者组。

**结论**：EPIG 优于随机 → 支持；优于最大不确定性 → 支持（raw），去深度在组层面不显著；收益来自 EPIG 的模型信息增益本身 → 不支持（无模型分布匹配可复现同等收益）。

偏差：仅 ssGSEA；每轮 8 epochs；spot 预算为事后重算；EPIG 在子采样 spot 上计算。详见 [methods_active.md]({{artifact:eeef657c-8b77-4c74-9249-bf8718efe865}})、[al_paired_comparisons.csv]({{artifact:0ce33b2d-d1fa-46a9-b85d-137d13765035}})。

## 4. 各 claim 汇总

| 提案 claim | pilot 结论 | 关键证据 |
|---|---|---|
| ssGSEA 目标被共享成分主导，需程序特异目标 | 支持 | 随机基因集 raw 0.316 vs 真实 0.342（组 p = 0.21） |
| 联合程序估计（MLM）给出更好的目标 | 部分支持 | MLM 去深度 +0.030 vs ssGSEA；但 ULM 再 +0.056 |
| Spectra 等监督因子模型 | 无定论 | 仅 8 张切片、无预测测试 |
| CONCERT 式空间建模优于仅距离 / 非空间 | 不支持 | +0.010 raw / +0.005 去深度，小于 seed 噪声 0.020 |
| 可解析概率头可行 | 支持 | CRPS −23–27%，准确率不变 |
| Flow matching 优于可解析概率头 | 不支持 | raw −0.04，协方差与边界均不更好 |
| 按动物/患者评估校准 | 部分支持 | 覆盖 0.36–0.90 跨组，全部覆盖不足 |
| EPIG 优于随机与最大不确定性 | 支持 | AUC +0.058 / +0.053 raw |
| EPIG 收益来自模型信息增益 | 不支持 | 无模型 H&E 相似度持平 |

## 5. 提案 Aim 2 措辞修改建议（英文，可直接替换）

**程序表示段**（替换 "To construct molecular training targets, I will evaluate joint program estimation ..." 一句及其后）：

> In a pilot on 57 HEST-Bench Visium sections, ssGSEA targets shared a dominant within-section axis tracking sequencing depth (median PC1 fraction 0.60; |r| with log counts 0.94), and size-matched random gene sets reached held-out Pearson close to that of the curated programs (0.32 vs 0.34). I will therefore construct targets that explicitly remove this shared component, comparing depth-residualised scores, regression-based estimates (decoupleR univariate and multivariate linear models) and supervised factor models, and will report every accuracy against a size-matched random-gene-set null and after depth residualisation.

**空间建模段**（替换 "Drawing on CONCERT, I will combine physical distance, H&E features and tissue-compartment information ..." 一段中的比较部分）：

> In the same pilot, adding distance, H&E-similarity and compartment attention biases to SPARC-Map changed held-out accuracy by at most 0.01, less than seed-to-seed variation, so I will test tissue-context modelling where it is expected to matter — program-specific targets, heterogeneous training data and the perturbation-response component — using at least five training seeds and patient-level tests. A low-rank Gaussian head preserved accuracy, reduced CRPS by about 25% and restored inter-program covariance in samples; it will be the primary probabilistic head, with conditional flow matching as an exploratory comparison, and predictive intervals will be recalibrated at the animal level.

**主动选片段**（替换 "Equal-budget comparisons with random and maximum-uncertainty selection will precede a prospective profiling round."）：

> In a simulated equal-budget experiment, EPIG outperformed random and maximum-uncertainty section selection, but a model-free H&E-embedding similarity rule achieved the same gain. Equal-budget comparisons will therefore include random, maximum-uncertainty and distribution-matching baselines, with budgets counted per animal, before a prospective profiling round.

## 6. 局限

- 编码器为 H-optimus-0，不是 SPARC 所用的 H-optimus-1；官方 SPARC-Map 权重未获访问，基线为按论文与公开代码重训。
- 训练规模小（每折约 40 张）；CRC 全转录组仅 READ 4 张、2 个患者组，CRC 层面结论只能作描述。
- 多数比较 1–2 个训练 seed；空间 pilot 显示 seed 噪声与效应量同量级。
- 扰动响应模块、空间 ATAC 监督、CRC 特异程序和 SPARC-Risk 层面的改进未测试（需要 Aim 1/3 数据或 TCGA）。
- HEST-Bench 的 COAD（541 基因 Xenium panel）因 10/36 程序少于 5 个基因而未纳入主分析。
