# Track B — 空间上下文与预测头（提案 Aim 2 claim ii）方法与结果

## 0. 一句话结论
- **空间上下文（B1）**：CONCERT 式注意力偏置（距离 + H&E 相似度 + 区室）相对 SPARC-Map 仅距离基线的增益很小，并且小于训练种子噪声。ssgsea 上 raw Pearson +0.0097（39/57 张切片提升，Wilcoxon p=0.016；16 个患者组中 10 组提升，p=0.13），depth-residualised Pearson +0.0050（p=0.11；组水平 13/16，p=0.029）。mlm 上没有增益（+0.0062 / +0.0025，均 p>0.2）。增益**不集中于 READ**。k=48 也**没有帮助**。同一基线换一个随机种子，ssgsea depth-residualised Pearson 就变化 −0.0196（p=0.0013），比 CONCERT 效应还大。
- **预测头（B2）**：低秩 Gaussian 头与确定性头精度相当，CRPS 和能量分数则大幅改善（CRPS：ssgsea 15/16 组、mlm 16/16 组改善；能量分数两种目标均 16/16 组改善）。flow-matching 头精度显著下降（raw −0.038 / −0.043）。flow 的点预测过度平滑较少，但它的样本在协方差和 S 矩阵上都不如 Gaussian 样本。两种头在**所有患者组**都系统性欠覆盖：Gaussian 的 90% 区间实际覆盖在 ssgsea 上为 0.695（各组 0.36–0.90），在 mlm 上为 0.771。

## 1. 数据与评估
- 主分析：57 张 Visium 全转录组切片，16 个患者组，使用 `L.wt_folds()` 的 4 折，患者组不跨训练/测试。每折从训练折中取一个内部验证患者组（与 `run_baseline.py` 同一规则）做 early stopping，指标为验证组的平均 raw Pearson。
- 目标：预计算的 `ssgsea__*` 与 `mlm__*`，共 36 个 SPARC 程序，按训练集 median/IQR 做 robust 归一化（`L.robust_stats`）。
- 特征：H-optimus-0（1536 维），k=16 近邻（k=48 用于 e 组）。
- 指标：先对每张切片、每个程序计算 raw Pearson 与 depth-residualised Pearson（对 log1p(total counts) 回归后的残差，`L.evaluate`），再在程序间平均。有两个种子的变体先在种子间平均。READ（4 张切片，2 个患者组）单独报告。
- 配对检验：(i) 57 张切片的配对差，Wilcoxon signed-rank；(ii) 患者组水平：先在组内对切片差取均值，再对 16 组做 Wilcoxon，并报告提升的组数。只在双方都具有的 (fold, seed) 组合上配对。未做多重比较校正。

## 2. B1 空间上下文消融
### 2.1 模型（全部 30 epoch，AdamW wd=1e-2，梯度裁剪 1.0，loss = MSE + 0.5·(1−Pearson)）
| 名称 | 说明 | 参数量 |
|---|---|---|
| mlp | 仅中心 patch 的 MLP，1536→1024→512→256→36（LayerNorm+GELU+dropout 0.1） | 2.24 M |
| base_k16 | SPARC-Map 基线 `L.build_model`（相对坐标 MLP，3 层 transformer，H=256，8 头） | 2.82 M |
| concert_k16 | 相同 transformer（自写 pre-LN block，`scaled_dot_product_attention` + float 偏置），保留相对坐标编码，另加每层每头的加性注意力偏置：(1) token 间物理距离的 16 个 RBF 展开 → 线性映射；(2) token 间 H&E 特征余弦相似度（先减去训练特征均值）的 9 个 RBF 展开 → 线性映射；(3) 区室：在训练折全部非测试切片（训练 + 内部验证）上对特征 PCA-32 做 k-means（k=6），模型内按最近质心分配。用 K×K×(层×头) 的区室对偏置表，并把区室 embedding 加到 token 上。偏置的输出层零初始化，因此训练从仅距离模型出发 | 2.82 M |
| noDist / noCos / noComp | concert_k16 分别去掉 (1)、(2)、(3) | 2.82 M |
| base_k48 / concert_k48 | k=48 近邻 | 同上 |

### 2.2 结果（57 张切片均值；表 `B1_summary_by_variant.csv`、`B1_paired_tests.csv`）
| 目标 | 变体 | 种子 | raw | depth-res | READ raw | READ depth-res |
|---|---|---|---|---|---|---|
| ssgsea | base_k16 | 2 | 0.3622 | 0.1572 | 0.4353 | 0.2336 |
| ssgsea | mlp | 2 | 0.3732 | 0.1512 | 0.3867 | 0.1932 |
| ssgsea | concert_k16 | 2 | 0.3720 | 0.1622 | 0.4574 | 0.2395 |
| ssgsea | noDist_k16 | 2 | 0.3633 | 0.1569 | 0.4342 | 0.2132 |
| ssgsea | noCos_k16 | 2 | 0.3617 | 0.1496 | 0.4119 | 0.1977 |
| ssgsea | noComp_k16 | 2 | 0.3545 | 0.1474 | 0.4145 | 0.2018 |
| ssgsea | base_k48 | 1 | 0.3638 | 0.1442 | 0.4443 | 0.2036 |
| ssgsea | concert_k48 | 1 | 0.3639 | 0.1476 | 0.4045 | 0.1825 |
| mlm | base_k16 | 2 | 0.2646 | 0.1732 | 0.3654 | 0.2940 |
| mlm | mlp | 2 | 0.2527 | 0.1635 | 0.3424 | 0.2728 |
| mlm | concert_k16 | 2 | 0.2707 | 0.1758 | 0.3353 | 0.2795 |
| mlm | noDist_k16 | 2 | 0.2727 | 0.1766 | 0.3280 | 0.2685 |
| mlm | noCos_k16 | 2 | 0.2674 | 0.1759 | 0.3280 | 0.2720 |
| mlm | noComp_k16 | 2 | 0.2750 | 0.1758 | 0.3498 | 0.2777 |
| mlm | base_k48 | 1 | 0.2715 | 0.1725 | 0.3275 | 0.2802 |
| mlm | concert_k48 | 1 | 0.2689 | 0.1710 | 0.3389 | 0.2878 |

主要配对比较（参照 base_k16）：
- **concert_k16, ssgsea**：raw +0.0097（39/57，p=0.016；组 10/16，p=0.13）；depth-res +0.0050（35/57，p=0.11；组 13/16，p=0.029）。READ：raw +0.022，depth-res +0.006。
- **concert_k16, mlm**：raw +0.0062（30/57，p=0.30；组 6/16，p=0.43）；depth-res +0.0025（p=0.36；组 5/16，p=0.23）。READ：raw −0.030，depth-res −0.015。
- **mlp（非空间）**：ssgsea raw +0.0109（p=0.23），depth-res −0.0060（p=0.75；组 5/16，p=0.093），与距离基线无可分辨差异。mlm 上显著更差：raw −0.0118（19/57，p=0.0024；组 3/16，p=0.0076），depth-res −0.0098（p=0.0036；组 2/16，p=0.0027）。
- **消融（参照 concert_k16, ssgsea）**：去掉 H&E 相似度 depth-res −0.0126（p=1.3e-4；组 4/16，p=0.021）；去掉区室 −0.0147（p=6.1e-4；组 4/16，p=0.016）；去掉距离 −0.0053（p=0.27）。但 noCos 和 noComp 也低于 base_k16 本身（−0.0076，p=0.047；−0.0098，p=0.055），说明三项之间并不是简单可加的关系。在 mlm 上，各消融与 concert_k16 无差异。
- **k=48（仅种子 0，与 base_k16 种子 0 配对）**：ssgsea base_k48 raw +0.0024（p=0.97），depth-res −0.0031（p=0.66）；concert_k48 对 concert_k16 depth-res −0.0113（p=0.085）。mlm 上 concert_k48 对 concert_k16 depth-res −0.0042（p=0.040）。
- **种子噪声**（`B1_per_seed_effects.csv`）：base_k16 种子 0 对种子 1，ssgsea depth-res −0.0196（15/57，p=0.0013），切片平均绝对差：raw 0.059，depth-res 0.036；mlm depth-res −0.0099（p=2e-4）。concert 对 base 分种子看：ssgsea raw +0.0107 / +0.0088（两种子同号），depth-res +0.0115 / −0.0016（不同号）。**结论：切片级 p 值没有计入训练种子方差，B1 中 0.01 量级的差异不应解读为方法差异。**

图：`fig_B1_spatial_ablation.png`（每点 = 患者组均值，橙色 = READ，菱形 = 57 张切片均值 ± 95% CI）。

## 3. B2 预测头
- **Backbone**：concert_k16（B2 启动时 ssgsea tier-1 已完成 7/8 run，concert_k16 是唯一领先的变体。mlm 结果是之后出来的，显示 concert ≈ base）。
- (a) **det**：直接复用 B1 concert_k16 的预测（两种子），不重新训练。
- (b) **gauss**：mean + 低秩（rank 4）+ 对角协方差 Σ=FFᵀ+diag(d)，loss = 精确多元 Gaussian NLL（`LowRankMultivariateNormal`，除以 P）+ 对均值的基线 loss。
- (c) **flow**：条件 rectified flow。速度场 MLP 输入为 [x_t(36)、正弦时间编码(32)、backbone embedding 投影(512)]，结构 512→512→512→36，SiLU。源分布为 N(0,I)（在 robust 归一化空间），路径 x_t=(1−t)x0+t·y，目标 y−x0。推断用 32 步 Euler、每 spot 16 个样本，点预测为样本均值。loss = FM loss + 辅助均值头的基线 loss。early stopping 用 8 样本 × 16 步的点预测验证 Pearson。
- 训练：与 B1 相同（30 epoch，bs=1024，lr=2e-4，bf16），2 个种子 × 4 折 × 2 种目标。
- 指标（每张切片）：
  - raw / depth-res Pearson。
  - ‖C_pred − C_true‖_F（36×36 程序间相关矩阵；点预测与“单个样本”分别计算）。
  - S 矩阵：S(g1,g2) = Spearman(spot 上的 g1, 邻居均值的 g2)。邻居为 1.5×中位最近邻距离内的 spot，即 Visium 六边形的 6 邻，不是 8-邻（Visium 不是方格）。报告 ‖S_pred − S_true‖_F 与展平后的 Pearson。
  - 边界 Pearson，两种定义，均在分析阶段对每张切片只算一次标签：(i) 按任务定义，对 z-score 后的真实程序做 k-means（k=6），邻居中有不同标签即为边界，结果 ssgsea 88%、mlm 84% 的 spot 都算边界；(ii) 平滑定义，先对真实程序取“自身 + 6 邻”均值再做 k-means，边界占比为 64% / 51%，图中使用此定义。
  - 校准：central 50% / 90% 区间覆盖率（Gaussian 分别报告解析值和 16 样本经验分位数；flow 只能用 16 样本），按患者组汇总；CRPS 和能量分数在 robust 归一化单位下计算（det 的 CRPS = MAE，作为参照）；spread-skill = Spearman(样本 SD, |误差|)。

### 3.1 结果（57 张切片；`B2_head_summary.csv`、`B2_paired_tests.csv`、`B2_calibration_summary.csv`、`B2_calibration_by_patient_group.csv`）
| 目标 | 头 | raw | depth-res | 边界(平滑) | ‖ΔC‖ 点 | ‖ΔC‖ 样本 | S r 点 | S r 样本 | CRPS | 能量分数 | cov50 | cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssgsea | det | 0.372 | 0.162 | 0.328 | 11.43 | – | 0.668 | – | 0.567 | 3.95 | – | – |
| ssgsea | gauss | 0.365 | 0.166 | 0.324 | 11.41 | 4.96 | 0.674 | 0.793 | 0.439 | 3.01 | 0.355 (解析) / 0.319 (16 样本) | 0.715 / 0.628 |
| ssgsea | flow | 0.334 | 0.139 | 0.294 | 9.26 | 5.56 | 0.691 | 0.700 | 0.459 | 3.19 | 0.271 | 0.557 |
| mlm | det | 0.271 | 0.176 | 0.233 | 14.78 | – | 0.632 | – | 0.513 | 3.88 | – | – |
| mlm | gauss | 0.265 | 0.171 | 0.229 | 15.23 | 5.19 | 0.634 | 0.677 | 0.377 | 2.84 | 0.409 / 0.366 | 0.798 / 0.704 |
| mlm | flow | 0.228 | 0.139 | 0.194 | 11.43 | 5.44 | 0.642 | 0.653 | 0.390 | 2.94 | 0.330 | 0.653 |

- 真实程序间的平均非对角相关为 ssgsea 0.528、mlm 0.090。点预测把它放大到 0.738 / 0.215（det），说明点预测把程序“拉向共享成分”。单个样本更接近真实值（gauss 0.529 / 0.105，flow 0.551 / 0.101）。
- gauss vs det：精度无显著变化（ssgsea raw −0.0073，p=0.15，但组水平只有 2/16 提升，p=0.025；mlm depth-res −0.0047，p=0.0038）。CRPS −0.128 / −0.136，能量分数 −0.94 / −1.03（全部 p<1e-9；组 ≤1/16 更差）。
- flow vs det：raw −0.0376 / −0.0430，depth-res −0.0228 / −0.0368，平滑边界 −0.034 / −0.039（全部 p<1e-6；组 0–2/16 提升）。点预测 ‖ΔC‖ 较低（−2.17 / −3.35）。
- flow vs gauss（样本）：‖ΔC‖ 样本 +0.60（ssgsea，p=1.6e-4；组 p=0.051）/ +0.25（mlm，p=1.4e-3；组 p=0.21）；S r 样本 −0.093 / −0.024（组 p=0.025 / 0.011）；CRPS +0.020（p=0.078）/ +0.012（p=1.2e-3；组 p=0.19）。flow 在任何一项上都没有赢过 Gaussian 样本。
- 患者组校准：gauss 解析 90% 覆盖，ssgsea 均值 0.695（0.361–0.897，16/16 组欠覆盖），mlm 0.771（0.648–0.880）；flow 90% 覆盖，ssgsea 0.563（0.349–0.784），mlm 0.631（0.507–0.733）。spread-skill 只有 0.02–0.05，即逐 spot 的不确定度几乎不能预测误差大小。
- READ：ssgsea 上 gauss raw 0.338、flow 0.385，均低于 det 的 0.457；mlm 上 gauss 0.334 ≈ det 0.335。

图：`fig_B2_heads.png`。

## 4. 与任务规格的偏差（详见 deviations）
1. bs=1024 / lr=2e-4（√ 缩放）+ bf16 autocast，取代 bs=256 / lr=1e-4 / fp32。原因是 GPU 被另外两个作业占用：bs=256 时一次 run 需 20–40 min，并且显存溢出造成严重降速。所有变体（含基线）使用同一设置。基线 fold 0 在三种设置下分别得到 0.237（参考包）、0.343（bs256 bf16）、0.303（bs1024），说明折内方差大于设置差异。
2. k=48 两个变体只跑了种子 0（共 16 run，任务要求 2 种子）。
3. Gaussian 与 flow 头都加入了对均值的基线 loss，未运行纯 NLL / 纯 FM 对照。
4. det 头复用 B1 预测。B2 的 backbone 是在 mlm 结果出来之前，依据 ssgsea 的中间结果选定的。
5. 区室 k-means 在每折全部非测试切片（训练 + 内部验证组）上拟合，不含测试切片；在 PCA-32 空间进行。
6. S 矩阵用六边形 6 邻代替 8 连通邻；另加了平滑区室的边界定义；flow 的区间只能由 16 个样本估计，存在已知的低估偏差（同一 Gaussian 分布下，16 样本 90% 覆盖 0.628，解析值 0.715）。
7. 由于 GPU 争用，曾多次中止并重启 run，被中止的 run 全部从头重跑，不影响结果。未做多重比较校正。

## 5. 文件
代码：`spatial_models.py`、`run_b1.py`、`run_b2.py`、`run_queue.py`、`b2_det.py`、`analyze_b1.py`、`analyze_b2.py`（打包于 `trackB_code_and_runs.tar.gz`，内含每个 run 的逐切片 × 程序 CSV）。
