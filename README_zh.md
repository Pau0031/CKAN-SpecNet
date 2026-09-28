# CKAN-SpecNet 完整中文文档

## 一、项目简介

CKAN-SpecNet 是一款具备可解释性的多任务红外光谱官能团分析模型，可同时完成**官能团有无判定**与**粗粒度官能团数量分级预测**两大任务。

模型内置 KAN（Kolmogorov-Arnold Network）模块，并配套**跃迁证据可视化**工具，用于验证模型预测是否依赖化学上具备物理意义的红外特征波段，实现模型决策逻辑可解释。

## 二、环境安装

```bash
uv sync
```

图像数字化工具额外依赖（光谱图片转数值）：

```bash
uv sync --extra digitization
```

## 三、仓库目录结构

```text
ckan_specnet/                     核心代码包
  core.py                         任务目录、模型配置、公共常量
  data.py                         parquet 读取、预处理、标签构建
  model.py                        CKAN-SpecNet 模型结构
  eval.py                         loss（Poly1 / 交叉熵）、指标、集成评估
  plot.py                         跃迁证据图与光谱绘图
  paths.py                        命令行路径解析工具
  grad_track.py                   少数类梯度记录 + 训练 loss 记录

scripts/
  evaluate.py                     五折集成模型在发布测试集上的评估
  predict.py                      单样本推理
  plot_transition_evidence.py     KAN 跃迁证据图
  compute_enrichment.py           官能团富集因子计算
  digitize_epochs.py              光谱图片数字化（单张图片或整个文件夹）
  train.py                        五折训练（loss 切换 + 梯度记录）
  check_grad_tracking.py          梯度记录公式自检
  exp_process/                    MCR-ALS 反应监测试验
  row_and_digital_comparation/    原始谱 vs 数字化谱配对分析（1000 条 SDBS 光谱）
  swgdrug_data_process/           SWGDRUG JCAMP → 光谱矩阵 + SMILES 处理流程

examples/                         数字化演示用光谱样例图片
assets/                           本文档引用的示例图片
pyproject.toml
README.md
README_zh.md
```

`data/`、`models/`、`results/` 三个目录**不随仓库发布**：按下一节放置发布文件时会自动创建，本文件所有命令的输入与输出都放在这三个目录里。

`zenode/` 只是数据集发布前的整理目录，**不属于仓库**。

## 四、发布数据与模型：下载后放到哪里

发布的训练语料、评估数据集与五折模型权重打包在同一个发布包里：

```text
 https://doi.org/10.57760/sciencedb.0147d
```

解压后得到的目录结构即下表左列。把每个文件复制到右列路径即可：

| 下载包中的文件 | 复制到 | 被谁使用 |
|---|---|---|
| `test.parquet`（7,524 条光谱） | `data/test.parquet` | `scripts/evaluate.py --test`、`scripts/predict.py --test`、`scripts/plot_transition_evidence.py --test`、`scripts/compute_enrichment.py --test`、`scripts/train.py --test` |
| `all.parquet`（40,850 条光谱） | `data/all.parquet` | `scripts/train.py --parquet` |
| `model/manifest.json` | `models/manifest.json` | 所有 `--run-dir models` |
| `model/fold_1.pt` … `model/fold_5.pt` | `models/fold_1.pt` … `models/fold_5.pt` | 所有 `--run-dir models` |
| `raw_and_digital_comparation/raw_selected_spectra.parquet` | `scripts/row_and_digital_comparation/raw_selected_spectra.parquet` | `analyze_raw_vs_digital.py --raw` |
| `raw_and_digital_comparation/digital_selected_spectra.parquet` | `scripts/row_and_digital_comparation/digital_selected_spectra.parquet` | `analyze_raw_vs_digital.py --digital` |
| `raw_and_digital_comparation/selected_sdbs_id_smiles.csv` | `scripts/row_and_digital_comparation/selected_sdbs_id_smiles.csv` | 1000 条抽样 SDBS 记录清单 |
| `raw_and_digital_comparation/digitized_results.json` | `scripts/row_and_digital_comparation/digitized_results.json` | `plot_img and load _files_plot_histograms.ipynb` |
| `exp_data/digitized_results.json` | `scripts/exp_process/digitized_results.json` | `MCR-PLS and Predict.ipynb`（参考谱） |
| `exp_data/exp_ftir_snapshots.csv` | `scripts/exp_process/exp_ftir_snapshots.csv` | `MCR-PLS and Predict.ipynb`（28 张时序谱） |
| `swgdrug/smiles_result.txt` | `scripts/swgdrug_data_process/smiles_result.txt` | `swgdrug_data_process.ipynb` |

一句话原则：**属于哪个脚本文件夹的文件，就放进那个脚本文件夹**；只有语料、评估集和模型权重放 `data/` 与 `models/`。所有文件名保持发布时的原名，无需改名。

假设发布包解压到了 `zenode/`，且当前位于仓库根目录，一次性放置命令为：

```bash
mkdir -p data models
cp zenode/test.parquet              data/test.parquet
cp zenode/all.parquet               data/all.parquet
cp zenode/model/fold_*.pt zenode/model/manifest.json models/
cp zenode/raw_and_digital_comparation/* scripts/row_and_digital_comparation/
cp zenode/exp_data/*                    scripts/exp_process/
cp zenode/swgdrug/smiles_result.txt     scripts/swgdrug_data_process/
```

### 不在发布包里的输入

* `analyze_p_check.py` 还需要两次由你自己跑出来的评估结果（见下文《原始谱 vs 数字化谱配对分析》）。如果这两个目录不存在，脚本会把校验结果标为 `missing`，其余分析照常运行。
* `swgdrug_data_process.ipynb` 需要 SWGDRUG 官方 JCAMP 原始文件：下载 `https://www.swgdrug.org/IR/JCAMP_051524.zip`，解压到 notebook 同级目录，并把 notebook 里的路径指向解压出的文件夹（见下文《SWGDRUG 数据处理》）。

## 五、模型评估

运行五折集成模型完成发布测试集的复现评估：

```bash
uv run python scripts/evaluate.py --test data/test.parquet --run-dir models --out results/reproduce
```

评估报告全部输出至 `results/reproduce/`：

```text
summary.csv                             各评估子集整体汇总指标
<子集>_task_metrics.csv                 单任务指标
<子集>_task_metrics_with_std.csv        带五折标准差的单任务指标
<子集>_fold_summaries.csv               每一折的汇总指标
<子集>_per_class_metrics.csv            每类 precision/recall/f1/support
<子集>_per_class_metrics_with_std.csv   带五折标准差的每类指标
```

`<子集>` 是 `data/test.parquet` 中保存的三个评估子集之一：

```text
main_test        预留 NIST/SDBS 纯化合物标准红外光谱
swgdrug          独立外部 FTIR-ATR 毒品光谱
xps_digitized    仪器导出图谱数字化得到的外部光谱
```

```bash
uv run python scripts/evaluate.py --help
```

## 六、单样本预测推理

```bash
uv run python scripts/predict.py --test data/test.parquet --run-dir models --eval-name main_test --sample-index 0 --out results/sample0_prediction.csv
```

输出 CSV 对全部 33 个任务给出：真实类别、预测类别、两者标签名、预测类别概率、是否预测正确。

```bash
uv run python scripts/predict.py --help
```

## 七、模型可解释性工具：跃迁证据可视化

跃迁证据可视化用于高亮支撑**官能团类别/数量分级跃迁**的关键光谱区间。该图不代表分子确定性结构判定，仅作为诊断工具，判断模型是否聚焦化学合理的红外特征峰区。

用法一：自动筛选预测正确的样本绘图（指定类别与排序）：

```bash
uv run python scripts/plot_transition_evidence.py --test data/test.parquet --run-dir models --eval-name main_test --task alcohols_4class --class-id 3 --rank 1 --out results/transition_evidence/alcohols_4class
```

用法二：指定样本序号绘图：

```bash
uv run python scripts/plot_transition_evidence.py --test data/test.parquet --run-dir models --eval-name main_test --task ketones --sample-index 6636 --out results/transition_evidence/sample6636_ketones
```

从 `C<n-1> -> C<n>` 逐级到预测类别，每一级输出一张 PNG（同时输出 PDF/TIFF）到指定目录。

```bash
uv run python scripts/plot_transition_evidence.py --help
```

## 八、官能团富集因子计算

该脚本通过计算全部 21 个官能团分类任务的富集因子，量化 CKAN-SpecNet 的**化学可解释性**。

### 核心定义

富集因子衡量模型预测证据是否集中在官能团对应的化学特征红外波段：

$$
\text{富集因子} = \frac{\text{落在目标红外区间内的跃迁证据占比}}{\text{目标红外区间占整个光谱轴的比例}}
$$

- 富集因子 > 1：模型预测信号富集于化学有意义的波数区间（符合物理预期的理想行为）。
- 富集因子 ≤ 1：模型证据均匀分布，或偏向与结构无关的光谱噪声。

### 主要流程

1. 为每个官能团预定义特征红外波数区间（解析自参考红外峰位表）。
2. 加载五折集成模型，对预留测试数据推理。
3. 对每个官能团与每一种类别跃迁（二分类：C0→C1；三分类：C0→C1、C1→C2；四分类：C0→C1、C1→C2、C2→C3）：
   - 抽取目标类别中预测正确且置信度最高的样本。
   - 提取每个有效样本的 KAN 跃迁证据曲线。
   - 计算两个核心指标：
     1. `Evidence within regions (%)`：落在官能团特征红外区间的预测信号占比。
     2. `Spectral-axis coverage (%)`：目标红外区间占整个输入光谱轴的比例。
   - 对每种跃迁的所有有效样本求平均富集因子。
4. 导出汇总统计 CSV，供补充材料分析与论文绘图使用。

### 执行命令

```bash
uv run python scripts/compute_enrichment.py \
    --test data/test.parquet \
    --run-dir models \
    --eval-name main_test \
    --n-samples 10 \
    --smooth-window 15 \
    --out results/enrichment_result.csv
```

### 参数说明

| 参数 | 说明 |
|----------|-------------|
| `--test` | 发布评估 parquet 文件路径 |
| `--run-dir` | 存放五折集成权重与 `manifest.json` 的目录 |
| `--eval-name` | 目标评估子集（`main_test` / `swgdrug` / `xps_digitized`） |
| `--n-samples` | 每种类别跃迁最多使用的“预测正确且置信度最高”样本数 |
| `--smooth-window` | 原始跃迁证据曲线的平滑窗口宽度 |
| `--out` | 富集因子统计输出 CSV 路径 |
| `--batch-size` | 集成推理的 batch size |
| `--num-workers` | DataLoader 进程数；为兼容 Windows/Jupyter 建议保持 0 |

### 输出 CSV 列

```text
Functional group             官能团可读名称
Transition                   类别跃迁对（C0->C1 / C1->C2 / C2->C3）
Chemically relevant regions  拼接后的特征红外波数区间
Spectral-axis coverage (%)   目标红外区间占整个光谱轴的比例
Evidence within regions (%)  目标红外区间内模型预测信号的平均占比
Enrichment                   所有有效样本的平均富集因子
N samples                    参与平均的有效样本数
```

```bash
uv run python scripts/compute_enrichment.py --help
```

## 九、光谱图像数字化工具

`scripts/digitize_epochs.py` 将红外光谱图片转换为数值化光谱曲线：基于 `python-doctr` 识别坐标轴刻度，同时输出诊断可视化图。既支持单张图片，也支持整个图片文件夹。

单张图片（输出数值 CSV 与诊断图）：

```bash
uv run python scripts/digitize_epochs.py --input examples/example1.png --out results/digitize_example
```

```text
results/digitize_example/
  digitized_spectrum.csv    数值化光谱数据
  spectrum.png              提取后的光谱曲线图
  axis_debug.png            坐标轴识别调试图
```

整个文件夹（批量模式：所有图片并行数字化，汇总为一个 JSON）：

```bash
uv run python scripts/digitize_epochs.py --input /path/to/spectrum_images --out results/digitize_batch
```

```text
results/digitize_batch/
  digitized_results.json    { "图片文件名（不含扩展名）": {"x": [...], "y": [...]}, ... }
```

支持图片格式：PNG、JPG、JPEG、BMP、TIF、TIFF、WEBP、GIF。

```bash
uv run python scripts/digitize_epochs.py --help
```

## 十、模型训练

```bash
uv run python scripts/train.py \
    --parquet data/all.parquet \
    --test data/test.parquet \
    --out results/new_run \
    --epochs 300 \
    --patience 30
```

训练过程按 `_sample_id` 自动剔除 `data/test.parquet` 中的所有样本，避免数据泄露。发布模型使用的训练集是剔除后剩余的 **28,257** 条单组分 SDBS/NIST 光谱（每一折在其中 4/5 上训练，其余作为该折的验证集）。

### loss 选择

`--loss` 可在论文使用的 Poly1 loss 与普通交叉熵对照实验之间切换：

```text
poly1      原始 Poly1 loss（默认，复现发布模型）
poly1_ce   Poly1 + 附加的一份普通交叉熵
ce         普通加权交叉熵（对照实验）
```

配套参数：`--epsilon`（Poly1 的 epsilon）、`--ce-weight`（附加交叉熵项的系数）、`--ce-use-class-weight`（附加项是否也使用类权重）、`--class-weight balanced|none`。

### 少数类梯度与 loss 记录

每次 run 都会把 loss 曲线与少数类梯度指标**按行增量**写盘，训练途中也能随时查看：

```text
<out>/train_log.csv                每个 epoch 的 loss / 梯度范数 / 学习率 / 验证指标
<out>/train_loss_by_task.csv       每个 epoch 分任务的 loss 分解
<out>/minority_grad_by_epoch.csv   每个 epoch 各少数类 (task, class) 的梯度指标
<out>/minority_grad_specs.json     少数类定义、实际类别频率、指标含义
<out>/manifest.json                模型配置、任务定义、各折文件名、日志文件名
<out>/fold_<n>.pt                  五折模型权重
```

相关参数：`--grad-track off|logit|full`（`full` 额外记录分类器末层参数梯度归因；`logit` 只记录 logits 空间梯度）、`--minority-mode group|class`（少数类按官能团存在率表判定或按训练集实际类别频率判定）、`--minority-threshold`（存在率/频率阈值百分比，默认 5）、`--no-val-loss`（省去验证集 loss 的一次前向）。

```bash
uv run python scripts/train.py --help
```

## 十一、梯度记录公式自检

`scripts/check_grad_tracking.py` 校验梯度记录用到的全部公式与 autograd 严格一致：三种 loss 变体的数值、逐样本 logits 梯度、参数空间归因、回传共享表示的梯度，以及少数类挑选结果：

```bash
uv run python scripts/check_grad_tracking.py
```

## 十二、原始谱 vs 数字化谱配对分析

论文对同 1000 条 SDBS 光谱做了数字化前后的对比。复现该对比分三步，全部使用已放置在 `scripts/row_and_digital_comparation/` 的发布文件。

第一步：产出校验用的两次评估结果：

```bash
uv run python scripts/evaluate.py \
    --test scripts/row_and_digital_comparation/raw_selected_spectra.parquet \
    --run-dir models --out results/raw_1000_test_9_1

uv run python scripts/evaluate.py \
    --test scripts/row_and_digital_comparation/digital_selected_spectra.parquet \
    --run-dir models --out results/digital_1000_test_9_1
```

第二步：对两个 parquet 重新推理，按 `sample_id` 配对对齐，计算配对一致性指标（逐样本概率缓存在 `results/anylize/raw_and_digital/predictions/`）：

```bash
uv run python scripts/row_and_digital_comparation/analyze_raw_vs_digital.py \
    --raw scripts/row_and_digital_comparation/raw_selected_spectra.parquet \
    --digital scripts/row_and_digital_comparation/digital_selected_spectra.parquet \
    --run-dir models \
    --out-dir results/anylize/raw_and_digital
```

输出：`summary.json`、`tables/` 下的补充表格（`tableS5*`、`tableS6*`、`tableS9*`、`tableS10*`、`tableS11*`），以及 `figures/` 下的图（`figS1`–`figS6`）。

第三步：运行配对显著性检验（输出到 `results/anylize/p_check/`）：

```bash
uv run python scripts/row_and_digital_comparation/analyze_p_check.py
```

表格与图分别写入 `results/anylize/p_check/tables/` 与 `results/anylize/p_check/figures/`。该脚本还会把自己重算的指标与第一步产出的两次评估结果对照校验；若这两个目录不存在，会把 `raw` / `dig` 标为 `missing`。

notebook `scripts/row_and_digital_comparation/plot_img and load _files_plot_histograms.ipynb` 负责把 1000 条光谱渲染成图片（数字化的输入），并根据 `digitized_results.json` 绘制数字化质量直方图。

## 十三、SWGDRUG 数据处理

`scripts/swgdrug_data_process/swgdrug_data_process.ipynb` 把 SWGDRUG 官方 JCAMP 文件转成光谱矩阵，再合并化合物 SMILES：

1. 下载 `https://www.swgdrug.org/IR/JCAMP_051524.zip` 并解压到 notebook 同级目录（notebook 默认读取 `./JCAMP_051524.extracted`；本仓库只提供处理代码，光谱需自行从 SWGDRUG 下载）。
2. 前面的 cell 逐个读取 `.jdx` 文件，把光谱插值到统一网格 `552–3842 cm⁻¹`（步长 2 cm⁻¹），写出 `spectral_matrix_interp.csv`。
3. 后续 cell 将其与 `smiles_result.txt`（随数据集发布，831 个化合物）合并，并用 RDKit 统计每条 SMILES 的官能团数量，写出 `full_swgdrug_data.csv`。

## 十四、反应监测试验（MCR-ALS）

`scripts/exp_process/MCR-PLS and Predict.ipynb` 复现概念验证性的反应监测试验。把 `exp_ftir_snapshots.csv`（28 张时序谱）与 `digitized_results.json`（参考谱）放在同一目录后，notebook 用参考谱引导的 MCR-ALS 解析混合谱，并给出逐扫描的官能团多重度预测：

```text
contrib_mDNB/mNA/mPDA/MeOH.csv   各组分逐扫描贡献谱 C_ij * ST_j
resolved_component_spectra.csv   4 张解析纯谱（模型唯一输入）
resolved_concentrations.csv      MCR-ALS 解析出的组分浓度
resolved_solute_spectra.csv      去除溶剂后的逐扫描混合谱（不进入模型）
pipeline_meta.json               拟合质量与参考谱数据
```

## 十五、发布评估数据集说明

`data/test.parquet` 是固定发布的评估数据集，共 7,524 条光谱，包含光谱向量、标签、数据来源与评估子集标识：

```text
main_test        7,066 条预留 NIST/SDBS 纯化合物光谱
swgdrug            358 条独立外部 FTIR-ATR 光谱
xps_digitized      100 条仪器图谱数字化得到的外部光谱
```

`data/all.parquet` 是发布语料（40,850 条光谱），训练数据从中抽取：

```text
nist_gas         8,271
sdbs            32,033
swgdrug            446   （外部评估来源，不参与训练）
xps_digitized      100   （外部评估来源，不参与训练）
```

训练只使用 SDBS 与 NIST 气相两个来源、只保留单组分光谱，并剔除所有出现在 `data/test.parquet` 中的样本。

评估子集由 `_eval_name` 区分：

```text
main_test        预留 NIST/SDBS 纯化合物标准红外光谱
swgdrug          独立外部 FTIR-ATR 光谱
xps_digitized    仪器导出图谱数字化得到的外部光谱
```

每行的原始数据来源保存在 `source_name`：

```text
sdbs             SDBS 数据库光谱
nist_gas         NIST 气相红外光谱
swgdrug          SWGDRUG 光谱库
xps_digitized    仪器导出数字化光谱
```

### 核心数据字段

```text
spectrum          预处理完毕的红外光谱一维向量（1646 点，552-3842 cm^-1，步长 2 cm^-1）
source_name       原始数据源名称
source_record_id  数据源原始样本编号
source_path       本地存储路径/生成路径
compound_name     化合物名称（有数据时填充）
smiles            分子 SMILES 结构式
component_count   分子组分数量
sample_id         可读样本编号
_sample_id        全局唯一样本 ID（缺失时在加载阶段自动重算）
_eval_name        评估子集标识
label columns     官能团标签列
```

标签列即 21 个官能团标签列（`alkane`、`alkene`、`alkyne`、`aromatics`、`alkyl_halides`、`alcohols`、`esters`、`ketones`、`aldehydes`、`carbonyl_oxygen`、`ether`、`acyl_halides`、`amines`、`amides`、`nitriles`、`nitro`、`isocyanate`、`isothiocyanate`、`ortho`、`meta`、`para`）。它们构成 33 个任务：21 个二分类“有无”任务、7 个 `_3class` 数量任务（aldehydes、acyl_halides、amides、nitriles、nitro、isocyanate、isothiocyanate）与 5 个 `_4class` 数量任务（alkyl_halides、alcohols、ether、amines、carbonyl_oxygen）。

`amines` 列采用修正后的官能团计数口径：硝基上的氮原子不计入氨基。

原始谱与数字化谱对比使用两个配对的 parquet：`raw_selected_spectra.parquet`（直接取自数据源的原始谱）与 `digital_selected_spectra.parquet`（同一条光谱经 `scripts/digitize_epochs.py` 从渲染图片中还原）。两者标签与样本编号完全一致，只有光谱数值不同。

## 十六、模型结构

发布模型由 5 个 checkpoint 按预测概率软投票集成（见 `manifest.json`，`ensemble.method = soft_voting_probability_mean`，验证集得分 96.05 ± 0.09）。网络结构为四段 CNN（32/64/128/256 通道），后两段带 ECA 注意力，自适应平均-最大池化到 64 维，接 1024 维全连接层，以及每个任务各自的预测头，其贡献分支为 KAN（`grid_size=3`、`spline_order=3`、64 个基函数、32 个隐藏单元）。

## 十七、可视化示例说明

### 1. 跃迁证据可视化图

图中高亮波段代表模型判定对应官能团数量发生跃迁时依赖的红外特征区间。

<center>醇类 0→1 羟基跃迁图、醇类 1→2 羟基跃迁图、醇类 2→3 羟基跃迁图、酮类 0→1 羰基跃迁图</center>

### 2. 光谱数字化效果图

动图展示图片识别 → 坐标轴校正 → 曲线提取全过程；静态图输出最终数值光谱曲线。

## 十八、数据来源说明

本研究数据集来源：

1. NIST Chemistry WebBook：https://webbook.nist.gov/chemistry/
2. SDBS 数据库：https://sdbs.db.aist.go.jp
3. SDBS 爬虫工具改编自 spectra-scraper：https://github.com/jgmotta98/spectra-scraper
4. SWGDRUG 光谱库：https://www.swgdrug.org

发布的 `data/test.parquet` 中还包含由商用仪器导出图谱数字化得到的光谱。

> 注意：仓库未开放全部原始数据源。因数据库、网页资源、仪器导出文件存在版权与分发限制，仅提供上文发布包中的文件用于实验复现。

---

# 术语对照表（论文专用）

| 英文原文 | 标准中文译名 |
| ---- | ---- |
| multi-task model | 多任务模型 |
| interpretable | 可解释性 |
| IR spectral | 红外光谱 |
| functional-group | 官能团 |
| coarse-grained count levels | 粗粒度数量分级 |
| transition-evidence visualization | 跃迁证据可视化 |
| five-fold ensemble | 五折集成模型 |
| held-out set | 预留测试集 |
| external dataset | 外部独立数据集 |
| digitization | 图谱数字化 |
| spectral region | 光谱波段/特征区间 |
| standard deviation | 标准差 |
| preprocessed spectrum vector | 预处理光谱向量 |
| binary presence label | 二元有无标签 |
| cross-validation fold | 交叉验证折次 |
| SMILES | SMILES 分子结构式 |
| Poly1 loss | Poly1 损失 |
| plain cross-entropy | 普通交叉熵 |
| class weight | 类权重 |
| minority class | 少数类 |
| gradient attribution | 梯度归因 |
| enrichment factor | 富集因子 |
| spectral fidelity | 光谱保真度 |
| paired agreement | 配对一致性 |
| quadratic weighted kappa (QWK) | 二次加权 kappa |
| MCR-ALS | 多元曲线分辨-交替最小二乘 |
| in-situ FTIR | 原位红外 |
