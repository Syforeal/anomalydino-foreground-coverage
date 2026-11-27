# AnomalyDINO：前景一致双向覆盖

课程项目的AnomalyDINO改进方案。针对**缺少正常应有内容**时的漏检，在原版AnomalyDINO上增加反向覆盖检查：既判断“测试图里的内容是否正常”，也判断“正常参考里的内容在测试图中是否仍然存在”。

```text
异常分数 = 原版前向分数 F + 0.25 × 反向覆盖分数 R
```

DINOv2-S/14保持冻结；每类用1张或4张正常图建库，两个方向取差异最大的1%图块平均。新增参考前景筛选沿用原方法的前景规则，不增加分割模型。原版前向定位图保持不变。

**阅读入口：** [简要课程报告](docs/课程项目报告.md) · [复现指南](docs/复现指南.md) · [逐类结果](results/per_category.csv) · [示例与失败情况](figures/example.png)

## 实验结果

表中为**图像AUROC（%，↑）**，越高越好；不是分类准确率。1/4-shot分别指每类使用1/4张正常参考图，每项对3组参考图取平均。同一shot设置下，原版在上、本方法在下，**仅本方法优于原版的结果加粗**。MVTec AD、VisA统计全部异常；LOCO按异常类型分别统计。

| 设置 | 方法 | MVTec AD | VisA | LOCO：逻辑异常 | LOCO：表面/结构异常 |
|:---:|:---|---:|---:|---:|---:|
| 1-shot | AnomalyDINO（原版） | 96.52 | 85.65 | 54.31 | 70.77 |
| 1-shot | 本方法 | 96.42 | 85.49 | **56.75** | **70.87** |
| 4-shot | AnomalyDINO（原版） | 97.55 | 91.22 | 62.84 | 78.52 |
| 4-shot | 本方法 | 97.50 | 91.20 | **65.39** | **78.80** |

换用60张未进入原参考库的正常图后，LOCO逻辑异常仍提高 **3.19 / 2.40个百分点**。新增参考图与原参考库的文件和内容哈希均不重叠；评价图像和评分规则保持一致。

参考库仅使用正常训练图，DINOv2编码器参数保持冻结。方法配置通过上述公开基准上的对比实验确定，原版和本改进方案在相同测试划分上计算指标。补充实验仅更换正常参考图，用于考察参考样本选择对结果的影响。

本改进方案对数量、位置关系及前景筛选错误仍有局限，具体示例见课程报告。

完整数据：[总体统计](results/main_results.csv)、[逐参考组](results/per_reference.csv)、[每张图的分数](results/predictions/)、[成本](results/cost.csv)、[来源与参数](results/provenance.json)。实验结果沿用已完成的实测，整理版已核对分数和指标一致性，未重新计时。

## 快速开始

Python 3.10环境；从仓库根目录执行。原始数据和预训练权重单独准备，不包含在Git仓库中。

```bash
git clone https://github.com/Syforeal/anomalydino-foreground-coverage.git
cd anomalydino-foreground-coverage
python3 -m venv work/venv
work/venv/bin/pip install -r requirements.txt
```

先核对随仓库保存的分数与指标，不需要GPU或原始图片：

```bash
work/venv/bin/python scripts/verify_release.py
```

按[复现指南](docs/复现指南.md)准备DINOv2代码、权重和cable数据后，即可运行随仓库提供的1-shot库：

```bash
work/venv/bin/python scripts/coverage_demo.py predict   --bank deployment/cable_1shot --data-root work/data/mvtec   --dino-repo work/dinov2 --weights work/dinov2_vits14_pretrain.pth   --images work/data/mvtec/cable/test/good/000.png work/data/mvtec/cable/test/cable_swap/000.png   --device cpu --output work/demo
```

默认`--head coverage`为最终方法，`--head baseline`为原版对照。输出连续分数和前向patch map，没有标定二值判定阈值。GPU运行可用`--device cuda:0`；Slurm服务器应在分配的作业内执行。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `src/` | 最终评分、正常库构建、特征提取与公共工具 |
| `scripts/` | 预测、完整评价、数据准备、统计核对和打包 |
| `configs/` | 三个数据集及额外正常参考组的固定配置 |
| `results/` | 原版对照与最终方法的逐图分数、指标和既有成本 |
| `docs/` | 简要课程报告和复现指南 |
| `deployment/` | 可直接加载的cable 1-shot示例库 |
| `figures/` | 最终结果、解释例及失败例 |
| `vendor/` | 保留来源与许可的官方基础代码 |

## 来源与数据归属

基础方法：[AnomalyDINO](https://github.com/dammsi/AnomalyDINO)；冻结编码器：[DINOv2](https://github.com/facebookresearch/dinov2)。数据来自MVTec AD、MVTec LOCO AD及VisA，固定下载来源、哈希和实际样本数在`results/data_manifest_*.json`中。

第三方代码、图像示例和派生特征的归属见[第三方说明](THIRD_PARTY_NOTICES.md)。本仓库未为新增代码另行指定开源许可证。
