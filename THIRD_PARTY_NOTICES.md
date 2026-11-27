# 第三方代码与数据归属

## 基础代码

`vendor/AnomalyDINO/src/backbones.py`和`utils.py`来自[官方仓库](https://github.com/dammsi/AnomalyDINO)，固定提交`dd35b0f1f6ff6146d196fe4efd3457a58ebcb1e0`。实现原样保留，Apache 2.0许可见`vendor/AnomalyDINO/LICENSE`。

DINOv2源码和权重另行下载，固定来源、版本与权重SHA256见复现指南；不随本仓库附带。

## 数据、图表与示例库

- MVTec AD和MVTec LOCO AD归属MVTec及原论文作者，原数据按[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)提供。`figures/`中的相应图像示例、`deployment/cable_1shot/`的MVTec特征派生物及相关数据结果遵循对应数据许可并保留署名。
- VisA来自[Spot-the-Difference](https://github.com/amazon-science/spot-diff)，原数据按其CC BY 4.0许可及原论文要求署名；本实验未打包原始图片。
- 本实验的LOCO和VisA使用固定公开镜像；来源、哈希与实际计数在`results/data_manifest_*.json`中。LOCO镜像总数与官方页面存在差异，具体口径在复现指南说明。

## 新增代码

本仓库当前没有为新增代码单独指定开源许可证。第三方代码和数据的现有许可不因公开发布而改变。
