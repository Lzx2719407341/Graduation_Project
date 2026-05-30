# 基于无人机图像的低压电线杆自动识别与二维几何量测系统

本毕业设计针对无人机巡检视角下的低压电线杆多视角偏置、长宽比悬殊及远景小目标漏检等难题，提出了一种基于**YOLOv8s-HyperACE-CA-P2**改进网络的旋转目标检测（OBB）方案。

## 核心改进亮点

1. **HyperACE 模块定制** ：深度融入自定义的 `HyperACE_Module`，自适应增强复杂背景下的特征鲁棒性，抑制噪声干扰。
2. **CoordAtt 坐标注意力机制** ：集成坐标注意力机制（`CoordAtt`），精准捕获横向与纵向的杆塔边缘空间位置信息，大幅提升狭长型电线杆的定位精度。
3. **P2 微小目标检测头** ：重构Head层引入P2浅层高分辨率检测头，强迫模型聚焦远距离、高空俯视下的微小杆塔目标。
4. **混合领域渐进式训练策略** ：通过冻结骨干网络、余弦退火学习率以及动态调控数据增强（Mosaic/Mixup）的生命周期，使模型成功克服地拍图与空拍图之间的视角偏置。

## 核心目录结构

**Plaintext**

```
├── core/  
│   ├── detector.py              # 模型推理与目标检测
│   ├── measurement.py           # 基于检测框的电线杆物理参数测量算法
│   └── history_manager.py       # 历史检测与数据测量
├── models/
│   ├── architecture/
│   │   ├── backbone.py          # 解析自定义模型结构的骨干网络脚本
│   │   ├── hyperace_ops.py      # 自定义HyperACE算子实现
│   │   └── ca_ops.py            # 自定义坐标注意力机制(CoordAtt)算子
│   └── yolov8s_hyperace_ca_p2.yaml # 融合多策略的最终模型网络配置文件
├── scripts/                         # 辅助工具
│   ├── split_dataset.py             # 自动划分训练集、验证集
│   ├── check_labels.py              # 检查标签合规性，剔除格式错误的标注
│   ├── bad_labels_list.txt          # 记录被检测出的异常/坏标签文件列表
│   ├── collect_bad_data.py          # 自动隔离或收集格式损坏、无法读取的坏数据
│   ├── find_unlabeled_images.py     # 检索无标签文件的图片
│   ├── generate_empty_labels.py     # 为背景负样本图片批量生成空白标签文件
│   ├── clean_ls_filenames.py        # 清洗Label Studio导出时带有随机前缀的文件名
│   ├── rotate_obb_180.py            # 针对旋转框（OBB）进行180度坐标修正
│   ├── view_xml_labels.py           # 可视化检查XML格式的电线杆标注数据
│   ├── batch_preprocess.py          # 图像尺寸缩放、归一化等批量预处理脚本
│   ├── prepare_batch_import.py      # 将外部数据整理为标准格式
│   ├── plot_metric_comparison.py    # 提取各版本总结文本，一键绘制消融实验对比图
│   ├── benchmark_system.py          # 针对整体检测与测量系统进行性能基准测试
│   └── last_progress.txt            # 记录数据处理的最后进度节点
├── web/
│   └── ui.py                    # 基于Web端的可视化交互推理与测量界面
├── data.yaml                    # 数据集路径与类别配置文件
├── train.py                     # 改进模型(YOLOv8s-HyperACE-CA-P2)的分阶段渐进式与恢复训练主程序
├── train_baseline.py            # 官方原生YOLOv8s-OBB基准模型训练程序
├── test_env.py                  # 运行环境与CUDA/GPU加载测试脚本
├── test_prediction.py           # 单张/批量图像的预测推理专项测试脚本
├── requirement.txt              # 项目第三方依赖包配置文件
├── yolov8s-obb.pt               # 官方YOLOv8s-OBB初始预训练权重文件
└── yolo26n.pt                   # 2026年版本的轻量化预训练权重
```

## 数据集配置与说明

本项目的数据集采用混合领域设计，专门针对电线杆检测定制，配置文件见 `data.yaml`：

* **任务类型** ：旋转目标检测（ `task: obb`）。
* **检测类别** ：`low_voltage_pole`（低压电线杆，共 1 类）。
* **数据分布** ：
* **训练集** ：包含UAV 正样本 + 地拍正样本 + 背景负样本。
* **验证集** ：包含纯 UAV 高空正样本。

## 运行指南

### 1. 环境准备

请确保硬件支持CUDA，并在Python 3.10环境下安装相关依赖：

**Bash**

```
pip install -r requirement.txt
```

### 2. 基准实验（Baseline训练）

运行官方原生的YOLOv8s-OBB模型，用于后续消融实验对照。程序会自动将最佳指标保存至 `LV_Pole_Project/exp_baseline_official_1024/best_metrics_summary.txt`。

**Bash**

```
python train_baseline.py
```

### 3. 本文改进模型训练（YOLOv8s-HyperACE-CA-P2）

`train.py` 内置了完整的增量训练链条（包括混合领域热身、定位惩罚调优以及P2头专项冲刺），默认从 `exp_p2_surge_v4` 阶段的断点自动恢复训练并输出最终的权威评审指标：

**Bash**

```
python train.py
```

### 4. 启动可视化测量UI界面

训练完成后，可以启动系统配套的Web推理交互终端，实时上传无人机图像并进行参数测量：

**Bash**

```
python web/ui.py
```

## 📈 实验结果自动化输出

本系统的训练脚本均集成了科学指标自动导出模块 。训练结束后，系统将在相应的输出目录下自动生成 `best_metrics_summary.txt`，其内容规范如下：

**Plaintext**

```
=== 最终性能指标 (Best Model) ===
Precision: 0.XXXX
Recall: 0.XXXX
mAP50: 0.XXXX
mAP50-95: 0.XXXX
-----------------------------------
结果文件夹: LV_Pole_Project/exp_p2_surge_v4
```
