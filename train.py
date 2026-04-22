# 自定义模型训练脚本 (train.py)
# 包含 HyperACE + CA 算子注入，并在训练结束后自动导出核心性能指标总结。

import sys
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import traceback

# 修复环境变量寻址
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 动态注入自定义算子
from models.architecture.hyperace_ops import HyperACE_Module
from models.architecture.ca_ops import CoordAtt
from models.architecture.backbone import parse_model as custom_parse_model

tasks.HyperACE_Module = HyperACE_Module
tasks.CoordAtt = CoordAtt
tasks.parse_model = custom_parse_model

def main():
    data_yaml = 'data.yaml'
    # 先跑50轮，观察训练曲线和性能指标，后续根据需要调整超参数
    # model = YOLO('models/yolov8s_hyperace_ca.yaml').load('yolov8s-obb.pt')
    
    # # 开启训练
    # results = model.train(
    #     data=data_yaml,
    #     epochs=50,         # 增加轮次以消化新增的混合特征
    #     imgsz=1024,         # 必须维持1024，否则地拍图的分辨率优势会丢失
    #     batch=4,            # 1024分辨率下，建议从4或8开始，防止显存溢出
    #     workers=4,
        
    #     # 核心策略：冻结与增强
    #     freeze=9,           # 冻结前9层Backbone，只训练Head和HyperACE/CA模块
    #     mosaic=1.0,         # 开启Mosaic，强制模型在小样本中寻找目标
    #     mixup=0.1,          # 适度开启Mixup，缓解地拍图带来的视角偏置
        
    #     # 优化器设置
    #     lr0=0.01,           # 初始学习率
    #     cos_lr=True,        # 使用余弦退火平滑下降
    #     warmup_epochs=5.0,  # 增加预热轮次，让模型平稳过渡到混合数据
    #     close_mosaic=20,    # 最后20轮关闭Mosaic，稳定OBB角度回归
        
    #     device=0,
    #     project='LV_Pole_Project',
    #     name='exp_mixed_domain_v1',
    #     exist_ok=True,
    #     cache=True          # 开启缓存，加快读取速度并稳定训练
    # )

    # 观察前50轮的训练曲线和性能指标后，解除冻结，降低学习率，继续微调。
    # 建议加载你之前 35% mAP 的那个 best.pt 权重进行增量训练
    # 重点：我们跳过表现不佳的 v2，直接继承表现最好的 v1 的“遗产”
    # 这样你的 v3 就能在一个 35.8% 的高起点上开始消化那 700 张新图
    # model_path = 'LV_Pole_Project/exp_mixed_domain_v1/weights/best.pt' 
    # model = YOLO(model_path)

    # results = model.train(
    #     # --- 基础配置 ---
    #     data=data_yaml,
    #     epochs=200,          # 700张图，200轮确保收敛
    #     imgsz=1024,          # 核心：必须1024
    #     batch=4,             # 若显存允许(>16G)可设为8
    #     workers=4,
    #     device=0,
    #     project='LV_Pole_Project',
    #     name='exp_final_v3', # 你的终极实验版本
    #     exist_ok=True,

    #     # --- 核心权重优化 (专门对付 Recall 漏检) ---
    #     box=10.0,            # 增加定位惩罚
    #     cls=2.0,             # 增加分类权重，大幅提升置信度分数
    #     dfl=2.0,             # 增加边界回归精度
        
    #     # --- 训练节奏控制 ---
    #     freeze=9,            # 维持冻结以保住 HyperACE 的预训练成果
    #     optimizer='AdamW',   # 对注意力机制更友好的优化器
    #     lr0=0.002,           # 精细初始学习率
    #     cos_lr=True,         # 余弦退火
    #     warmup_epochs=5.0,   # 充分预热
    #     close_mosaic=30,     # 最后30轮关闭增强进行角度微调
    #     patience=50,         # 自动早停

    #     # --- 数据增强 ---
    #     mosaic=1.0,
    #     mixup=0.15,
    #     scale=0.6,           # 强制模型学习远小目标
    #     fliplr=0.5,
    #     flipud=0.5,          # 增加无人机视角的随机性
        
    #     # --- 性能优化 ---
    #     cache=True,
    #     amp=True
    # )

    # # 1. 配置文件指向你刚改好的 P2 版本 YAML
    # model_yaml = 'models/yolov8s_hyperace_ca_p2.yaml' 
    
    # # 2. 初始权重：加载 v3 的 best.pt
    # # 它会自动匹配能用的层（Backbone），而新增加的 P2 检测头会进行随机初始化学习
    # model = YOLO(model_yaml).load('LV_Pole_Project/exp_final_v3/weights/best.pt') 

    # results = model.train(
    #     data='data.yaml',      # 继续使用包含 700 张图的 v3 划分
    #     epochs=150,           # P2 头比较细碎，建议给 150 轮让它学透
    #     imgsz=1024,           # 必须 1024！P2 层的威力和 1024 是绝配
    #     batch=2,              # [警告] P2 层非常吃显存，如果显存报错请设为 2
        
    #     # --- 针对 P2 头的特殊策略 ---
    #     freeze=9,             # 建议冻结前 9 层，强迫模型把注意力放在新加的 P2 检测头上
    #     lr0=0.002,            # 适中的学习率
    #     box=12.0,             # [优化] 调高定位权重，P2 就是为了把框卡得更死、更准
    #     cls=2.0,              # 维持高分类权重，冲刺 90% 置信度
        
    #     project='LV_Pole_Project',
    #     name='exp_p2_surge_v4', # 我们的第四阶段：P2 冲刺版
        
    #     # --- 增强与效率 ---
    #     mosaic=1.0,
    #     close_mosaic=30,      # 最后 30 轮关闭增强，对测量任务的角度回归极度重要
    #     cache=True,
    #     amp=True,             # 必须开启，否则显存必爆
    #     optimizer='AdamW',
    #     deterministic=False,
    #     workers=8
    # )

    last_weights = 'LV_Pole_Project/exp_p2_surge_v4/weights/last.pt'
    model = YOLO(last_weights)

    # 恢复训练
    results = model.train(
        resume=True,         # 开启恢复模式，它会自动接在 120 轮往后跑
        
        # 为了防止再次 MemoryError，我们需要覆盖这两个参数
        cache=False,         # 别再往内存里塞图了，从硬盘读，保命要紧
        workers=2,           # 减少线程数，降低 spawn 进程的内存开销
        
        # 确保其他环境参数一致
        imgsz=1024,
        batch=2
    )

    # 自动导出核心指标总结
    summary_path = os.path.join(results.save_dir, "best_metrics_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== 最终性能指标 (Best Model) ===\n")
        f.write(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}\n")
        f.write(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}\n")
        f.write(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}\n")
        f.write(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}\n")
        f.write("-" * 35 + "\n")
        f.write(f"结果文件夹: {results.save_dir}\n")

    print(f"🏁 训练完成，核心指标已保存至: {summary_path}")

if __name__ == '__main__':
    main()