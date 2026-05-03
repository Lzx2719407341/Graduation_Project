# train.py
# 运行自定义的YOLOv8s-HyperACE-CA-P2模型，进行分阶段训练，并在每个阶段结束后自动导出核心性能指标总结

import sys
import os
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

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
    # 先跑50轮，观察训练曲线和性能指标
    # model = YOLO('models/yolov8s_hyperace_ca.yaml').load('yolov8s-obb.pt')
    
    # # 开启训练
    # results = model.train(
    #     data=data_yaml,
    #     epochs=50,         
    #     imgsz=1024,         
    #     batch=4,            
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

    # 加载之前35% mAP的那个best.pt权重进行增量训练
    # 跳过表现不佳的v2，直接继承表现最好的v1
    # model_path = 'LV_Pole_Project/exp_mixed_domain_v1/weights/best.pt' 
    # model = YOLO(model_path)

    # results = model.train(
    #     # --- 基础配置 ---
    #     data=data_yaml,
    #     epochs=200,          
    #     imgsz=1024,          
    #     batch=4,             
    #     workers=4,
    #     device=0,
    #     project='LV_Pole_Project',
    #     name='exp_final_v3', 
    #     exist_ok=True,

    #     box=10.0,            # 增加定位惩罚
    #     cls=2.0,             # 增加分类权重，大幅提升置信度分数
    #     dfl=2.0,             # 增加边界回归精度
        
    #     freeze=9,            # 维持冻结以保住HyperACE的预训练成果
    #     optimizer='AdamW',   # 对注意力机制更友好的优化器
    #     lr0=0.002,           # 精细初始学习率
    #     cos_lr=True,         # 余弦退火
    #     warmup_epochs=5.0,   # 充分预热
    #     close_mosaic=30,     # 最后30轮关闭增强进行角度微调
    #     patience=50,         # 自动早停

    #     mosaic=1.0,
    #     mixup=0.15,
    #     scale=0.6,           # 强制模型学习远小目标
    #     fliplr=0.5,
    #     flipud=0.5,          # 增加无人机视角的随机性
        
    #     cache=True,
    #     amp=True
    # )

    # # 配置文件yolov8s_hyperace_ca_p2版本YAML
    # model_yaml = 'models/yolov8s_hyperace_ca_p2.yaml' 
    
    # # 初始权重：加载v3的best.pt
    # model = YOLO(model_yaml).load('LV_Pole_Project/exp_final_v3/weights/best.pt') 

    # results = model.train(
    #     data='data.yaml',      
    #     epochs=150,           
    #     imgsz=1024,           
    #     batch=2,              
        
    #     freeze=9,             # 冻结前9层，强迫模型把注意力放在新加的P2检测头上
    #     lr0=0.002,            
    #     box=12.0,             
    #     cls=2.0,              
        
    #     project='LV_Pole_Project',
    #     name='exp_p2_surge_v4', 
        
    #     mosaic=1.0,
    #     close_mosaic=30,      # 最后30轮关闭增强
    #     cache=True,
    #     amp=True,             
    #     optimizer='AdamW',
    #     deterministic=False,
    #     workers=8
    # )

    # 从v4的last.pt继续训练，直到性能稳定，自动导出核心指标总结
    last_weights = 'LV_Pole_Project/exp_p2_surge_v4/weights/last.pt'
    model = YOLO(last_weights)

    # 恢复训练
    results = model.train(
        resume=True,         # 开启恢复模式

        cache=False,         
        workers=2,           
    
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
