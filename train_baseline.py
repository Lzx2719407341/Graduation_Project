# 基准测试代码 (train_baseline.py)
# 运行官方原生 YOLOv8s-OBB 模型，并在结束后自动导出对比用的核心性能指标。

import sys
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import traceback

sys.path.append(os.getcwd())

def main():
    data_yaml = 'data.yaml'
    model_weight = 'yolov8s-obb.pt' 
    project_name = 'LV_Pole_Project'

    try:
        model = YOLO(model_weight)
    except Exception:
        traceback.print_exc()
        return

    # 开启训练
    results = model.train(
        data=data_yaml,
        epochs=150,        
        batch=2,           
        imgsz=1024,        
        workers=2,         
        cache=False,       
        amp=True,          
        optimizer='AdamW', 
        device=0 if torch.cuda.is_available() else 'cpu',
        project=project_name,
        name='exp_baseline_official_1024', # 👈 改个名字，方便区分
        exist_ok=True
    )

    # 自动导出基准测试指标总结
    summary_path = os.path.join(results.save_dir, "best_metrics_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== 官方基准模型性能指标 ===\n")
        f.write(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}\n")
        f.write(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}\n")
        f.write(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}\n")
        f.write(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}\n")
        f.write("-" * 35 + "\n")
        f.write(f"结果文件夹: {results.save_dir}\n")

    print(f"🏁 基准测试完成，指标已保存至: {summary_path}")

if __name__ == '__main__':
    main()