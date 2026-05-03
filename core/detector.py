# core/detector.py
# 核心检测模块，负责加载v4模型并进行电线杆检测

import sys
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)                  
if root_dir not in sys.path:
    sys.path.append(root_dir)

# v4自定义算子注入
from models.architecture.hyperace_ops import HyperACE_Module
from models.architecture.ca_ops import CoordAtt
from models.architecture.backbone import parse_model as custom_parse_model

# 注入到Ultralytics中
tasks.HyperACE_Module = HyperACE_Module
tasks.CoordAtt = CoordAtt
tasks.parse_model = custom_parse_model

# PoleDetector类，封装了v4模型的加载和预测逻辑
class PoleDetector:
    def __init__(self, model_path=None, conf_thres=0.25):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 默认指向v4的最佳权重
        if model_path is None:
            model_path = os.path.join(root_dir, 'LV_Pole_Project', 'exp_p2_surge_v4', 'weights', 'best.pt')
        
        print(f"🚀 正在加载 v4 P2-增强型模型: {model_path}")

        if not os.path.exists(model_path):
            print(f"⚠️ 错误: 找不到权重文件 {model_path}")
            self.model = None
        else:
            try:
                # 加载模型
                self.model = YOLO(model_path)
                print("✅ v4 模型加载成功 (P2 + HyperACE + CA Ready)")
            except Exception as e:
                print(f"❌ 模型实例化失败: {e}")
                self.model = None
        
        self.conf = conf_thres

    # 预测函数，输入图像，输出检测结果和可视化图像
    def predict(self, image):
        results = self.model(image, imgsz=1024, conf=self.conf, task='obb')[0]
        plot_img = results.plot(labels=False, conf=False) 
        
        if len(results.obb) > 0:
            for i, obb in enumerate(results.obb):
                poly = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
                x, y = poly[0]
                conf_score = float(obb.conf[0])
                # 增强型标签：编号+置信度
                label = f"#{i+1} [{conf_score:.2f}]"
                # 亮黄色方案
                bg_color = (0, 255, 255) # Yellow
                text_color = (0, 0, 0)   # Black
                font_scale = 1.6
                thickness = 3
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # 绘制背景块
                cv2.rectangle(plot_img, (x, y - h - 20), (x + w + 10, y), bg_color, -1)
                # 绘制文字
                cv2.putText(plot_img, label, (x + 5, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        return results, plot_img
