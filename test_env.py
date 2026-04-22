# 测试代码 (test_env.py)

import torch
from models.architecture.backbone import DetectionModel

# 检查GPU是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")

# 尝试加载自定义模型配置
try:
    # yolov8s_hyperace.yaml 放到了 models 目录下
    model = DetectionModel(cfg='models/yolov8s_hyperace.yaml', ch=3, nc=1)
    print("✅ 模型加载成功！HyperACE 模块已识别。")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")