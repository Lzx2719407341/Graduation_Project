# scripts/benchmark_system.py
# 系统性能测试脚本，评估电线杆检测系统在推理速度

import time
import json
import cv2
import torch
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.detector import PoleDetector
from core.measurement import PoleMeasurer
from core.history_manager import HistoryManager

# 系统性能测试函数，输入测试图片路径和测试次数，输出推理时间、业务逻辑时间和总时间的统计结果
def run_benchmark(image_path, num_tests=50):
    # 初始化组件（会自动加载v4权重）
    detector = PoleDetector()
    measurer = PoleMeasurer()
    history = HistoryManager(db_path="data/test_history.db")
    
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"错误: 无法在 {image_path} 找到测试图片")
        return
    img_1024 = cv2.resize(raw_img, (1024, 1024))
    
    inference_times, business_times, total_times = [], [], []
    print(f"🚀 开始性能测试 (样本数: {num_tests})...")

    # 预热GPU
    _ = detector.predict(img_1024)

    for i in range(num_tests):
        start_total = time.perf_counter()

        # 推理耗时
        start_inf = time.perf_counter()
        results, _ = detector.predict(img_1024)
        inf_duration = (time.perf_counter() - start_inf) * 1000
        inference_times.append(inf_duration)
        
        # 业务逻辑耗时（量测+数据库写入）
        start_biz = time.perf_counter()
        current_measurements = []
        if len(results.obb) > 0:
            for obb in results.obb:
                m_res = measurer.measure(obb.xyxyxyxy[0], mode="国标法")
                if m_res: current_measurements.append(m_res)
            history.add_record("bench.jpg", len(current_measurements), json.dumps(current_measurements))
        
        biz_duration = (time.perf_counter() - start_biz) * 1000
        business_times.append(biz_duration)
        total_times.append((time.perf_counter() - start_total) * 1000)

    print("\n" + "="*40)
    print(f"⚡ 系统实时性测试报告 (1024x1024)")
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 40)
    print(f"模型推理均值: {np.mean(inference_times):.2f} ms")
    print(f"业务逻辑均值: {np.mean(business_times):.2f} ms")
    print(f"端到端总均值: {np.mean(total_times):.2f} ms")
    print(f"系统吞吐量: {1000 / np.mean(total_times):.2f} FPS")
    print("="*40)

if __name__ == "__main__":
    run_benchmark("test.jpg")