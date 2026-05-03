# scripts/plot_metric_comparison.py
# 比较不同模型版本在核心指标上的表现，读取各版本的CSV结果文件，提取mAP50、mAP50-95、Precision和Recall四个指标，并生成对比折线图

import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# 模型版本与数据路径
files = {
    "Baseline (Official YOLOv8s-OBB)": "LV_Pole_Project/results-baseline.csv",
    "v1 (Mixed Domain)": "LV_Pole_Project/results-v1.csv",
    "v2 (Full Fine-tuning)": "LV_Pole_Project/results-v2.csv",
    "v3 (HyperACE + CA)": "LV_Pole_Project/results-v3.csv",
    "v4 (v3 + Enhanced P2 Head)": "LV_Pole_Project/results-v4.csv"
}

# 核心指标映射字典：{CSV列名: 显示标签}
metrics_map = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall"
}

# 预加载并清洗数据
dataframes = {}
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns] # 清除列名空格
        dataframes[name] = df
    else:
        print(f"警告: 找不到文件 {path}")

# 循环生成四张图
for col, label in metrics_map.items():
    plt.figure(figsize=(10, 6))
    
    for name, df in dataframes.items():
        if col in df.columns:
            plt.plot(df['epoch'], df[col], label=name, linewidth=2)
    
    # 图表细节配置
    plt.title(f'Comparison of {label} across Model Versions', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    
    # 保存图片
    save_path = f"{label.replace('/', '_')}_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close() 
    print(f"已生成: {save_path}")