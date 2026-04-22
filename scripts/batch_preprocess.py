# 图像预处理脚本 (batch_preprocess.py)
# 对无人机航拍影像进行批量对比度受限自适应直方图均衡化 (CLAHE) 处理，增强暗部与细长杆体细节。

import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def batch_preprocess(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    # 创建目标文件夹
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    # 初始化 CLAHE (自适应直方图均衡化) 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 获取所有图片文件
    img_extensions = ('.jpg', '.png', '.jpeg', '.tiff')
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(img_extensions)]
    print(f"开始批处理，共计 {len(img_files)} 张影像...")
    for img_name in tqdm(img_files):
        # 图像处理
        img_path = os.path.join(src_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # 去噪处理：使用中值滤波滤除无人机影像噪点，同时保留电线杆边缘
        denoised = cv2.medianBlur(img, 3)

        # 增强处理：将图像转为LAB空间对亮度通道进行CLAHE增强
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        final_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        cv2.imwrite(os.path.join(dst_img_dir, img_name), final_img)
        
        # 寻找对应的.txt标签文件
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label_path = os.path.join(src_label_dir, label_name)
        dst_label_path = os.path.join(dst_label_dir, label_name)
        if os.path.exists(src_label_path):
            # 直接复制OBB标签
            shutil.copy(src_label_path, dst_label_path)
    print("批处理与标签同步完成！数据已存入 data/processed_all/")

if __name__ == "__main__":
    batch_preprocess(
        # 源数据路径
        src_img_dir='data/labeled_raw/images', 
        src_label_dir='data/labeled_raw/labels',
        # 处理后的输出路径
        dst_img_dir='data/processed_all/images',
        dst_label_dir='data/processed_all/labels'
    )