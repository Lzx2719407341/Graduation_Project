import os
import shutil
import random
from tqdm import tqdm

# 配置路径
base_path = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data"
uav_dir = os.path.join(base_path, "processed_all")
bg_dir = os.path.join(base_path, "processed_all", "unlabeled_images")
ground_dir = os.path.join(base_path, "processed_all", "地拍电线杆-100")
dst_dir = os.path.join(base_path, "final_dataset")

def prepare_data():
    # 1. 划分 UAV 数据 (9:1)
    uav_imgs = [f for f in os.listdir(os.path.join(uav_dir, 'images')) if f.endswith(('.jpg', '.JPG', '.png'))]
    random.seed(42)
    random.shuffle(uav_imgs)
    split_idx = int(len(uav_imgs) * 0.9)
    train_uav = uav_imgs[:split_idx]
    val_uav = uav_imgs[split_idx:]

    # 2. 复制函数
    def copy_group(file_list, src_root, subset):
        os.makedirs(os.path.join(dst_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, subset, 'labels'), exist_ok=True)
        for f in file_list:
            # 复制图
            shutil.copy(os.path.join(src_root, 'images', f), os.path.join(dst_dir, subset, 'images', f))
            # 复制标签 (如果不存在则创建空)
            lbl = os.path.splitext(f)[0] + '.txt'
            src_lbl = os.path.join(src_root, 'labels', lbl)
            dst_lbl = os.path.join(dst_dir, subset, 'labels', lbl)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                open(dst_lbl, 'w').close()

    # 3. 执行汇总
    print("正在移动 UAV 数据...")
    copy_group(train_uav, uav_dir, 'train')
    copy_group(val_uav, uav_dir, 'val')

    print("正在移动地拍图与背景图到训练集...")
    # 地拍图
    ground_imgs = [f for f in os.listdir(os.path.join(ground_dir, 'images'))]
    copy_group(ground_imgs, ground_dir, 'train')
    # 背景图
    bg_imgs = [f for f in os.listdir(os.path.join(bg_dir, 'images'))]
    copy_group(bg_imgs, bg_dir, 'train')

if __name__ == "__main__":
    prepare_data()