# 脚本功能：根据 check_labels.py 生成的异常名单，批量整理这些异常样本到指定目录，方便后续分析和处理。
import os
import shutil
import math

# ================= 1. 配置区 =================
# 原数据集路径
SOURCE_IMG_DIR = r"data/processed_all/images"
SOURCE_LBL_DIR = r"data/processed_all/labels"

# 记录异常名单的文件
BAD_LIST_FILE = r"scripts/bad_labels_list.txt"

# 移动的目标根目录
TARGET_ROOT_DIR = r"data/bad_samples_grouped"

# 每组包含的图片数量
BATCH_SIZE = 20
# =============================================

def collect_data():
    # 检查名单是否存在
    if not os.path.exists(BAD_LIST_FILE):
        print(f"错误: 找不到名单文件 {BAD_LIST_FILE}")
        return

    # 读取被标记的文件名
    with open(BAD_LIST_FILE, 'r', encoding='utf-8') as f:
        bad_images = [line.strip() for line in f if line.strip()]

    if not bad_images:
        print("提示: 名单为空，没有需要移动的文件。")
        return

    total_files = len(bad_images)
    num_batches = math.ceil(total_files / BATCH_SIZE)
    print(f"检测到 {total_files} 个异常样本，将分为 {num_batches} 组进行存放。")

    for idx, img_name in enumerate(bad_images):
        # 1. 计算当前属于第几组 (从 1 开始)
        batch_id = (idx // BATCH_SIZE) + 1
        
        # 2. 构建目标子文件夹路径
        batch_dir = os.path.join(TARGET_ROOT_DIR, f"batch_{batch_id}")
        target_img_dir = os.path.join(batch_dir, "images")
        target_lbl_dir = os.path.join(batch_dir, "labels")

        # 自动创建目标目录结构
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        # 3. 处理图片移动
        src_img_path = os.path.join(SOURCE_IMG_DIR, img_name)
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, os.path.join(target_img_dir, img_name))
        else:
            print(f"[警告] 找不到原图: {src_img_path}")

        # 4. 处理标注文件移动
        # 假设标注文件名与图片名一致，仅后缀为 .txt
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl_path = os.path.join(SOURCE_LBL_DIR, lbl_name)
        if os.path.exists(src_lbl_path):
            shutil.move(src_lbl_path, os.path.join(target_lbl_dir, lbl_name))
        else:
            print(f"[警告] 找不到对应的标注文件: {src_lbl_path}")

        # 进度反馈
        if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
            print(f"进度: {idx + 1}/{total_files} 已完成移动...")

    print("-" * 30)
    print(f"数据整理完成！所有异常样本已移动至: {TARGET_ROOT_DIR}")

if __name__ == "__main__":
    collect_data()