import os

# ================= 1. 配置区 =================
# 图片所在的子目录
IMG_DIR = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data\processed_all\unlabeled_images\images"
# 空标签存放的子目录
LBL_DIR = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data\processed_all\unlabeled_images\labels"
# =============================================

def generate_labels():
    # 1. 检查图片路径是否存在
    if not os.path.exists(IMG_DIR):
        print(f"[错误] 找不到图片目录: {IMG_DIR}")
        return

    # 2. 自动创建标签目录（如果不存在）
    if not os.path.exists(LBL_DIR):
        os.makedirs(LBL_DIR)
        print(f"[系统] 已创建标签存放目录: {LBL_DIR}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG')
    count = 0

    print(f"正在扫描图片并生成空标签...")

    # 3. 遍历图片目录
    for filename in os.listdir(IMG_DIR):
        if filename.lower().endswith(image_extensions):
            # 获取文件名（不含后缀）
            file_base = os.path.splitext(filename)[0]
            label_name = file_base + ".txt"
            label_path = os.path.join(LBL_DIR, label_name)

            # 如果该标签文件不存在，则创建一个空的 .txt
            if not os.path.exists(label_path):
                with open(label_path, 'w', encoding='utf-8') as f:
                    pass  # 创建空文件，代表背景样本
                count += 1

    print("-" * 30)
    print(f"任务完成！")
    print(f"共检测到图片: {len([f for f in os.listdir(IMG_DIR) if f.lower().endswith(image_extensions)])} 张")
    print(f"新生成的空标签: {count} 个")

if __name__ == "__main__":
    generate_labels()