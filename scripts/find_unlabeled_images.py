# 这个脚本用于检查指定文件夹中的图片是否都有对应的标签文件（.txt）。
# 如果没有标签，或者标签文件为空，则将它们移动到指定的处理文件夹中。
import os
import shutil

# ================= 1. 配置区 =================
# 填入你想要检查的文件夹路径
TARGET_DIR = r"D:/lzx/tongji/Graduation_Project/Code/LV_Pole_UAV_Det_Meas/data/processed_all"

# 结果存放文件夹（脚本会自动在此线下创建 images 和 labels 子目录）
OUTPUT_ROOT = os.path.join(TARGET_DIR, "unlabeled_images")
# =============================================

def find_and_move_unlabeled():
    img_dir = os.path.join(TARGET_DIR, "images")
    lbl_dir = os.path.join(TARGET_DIR, "labels")
    
    # 定义输出的子目录
    output_img_dir = os.path.join(OUTPUT_ROOT, "images")
    output_lbl_dir = os.path.join(OUTPUT_ROOT, "labels")

    # 1. 基础检查
    if not os.path.exists(img_dir):
        print(f"[错误] 找不到图片目录: {img_dir}")
        return
    
    if not os.path.exists(lbl_dir):
        print(f"[警告] 找不到标签目录: {lbl_dir}")
        os.makedirs(lbl_dir, exist_ok=True)

    # 2. 获取所有图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG')
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(image_extensions)]
    
    missing_count = 0  # 缺失标签的数量
    empty_count = 0    # 标签为空的数量
    
    print(f"开始检查路径: {TARGET_DIR}")
    print(f"总图片数: {len(images)}")

    # 3. 检查并移动
    for img_name in images:
        file_base = os.path.splitext(img_name)[0]
        label_name = file_base + ".txt"
        label_path = os.path.join(lbl_dir, label_name)
        
        src_img_path = os.path.join(img_dir, img_name)
        
        should_move_img = False
        should_move_lbl = False

        # 情况 A: 标签文件不存在
        if not os.path.exists(label_path):
            should_move_img = True
            missing_count += 1
        
        # 情况 B: 标签文件存在，但检查是否为空
        else:
            # 首先检查文件大小是否为0，如果不是0再看内容是否全是空格/换行
            is_empty = False
            if os.path.getsize(label_path) == 0:
                is_empty = True
            else:
                with open(label_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        is_empty = True
            
            if is_empty:
                should_move_img = True
                should_move_lbl = True
                empty_count += 1

        # 执行移动操作
        if should_move_img:
            # 确保输出目录存在
            os.makedirs(output_img_dir, exist_ok=True)
            try:
                shutil.move(src_img_path, os.path.join(output_img_dir, img_name))
            except Exception as e:
                print(f"  [错误] 无法移动图片 {img_name}: {e}")

        if should_move_lbl:
            os.makedirs(output_lbl_dir, exist_ok=True)
            try:
                shutil.move(label_path, os.path.join(output_lbl_dir, label_name))
            except Exception as e:
                print(f"  [错误] 无法移动标签文件 {label_name}: {e}")

    print("-" * 30)
    print(f"检查完毕：")
    print(f" - 缺失标签并移动的图片: {missing_count} 张")
    print(f" - 标签内容为空并移动的(图片+标签): {empty_count} 组")
    
    if (missing_count + empty_count) > 0:
        print(f"所有待处理文件已移至: {OUTPUT_ROOT}")
    else:
        print("恭喜！所有图片均有有效标注。")

if __name__ == "__main__":
    find_and_move_unlabeled()