# scripts/rotate_obb_180.py
# 对指定批次的图像和标注文件进行180度旋转

import cv2
import os

# 旋转函数，接受图像路径、标注路径和输出目录，进行180度旋转并保存
def rotate_180(image_path, label_path, save_img_dir, save_label_dir):
    # 读取并旋转图片
    img = cv2.imread(image_path)
    if img is None: return
    rotated_img = cv2.flip(img, -1) 
    
    img_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(save_img_dir, img_name), rotated_img)

    # 读取并转换标注文件
    if not os.path.exists(label_path): return
    
    new_lines = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = parts[0]
            coords = list(map(float, parts[1:]))
            
            # 核心逻辑：如果是归一化坐标，180度旋转就是用1.0减去原坐标
            new_coords = [1.0 - c if i % 2 == 0 else 1.0 - c for i, c in enumerate(coords)]
            
            # 重新拼凑成字符串
            new_line = f"{cls} " + " ".join([f"{c:.6f}" for c in new_coords])
            new_lines.append(new_line)

    label_name = os.path.basename(label_path)
    with open(os.path.join(save_label_dir, label_name), 'w') as f:
        f.write("\n".join(new_lines))

def process_list(txt_list_path, img_dir, label_dir, out_img_dir, out_label_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    with open(txt_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    for name in filenames:
        # 支持处理带后缀或不带后缀的文件名
        base_name = os.path.splitext(name)[0]
        img_path = os.path.join(img_dir, base_name + ".jpg") # 假设是jpg，可根据实际修改
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        print(f"正在处理: {base_name}")
        rotate_180(img_path, label_path, out_img_dir, out_label_dir)

process_list(
    txt_list_path="data/bad_samples_grouped/batch_55/todo_list.txt",  # 存有文件名的列表文件
    img_dir="data/bad_samples_grouped/batch_55/images",       # 原图路径
    label_dir="data/bad_samples_grouped/batch_55/labels",     # 原标注路径
    out_img_dir="data/bad_samples_grouped/batch_55/images",   # 旋转后图片存放处
    out_label_dir="data/bad_samples_grouped/batch_55/labels"  # 旋转后标注存放处
)