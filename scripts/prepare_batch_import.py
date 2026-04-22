# 脚本功能：将 Label Studio 导出的 JSON 文件中的 ID 与本地批次文件夹中的图像文件名进行匹配，并生成新的 JSON 文件，包含正确的 ID 和对应的标注信息，方便后续批量导入 Label Studio 进行复审和修正。
import json
import os
import math
import numpy as np

# ================= 1. 配置区 =================
# 请确保这个 JSON 文件路径是绝对正确的
LS_EXPORT_FILE = r"data/bad_samples_grouped/batch_55/project-40-at-2026-04-15-16-14-505ce747.json" 
START_BATCH = 39
END_BATCH = 55
BASE_DIR = r"data/bad_samples_grouped"
CLASS_NAMES = ["low_voltage_pole"]
# =============================================

def get_ls_rotation(pts):
    p1, p2, p4 = pts[0], pts[1], pts[3]
    width = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    height = math.sqrt((p4[0] - p1[0])**2 + (p4[1] - p1[1])**2)
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return p1[0]*100, p1[1]*100, width*100, height*100, angle

def load_id_map(export_file):
    mapping = {}
    if not os.path.exists(export_file):
        print(f"[错误] 找不到 Label Studio 导出文件: {export_file}")
        return None
    
    with open(export_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for task in data:
            full_path = task['data']['image']
            # 提取原始文件名：处理 Label Studio 自动添加的哈希前缀
            # 例如 /data/upload/1/abc-DJI_001.JPG -> DJI_001.JPG
            original_filename = full_path.split('-')[-1]
            mapping[original_filename] = {
                "id": task['id'],
                "full_path": full_path
            }
    print(f"[系统] 已成功加载 {len(mapping)} 个任务 ID 映射。")
    return mapping

def process_range():
    # 打印初始反馈
    print(f"正在启动 ID 匹配处理 (Batch {START_BATCH} - {END_BATCH})...")
    
    id_map = load_id_map(LS_EXPORT_FILE)
    if not id_map: 
        print("[中断] 映射表为空，请检查导出 JSON 路径。")
        return

    for b_id in range(START_BATCH, END_BATCH + 1):
        batch_folder = f"batch_{b_id}"
        batch_path = os.path.join(BASE_DIR, batch_folder)
        img_dir = os.path.join(batch_path, "images")
        lbl_dir = os.path.join(batch_path, "labels")
        
        if not os.path.exists(batch_path): continue

        ls_data = []
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in images:
            if img_name not in id_map:
                # 尝试不区分大小写匹配
                matched_key = next((k for k in id_map if k.lower() == img_name.lower()), None)
                if not matched_key:
                    print(f"  [跳过] {img_name} 不在项目 ID 列表中")
                    continue
                info = id_map[matched_key]
            else:
                info = id_map[img_name]
                
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
            
            task = {
                "id": info["id"],
                "data": {"image": info["full_path"]},
                "predictions": [{"result": []}]
            }

            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9: continue
                        pts = np.array([float(x) for x in parts[1:]]).reshape(4, 2)
                        x, y, w, h, rot = get_ls_rotation(pts)
                        task["predictions"][0]["result"].append({
                            "value": {
                                "x": x, "y": y, "width": w, "height": h, "rotation": rot,
                                "rectanglelabels": [CLASS_NAMES[int(parts[0])]]
                            },
                            "type": "rectanglelabels",
                            "from_name": "label", "to_name": "image"
                        })
            ls_data.append(task)

        output_path = os.path.join(batch_path, f"{batch_folder}_ID_SYNC.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ls_data, f, indent=2)
        print(f"  [完成] -> {batch_folder} (匹配到 {len(ls_data)} 个 ID)")

# === 关键：必须调用函数 ===
if __name__ == "__main__":
    process_range()