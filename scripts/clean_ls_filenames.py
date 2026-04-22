import os

# ================= 1. 配置区 =================
# 在这里填入你想要处理的文件夹完整路径
TARGET_DIR = r"D:/lzx/tongji/Graduation_Project/Code/LV_Pole_UAV_Det_Meas/data/bad_samples_new/39"

# 需要同步处理的子文件夹
SUB_FOLDERS = ["images", "labels"]
# =============================================

def clean_ls_prefix():
    # 检查目标路径是否存在
    if not os.path.exists(TARGET_DIR):
        print(f"[错误] 找不到路径: {TARGET_DIR}")
        return

    total_renamed = 0
    print(f"正在清理文件夹: {TARGET_DIR}")

    for sub in SUB_FOLDERS:
        sub_path = os.path.join(TARGET_DIR, sub)
        
        if not os.path.exists(sub_path):
            print(f"[跳过] 找不到子文件夹: {sub}")
            continue
            
        print(f"正在处理子目录: {sub}")
        count = 0
        
        for filename in os.listdir(sub_path):
            # 匹配 Label Studio 的前缀特征：包含横杠且后面跟着 DJI
            if "-" in filename and "DJI_" in filename:
                # 提取第一个横杠之后的内容
                # 示例: "0d25a42c-DJI_0003_V.JPG" -> "DJI_0003_V.JPG"
                new_name = filename.split('-', 1)[1]
                
                old_file = os.path.join(sub_path, filename)
                new_file = os.path.join(sub_path, new_name)
                
                # 如果目标文件名已存在，则不进行覆盖操作，确保安全
                if os.path.exists(new_file):
                    continue
                    
                try:
                    os.rename(old_file, new_file)
                    count += 1
                    total_renamed += 1
                except Exception as e:
                    print(f"  [错误] 无法重命名 {filename}: {e}")
        
        print(f"  -> {sub} 目录下已完成 {count} 个文件的还原。")

    print("-" * 30)
    print(f"任务结束！共计还原文件数: {total_renamed}")

if __name__ == "__main__":
    clean_ls_prefix()