# 标签可视化检查工具 (check_labels.py) - 增强版
import cv2
import os
import numpy as np
import time

# --- 全局视图控制变量 ---
WINDOW_NAME = "Check Labels"
VIEW_W, VIEW_H = 3840, 2160  # 高清画布

g_base_img = None
g_status_text = ""
g_is_flagged = False
g_is_rotated = False

# 相机状态
g_scale = 1.0
g_x = 0.0
g_y = 0.0

# 交互状态
g_is_dragging = False
g_is_scrolling = False
g_last_mouse_x = 0
g_last_mouse_y = 0
g_last_scroll_time = 0

def draw_obb(image, label_path):
    """读取 OBB 标签并绘制"""
    if not os.path.exists(label_path):
        cv2.putText(image, "No Label", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return image
    with open(label_path, 'r') as f:
        lines = f.readlines()
    h, w = image.shape[:2]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9: continue
        coords = list(map(float, parts[1:]))
        # 转换坐标
        pts = np.array([(int(coords[i]*w), int(coords[i+1]*h)) for i in range(0, 8, 2)], np.int32).reshape((-1, 1, 2))
        # 绘制多边形
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        # 绘制起点（绿色点），方便检查方向
        cv2.circle(image, tuple(pts[0][0]), 5, (0, 255, 0), -1) 
    return image

def mouse_callback(event, x, y, flags, param):
    global g_scale, g_x, g_y, g_is_dragging, g_last_mouse_x, g_last_mouse_y, g_last_scroll_time, g_is_scrolling

    # 屏幕坐标转图像坐标
    tx = VIEW_W / 2 - g_scale * g_x
    ty = VIEW_H / 2 - g_scale * g_y
    img_x = (x - tx) / g_scale
    img_y = (y - ty) / g_scale

    if event == cv2.EVENT_MOUSEWHEEL:
        g_is_scrolling = True
        g_last_scroll_time = time.time()
        if flags > 0: g_scale *= 1.15
        else: g_scale /= 1.15
        g_scale = max(0.005, min(g_scale, 100.0))
        # 调整偏移以实现以鼠标为中心缩放
        g_x = (VIEW_W / 2 - x) / g_scale + img_x
        g_y = (VIEW_H / 2 - y) / g_scale + img_y
        redraw(interp=cv2.INTER_NEAREST)

    elif event == cv2.EVENT_LBUTTONDOWN:
        g_is_dragging = True
        g_last_mouse_x, g_last_mouse_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if g_is_dragging:
            g_x -= (x - g_last_mouse_x) / g_scale
            g_y -= (y - g_last_mouse_y) / g_scale
            g_last_mouse_x, g_last_mouse_y = x, y
            redraw(interp=cv2.INTER_NEAREST)

    elif event == cv2.EVENT_LBUTTONUP:
        g_is_dragging = False
        redraw(interp=cv2.INTER_LINEAR)

def redraw(interp=cv2.INTER_LINEAR):
    global g_scale, g_x, g_y, g_base_img, g_status_text, g_is_flagged, g_is_rotated
    if g_base_img is None: return
        
    img_to_warp = cv2.flip(g_base_img, -1) if g_is_rotated else g_base_img
    
    tx = VIEW_W / 2 - g_scale * g_x
    ty = VIEW_H / 2 - g_scale * g_y
    M = np.array([[g_scale, 0, tx], [0, g_scale, ty]], dtype=np.float32)

    # 渲染主画布
    canvas = cv2.warpAffine(img_to_warp, M, (VIEW_W, VIEW_H), flags=interp, borderValue=(40, 40, 40))
    
    # --- 1. 顶部状态栏反馈 ---
    rot_str = " | ROT: 180" if g_is_rotated else ""
    info = f"{g_status_text}{rot_str} | Scale: {g_scale:.2f}x"
    # 背景阴影提升文字可读性
    cv2.putText(canvas, info, (32, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(canvas, info, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # --- 2. 标记反馈 (Flagged) ---
    if g_is_flagged:
        cv2.rectangle(canvas, (25, 95), (280, 155), (0, 0, 255), -1)
        cv2.putText(canvas, "BAD LABEL", (45, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # --- 3. 右侧操作提示面板 ---
    controls = [
        ("CONTROLS", (0, 255, 255)),
        ("Space / N : Next", (200, 200, 200)),
        ("P         : Prev", (200, 200, 200)),
        ("F         : Flag / Unflag", (200, 200, 200)),
        ("R         : Rotate 180", (200, 200, 200)),
        ("J         : Jump (Terminal)", (200, 200, 200)),
        ("Scroll    : Zoom", (200, 200, 200)),
        ("Drag      : Pan", (200, 200, 200)),
        ("Q / ESC   : Exit", (100, 100, 255))
    ]
    for i, (text, color) in enumerate(controls):
        cv2.putText(canvas, text, (VIEW_W - 450, 100 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 3) # 阴影
        cv2.putText(canvas, text, (VIEW_W - 452, 98 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    cv2.imshow(WINDOW_NAME, canvas)

def main():
    global g_scale, g_x, g_y, g_base_img, g_status_text, g_is_flagged, g_is_rotated, g_is_scrolling, g_last_scroll_time
    
    # 配置路径（根据实际修改）
    img_dir = r"data/processed_all/images"
    label_dir = r"data/processed_all/labels"
    # img_dir = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data\p100\images"
    # label_dir = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data\p100\labels"
    # img_dir = r"data/bad_samples_new/44/images"
    # label_dir = r"data/bad_samples_new/44/labels"
    # output_bad_list = r"scripts/bad_labels_list.txt"
    output_bad_list = r"scripts/bad_labels_list_new.txt"
    progress_file = r"scripts/last_progress.txt"

    if not os.path.exists(img_dir):
        print(f"错误: 找不到目录 {img_dir}")
        return

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not images:
        print("错误: 目录下没有图片。")
        return

    # 加载已标记的黑名单
    flagged_images = set()
    if os.path.exists(output_bad_list):
        with open(output_bad_list, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): flagged_images.add(line.strip())

    # 载入上次进度
    start_idx = 0
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                last_name = f.read().strip()
                if last_name in images: start_idx = images.index(last_name)
        except: pass

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    i = start_idx
    while i < len(images):
        img_name = images[i]
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        # 记录进度
        try:
            with open(progress_file, 'w', encoding='utf-8') as f: f.write(img_name)
        except: pass

        frame = cv2.imread(img_path)
        if frame is None:
            i += 1
            continue

        # 预渲染 OBB
        frame = draw_obb(frame, label_path)
        g_base_img = frame
        h, w = frame.shape[:2]
        
        # 重置视角：自适应窗口并居中
        g_scale = min(VIEW_W/w, VIEW_H/h, 1.0)
        g_x, g_y = w/2.0, h/2.0
        g_is_rotated = False
        g_status_text = f"[{i+1}/{len(images)}] {img_name}"
        g_is_flagged = img_name in flagged_images

        redraw()

        # 交互循环
        while True:
            key = cv2.waitKey(20) & 0xFF
            
            # 缩放平滑切换：停止滚动 100ms 后切换到高质量插值
            if g_is_scrolling and (time.time() - g_last_scroll_time > 0.1):
                if not g_is_dragging: redraw(interp=cv2.INTER_LINEAR)
                g_is_scrolling = False

            if key == 255: continue

            if key in (27, ord('q')): # 退出
                i = len(images)
                break
            elif key in (32, ord('n')): # 下一张
                i += 1
                break
            elif key == ord('p'): # 上一张
                i = max(0, i - 1)
                break
            elif key == ord('r'): # 旋转
                g_is_rotated = not g_is_rotated
                redraw()
            elif key == ord('f'): # 标记
                if img_name in flagged_images:
                    flagged_images.remove(img_name)
                    print(f"[UNFLAG] {img_name}")
                else:
                    flagged_images.add(img_name)
                    print(f"[FLAG] {img_name}")
                g_is_flagged = img_name in flagged_images
                redraw()
            elif key == ord('j'): # 跳转
                print("\n" + "="*30)
                target = input("请输入跳转文件名的关键字: ").strip()
                found = False
                for idx, name in enumerate(images):
                    if target in name:
                        i = idx
                        found = True
                        print(f"成功定位到第 {idx+1} 张图片。")
                        break
                if not found: print("未找到匹配文件。")
                break

    cv2.destroyAllWindows()
    # 保存结果
    with open(output_bad_list, 'w', encoding='utf-8') as f:
        for name in sorted(list(flagged_images)): f.write(f"{name}\n")
    print("\n" + "-"*30)
    print(f"检查任务结束！")
    print(f"累计异常标签数: {len(flagged_images)}")
    print(f"清单已保存至: {output_bad_list}")

if __name__ == "__main__":
    main()