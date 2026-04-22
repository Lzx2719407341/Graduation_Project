import os
import cv2
import xml.etree.ElementTree as ET

# ================= 1. 配置区 =================
# 图片所在的路径 (根据你的截图推测)
IMG_DIR = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data\pole\test\images"
# XML 标注所在的路径
XML_DIR = r"D:\lzx\tongji\Graduation_Project\Code\LV_Pole_UAV_Det_Meas\data\pole\test\annotations\xmls"

# 窗口显示大小
VIEW_W, VIEW_H = 1280, 720
# =============================================

def parse_xml(xml_path):
    """解析 XML 文件，提取类别和坐标"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        # 提取标准 VOC 坐标
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        objs.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})
    return objs

def view_labels():
    # 获取所有 XML 文件
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    if not xml_files:
        print(f"[错误] 在 {XML_DIR} 中没找到 XML 文件")
        return

    idx = 0
    print("控制说明: [D]下一张, [A]上一张, [Q]退出")

    while True:
        xml_name = xml_files[idx]
        file_base = os.path.splitext(xml_name)[0]
        
        # 尝试匹配不同的图片后缀
        img_path = None
        for ext in ['.jpg', '.JPG', '.png', '.png']:
            tmp_path = os.path.join(IMG_DIR, file_base + ext)
            if os.path.exists(tmp_path):
                img_path = tmp_path
                break

        if img_path:
            img = cv2.imread(img_path)
            objs = parse_xml(os.path.join(XML_DIR, xml_name))
            
            # 绘制每一个框
            for obj in objs:
                b = obj["bbox"]
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(img, obj["name"], (b[0], b[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 缩放并显示
            show_img = cv2.resize(img, (VIEW_W, VIEW_H))
            cv2.imshow("XML Label Viewer", show_img)
            print(f"[{idx+1}/{len(xml_files)}] 查看中: {file_base}")
        else:
            print(f"[警告] 找不到对应的图片: {file_base}")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):
            idx = (idx + 1) % len(xml_files)
        elif key == ord('a'):
            idx = (idx - 1) % len(xml_files)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_labels()