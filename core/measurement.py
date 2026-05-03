# core/measurement.py
# 量测模块，负责根据检测结果计算电线杆的实际高度，支持国标法和手工标定两种模式

import numpy as np

# PoleMeasurer类，负责根据检测结果计算电线杆的实际高度，支持国标法和手工标定两种模式
class PoleMeasurer:
    def __init__(self, standard_diameter_mm=190.0):
        self.STD_DIAMETER_MM = standard_diameter_mm
    
    # 计算OBB的宽高（像素），输入是4个顶点坐标
    def calculate_obb_dims(self, box_points):
        dists = []
        for i in range(4):
            p1 = box_points[i]
            p2 = box_points[(i+1)%4]
            d = np.linalg.norm(p1 - p2)
            dists.append(d)
        dists.sort()
        width_px = np.mean(dists[:2])   # 像素直径
        height_px = np.mean(dists[2:])  # 像素高度
        return height_px, width_px

    # 量测函数，输入OBB的tensor，输出实际高度和相关信息
    def measure(self, obb_tensor, mode="国标法", ref_type="直径", ref_val=190.0):
        try:
            box = obb_tensor.cpu().numpy().reshape(4, 2).astype(float)
            h_px, w_px = self.calculate_obb_dims(box)
        
            if mode == "国标法":
                # 强制使用直径基准
                k = self.STD_DIAMETER_MM / w_px
                real_h_m = (h_px * k) / 1000.0
                info = f"标准直径 {self.STD_DIAMETER_MM}mm"
            else:
                # 手工模式：根据用户选的基准计算比例因子k
                if ref_type == "直径":
                    k = ref_val / w_px  # k 是 mm/px
                    real_h_m = (h_px * k) / 1000.0
                else: # 已知高度
                    # 此时ref_val应当是米（m），转化成mm统一计算
                    k = (ref_val * 1000.0) / h_px 
                    real_h_m = ref_val
                info = f"人工{ref_type}标定 {ref_val}"

            return {
                "结果_高度_m": round(real_h_m, 2),
                "比例因子_k": round(k, 4),
                "像素高度_px": int(h_px),
                "像素直径_px": int(w_px),
                "测量基准": info
            }
        except:
            return None
