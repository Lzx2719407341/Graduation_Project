# web/ui.py
import streamlit as st
import cv2
import numpy as np
import sys
import os
import json

# --- 路径修复：确保能找到项目根目录下的 core 模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # web/
root_dir = os.path.dirname(current_dir)                  # 项目根目录
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.detector import PoleDetector
from core.measurement import PoleMeasurer
from core.history_manager import HistoryManager

# 页面全局配置
st.set_page_config(page_title="无人机电线杆智能量测系统 v4", layout="wide")

# 初始化后端组件（使用缓存避免重复加载模型）
@st.cache_resource
def init_backend():
    # 自动加载 v4 模型、几何量测器和 SQLite 历史管理器
    return PoleDetector(), PoleMeasurer(), HistoryManager()

detector, measurer, history = init_backend()

if "current_view" not in st.session_state:
    st.session_state.current_view = "gallery"
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None
if "last_plot" not in st.session_state:
    st.session_state.last_plot = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# 界面主标题
st.title("🚁 无人机电线杆识别与量测系统")
st.markdown("---")

# 创建功能分页
tab_main, tab_history = st.tabs(["🔍 实时检测与量测", "📜 历史巡检记录"])

# --- Tab 1: 实时检测与交互量测 ---
# --- Tab 1: 实时检测 ---
with tab_main:
    # 👆 修复位置 1：在进入逻辑前初始化，防止全局 NameError
    res_obj = None 
    
    # 侧边栏配置保持不变
    st.sidebar.header("🛠️ 核心配置")
    conf_val = st.sidebar.slider("模型置信度阈值", 0.1, 1.0, 0.25)
    if detector:
        detector.conf = conf_val

    # 批量上传
    uploaded_files = st.file_uploader("批量上传巡检图片", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        # 返回按钮逻辑
        if st.session_state.current_view == "detail":
            if st.button("⬅️ 返回图片列表"):
                st.session_state.current_view = "gallery"
                st.rerun()

        # --- 视图 A：图片相册总览 ---
        if st.session_state.current_view == "gallery":
            st.subheader(f"📂 已导入图片 ({len(uploaded_files)} 张)")
            cols = st.columns(4) 
            for i, file in enumerate(uploaded_files):
                with cols[i % 4]:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    file.seek(0)
                    img = cv2.imdecode(file_bytes, 1)
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
                    if st.button(f"查看详情", key=f"btn_{i}", width='stretch'):
                        st.session_state.selected_idx = i
                        st.session_state.current_view = "detail"
                        st.rerun()

        # --- 视图 B：单张图片详细界面 ---
        elif st.session_state.current_view == "detail":
            idx = st.session_state.selected_idx
            uploaded_file = uploaded_files[idx]
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # 运行检测与缓存
            if "last_results" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
                with st.spinner("v4 模型识别中..."):
                    results, plot_img = detector.predict(image)
                    st.session_state.last_results = results
                    st.session_state.last_plot = plot_img
                    st.session_state.last_file = uploaded_file.name
            
            # 👆 修复位置 2：在详情模式下，从 session_state 提取结果给 res_obj
            res_obj = st.session_state.last_results

            # 👆 修复位置 3：【重点】以下所有 UI 展示代码必须全部缩进到这个 elif 分支内
            col_img, col_data = st.columns([1.8, 1])
            
            with col_img:
                st.subheader("🖼️ 识别结果可视化")
                if st.session_state.last_plot is not None:
                    st.image(
                        cv2.cvtColor(st.session_state.last_plot, cv2.COLOR_BGR2RGB), 
                        caption=f"当前图片: {uploaded_file.name}", 
                        width='stretch'
                    )
                else:
                    st.warning("正在准备图像数据...")

            with col_data:
                st.subheader("📊 目标明细与量测报告")
                
                # 现在 res_obj 已经在上方被定义了，不会报 NameError
                if res_obj and len(res_obj.obb) > 0:
                    current_measurements = []
                    for i, obb in enumerate(res_obj.obb):
                        conf_score = float(obb.conf[0])
                        expander_label = f"📍 电线杆 #{i+1} (置信度: {conf_score:.2f})"
                        
                        with st.expander(expander_label, expanded=True):
                            st.markdown(f"""
                                <div style='display:flex; justify-content:space-between; align-items:center;'>
                                    <span style='font-size:26px; font-weight:bold; color:#FFD700;'># {i+1}</span>
                                    <span style='background-color:#333; color:white; padding:2px 10px; border-radius:10px; font-size:14px;'>
                                        得分: {conf_score:.4f}
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.divider()
                            m_mode = st.toggle(f"启用手工标定 (目标 #{i+1})", key=f"tog_{i}")
                            
                            if not m_mode:
                                m_res = measurer.measure(obb.xyxyxyxy[0], mode="国标法")
                            else:
                                c1, c2 = st.columns(2)
                                ref_type = c1.selectbox("标定基准", ["直径 (mm)", "高度 (m)"], key=f"type_{i}")
                                if ref_type == "直径 (mm)":
                                    ref_val = c2.number_input("输入实际直径", value=190.0, step=5.0, key=f"val_{i}")
                                    m_res = measurer.measure(obb.xyxyxyxy[0], mode="手工", ref_type="直径", ref_val=ref_val)
                                else:
                                    ref_val = c2.number_input("输入实际高度", value=10.0, step=0.5, key=f"val_{i}")
                                    m_res = measurer.measure(obb.xyxyxyxy[0], mode="手工", ref_type="高度", ref_val=ref_val)

                            if m_res:
                                v1, v2 = st.columns(2)
                                v1.metric("估算高度", f"{m_res['结果_高度_m']} m")
                                v2.metric("像素占比", f"{m_res['像素高度_px']} px")
                                with st.expander("详细几何参数"):
                                    st.write(m_res)
                                current_measurements.append(m_res)

                    st.markdown("---")
                    if st.button("💾 存档本次识别结果", width='stretch'):
                        history.add_record(uploaded_file.name, len(current_measurements), json.dumps(current_measurements, ensure_ascii=False))
                        st.success("数据已存入数据库")
                else:
                    st.info("💡 未识别到电线杆目标")

# --- Tab 2: 历史记录查询 ---
with tab_history:
    st.subheader("📋 历史巡检任务列表")
    # 获取数据
    df_raw = history.get_all()
    
    if not df_raw.empty:
        # 1. 关键：提取需要的列，并强制转为字符串类型
        df_display = df_raw[["id", "time", "filename", "pole_count"]].astype(str)
        
        # 2. 渲染表格
        st.dataframe(
            df_display, 
            width='stretch', # 替代旧版的 width='stretch'
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("ID", alignment="left"),
                "time": st.column_config.TextColumn("检测时间", alignment="left"),
                "filename": st.column_config.TextColumn("文件名", alignment="left"),
                "pole_count": st.column_config.TextColumn("检测数量", alignment="left"),
            }
        )
        
        st.divider()
        
        # --- 详细查询部分也同步修改 ---
        search_id = st.number_input("输入记录 ID 查看详细测量报表", 
                                    min_value=int(df_raw['id'].min()), 
                                    max_value=int(df_raw['id'].max()))
        
        if st.button("查询详细数据"):
            target_record = df_raw[df_raw['id'] == search_id]
            if not target_record.empty:
                results_data = json.loads(target_record['results'].values[0])
                st.markdown(f"**记录文件:** `{target_record['filename'].values[0]}` | **时间:** `{target_record['time'].values[0]}`")
                
                # 将结果转为 DataFrame 并强制转为字符串，确保详细表也是左对齐
                import pandas as pd
                df_detail = pd.DataFrame(results_data).astype(str)
                st.dataframe(df_detail, width='stretch', hide_index=True)