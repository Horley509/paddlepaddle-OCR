# app.py

import streamlit as st
import os
import cv2
import json
import numpy as np
import shutil  # 用于打包zip文件
import time  # 用于模拟延时，让进度条更明显

# 导入我们的核心AI模块！
import ocr_core

# --- Streamlit 页面基础配置 ---
st.set_page_config(page_title="多策略OCR工具", page_icon="🤖", layout="wide")
st.title("多策略OCR标注与批量处理工具")


# ==============================================================================
# --- 模型加载 (保持不变) ---
@st.cache_resource
def cached_load_models():
    return ocr_core.load_ocr_models()


with st.spinner("首次启动，正在加载所有语言模型..."):
    ocr_experts = cached_load_models()

if 'models_loaded' not in st.session_state:
    st.toast("所有模型加载成功，并已缓存！", icon="✅")
    st.session_state.models_loaded = True

# --- 侧边栏参数配置 (保持不变，所有模式共用) ---
with st.sidebar:
    st.header("⚙️ OCR 参数配置 (通用)")
    st.info("这里的参数对“互动模式”和“批量模式”都有效。")

    st.subheader("“品控中心”")
    line_min_conf = st.slider("最低置信度 (Confidence)", 0.0, 1.0, 0.4, 0.05)
    line_min_len = st.number_input("最短文本长度", min_value=1, value=1)
    line_min_area_ratio = st.slider("最小文本区域占比", 0.0, 0.01, 0.0001, 0.0001, format="%.4f")

    st.subheader("“多语言专家委员会”")
    available_langs = list(ocr_experts.keys())
    # 这里的 multiselect 对两种模式都适用，更加灵活
    selected_langs = st.multiselect("选择要参与评测的语言模型", available_langs, default=available_langs)

    st.subheader("“预处理专家团队”")
    available_strategies = ["Original", "CLAHE", "Sharpen", "Closing"]
    selected_strategies = st.multiselect("选择要尝试的预处理策略", available_strategies, default=available_strategies)

# --- 主页面：使用标签页分离不同模式 ---
tab1, tab2 = st.tabs(["互動模式 (上传图片)", "批量模式 (处理文件夹)"])

# --- 标签页1: 互动模式 ---
with tab1:
    st.header("处理单张或多张上传的图片")
    uploaded_files = st.file_uploader("上传一张或多张图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if not uploaded_files:
        st.info("请上传图片开始处理。")
    elif not selected_langs:
        st.warning("请在左侧选择至少一种语言模型。")
    else:
        for uploaded_file in uploaded_files:
            # (这里的代码逻辑和您之前的 app.py 主页面部分完全一样，我直接复制过来)
            st.subheader(f"处理结果: `{uploaded_file.name}`")
            bytes_data = uploaded_file.getvalue()
            file_bytes = np.frombuffer(bytes_data, np.uint8)
            original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner(f"正在分析 `{uploaded_file.name}`..."):
                visual_image, output_data, result_summary = ocr_core.find_best_ocr_result(
                    original_image, ocr_experts, selected_langs, selected_strategies,
                    line_min_conf, line_min_len, line_min_area_ratio
                )
            st.success("处理完成！")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 诊断总结")
                lang_label = "最佳语言专家" if len(selected_langs) > 1 else "选用语言专家"
                summary_text = (f"**{lang_label}**: `{result_summary['language_expert']}`\n\n"
                                f"**最佳预处理策略**: `{result_summary['preprocess_strategy']}`\n\n"
                                f"**识别出高质量文本行数**: `{result_summary['line_count']}`")
                st.markdown(summary_text)

                st.markdown("##### 识别出的文本内容")
                if output_data:
                    text_to_display = "\n".join([f"- {item['text']}" for item in output_data])
                    st.text_area("文本结果", text_to_display, height=200)
                else:
                    st.warning("未能识别出任何符合质量要求的文本。")

            with col2:
                st.markdown("##### 可视化结果")
                st.image(cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB), caption="标注了文本框的图片",
                         use_column_width=True)

            json_string = json.dumps(output_data, ensure_ascii=False, indent=4)
            st.download_button(label="下载 JSON 文件", file_name=f"{os.path.splitext(uploaded_file.name)[0]}.json",
                               mime="application/json", data=json_string)
            st.divider()

# --- 标签页2: 批量模式 ---
with tab2:
    st.header("处理服务器本地文件夹中的所有图片")
    st.warning("注意：这里使用的是服务器上的文件夹路径，而不是您本地电脑的。", icon="⚠️")

    # 定义默认路径
    default_input_dir = "ar_jpg"
    default_output_dir = "results_from_app"

    input_dir = st.text_input("输入文件夹路径", value=default_input_dir,
                              help=f"包含图片的文件夹，例如：`{default_input_dir}`")
    output_dir = st.text_input("输出文件夹路径", value=default_output_dir,
                               help="用于存放结果的文件夹，如果不存在会自动创建。")

    start_batch = st.button("开始批量处理", type="primary", use_container_width=True)

    if start_batch:
        if not selected_langs:
            st.error("请在左侧的参数配置中至少选择一种语言模型！")
        elif not os.path.isdir(input_dir):
            st.error(f"输入文件夹 '{input_dir}' 不存在！请检查路径是否正确。")
        else:
            # 准备输出文件夹
            output_visual_dir = os.path.join(output_dir, "visual")
            output_json_dir = os.path.join(output_dir, "json")
            os.makedirs(output_visual_dir, exist_ok=True)
            os.makedirs(output_json_dir, exist_ok=True)

            # 查找所有符合条件的图片文件
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(image_files)

            if total_files == 0:
                st.warning("输入文件夹中没有找到任何图片文件。")
            else:
                progress_bar = st.progress(0, text=f"准备处理 {total_files} 张图片...")

                # 使用 st.status 提供更详细的日志
                with st.status("正在处理中...", expanded=True) as status:
                    for i, filename in enumerate(image_files):
                        progress_text = f"正在处理第 {i + 1}/{total_files} 张图片: {filename}"
                        st.write(progress_text)  # 在status中打印日志
                        progress_bar.progress((i + 1) / total_files, text=progress_text)

                        image_path = os.path.join(input_dir, filename)
                        original_image = cv2.imread(image_path)

                        if original_image is None:
                            st.write(f"> 跳过无法读取的文件: {filename}")
                            continue

                        # 调用核心AI逻辑
                        visual_image, output_data, result_summary = ocr_core.find_best_ocr_result(
                            original_image, ocr_experts, selected_langs, selected_strategies,
                            line_min_conf, line_min_len, line_min_area_ratio
                        )

                        # 保存结果
                        cv2.imwrite(os.path.join(output_visual_dir, f"vis_{filename}"), visual_image)
                        json_path = os.path.join(output_json_dir, f"{os.path.splitext(filename)[0]}.json")
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=4)

                    status.update(label="所有图片处理完成！", state="complete")

                st.success(f"批量处理成功！结果已保存到 `{output_dir}` 文件夹中。")

                # --- 提供打包下载功能 ---
                st.subheader("下载打包结果")
                col_zip1, col_zip2 = st.columns(2)

                with col_zip1:
                    try:
                        shutil.make_archive(f"{output_dir}_json", 'zip', output_json_dir)
                        with open(f"{output_dir}_json.zip", "rb") as f:
                            st.download_button(
                                "下载所有JSON结果 (.zip)", f, file_name=f"{output_dir}_json.zip",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"打包JSON文件失败: {e}")

                with col_zip2:
                    try:
                        shutil.make_archive(f"{output_dir}_visual", 'zip', output_visual_dir)
                        with open(f"{output_dir}_visual.zip", "rb") as f:
                            st.download_button(
                                "下载所有可视化图片 (.zip)", f, file_name=f"{output_dir}_visual.zip",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"打包图片文件失败: {e}")