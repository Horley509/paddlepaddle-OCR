import streamlit as st
import os
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR
import time

# --- Streamlit 页面基础配置 ---
st.set_page_config(page_title="多策略OCR标注工具", page_icon="🤖", layout="wide")
st.title("多策略OCR标注辅助工具 (Streamlit版)")
st.markdown("上传图片，选择不同的OCR语言模型和图像预处理策略，找到最佳的文本识别结果。")


# ==============================================================================
# --- 模型加载 ---
@st.cache_resource
def load_ocr_models():
    print("--- 正在执行模型加载并缓存... ---")

    # 【最终修复】移除了无效的 'use_gpu': True 参数。
    # PaddlePaddle-GPU 版本会自动使用 GPU，无需手动指定。
    common_params = {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}

    ocr_experts = {
        'english': PaddleOCR(lang='en', **common_params),
        'russian': PaddleOCR(lang='ru', **common_params),
        'arabic': PaddleOCR(lang='ar', **common_params),
        'korean': PaddleOCR(lang='korean', **common_params),
        'Spanish': PaddleOCR(lang='es', **common_params),
    }
    print("--- 所有模型已加载并存入缓存。 ---")
    return ocr_experts


with st.spinner("首次启动，正在加载所有语言模型 (加载一次后会缓存，请稍候)..."):
    ocr_experts = load_ocr_models()

if 'models_loaded' not in st.session_state:
    st.toast("所有模型加载成功，并已缓存！", icon="✅")
    st.session_state.models_loaded = True


# --- 预处理函数 (无改动) ---
def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


def preprocess_sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_closing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    closed_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)


# --- 核心处理逻辑 (无改动) ---
def process_image(original_image, selected_langs, selected_strategies, line_min_conf, line_min_len,
                  line_min_area_ratio):
    overall_best_result = {"results": [], "line_count": -1, "language_expert": "None", "preprocess_strategy": "None"}
    image_height, image_width, _ = original_image.shape
    total_area = image_width * image_height

    for lang_name in selected_langs:
        ocr_engine = ocr_experts[lang_name]
        pipelines = {
            "Original": original_image, "CLAHE": preprocess_clahe(original_image),
            "Sharpen": preprocess_sharpen(original_image), "Closing": preprocess_closing(original_image)
        }
        selected_pipelines = {name: func for name, func in pipelines.items() if name in selected_strategies}
        best_result_for_this_lang = {"results": [], "line_count": -1, "strategy": "None"}

        for strategy_name, img in selected_pipelines.items():
            ocr_results = ocr_engine.predict(img)
            high_quality_lines = []

            if ocr_results and ocr_results[0] is not None:
                results_dict = ocr_results[0]
                boxes = results_dict.get('dt_polys', [])
                texts = results_dict.get('rec_texts', [])
                scores = results_dict.get('rec_scores', [])

                for bbox, text, confidence in zip(boxes, texts, scores):
                    if confidence < line_min_conf or len(text.strip()) < line_min_len:
                        continue
                    if not isinstance(bbox, np.ndarray) and not isinstance(bbox, list): continue

                    box_np = np.array(bbox).astype(np.int32)
                    if (cv2.contourArea(box_np) / total_area) < line_min_area_ratio:
                        continue

                    high_quality_lines.append([bbox, (text, confidence)])

            line_count = len(high_quality_lines)
            if line_count > best_result_for_this_lang["line_count"]:
                best_result_for_this_lang = {"results": high_quality_lines, "line_count": line_count,
                                             "strategy": strategy_name}

        if best_result_for_this_lang['line_count'] > overall_best_result['line_count']:
            overall_best_result = {
                "results": best_result_for_this_lang['results'],
                "line_count": best_result_for_this_lang['line_count'],
                "language_expert": lang_name,
                "preprocess_strategy": best_result_for_this_lang['strategy']
            }

    visual_image = original_image.copy()
    output_data = []
    lang_label = "最佳语言专家" if len(selected_langs) > 1 else "选用语言专家"
    summary_text = f"**{lang_label}**: `{overall_best_result['language_expert']}`\n\n"
    summary_text += f"**最佳预处理策略**: `{overall_best_result['preprocess_strategy']}`\n\n"
    summary_text += f"**识别出高质量文本行数**: `{overall_best_result['line_count']}`"

    if overall_best_result['line_count'] > 0:
        for line_data in overall_best_result['results']:
            bbox, (text, confidence) = line_data
            box = np.array(bbox).astype(np.int32)
            cv2.polylines(visual_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            min_x, min_y = int(min(p[0] for p in bbox)), int(min(p[1] for p in bbox))
            max_x, max_y = int(max(p[0] for p in bbox)), int(max(p[1] for p in bbox))
            simple_bbox = [[min_x, min_y], [max_x, max_y]]
            output_data.append({'bbox': simple_bbox, 'text': text})

    return visual_image, output_data, summary_text


# --- 界面布局 (无改动) ---
with st.sidebar:
    st.header("⚙️ 参数配置")
    uploaded_files = st.file_uploader("上传一张或多张图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    st.subheader("“品控中心”")
    line_min_conf = st.slider("最低置信度 (Confidence)", 0.0, 1.0, 0.4, 0.05)
    line_min_len = st.number_input("最短文本长度", min_value=1, value=1)
    line_min_area_ratio = st.slider("最小文本区域占比", 0.0, 0.01, 0.0001, 0.0001, format="%.4f")

    st.subheader("“多语言专家委员会”")
    detection_mode = st.radio("语言选择模式", ("自动寻找最佳语言", "手动指定语言"), key="detection_mode")
    available_langs = list(ocr_experts.keys())
    selected_langs = []

    if detection_mode == "手动指定语言":
        manual_lang = st.selectbox("请选择一个语言模型", available_langs, index=0)
        selected_langs = [manual_lang]
    else:
        selected_langs = st.multiselect("选择要参与评测的语言模型", available_langs, default=available_langs)

    st.subheader("“预处理专家团队”")
    available_strategies = ["Original", "CLAHE", "Sharpen", "Closing"]
    selected_strategies = st.multiselect("选择要尝试的预处理策略", available_strategies, default=available_strategies)

# --- 主页面：显示结果 (无改动) ---
if not uploaded_files:
    st.info("请在左侧上传图片开始处理。")
elif not selected_langs:
    st.warning("请在左侧选择至少一种语言模型。")
else:
    for uploaded_file in uploaded_files:
        st.header(f"处理结果: `{uploaded_file.name}`")
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.frombuffer(bytes_data, np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner(f"正在分析 `{uploaded_file.name}`... 这可能需要一些时间。"):
            visual_image, output_data, summary_text = process_image(
                original_image, selected_langs, selected_strategies, line_min_conf, line_min_len, line_min_area_ratio)
        st.success("处理完成！")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("诊断总结")
            st.markdown(summary_text)
            st.subheader("识别出的文本内容")
            if output_data:
                for item in output_data:
                    st.text(f"- {item['text']}")
            else:
                st.warning("未能识别出任何符合质量要求的文本。")
        with col2:
            st.subheader("可视化结果")
            st.image(cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB), caption="标注了文本框的图片", use_column_width=True)
        st.subheader("JSON 结果下载")
        json_string = json.dumps(output_data, ensure_ascii=False, indent=4)
        st.json(json_string)
        st.download_button(label="下载 JSON 文件", file_name=f"{os.path.splitext(uploaded_file.name)[0]}.json",
                           mime="application/json", data=json_string)
        st.divider()