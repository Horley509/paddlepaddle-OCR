# ocr_core.py

import cv2
import json
import numpy as np
from paddleocr import PaddleOCR


# --- 模型加载 ---
# 这个函数可以被任何其他脚本导入和调用
def load_ocr_models():
    """
    加载所有需要的PaddleOCR模型。
    返回一个包含已加载模型的字典。
    """
    print("--- 正在加载 PaddleOCR 模型... ---")
    common_params = {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}

    ocr_experts = {
        'english': PaddleOCR(lang='en', **common_params),
        'russian': PaddleOCR(lang='ru', **common_params),
        'arabic': PaddleOCR(lang='ar', **common_params),
        'korean': PaddleOCR(lang='korean', **common_params),
        'Spanish': PaddleOCR(lang='es', **common_params),
    }
    print("--- 所有模型已加载。 ---")
    return ocr_experts


# --- 预处理函数 ---
# 这些是纯粹的图像处理函数，非常适合放在核心模块里
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


# --- 核心处理逻辑 ---
# 这是最关键的函数，它封装了所有的处理流程
def find_best_ocr_result(
        original_image,
        ocr_experts,
        selected_langs,
        selected_strategies,
        line_min_conf,
        line_min_len,
        line_min_area_ratio
):
    """
    接收一张图片和配置，返回最佳的识别结果。
    """
    overall_best_result = {"results": [], "line_count": -1, "language_expert": "None", "preprocess_strategy": "None"}
    image_height, image_width, _ = original_image.shape
    total_area = image_width * image_height

    # 预处理流水线
    pipelines = {
        "Original": original_image,
        "CLAHE": preprocess_clahe(original_image),
        "Sharpen": preprocess_sharpen(original_image),
        "Closing": preprocess_closing(original_image)
    }

    for lang_name in selected_langs:
        ocr_engine = ocr_experts[lang_name]
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
                    if confidence < line_min_conf or len(text.strip()) < line_min_len: continue
                    box_np = np.array(bbox).astype(np.int32)
                    if (cv2.contourArea(box_np) / total_area) < line_min_area_ratio: continue
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

    # --- 结果生成 ---
    # 同样返回结构化的数据，让调用者决定如何使用
    visual_image = original_image.copy()
    output_data = []

    if overall_best_result['line_count'] > 0:
        for line_data in overall_best_result['results']:
            bbox, (text, confidence) = line_data
            box = np.array(bbox).astype(np.int32)
            cv2.polylines(visual_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            min_x, min_y = int(min(p[0] for p in bbox)), int(min(p[1] for p in bbox))
            max_x, max_y = int(max(p[0] for p in bbox)), int(max(p[1] for p in bbox))
            simple_bbox = [[min_x, min_y], [max_x, max_y]]
            output_data.append({'bbox': simple_bbox, 'text': text})

    return visual_image, output_data, overall_best_result