import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import json
from paddleocr import PaddleOCR
import numpy as np
# --- 核心配置 ---
INPUT_DIR = 'fr_jpg'
OUTPUT_VISUAL = 'result_fr'
OUTPUT_JSON = 'result_fr'
for d in [OUTPUT_VISUAL, OUTPUT_JSON]:
    if not os.path.exists(d): os.makedirs(d)



LINE_MIN_CONFIDENCE = 0.1  # 整行文本的最低置信度
LINE_MIN_TEXT_LENGTH = 1  # 最短文本行长度 (去除空格后)
LINE_MIN_AREA_RATIO = 0.0001  # 文本框面积占总图片面积的最小比例 (过滤微小噪点)

# --- 加载模型 ---
print("正在加载 PaddleOCR 模型...")
ocr_engine = PaddleOCR(use_textline_orientation=False,
    lang='ru',
    # --- 关键修正：使用正确的参数名来禁用文档预处理 ---
    # 关闭文档图像方向整体分类
    use_doc_orientation_classify=False,
    # 关闭文档图像的扭曲校正
    use_doc_unwarping=False)
print("模型加载完毕。")


def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 使用 .apply() 方法处理灰度图 ***
    enhanced_gray = clahe.apply(gray)

    # 因为PaddleOCR接受3通道，我们将增强后的灰度图（这才是真正的图片）转回去
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


def preprocess_sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)



for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        print(f"\n--- 正在处理图片: {filename} ---")

        original_image = cv2.imread(image_path)
        if original_image is None: continue

        pipelines = {"Original": original_image, "CLAHE": preprocess_clahe(original_image),
                     "Sharpen": preprocess_sharpen(original_image)}
        best_result = {"results": [], "line_count": -1, "strategy": "None"}

        image_height, image_width, _ = original_image.shape
        total_area = image_width * image_height

        for name, img in pipelines.items():
            print(f"  > 尝试策略: '{name}'...")
            ocr_results = ocr_engine.predict(img)

            high_quality_lines = []

            # 检查是否有识别结果
            if ocr_results and ocr_results[0]:
                results_dict = ocr_results[0]

                # 从字典中安全地取出三个核心列表
                boxes = results_dict.get('dt_polys', [])
                texts = results_dict.get('rec_texts', [])
                scores = results_dict.get('rec_scores', [])

                # 使用 zip 将三个列表打包，进行一次遍历
                for bbox, text, confidence in zip(boxes, texts, scores):
                    # --- 在这里进行高质量的行筛选 ---

                    # 置信度 & 文本长度 检查
                    if confidence < LINE_MIN_CONFIDENCE or len(text.strip()) < LINE_MIN_TEXT_LENGTH:
                        continue

                    # 几何面积检查
                    # bbox 本身就是 numpy array，可以直接用
                    box_np = np.array(bbox).astype(np.int32)
                    area = cv2.contourArea(box_np)
                    if (area / total_area) < LINE_MIN_AREA_RATIO:
                        continue

                    # 将通过筛选的数据，重新组合成后续代码期望的格式
                    # 格式: [bbox, (text, confidence)]
                    line_data = [bbox.tolist(), (text, confidence)]
                    high_quality_lines.append(line_data)

            # 根据高质量文本行的数量，选择最佳策略
            line_count = len(high_quality_lines)
            if line_count > best_result["line_count"]:
                best_result = {"results": high_quality_lines, "line_count": line_count, "strategy": name}
                print(f"    * 新的最佳策略! 找到 {line_count} 个高质量文本行。")

        print(f"--- 最佳预处理策略: '{best_result['strategy']}' ---")

        # --- 最终产出生成 ---
        visual_image = original_image.copy()
        output_data = []

        if best_result['line_count'] > 0:
            for line_data in best_result['results']:
                bbox, (text, confidence) = line_data

                # 绘制视觉结果
                box = np.array(bbox).astype(np.int32)
                cv2.polylines(visual_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

                # 准备JSON数据
                # 将四点坐标转换为简单的 [ [min_x, min_y], [max_x, max_y] ] 格式，方便标注
                min_x = min([p[0] for p in bbox])
                min_y = min([p[1] for p in bbox])
                max_x = max([p[0] for p in bbox])
                max_y = max([p[1] for p in bbox])
                simple_bbox = [[int(min_x), int(min_y)], [int(max_x), int(max_y)]]

                output_data.append({'bbox': simple_bbox, 'text': text})
                print(f"  > 识别结果: '{text}' (置信度: {confidence:.4f})")

        # 保存视觉结果
        cv2.imwrite(os.path.join(OUTPUT_VISUAL, f"vis_{filename}"), visual_image)

        # 保存JSON数据
        json_path = os.path.join(OUTPUT_JSON, f"{os.path.splitext(filename)[0]}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"  > 已生成 {len(output_data)} 个高质量文本行, 结果保存至 JSON 和图片。")

print("\n\n所有图片处理完毕！")