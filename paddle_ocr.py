import os
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR
import time
import sys
import argparse

# --- 命令行参数解析 ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='多语言OCR文本识别工具')
    parser.add_argument('input_dir', nargs='?', type=str, help='存储图片文件的根目录路径')
    parser.add_argument('--output_visual', type=str, default=None, help='可视化结果输出目录')
    parser.add_argument('--output_json', type=str, default=None, help='JSON结果输出目录')
    return parser.parse_args()

# --- 主程序 ---
def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果没有提供输入目录，使用默认值或提示用户
    INPUT_DIR = args.input_dir
    if not INPUT_DIR:
        print("\n===== 多语言OCR文本识别工具 =====")
        print("请在命令行中提供图片目录路径，例如:")
        print("python paddle_ocr.py C:\\Users\\用户名\\Pictures\\多语言图片")
        print("\n您也可以直接将文件夹拖放到此窗口中，然后按回车键。")
        
        try:
            INPUT_DIR = input("\n请输入存储图片文件的根目录路径: ").strip().strip('"\'')
        except Exception as e:
            print(f"输入错误: {e}")
            print("请使用命令行参数指定输入目录:")
            print("python paddle_ocr.py <图片目录路径>")
            return
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 '{INPUT_DIR}' 不存在!")
        return
    
    if not os.path.isdir(INPUT_DIR):
        print(f"错误: '{INPUT_DIR}' 不是一个目录!")
        return
    
    # 如果未指定输出目录，则使用输入目录名称创建对应的输出目录
    input_dir_name = os.path.basename(os.path.normpath(INPUT_DIR))
    OUTPUT_VISUAL = args.output_visual if args.output_visual else f'result_{input_dir_name}_visual'
    OUTPUT_JSON = args.output_json if args.output_json else f'result_{input_dir_name}_json'
    
    print(f"\n输入目录: {INPUT_DIR}")
    print(f"可视化输出目录: {OUTPUT_VISUAL}")
    print(f"JSON输出目录: {OUTPUT_JSON}")
    
    # 创建输出目录
    for d in [OUTPUT_VISUAL, OUTPUT_JSON]:
        if not os.path.exists(d): os.makedirs(d)
    
    # ==============================================================================
    # --- "品控中心" ---
    LINE_MIN_CONFIDENCE = 0.4
    LINE_MIN_TEXT_LENGTH = 1
    LINE_MIN_AREA_RATIO = 0.0001
    # ==============================================================================
    # --- 加载"多语言专家委员会" ---
    print("\n正在加载 PaddleOCR 模型 (精简多语言模式)...")
    common_params = {
        # 关闭文档图像方向整体分类
        'use_doc_orientation_classify':False,
        # 关闭文档图像的扭曲校正
        'use_doc_unwarping':False,
    }
    
    # PaddleOCR() 的用法本身是完全正确的，我们保持原样
    ocr_experts = {
        'russian': PaddleOCR(lang='ru', **common_params),
        'arabic': PaddleOCR(lang='ar', **common_params),
        'english': PaddleOCR(lang='en', **common_params),
        #'japan': PaddleOCR(lang='japan', **common_params),
        'korean': PaddleOCR(lang='korean', **common_params),
        'Spanish': PaddleOCR(lang='es', **common_params),
    }
    print("所有语言模型加载完毕。")
    
    # --- 预处理专家团队 ---
    def preprocess_clahe(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    
    def preprocess_sharpen(image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    
    def preprocess_closing(image):
        """
        使用形态学闭运算来连接断裂的字符笔画，并使字体更清晰。
        """
        # 首先，为了让形态学操作更有效，我们通常在灰度图上进行
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # 定义一个"结构元素"或"内核"。它定义了膨胀/腐蚀操作的邻域范围。
        # 2x2 或 3x3 的矩形内核对于常规大小的文本效果最好。
        kernel = np.ones((2, 2), np.uint8)
    
        # 执行闭运算
        # cv2.morphologyEx 是一个多功能的形态学函数
        # cv2.MORPH_CLOSE 指定我们要做的是闭运算
        closed_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
        # 将处理后的灰度图转换回BGR格式，以匹配其他管道的输出
        return cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)
    
    # --- 处理图片文件 ---
    image_files = []
    # 递归遍历目录，查找所有图片文件
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, filename))
    
    if not image_files:
        print(f"警告: 在目录 '{INPUT_DIR}' 及其子目录中未找到任何图片文件!")
        return
    
    print(f"\n找到 {len(image_files)} 个图片文件，开始处理...")
    
    for image_path in image_files:
        # 获取相对于输入目录的路径，用于创建相同的输出目录结构
        rel_path = os.path.relpath(image_path, INPUT_DIR)
        filename = os.path.basename(image_path)
        output_dir_visual = os.path.join(OUTPUT_VISUAL, os.path.dirname(rel_path))
        output_dir_json = os.path.join(OUTPUT_JSON, os.path.dirname(rel_path))
        
        # 确保输出子目录存在
        os.makedirs(output_dir_visual, exist_ok=True)
        os.makedirs(output_dir_json, exist_ok=True)
        
        print(f"\n--- 正在处理图片: {rel_path} ---")
    
        original_image = cv2.imread(image_path)
        if original_image is None: 
            print(f"  > 无法读取图片: {image_path}")
            continue
    
        overall_best_result = {"results": [], "line_count": -1, "language_expert": "None",
                               "preprocess_strategy": "None"}
        image_height, image_width, _ = original_image.shape
        total_area = image_width * image_height
    
        for lang_name, ocr_engine in ocr_experts.items():
            print(f"\n  --- 语言专家 '{lang_name}' 开始诊断 ---")
    
            pipelines = {
                "Original": original_image,
                "CLAHE": preprocess_clahe(original_image),
                "Sharpen": preprocess_sharpen(original_image),
                "Closing": preprocess_closing(original_image)
            }
            best_result_for_this_lang = {"results": [], "line_count": -1, "strategy": "None"}
    
            for strategy_name, img in pipelines.items():
                print(f"    > 尝试预处理策略: '{strategy_name}'...")
                # 使用推荐的 predict 方法
                ocr_results = ocr_engine.predict(img)
    
                lines_this_image = ocr_results[0] if ocr_results and ocr_results[0] is not None else []
    
                high_quality_lines = []
                if ocr_results and ocr_results[0]:
                    results_dict = ocr_results[0]
                    boxes = results_dict.get('dt_polys', [])
                    texts = results_dict.get('rec_texts', [])
                    scores = results_dict.get('rec_scores', [])
    
                    for bbox, text, confidence in zip(boxes, texts, scores):
                        if confidence < LINE_MIN_CONFIDENCE or len(text.strip()) < LINE_MIN_TEXT_LENGTH: continue
                        box_np = np.array(bbox).astype(np.int32)
                        if (cv2.contourArea(box_np) / total_area) < LINE_MIN_AREA_RATIO: continue
    
                        # 重新组合成 [bbox, (text, confidence)] 格式
                        high_quality_lines.append([bbox.tolist(), (text, confidence)])
    
                line_count = len(high_quality_lines)
                if line_count > best_result_for_this_lang["line_count"]:
                    best_result_for_this_lang = {"results": high_quality_lines, "line_count": line_count,
                                                 "strategy": strategy_name}
    
            print(
                f"    > '{lang_name}' 专家诊断完毕，最佳策略 '{best_result_for_this_lang['strategy']}' 找到 {best_result_for_this_lang['line_count']} 行。")
    
            if best_result_for_this_lang['line_count'] > overall_best_result['line_count']:
                print("  *** 新的全局最佳结果! ***")
                overall_best_result = {
                    "results": best_result_for_this_lang['results'],
                    "line_count": best_result_for_this_lang['line_count'],
                    "language_expert": lang_name,
                    "preprocess_strategy": best_result_for_this_lang['strategy']
                }
    
        print(f"\n--- 图片 '{filename}' 最终诊断 ---")
        print(f"  > 胜出语言专家: '{overall_best_result['language_expert']}'")
        print(f"  > 胜出预处理策略: '{overall_best_result['preprocess_strategy']}'")
    
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
                print(f"  > 识别结果: '{text}' (置信度: {confidence:.2f})")
    
        # 保存结果到对应的子目录
        output_visual_path = os.path.join(output_dir_visual, f"vis_{filename}")
        output_json_path = os.path.join(output_dir_json, f"{os.path.splitext(filename)[0]}.json")
        
        cv2.imwrite(output_visual_path, visual_image)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"  > 已生成 {len(output_data)} 个高质量文本行。")
        print(f"  > 可视化结果保存至: {output_visual_path}")
        print(f"  > JSON结果保存至: {output_json_path}")
    
    print("\n\n所有图片处理完毕！")

if __name__ == "__main__":
    main()