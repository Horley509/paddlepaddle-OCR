import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
from paddleocr import PaddleOCR
import json  # 导入json，用于“美化”打印，让结构更清晰

# --- 1. 加载模型 ---
print("正在加载 PaddleOCR 模型...")
try:
    ocr_engine = PaddleOCR(use_textline_orientation=False, lang='ru')
    print("模型加载完毕。")
except Exception as e:
    print(f"!!! 模型加载失败: {e}")
    exit()

# --- 2. 选择一张简单的“证物”图片 ---
# 我们只处理一张图片，确保输出最干净、最聚焦
# 请确保这张图片 `test1.png` 存在于 `russian_jpg` 文件夹中
image_path = 'fr_jpg/test11.png'
print(f"正在处理单张图片: {image_path}")
image = cv2.imread(image_path)

# --- 3. 执行核心操作，并立刻“亮出证据” ---
if image is not None:
    try:
        # 调用 predict 函数，获取原始返回结果
        ocr_results = ocr_engine.predict(image)

        # --- 这就是我们点亮的那盏灯！ ---
        print("\n\n" + "=" * 50)
        print("           >>> ocr_engine.predict() 的原始返回值 <<<")
        print("=" * 50)

        # 使用 json.dumps 进行格式化打印，这样嵌套结构一目了然
        # indent=4 让它像JSON文件一样拥有漂亮的缩进，非常清晰
        # ensure_ascii=False 确保俄语字符能正常显示
        # 我们用一个try-except来处理任何可能无法被JSON序列化的意外情况
        try:
            print(json.dumps(ocr_results, indent=4, ensure_ascii=False))
        except TypeError:
            print("--- 返回结果无法被JSON序列化，正在尝试直接打印 ---")
            import pprint

            pprint.pprint(ocr_results)

        print("\n\n" + "=" * 50)
        print("           >>> 分析结束，请将以上内容发给我 <<<")
        print("=" * 50)


    except Exception as e:

        print(f"!!! 在执行 predict 时发生错误: {e}")
else:
    print(f"!!! 错误：无法加载图片 {image_path}")