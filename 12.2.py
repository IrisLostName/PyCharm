import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import requests
from io import BytesIO

# --- 全局设置 ---
# 设置 Matplotlib 字体以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 函数定义 ---

def preprocess_image_from_url(url: str, target_size: tuple = (224, 224)) -> tuple:
    """
    从给定的URL下载图片，并进行预处理以适配MobileNetV2模型。

    Args:
        url (str): 图片的URL地址。
        target_size (tuple): 模型输入所需的目标图片尺寸。

    Returns:
        tuple: 包含原始PIL图片对象和预处理后的numpy数组。
               如果下载或处理失败，则返回 (None, None)。
    """
    try:
        # 下载图片
        print(f"正在从URL下载图片: {url}")
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败 (如 404), 则会抛出异常

        # 从二进制内容中打开图片
        original_img = Image.open(BytesIO(response.content))

        # 调整图片大小并转换为numpy数组
        resized_img = original_img.resize(target_size)
        img_array = image.img_to_array(resized_img)

        # 添加批次维度 (从(H, W, C)变为(1, H, W, C))
        img_array = np.expand_dims(img_array, axis=0)

        # 应用模型特定的预处理
        processed_img_array = preprocess_input(img_array)

        print("✅ 图片预处理完成！")
        print(f"处理后的图片形状: {processed_img_array.shape}")

        return original_img, processed_img_array

    except requests.exceptions.RequestException as e:
        print(f"❌ 图片下载失败: {e}")
        return None, None
    except Exception as e:
        print(f"❌ 图片处理时发生错误: {e}")
        return None, None


def get_predictions(model, processed_img_array: np.ndarray, top: int = 3) -> list:
    """
    使用模型对预处理后的图片进行预测。

    Args:
        model: 预训练的Keras模型。
        processed_img_array (np.ndarray): 预处理后的图片数组。
        top (int): 需要返回的最高置信度的预测数量。

    Returns:
        list: 解码后的预测结果列表。
    """
    print("\n✅ AI模型正在分析图片...")
    predictions = model.predict(processed_img_array)
    decoded_predictions = decode_predictions(predictions, top=top)[0]

    print("🎯 AI识别结果：")
    print("=" * 40)
    for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
        print(f"{i+1}. {label}: {confidence*100:.2f}% 置信度")
    print("=" * 40)

    return decoded_predictions


def display_results(original_img: Image.Image, predictions: list):
    """
    使用 Matplotlib 可视化输入图片和模型的预测结果。

    Args:
        original_img (Image.Image): 原始的PIL图片对象。
        predictions (list): 模型的预测结果列表。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1: 显示原始图片
    ax1.imshow(original_img)
    ax1.set_title('输入图片')
    ax1.axis('off')

    # 子图2: 显示预测结果的条形图
    labels = [pred[1] for pred in predictions]
    confidences = [pred[2] * 100 for pred in predictions]
    colors = ['#FF9999', '#66B2FF', '#99FF99']

    bars = ax2.barh(range(len(labels)), confidences, color=colors)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('置信度 (%)')
    ax2.set_title('AI识别结果')
    ax2.invert_yaxis()  # 反转Y轴，让最高置信度显示在最上面

    # 在条形图上添加置信度数值
    for bar, confidence in zip(bars, confidences):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{confidence:.1f}%', ha='left', va='center')

    plt.tight_layout()
    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 加载预训练模型
    print("正在加载 MobileNetV2 模型...")
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    print("✅ 模型加载完成！\n")

    # 2. 定义图片URL并进行预处理
    cat_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Shiba_inu_taiki.jpg/1018px-Shiba_inu_taiki.jpg"
    original_cat_img, processed_img = preprocess_image_from_url(cat_url)

    # 3. 如果图片处理成功，则进行预测和展示
    if original_cat_img and processed_img is not None:
        # 4. 获取预测结果
        top_predictions = get_predictions(model, processed_img, top=3)

        # 5. 可视化结果
        print("\n✅ 正在生成结果可视化图表...")
        display_results(original_cat_img, top_predictions)
        print("✅ 操作完成！")

