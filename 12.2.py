import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import requests
from io import BytesIO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
model = mobilenet_v2.MobileNetV2(weights='imagenet')

cat_url = "https://qcloud.dpfile.com/pc/bY0KHturZh_7sd0Cle2eqGN2ajzMkSSWdzxH6AWUw2qdxBVf06W34QcwP5N5Ww6B.jpg"
# 下载图片
response = requests.get(cat_url)
cat_img = Image.open(BytesIO(response.content))
plt.figure(figsize=(8, 8))
plt.imshow(cat_img)
plt.axis('off')
plt.title('测试图片：一只可爱的猫咪')
plt.show()


# 调整图片大小
cat_img_resized = cat_img.resize((224, 224))

# 转换为numpy数组
img_array = image.img_to_array(cat_img_resized)

# 添加批次维度（从(224,224,3)变成(1,224,224,3)）
img_array = np.expand_dims(img_array, axis=0)

# 应用模型特定的预处理
img_array = preprocess_input(img_array)

print("✅ 图片预处理完成！")
print(f"处理后的图片形状: {img_array.shape}")


predictions = model.predict(img_array)

print("✅ AI模型已完成分析！")
print("模型正在思考这张图片是什么...")
decoded_predictions = decode_predictions(predictions, top=3)[0]

print("🎯 AI识别结果：")
print("=" * 40)

# 显示前3个预测结果
for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
    print(f"{i+1}. {label}: {confidence*100:.2f}% 置信度")

print("=" * 40)
print("✅ 识别完成！AI认为这最可能是一只猫！")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 显示原始图片
ax1.imshow(cat_img)
ax1.set_title('输入图片')
ax1.axis('off')

# 显示预测结果的条形图
labels = [pred[1] for pred in decoded_predictions]
confidences = [pred[2] * 100 for pred in decoded_predictions]
colors = ['#FF9999', '#66B2FF', '#99FF99']

bars = ax2.barh(range(len(labels)), confidences, color=colors)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels)
ax2.set_xlabel('置信度 (%)')
ax2.set_title('AI识别结果')
ax2.invert_yaxis()  # 让最高置信度显示在最上面

# 在条形图上添加数值标签
for i, (bar, confidence) in enumerate(zip(bars, confidences)):
    width = bar.get_width()
    ax2.text(
        width + 1, bar.get_y() + bar.get_height()/2,
            f'{confidence:.1f}%', ha='left', va='center'
            )

plt.tight_layout()
plt.show()

