import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTImageProcessor

# 加载预训练的 ViT 模型
model = ViTModel.from_pretrained('google/vit-base-patch16-224', output_attentions=True)
model.eval()

# 加载图像并进行预处理
image_path = "E:\\Code\\CVIU-17\\cviu7\\super-resolved_images\\12-29030.png"  # 你的图像路径
image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 格式

# Resize the image to 224x224 to match the model input size
image = image.resize((224, 224))

image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
inputs = image_processor(images=image, return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)
    attention_weights = outputs.attentions  # 获取注意力权重

# 选择最后一层的注意力权重
if attention_weights is not None and len(attention_weights) > 0:
    last_attention = attention_weights[-1][0]  # (num_heads, num_patches, num_patches)

    # 计算每个位置的注意力权重
    attention_map = last_attention.mean(dim=0).cpu().numpy()  # (num_patches, num_patches)

    # 获取输入图像的patch数量
    num_patches = int(attention_map.shape[0]**0.5)

    # 确保注意力图形状正确
    attention_map = attention_map.reshape(num_patches, num_patches)

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(attention_map, alpha=0.5, cmap='jet')  # 将注意力图叠加到原图上
    plt.axis('off')
    plt.title('Attention Map')
    plt.show()
else:
    print("No attention weights returned from the model.")
