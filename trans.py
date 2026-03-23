import cv2
import os
import matplotlib.pyplot as plt

# 定义输入文件夹和输出文件夹
input_folder = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/validation/images'
output_folder = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/validation/heatmaps'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 检查文件是否为图像格式
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'heatmap_{filename}')

        # 读取图像
        image = cv2.imread(input_path)

        # 检查图像是否加载成功
        if image is None:
            print(f"无法加载图像: {input_path}")
            continue

        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 生成热力图
        heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

        # 保存热力图
        cv2.imwrite(output_path, heatmap)

        # 可视化热力图（可选）
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 关闭坐标轴
        plt.title(f"Heatmap: {filename}")
        plt.show()

        print(f"已生成热力图: {output_path}")

print("所有图像处理完成！")
