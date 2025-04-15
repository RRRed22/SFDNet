import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_depth(npy_file_path, output_image_path):
    """
    Visualize depth from a .npy file and save it as an image using magma colormap.

    Args:
        npy_file_path (str): Path to the .npy file containing depth data.
        output_image_path (str): Path to save the output image.
    """
    # 读取 .npy 文件
    depth = np.load(npy_file_path)
    
    # 归一化深度值到 [0, 1]
    depth_normalized = (depth - 0.1) / (40-0.1)
    
    # 使用 magma colormap
    cmap = plt.get_cmap('magma')
    depth_colored = cmap(depth_normalized)  # 返回 [H, W, 4]，最后一维为 RGBA
    
    # 转换为 RGB 格式
    depth_colored_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)  # 忽略 Alpha 通道
    
    # 保存为图片
    cv2.imwrite(output_image_path, cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR))
    print(f"Depth visualization saved to {output_image_path}")

# 示例使用
npy_file = "aug_depth.npy"
output_image = "depth_visualization.png"
visualize_depth(npy_file, output_image)
