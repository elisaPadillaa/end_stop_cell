import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageDraw
from scipy.ndimage import convolve

plt.rcParams['font.family'] = 'Meiryo'

#=================================================#
#               フィルタを表示する                 #
#=================================================#
# def print_filter(dic = None):
#     if dic is None:
#         return
#     fig, axes = plt.subplots(1, len(dic), figsize=(15, 5))

#     # 各フィルターと結果の表示
#     axes[0].imshow(dic["gabor_even"], cmap='gray')
#     axes[0].set_title('Gabor(Even)')

#     axes[1].imshow(dic["gabor_odd"], cmap='gray')
#     axes[1].set_title('Gabor(Odd)')

#     axes[2].imshow(dic["complex_cell"], cmap='gray')
#     axes[2].set_title('complex_cell')

#     # 軸を非表示に
#     for ax in axes:
#         ax.axis('off')
#     plt.tight_layout()
#     return

def create_grayscale_circle_array(size, diameter):
    """
    中心に白い円があるグレースケール画像をNumPy配列として生成します。

    Parameters:
    - size: 画像のサイズ (width, height)
    - diameter: 円の直径

    Returns:
    - 生成されたNumPy配列 (shape: (size, size))
    """
    # 空のグレースケール画像を作成
    img = Image.new('L', (size, size), 0)  # 0: 黒
    draw = ImageDraw.Draw(img)

    # 円の左上と右下の座標を計算
    radius = diameter // 2
    left_up_point = (size // 2 - radius, size // 2 - radius)
    right_down_point = (size // 2 + radius, size // 2 + radius)

    # 円を描画 (255: 白)
    draw.ellipse([left_up_point, right_down_point], fill=255)

    # 画像をNumPy配列に変換
    return np.array(img)

def create_grayscale_circle_array2(size, diameter,diameter2):
    """
    中心に白い円があるグレースケール画像をNumPy配列として生成します。

    Parameters:
    - size: 画像のサイズ (width, height)
    - diameter: 円の直径

    Returns:
    - 生成されたNumPy配列 (shape: (size, size))
    """
    # 空のグレースケール画像を作成
    img = Image.new('L', (size, size), 0)  # 0: 黒
    draw = ImageDraw.Draw(img)

    # 円の左上と右下の座標を計算
    radius = diameter // 2
    left_up_point = (size // 2 - radius, size // 2 - radius)
    right_down_point = (size // 2 + radius, size // 2 + radius)

    # 円を描画 (255: 白)
    draw.ellipse([left_up_point, right_down_point], fill=255)

    # 円の左上と右下の座標を計算
    radius = diameter2 // 2
    left_up_point = (size // 2 - radius, size // 2 - radius)
    right_down_point = (size // 2 + radius, size // 2 + radius)

    draw.ellipse([left_up_point, right_down_point], fill=0)

    # 画像をNumPy配列に変換
    return np.array(img)

def apply_filter(image, filter_kernel):
    """
    自作のフィルターを画像に適用する関数。

    Parameters:
    image (numpy.ndarray): 入力画像（2Dアレイ）。
    filter_kernel (numpy.ndarray): フィルターカーネル（例えば3x3の行列）。

    Returns:
    numpy.ndarray: フィルタリング後の画像。
    """
    # フィルターを画像に適用
    filtered_image = filtered_image = convolve(image, filter_kernel, mode='constant', cval=0)
    
    # print(np.max(filtered_image),np.min(filtered_image))
    return filtered_image