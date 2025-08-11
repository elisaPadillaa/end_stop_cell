import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageDraw
from scipy.ndimage import convolve

from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'

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

def filter_img(img, sigma, theta):
    simple_odd = [] 
    simple_even = []
    filtered_img = []
    threshold = 200

    """ Calculate Gabor's response """

    for i in theta:
        # Calculation of response of simple_cell
        # filtered_image_odd = FeatureExtraction(img, s_cell["sigma_x"], i ,"Odd", s_cell["AR"])      #sigma_x = 3 s_cell = 2
        filtered_image_even = FeatureExtraction(img, sigma, i ,"Even")    #sigma_x = 3 s_cell = 2

        # append results
        # simple_odd.append(filtered_image_odd)
        simple_even.append(filtered_image_even)

    # Convert to np array
    # simple_odd = np.array(simple_odd) 
    simple_even = np.array(simple_even)
    avg_img = simple_even

    # # set binary
    img_norm = ((avg_img - avg_img.min()) / (avg_img.max() - avg_img.min()) * 255).astype(np.uint8)
    binary_img = np.where(img_norm > threshold, 1, 0).astype(np.uint8)

    return simple_even

def rotate_point_around_center(x, y, x0, y0, angle_deg):
        angle_rad = math.radians(angle_deg)
        dx = x - x0
        dy = y - y0
        x_rot = math.cos(angle_rad) * dx - math.sin(angle_rad) * dy + x0
        y_rot = math.sin(angle_rad) * dx + math.cos(angle_rad) * dy + y0
        return x_rot, y_rot
