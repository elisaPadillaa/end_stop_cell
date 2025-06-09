import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams['font.family'] = 'Meiryo'

def gabor_filter(size, sigma_x,sigma_y, theta, lambda_,psi=0):
    '''
    size : フィルタサイズ
    sigma : ガウスの標準偏差
    theta : 抽出する線分の角度
    lamda_ : 波長
    psi : 位相
    '''

    # フィルタのサイズに基づいて、xとyの座標を生成
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    y = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(x, y)

    # 座標を回転
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # ガウス関数の計算（アスペクト比をγで調整）
    gaussian = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * ((x_theta**2) / (sigma_x**2) + (y_theta**2) / (sigma_y**2)))
    # gaussian = np.exp(-0.5 * ((x_theta**2) / (sigma_x**2) + (y_theta**2) / (sigma_y**2)))
    
    # 正弦波の計算
    sinusoidal = np.cos(2 * np.pi * x_theta / lambda_ + psi)

    # ガボールフィルタの生成
    gabor = gaussian * sinusoidal

    #一時的なもの
    gabor = np.where(gabor < 0, 1.25 * gabor,gabor)
    # print(np.max(gabor),np.min(gabor))

    return gabor

# ガウス関数を定義
def gauss(x, a=1, mu=0, sigma=1):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))


def convolution2d(image, kernel, padding_type='constant', padding_value=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # パディングサイズの計算
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # ゼロパディングを施した新しい画像の作成
    if padding_type == 'constant':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=padding_value)
    elif padding_type == 'reflect':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    else:
        raise ValueError("Unsupported padding type")
    
    # 畳み込み結果の保存用行列
    output = np.zeros_like(image, dtype=np.float32)
    
    # フィルタをスライドさせて畳み込み演算を実行
    for i in range(image_height):
        for j in range(image_width):
            # フィルタと重ねる部分を抽出し、要素ごとの積を計算して合計
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output
