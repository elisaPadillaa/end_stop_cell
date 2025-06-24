import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from cells import SCell
from funcs import *
from scipy.ndimage import convolve
from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'


if __name__ == "__main__":
    # Example: Draw a circle with a diameter of 50 on an image with img_size=200
    img_size = 200

    img = cv2.imread("circle/suji.png", cv2.IMREAD_GRAYSCALE)

    # Resize img to img_size x img_size
    img = cv2.resize(img, (img_size, img_size))

    simple_cell_Params = {
        'AR' : 2, #aspect ratio
        'sigma_x' : 2, #more or less defined
    }

    filtered_img = filter_img(img, simple_cell_Params)

    max = filtered_img.max()
    min = filtered_img.min()
    
    simple_cell = SCell(10,20,135)
    # resp_1 = simple_cell.get_response(filtered_img[1], 57,68)
    # resp_2 = simple_cell.get_response(filtered_img[1], 75,49)
    resp_3, grid = simple_cell.get_response(filtered_img[3], 152,70)
    # print(resp_1)
    # print(resp_2)
    print(f"resp = {resp_3}")

    fig, axes = plt.subplots(1, 5 , figsize=(15, 5))

    # 各フィルターと結果の表示
    axes[0].imshow(filtered_img[0] , cmap='gray')
    axes[0].set_title('0º')
    
    axes[1].imshow(filtered_img[1] , cmap='gray')
    axes[1].set_title('45º')

    axes[2].imshow(filtered_img[2] , cmap='gray')
    axes[2].set_title('90º')

    axes[3].imshow(filtered_img[3] , cmap='gray')
    axes[3].set_title('135º')

    axes[4].imshow(grid , cmap='gray', vmin = min, vmax = max)
    axes[4].set_title('grid')
    
    plt.show()


