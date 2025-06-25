import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from cells import *
from funcs import *
from scipy.ndimage import convolve
from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'

def test():
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
    
    data = [(0, 0), (45, 1), (90, 2), (135, 3)]
    
    c_x = 158
    c_y = 144    
    fig, axes = plt.subplots(1, 5 , figsize=(15, 5))

    for i, (angle, img_n) in enumerate(data):
        simple_cell = SCell(5,20,angle)
        complex_cell = CCells(5, 5, 0, simple_cell)
        resp, grid = simple_cell.get_response(filtered_img[img_n], c_x,c_y)
        centers = complex_cell.get_centers(c_x,c_y)
        print(f"resp = {resp}")

        axes[img_n].imshow(filtered_img[img_n] , cmap='gray')
        axes[i].set_title(f'{angle}º')

        e, f = rotate_point_around_center(c_x, c_y + 10, c_x, c_y, angle)
        p, q = rotate_point_around_center(c_x, c_y - 10, c_x, c_y, angle)

        axes[img_n].plot(c_x, c_y, 'bo', markersize = 1.5)
        axes[img_n].plot(e, f, 'bo', markersize = 1.5)
        axes[img_n].plot(p, q, 'bo', markersize = 1.5)
        for i, points in enumerate(centers):
            for x, y in points:
                color = 'ro' if i == 0 else 'yo'
                axes[img_n].plot(x, y, color, markersize=1.5)


    # 各フィルターと結果の表示
    # axes[0].imshow(filtered_img[0] , cmap='gray')
    # axes[0].set_title('0º')
    
    
    # axes[1].imshow(filtered_img[1] , cmap='gray')
    # axes[1].set_title('45º')

    # axes[2].imshow(filtered_img[2] , cmap='gray')
    # axes[2].set_title('90º')

    # axes[3].imshow(filtered_img[3] , cmap='gray')
    # axes[3].set_title('135º')

    axes[4].imshow(grid , cmap='gray', vmin = min, vmax = max)
    axes[4].set_title('grid')
    # e, f = rotate_point_around_center(c_x, c_y + 10, c_x, c_y, angle)
    # p, q = rotate_point_around_center(c_x, c_y - 10, c_x, c_y, angle)

    # axes[img_n].plot(c_x, c_y, 'bo', markersize = 1.5)
    # axes[img_n].plot(e, f, 'bo', markersize = 1.5)
    # axes[img_n].plot(p, q, 'bo', markersize = 1.5)
    # for i, points in enumerate(centers):
    #     for j, (x, y) in enumerate(points):
    #         color = 'ro' if i == 0 else 'yo'
    #         axes[img_n].plot(x, y, color, markersize=1.5)
    
    plt.show()



if __name__ == "__main__":
    test()



