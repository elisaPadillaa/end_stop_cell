import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from cells import *
from end_stop_cell import *
from funcs import *
from scipy.ndimage import convolve
from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'

def visualize_points():
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
    
    data = [(0, 0, 74, 76), (45, 1, 79, 69), (90, 2, 84, 67), (135, 3, 130, 55)]
    
    # c_x = 74
    # c_y = 76    
    fig, axes = plt.subplots(1, 5 , figsize=(15, 5))

    for i, (angle, img_n, c_x, c_y) in enumerate(data):
        simple_cell = SCell(5,20,angle)
        complex_cell = CCells(5, 5, -45, simple_cell)
        grid = simple_cell.get_patch(filtered_img[img_n], c_x,c_y)
        resp = simple_cell.get_response(filtered_img[img_n], c_x,c_y)
        c_respL, c_respR = complex_cell.get_response(filtered_img[img_n], c_x, c_y)
        centers = complex_cell.get_centers(c_x,c_y)
        print(f"resp simple {i} = {resp}")
        print(f"resp compl {i} = {c_respL}, {c_respR}")

        print (filtered_img.min(), filtered_img.max())
        axes[img_n].imshow(filtered_img[img_n] , cmap='gray')
        axes[i].set_title(f'{angle}º')

        e, f = rotate_point_around_center(c_x, c_y + 10, c_x, c_y, angle)
        p, q = rotate_point_around_center(c_x, c_y - 10, c_x, c_y, angle)

        axes[img_n].plot(c_x, c_y, 'bo', markersize = 1.5)
        axes[img_n].plot(e, f, 'bo', markersize = 1.5)
        axes[img_n].plot(p, q, 'bo', markersize = 1.5)
        for j, points in enumerate(centers):
            for x, y in points:
                color = 'ro' if j == 0 else 'yo'
                axes[img_n].plot(x, y, color, markersize=1.5)

    axes[4].imshow(grid , cmap='gray', vmin = min, vmax = max)
    axes[4].set_title('grid')

    plt.show()

def visualize_end_stopped_cell(img, data):
    values = [(0, 0), (45, 1), (90, 2), (135, 3)]
    for coord, cell_type in data:
        fig, axes = plt.subplots(1, 5 , figsize=(15, 5))
        c_x, c_y = coord
        for i, (angle, img_n) in enumerate(values):
            params = {
                "s_cell_width": 5,
                "s_cell_height": 20,
                "esc_angle": angle,
                "c_cell_overlap": 5,
                "num_c_cells": 5,
                "gains": [1.0, 0.8, 0.8],
            }
            esc_cell = cell_type(**params)
            if img_n == 2: grid = esc_cell.s_cell.get_patch(img[img_n], c_x,c_y)
            axes[img_n].imshow(img[img_n] , cmap='gray')
            axes[i].set_title(f'{angle}º')
            points = esc_cell.plot_points(c_x, c_y)
            resp = esc_cell.get_response(img[img_n], c_x, c_y)
            print(f"result = {resp}")
            for x, y in points:
                # color = 'ro' if j == 0 else 'yo'
                axes[img_n].plot(x, y, 'ro', markersize=1.5)

        min = img.min()
        max = img.max()
        axes[4].imshow(grid , cmap='gray', vmin = min, vmax = max)
        axes[4].set_title('grid')

    plt.show()




if __name__ == "__main__":
    img_size = 200

    img = cv2.imread("circle/suji.png", cv2.IMREAD_GRAYSCALE)

    # Resize img to img_size x img_size
    img = cv2.resize(img, (img_size, img_size))

    simple_cell_Params = {
        'AR' : 2, #aspect ratio
        'sigma_x' : 2, #more or less defined
    }

    filtered_img = filter_img(img, simple_cell_Params)
    data = [((74, 76), DegreeCurveESCell), ((80, 80), SignCurveESCell)]
    visualize_end_stopped_cell(filtered_img, data)
    
    # visualize_points()



