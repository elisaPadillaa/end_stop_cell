import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from cells import *
from curvature_cell import CurvatureCell
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

def visualize_end_stopped_cell():
    img_size = 200

    img = cv2.imread("circle/cloud.png", cv2.IMREAD_GRAYSCALE)
    theta = [0, 45 , 90 , 135]

    # Resize img to img_size x img_size
    img = cv2.resize(img, (img_size, img_size))

    simple_cell_Params = {
        'AR' : 2, #aspect ratio
        'sigma_x' : 2, #more or less defined
    }

    filtered_img = filter_img(img, simple_cell_Params, theta)
    data = [((74, 76), DegreeCurveESCell), ((30, 83), SignCurveESCell)]
    
    values = [(0, 0), (45, 1), (90, 2), (135, 3)]
    for coord, cell_type in data:
        fig, axes = plt.subplots(1, 5 , figsize=(15, 5))
        c_x, c_y = coord
        for i, (angle, img_n) in enumerate(values):
            params = {
                "s_cell_width": 3,
                "s_cell_height": 15,
                "esc_angle": angle,
                "c_cell_overlap": 5,
                "num_c_cells": 5,
                "gains": [1.0, 0.8, 0.8],
            }
            esc_cell = cell_type(**params)
            if img_n == 2: grid = esc_cell.s_cell.get_patch(filtered_img[img_n], c_x,c_y)
            axes[img_n].imshow(filtered_img[img_n] , cmap='gray')
            axes[i].set_title(f'{angle}º')
            points = esc_cell.plot_points(c_x, c_y, filtered_img[img_n])
            print(np.array(points).shape)

            resp = esc_cell.get_response(filtered_img[img_n], c_x, c_y)
            if isinstance(esc_cell, SignCurveESCell): 
                resp = str(resp[0]) + " - " + str(resp[1])
                points = np.concatenate([points[0], points[1]], axis=0)
            print(f"result = {resp}")
            for l, (x, y) in enumerate(points):
                color = 'ro'
                color = 'ro' if  l < 13 else 'yo'
                axes[img_n].plot(x, y, color, markersize=1.5)

        min = filtered_img.min()
        max = filtered_img.max()
        axes[4].imshow(grid , cmap='gray', vmin = min, vmax = max)
        axes[4].set_title('grid')

    plt.show()

def visualize_esc_responses(use_img, sigma, curvature_cells):
    img_size = 200
    img = cv2.imread(use_img, cv2.IMREAD_GRAYSCALE)
    theta = [0, 45 , 90 , 135]

    # Resize img to img_size x img_size
    img = cv2.resize(img, (img_size, img_size))

    show_original_img(img)

    filtered_imgs = filter_img(img, sigma, theta)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    processed_esc_imgs = []

    filtered_imgs = torch.tensor(filtered_imgs, dtype=torch.float32)

    for ori, cell in curvature_cells:
        pos_img, neg_img = cell.get_response(filtered_imgs)
        processed_esc_imgs.append(pos_img)
        processed_esc_imgs.append(neg_img)

    curv = 0
    for i, img in enumerate(filtered_imgs):
        axes[0][i].imshow(img , cmap='gray')
        axes[0][i].set_title(f'Gabor img angle {curv}º')
        curv += 45

    curv = 0
    for col in range(4):
        axes[1][col].imshow(processed_esc_imgs[2*col], cmap = 'grey')
        axes[2][col].imshow(processed_esc_imgs[2*col + 1], cmap = 'grey')
        axes[1][col].set_title(f'Pos sign curve {curv}º')
        axes[2][col].set_title(f'Pos sign curve {curv}º')
        curv += 45




def show_filtered_img(imgs):
    fig, axes = plt.subplots(1, 4 , figsize=(20, 4))
    for i, img in enumerate(imgs):
        axes[i].imshow(img , cmap='gray')
    return axes


def show_original_img(img):
    plt.figure("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    plt.show(block=False)

    



if __name__ == "__main__":
    # visualize_end_stopped_cell()
    # imgs = ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma = 6
    esc_overlap = 5
    num_c_cells = 5
    gains = [1.0, 0.8, 0.8]
    curvature_cell_0 = CurvatureCell(sigma, 0, esc_overlap, num_c_cells, gains, device)
    curvature_cell_45 = CurvatureCell(sigma, 45, esc_overlap, num_c_cells, gains, device)
    curvature_cell_90 = CurvatureCell(sigma, 90, esc_overlap, num_c_cells, gains, device)
    curvature_cell_135 = CurvatureCell(sigma, 135, esc_overlap, num_c_cells, gains, device)

    curvature_cells = [
        (0, curvature_cell_0),
        (45, curvature_cell_45),
        (90, curvature_cell_90),
        (135, curvature_cell_135),
    ]

    # for sigma in sigmas:
    visualize_esc_responses(f"circle/cloud.png", sigma, curvature_cells)
    
    # visualize_points()
    
    plt.show()



