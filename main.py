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

def visualize_esc_responses(use_img, sigma):
    img_size = 200

    # img = cv2.imread("circle/circles_1.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(use_img, cv2.IMREAD_GRAYSCALE)
    theta = [0, 45 , 90 , 135]

    # Resize img to img_size x img_size
    img = cv2.resize(img, (img_size, img_size))

    filtered_img = filter_img(img, sigma, theta)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    degree_img = np.zeros((img_size, img_size), dtype=np.float32)
    pos_sign_img = np.zeros_like(degree_img, dtype=np.float32)
    neg_sign_img = np.zeros_like(degree_img, dtype=np.float32)
    axis_2 = show_filtered_img(filtered_img)

    for i, angle in enumerate(theta):
        simple_cell = SCell(sigma, angle)
        params = {
                "s_cell_type": sigma,
                "esc_angle": angle,
                "c_cell_overlap": 2,
                "num_c_cells": 5,
                "gains": [1.0, 1.8, 1.8],
                # "s_cell": simple_cell
            }
        degree_esc = DegreeCurveESCell(**params)
        sign_escs = SignCurveESCell(**params)
        
        print(f'{angle} starting')
        points_d = []
        points_p_s = []
        points_n_s = []

        for x in range(filtered_img[i].shape[0]):
            for y in range(filtered_img[i].shape[1]):
                s_resp = simple_cell.get_response(filtered_img[i], x,y) 
                # if y == 1 and x == 2 :
                #      print(filtered_img[i][x][y])
                degree_img[y,x] = degree_esc.get_response(filtered_img[i], x, y, s_resp)
                pos_resp, neg_resp = sign_escs.get_response(filtered_img[i], x, y, s_resp)
                pos_sign_img[y,x] = pos_resp
                neg_sign_img[y,x] = neg_resp

            y, x = np.unravel_index(np.argmax(degree_img), degree_img.shape)
            points_d = degree_esc.plot_points(x, y, filtered_img[i])
            y, x = np.unravel_index(np.argmax(pos_sign_img), pos_sign_img.shape)
            a, b = sign_escs.plot_points(x, y, filtered_img[i])
            points_p_s = a
            y, x = np.unravel_index(np.argmax(neg_sign_img), neg_sign_img.shape)
            a, b = sign_escs.plot_points(x, y, filtered_img[i])
            points_n_s = b
        
        axes[0, i].imshow(degree_img, cmap = 'grey')
        axes[0, i].set_title(f'degree curv {angle}º')

        for l, (x, y) in enumerate(points_d):
                axes[0, i].plot(x, y, 'ro', markersize=1.5)
                axis_2[i].plot(x, y, 'ro', markersize=1.5)

        axes[1, i].imshow(pos_sign_img, cmap = 'grey')
        axes[1, i].set_title(f'pos sign curv {angle}º')

        for l, (x, y) in enumerate(points_p_s):
                axes[1, i].plot(x, y, 'yo', markersize=1.5)
                axis_2[i].plot(x, y, 'yo', markersize=1.5)

        axes[2, i].imshow(neg_sign_img, cmap = 'grey')
        axes[2, i].set_title(f'neg sign curv {angle}º')

        for l, (x, y) in enumerate(points_n_s):
                color = 'ro'
                color = 'ro' if  l < 13 else 'yo'
                axes[2, i].plot(x, y, 'go', markersize=1.5)
                axis_2[i].plot(x, y, 'go', markersize=1.5)
        print(f'{angle} finshed')


    # plt.show()

def show_filtered_img(imgs):
    fig, axes = plt.subplots(1, 4 , figsize=(20, 4))
    for i, img in enumerate(imgs):
        axes[i].imshow(img , cmap='gray')
    return axes

    



if __name__ == "__main__":
    # visualize_end_stopped_cell()
    # imgs = ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"]
    sigmas = [1, 2, 3, 4, 5, 6]
    for sigma in sigmas:
        visualize_esc_responses(f"circle/9.png", sigma)
    
    # visualize_points()
    plt.show()



