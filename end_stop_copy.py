import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from funcs import *
from scipy.ndimage import convolve
from functions.GaborFilter import FeatureExtraction
import pandas as pd
import pylab

plt.rcParams['font.family'] = 'Meiryo'


#============================================================================#
#                      End_stopの応答を算出する関数
#============================================================================#

def End_Stopped_cell(img , img_size , c_cell , s_cell , end_stop_Params , overlap_dic):



    theta = [0, 45 , 90 , 135]
    simple_odd = [] 
    simple_even = []
    comp_simple_odd = [] 
    comp_simple_even = []
    complex_cell = []
    
    # kernel = np.zeros((201, 201))
    # kernel[100, 100] = 1
    
    


    for i in theta:
        #simple_cellの応答の算出
        filtered_image_odd = FeatureExtraction(img, s_cell["sigma_x"], i ,"Odd", s_cell["AR"])      #sigma_x = 3 s_cell = 2
        filtered_image_even = FeatureExtraction(img, s_cell["sigma_x"], i ,"Even", s_cell["AR"])    #sigma_x = 3 s_cell = 2
        
        # imKernel = FeatureExtraction(kernel, s_cell["sigma_x"], i ,"Odd", s_cell["AR"])
        # imKernelRange = np.max(np.abs(imKernel))
        
        # fig, ax = plt.subplots(1, 1)
        # ax.set_title("Kernel")
        # ax.imshow(imKernel, cmap="gray", vmin=-imKernelRange, vmax=imKernelRange)
        # ax.axis("off")
        # plt.show()

        
        # 結果を保存
        simple_odd.append(filtered_image_odd)
        simple_even.append(filtered_image_even)

        #complex_cellの応答の算出
        filtered_image_odd = FeatureExtraction(img, c_cell["sigma_x"], i ,"Odd", c_cell["AR"])
        filtered_image_even = FeatureExtraction(img, c_cell["sigma_x"], i ,"Even", c_cell["AR"])    


        # 結果を保存
        comp_simple_odd.append(filtered_image_odd)
        comp_simple_even.append(filtered_image_even)
        
        #complexの応答を計算し，保存
        complex_cell.append(np.sqrt(filtered_image_odd ** 2 + filtered_image_odd ** 2))


    #numpy配列に変換
    simple_odd = np.array(simple_odd) 
    simple_even = np.array(simple_even)
    comp_simple_odd = np.array(comp_simple_odd) 
    comp_simple_even = np.array(comp_simple_even)
    complex_cell = np.array(complex_cell)


    #overlapの大きさに応じて，complexの応答を拡張する
    comp_copy = complex_cell.copy()
    comp_copy_st = complex_cell.copy()
    complex_cell = []
    complex_cell_st = []

    #直線用 1 
    over_st = np.array(overlap_dic["st"])
    over_side = np.array(overlap_dic["side"]) 
    over_verticality = np.array(overlap_dic["verticality"])
    #over_side と　over_verticalityの比較で大きいほうをoverlapに
    overlap = np.where(over_side > over_verticality , over_side , over_verticality)
    
    for i in range(len(theta)):
        complex_cell_st.append(np.pad(comp_copy_st[i],(over_st), mode= 'constant', constant_values = 0))
        complex_cell.append(np.pad(comp_copy[i], ((overlap[1], overlap[1]), (overlap[0], overlap[0])), mode='constant', constant_values=0))
        
    complex_cell_st = np.array(complex_cell_st)
    complex_cell = np.array(complex_cell)
    
    # for i in range(len(theta)):
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(comp_copy_st[i], cmap='gray')
    #     plt.title(f'Before Padding - Direction {theta[i]}°')
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(complex_cell[i], cmap='gray')
    #     plt.title(f'After Padding - Direction {theta[i]}°')
    #     plt.axis('off')

    # plt.show()

    
    

    
    #===================================================================================
    #end_stop作成


    end_stop_st0 = simple_even[0, :, :] -  end_stop_Params["st_gain"] * \
        (complex_cell_st[0, over_st:over_st + img_size , :img_size] + complex_cell_st[0, -over_st-img_size:-over_st, :img_size] )    
        
    end_stop_st90 = simple_even[2,:,:] - end_stop_Params["st_gain"] * \
        (complex_cell_st[2, :img_size , over_st + img_size] + complex_cell[2 , :img_size , over_st + img_size])
        
    end_stop_st0 = np.where(end_stop_st0 <= 0 , 0 , end_stop_st0)
    end_stop_st90 = np.where(end_stop_st90 <= 0 , 0 , end_stop_st90)
    print(np.min(end_stop_st0) , np.max(end_stop_st0))
        
    

        
    #右向きの弧に応答を示すend-stop   
    end_stop_right = simple_even[0,:,:] - end_stop_Params["gain"] * \
        (complex_cell[1, over_side[1]:over_side[1]+img_size , -img_size:]  + complex_cell[3 , -over_side[1]-img_size:-over_side[1] ,-img_size:] )
        

    
    #左向きの弧に応答を示すend-stop
    end_stop_left = simple_even[0,:,:] - end_stop_Params["gain"] * \
        (complex_cell[3 , over_side[1]:over_side[1]+img_size ,:img_size]  + complex_cell[1 , -over_side[1]-img_size:-over_side[1] , :img_size])


    #上向きの弧に応答を示すend-stop
    end_stop_top = simple_even[2,:,:] - end_stop_Params["gain"] * \
        (complex_cell[1 , :img_size , over_verticality[0]:img_size+over_verticality[0]]  + complex_cell[3 , :img_size , over_verticality[0]:img_size+over_verticality[0]])
        

    #下向きの弧に応答を示すend-stop
    end_stop_bottom = simple_even[2,:,:] - end_stop_Params["gain"] * \
        (complex_cell[3 , -img_size: , over_verticality[0]:img_size+over_verticality[0]]  + complex_cell[1 , -img_size: , -over_verticality[0]-img_size:-over_verticality[0]])
        
        
    end_stop_right = np.where(end_stop_right <= 0 , 0 , end_stop_right)
    end_stop_left = np.where(end_stop_left <= 0 , 0 , end_stop_left)
    end_stop_top = np.where(end_stop_top <= 0, 0, end_stop_top)
    end_stop_bottom = np.where(end_stop_bottom <= 0, 0, end_stop_bottom)
    print(np.min(end_stop_right) , np.max(end_stop_right))
    print(np.min(end_stop_left) , np.max(end_stop_left))
    print(np.min(end_stop_top) , np.max(end_stop_top))
    print(np.min(end_stop_bottom) , np.max(end_stop_bottom))
        
        

        
    # print(f"over_side: {over_side}")
    # print(f"over_side[1]: {over_side[1]}")
    # print(f"スライス範囲: {over_side[1]}:{over_side[1] + img_size}")
    

    end_stop = []
    end_stop.append(end_stop_right) 
    end_stop.append(end_stop_left)
    end_stop.append(end_stop_top) 
    end_stop.append(end_stop_bottom)

    end_stop_curve = []
    end_stop_curve.append(end_stop_st0)
    end_stop_curve.append(end_stop_st90)

    filter_dic = {
        "gabor_s_odd" : simple_odd,
        "gabor_s_even" : simple_even,
        "gabor_c_odd" : comp_simple_odd,
        "gabor_c_even" : comp_simple_even,
        "complex" : comp_copy
    }

    return end_stop , filter_dic , end_stop_curve


##中心からの距離によって形状ニューロンの応答をより正確にする##

# def calculate_shape_response(img, img_size, end_stop, center):
# 
#     """
#     形状ニューロンの応答を計算
#     - img: 入力画像
#     - img_size: 画像サイズ
#     - end_stop: end_stop応答 (list)
#     - center: 形状の中心 (x, y)
#     """
#     # 形状ニューロンの応答を初期化
#     shape_response = np.zeros((img_size, img_size))

#     # 曲率の位置ごとの重み付け (Gaussian weight)
#     x_coords, y_coords = np.meshgrid(np.arange(img_size), np.arange(img_size))
#     distances = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
#     c_i = np.exp(-distances**2 / (2 * (img_size / 4)**2))  # 中心からの距離に基づく重み (例)

#     # 各end_stop応答に重みを適用し、形状応答を計算
#     for response in end_stop:
#         weighted_response = c_i * response  # 重み付け
#         shape_response += weighted_response  # 応答を統合

#     return shape_response

#============================================================================#
#                      フィルタリング結果を格納する関数
#============================================================================#


def filter_result(filter_dic):
    theta = [0,45,90,135]
    fig, axes = plt.subplots(2,4 , figsize=(15, 5))

    fig1, axes1 = plt.subplots(3,4 , figsize=(15, 5))

    for i in range(len(theta)):
        # 各フィルターと結果の表示
        s_odd = filter_dic["gabor_s_odd"][i] + 128
        s_odd[s_odd > 255] = 255
        s_odd[s_odd < 0] = 0
        s_odd = s_odd.astype(np.uint8)
        
        axes[0,i].imshow(s_odd, cmap='gray', vmin = 0, vmax = 255)
        axes[0,i].set_title(str(i * 45) + '_odd_simple')
        axes[0,i].set_xticks([]) ; axes[0, i].set_yticks([])
        
        s_even = filter_dic["gabor_s_even"][i] + 128
        s_even[s_even > 255] = 255
        s_even[s_even < 0] = 0
        s_even = s_even.astype(np.uint8)

        axes[1,i].imshow(s_even, cmap='gray', vmin = 0, vmax = 255)
        axes[1,i].set_title(str(i * 45) + '_even_simple')
        axes[1,i].set_xticks([]) ; axes[1, i].set_yticks([])
        
        c_odd = filter_dic["gabor_c_odd"][i] + 128
        c_odd[c_odd > 255] = 255
        c_odd[c_odd < 0] = 0
        c_odd = c_odd.astype(np.uint8)

        axes1[0,i].imshow(c_odd , cmap='gray',vmin = 0, vmax = 255)
        axes1[0,i].set_title(str(i * 45) + '_odd_comp')
        axes1[0,i].set_xticks([]) ; axes1[0, i].set_yticks([])
        
        c_even = filter_dic["gabor_c_even"][i] + 128
        c_even[c_even > 255] = 255
        c_even[c_even < 0] = 0
        c_even = c_even.astype(np.uint8)

        axes1[1,i].imshow(c_even , cmap='gray', vmin = 0, vmax = 255)
        axes1[1,i].set_title(str(i * 45) + '_even_comp')
        axes1[1,i].set_xticks([]) ; axes1[1, i].set_yticks([])

        axes1[2,i].imshow(filter_dic["complex"][i] , cmap='gray')
        axes1[2,i].set_title(str(i * 45) + '_complex')
        axes1[2,i].set_xticks([]) ; axes1[2, i].set_yticks([])
    return

if __name__ == "__main__":
    # 使用例: size=200の画像で直径50の円を描く
    img_size = 200

    img = cv2.imread("circle/circle_20.png", cv2.IMREAD_GRAYSCALE)
    # imgをimg_size x img_sizeにリサイズ
    img = cv2.resize(img, (img_size, img_size))
    
    # ここにコントラストの反転を挿入
    if np.mean(img) > 128:  # 白背景の場合（平均値が高い場合）
        img = 255 - img  # コントラストを反転（白背景→黒背景）

    # AR -3～3, 0のときは縦横比1:1
    #sigma_xの値  
    # 2: 2.201,
    # 3: 3.128,
    # 4: 4.467,
    # 5: 6.322,
    # 6: 8.971,
    # 7: 12.694,
    # 8: 17.964,
    # 9: 25.419,
    # 10: 35.941
    simple_cell_Params = {
        'AR' : 2,
        'sigma_x' : 3,
    }

    complex_cell_params = {
        'AR' : 3,  #おかしい
        'sigma_x' : 3,
    }

    end_stop_Params = {
        'st_gain' : 2,
        'gain' : 3 ,
    }

    #overlapの大きさを決定する([x,y])
    overlap_dic = {
        "side" : [2,1],
        "verticality" : [1,2],
        "st" : 1
    }

    end_stop , filter_result_dic , end_stop_curve = End_Stopped_cell(img , img_size , complex_cell_params , simple_cell_Params , end_stop_Params , overlap_dic)
    
    # center = (img_size // 2, img_size // 2)
    
    filter_result(filter_result_dic)
    
    # shape_response = calculate_shape_response(img, img_size, end_stop, center)

    
    fig, axes = plt.subplots(1, 5 , figsize=(15, 5))

    for i in range(4):
        end_stop[i] = np.where(end_stop[i] < 0 , 0,end_stop[i])

    # 各フィルターと結果の表示
    axes[0].imshow(end_stop[0] , cmap='gray')
    axes[0].set_title('endstop_right')
    axes[0].set_xticks([]) ; axes[0].set_yticks([])

    axes[1].imshow(end_stop[1] , cmap='gray')
    axes[1].set_title('endstop_left')
    axes[1].set_xticks([]) ; axes[1].set_yticks([])

    axes[2].imshow(end_stop[2] , cmap='gray')
    axes[2].set_title('endstop_top')
    axes[2].set_xticks([]) ; axes[2].set_yticks([])

    axes[3].imshow(end_stop[3] , cmap='gray')
    axes[3].set_title('endstop_bottom')
    axes[3].set_xticks([]) ; axes[3].set_yticks([])

    axes[4].imshow(img , cmap='gray')
    axes[4].set_title('endstop_input')
    axes[4].set_xticks([]) ; axes[4].set_yticks([])
    
    
    
    fig, axes = plt.subplots(1, 2 , figsize=(15, 5))

    # 各フィルターと結果の表示
    axes[0].imshow(end_stop_curve[0] , cmap='gray')
    axes[0].set_title('endstop_right')

    axes[1].imshow(end_stop_curve[1] , cmap='gray')
    axes[1].set_title('endstop_left')
    
    
    plt.show()

