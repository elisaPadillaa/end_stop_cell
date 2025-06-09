import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from funcs import *
from scipy.ndimage import convolve
from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'


#============================================================================#
#                      End_stopの応答を算出する関数
#============================================================================#

def End_Stopped_cell(img , 
                    img_size , 
                    c_cell , 
                    s_cell , 
                    end_stop_Params , 
                    overlap_dic):



    theta = [0, 45 , 90 , 135]
    simple_odd = [] ; simple_even = []
    comp_simple_odd = [] ; comp_simple_even = []
    complex_cell = []


    for i in theta:
        #simple_cellの応答の算出
        filtered_image_odd = FeatureExtraction(img, s_cell["sigma_x"], i ,"Odd", s_cell["AR"])
        filtered_image_even = FeatureExtraction(img, s_cell["sigma_x"], i ,"Even", s_cell["AR"])

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
    simple_odd = np.array(simple_odd) ; simple_even = np.array(simple_even)
    comp_simple_odd = np.array(comp_simple_odd) ; comp_simple_even = np.array(comp_simple_even)
    complex_cell = np.array(complex_cell)


    #overlapの大きさに応じて，complexの応答を拡張する
    comp_copy = complex_cell.copy()
    complex_cell = []

    for i in range(len(theta)):
        complex_cell.append(np.pad(comp_copy[i], ((overlap_dic["y"], overlap_dic["y"]), (overlap_dic["x"], overlap_dic["x"])), mode='constant', constant_values=0))

    complex_cell = np.array(complex_cell)

    
    #===================================================================================
    #end_stop作成
    #右向きの弧に応答を示すend-stop
    end_stop_right = simple_even[0,:,:] - end_stop_Params["gain"] * (complex_cell[1,:img_size, -img_size:]  + complex_cell[3,-img_size:,-img_size:])
    #左向きの弧に応答を示すend-stop
    end_stop_left = simple_even[0,:,:] - end_stop_Params["gain"] * (complex_cell[3,:img_size,:img_size]  + complex_cell[1,-img_size:,:img_size])
    #上向きの弧に応答を示すend-stop
    end_stop_top = simple_even[2,:,:] - end_stop_Params["gain"] * (complex_cell[3,-img_size:,:img_size]  + complex_cell[1,-img_size:,-img_size:])
    #下向きの弧に応答を示すend-stop
    end_stop_bottom = simple_even[2,:,:] - end_stop_Params["gain"] * (complex_cell[1,:img_size,:img_size]  + complex_cell[3,:img_size,-img_size:])
    

    end_stop = []
    end_stop.append(end_stop_right) ; end_stop.append(end_stop_left)
    end_stop.append(end_stop_top) ; end_stop.append(end_stop_bottom)

    filter_dic = {
        "gabor_s_odd" : simple_odd,
        "gabor_s_even" : simple_even,
        "gabor_c_odd" : comp_simple_odd,
        "gabor_c_even" : comp_simple_even,
        "complex" : comp_copy
    }

    return end_stop , filter_dic

#============================================================================#
#                      フィルタリング結果を格納する関数
#============================================================================#
def filter_result(filter_dic):
    theta = [0,45,90,135]
    fig, axes = plt.subplots(2,4 , figsize=(15, 5))

    fig1, axes1 = plt.subplots(3,4 , figsize=(15, 5))

    for i in range(len(theta)):
        # 各フィルターと結果の表示
        axes[0,i].imshow(filter_dic["gabor_s_odd"][i] , cmap='gray')
        axes[0,i].set_title(str(i * 45) + '_odd_simple')
        axes[0, i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes[1,i].imshow(filter_dic["gabor_s_even"][i] , cmap='gray')
        axes[1,i].set_title(str(i * 45) + '_even_simple')
        axes[1, i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes1[0,i].imshow(filter_dic["gabor_c_odd"][i] , cmap='gray')
        axes1[0,i].set_title(str(i * 45) + '_odd_comp')
        axes1[0, i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes1[1,i].imshow(filter_dic["gabor_c_even"][i] , cmap='gray')
        axes1[1,i].set_title(str(i * 45) + '_even_comp')
        axes1[1, i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes1[2,i].imshow(filter_dic["complex"][i] , cmap='gray')
        axes1[2,i].set_title(str(i * 45) + '_complex')
        axes1[2, i].set_xticks([]) ; axes[0, i].set_yticks([])
    return

if __name__ == "__main__":
    # 使用例: size=200の画像で直径50の円を描く
    img_size = 200

    img = cv2.imread("circle/circle_9.png", cv2.IMREAD_GRAYSCALE)
    # imgをimg_size x img_sizeにリサイズ
    img = cv2.resize(img, (img_size, img_size))

    simple_cell_Params = {
    'AR' : 2,
    'sigma_x' : 3,
    }

    complex_cell_params = {
        'AR' : 3,
        'sigma_x' : 3,
    }

    end_stop_Params = {
        'gain' : 2.0 ,
        'overlap_x' : 3,
        'overlap_y' : 1
    }

    overlap_dic = {
        "x" : 2,
        "y" : 1
    }

    overlap_dic2 = {
        "side" : [2,1],
        "verticality" : [1,2]
    }

    end_stop , filter_result_dic = End_Stopped_cell(img , 
                            img_size , 
                            complex_cell_params , 
                            simple_cell_Params , 
                            end_stop_Params , 
                            overlap_dic)
    
    filter_result(filter_result_dic)
    
    fig, axes = plt.subplots(1,5 , figsize=(15, 5))

    for i in range(4):
        end_stop[i] = np.where(end_stop[i] < 0 , 0,end_stop[i])

    # 各フィルターと結果の表示
    axes[0].imshow(end_stop[0] , cmap='gray')
    axes[0].set_title('endstop_right')

    axes[1].imshow(end_stop[1] , cmap='gray')
    axes[1].set_title('endstop_left')

    axes[2].imshow(end_stop[2] , cmap='gray')
    axes[2].set_title('endstop_top')

    axes[3].imshow(end_stop[3] , cmap='gray')
    axes[3].set_title('endstop_bottom')

    axes[4].imshow(img , cmap='gray')
    axes[4].set_title('endstop_bottom')
    plt.show()

