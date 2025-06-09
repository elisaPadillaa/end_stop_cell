import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from Gabor_filter import *
from funcs import *
from scipy.ndimage import convolve
from PIL import Image, ImageOps
from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'


    #=============================================================#
    #            　　　End_stop_cell作成を行う関数
    #=============================================================#
class End_stop_cell_class:
    def __init__(self,size,simple_cell,complex_cell):
        self.size = size
        self.s_cell_dic = simple_cell
        self.c_cell_dic = complex_cell
        self.theta = [0, np.pi/4 , np.pi/2 , 3*np.pi/4]# 0, 45, 90, 135 degrees

  
    def Make_endstop(self,img,img_size,overlap_x,overlap_y,gain):

        image_odd_s = [] 
        image_even_s = [] 
        image_odd = [] 
        image_even = []
        c_cell = []
        
        theta = [0, np.pi/4 , np.pi/2 , 3*np.pi/4]# 0, 45, 90, 135 degrees


            #simple_cell作成
        for i in range(len(theta)):
            # フィルタリング結果を取得           
            filtered_image_odd = FeatureExtraction(img, self.s_cell_dic["sigma_x"], i * 45,"Odd", self.s_cell_dic["AR"])
            filtered_image_even = FeatureExtraction(img, self.s_cell_dic["sigma_x"], i * 45 ,"Even", self.s_cell_dic["AR"])      
    
            # 結果を保存
            image_odd_s.append(filtered_image_odd)
            image_even_s.append(filtered_image_even)

            # complex_cell作成
        for i in range(len(theta)):
            filtered_image_odd = FeatureExtraction(img, self.c_cell_dic["sigma_x"], i * 45,"Odd", self.c_cell_dic["AR"])
            filtered_image_even = FeatureExtraction(img, self.c_cell_dic["sigma_x"], i * 45 ,"Even", self.c_cell_dic["AR"])    
    
    
            # 結果を保存
            image_odd.append(filtered_image_odd)
            image_even.append(filtered_image_even)
            
            #complexの応答を計算
            c_cell.append(np.sqrt(image_odd_s[i] ** 2 + image_even_s[i] ** 2))

        # 変数を numpy 配列に変換
        image_odd = np.array(image_odd)
        image_even = np.array(image_even)
        image_odd_s = np.array(image_odd_s)
        image_even_s = np.array(image_even_s)
        c_cell = np.array(c_cell)

        #end_stop作成
        #右向きの弧に応答を示すend-stop
        end_stop_right = image_even_s[0,:,:] - gain * (c_cell[1,:img_size, -img_size:]  + c_cell[3,-img_size:,-img_size:])
        #左向きの弧に応答を示すend-stop
        end_stop_left = image_odd_s[0,:,:] - gain * (c_cell[3,:img_size,:img_size]  + c_cell[1,-img_size:,:img_size])
        #上向きの弧に応答を示すend-stop
        end_stop_top = image_odd_s[2,:,:] - gain * (c_cell[3,-img_size:,:img_size]  + c_cell[1,-img_size:,-img_size:])
        #下向きの弧に応答を示すend-stop
        end_stop_bottom = image_odd_s[2,:,:] - gain * (c_cell[1,:img_size,:img_size]  + c_cell[3,:img_size,-img_size:])
        

        end_stop = []
        end_stop.append(end_stop_right) ; end_stop.append(end_stop_left)
        end_stop.append(end_stop_top) ; end_stop.append(end_stop_bottom)

        #===================================================================================

        return end_stop
    