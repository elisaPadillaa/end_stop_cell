import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from funcs import *
from scipy.ndimage import convolve
from functions.GaborFilter import FeatureExtraction

plt.rcParams['font.family'] = 'Meiryo'


#============================================================================#
#                    Function to calculate End_stop response
#============================================================================#

def End_Stopped_cell(img , img_size , c_cell , s_cell , end_stop_Params , overlap_dic):



    theta = [0, 45 , 90 , 135]
    simple_odd = [] 
    simple_even = []
    comp_simple_odd = [] 
    comp_simple_even = []
    complex_cell = []

    """ Calculate Gabor's response """

    for i in theta:
        # Calculation of response of simple_cell
        filtered_image_odd = FeatureExtraction(img, s_cell["sigma_x"], i ,"Odd", s_cell["AR"])      #sigma_x = 3 s_cell = 2
        filtered_image_even = FeatureExtraction(img, s_cell["sigma_x"], i ,"Even", s_cell["AR"])    #sigma_x = 3 s_cell = 2

        # append results
        simple_odd.append(filtered_image_odd)
        simple_even.append(filtered_image_even)

        # Calculation of response of complex_cell
        filtered_image_odd = FeatureExtraction(img, c_cell["sigma_x"], i ,"Odd", c_cell["AR"])
        filtered_image_even = FeatureExtraction(img, c_cell["sigma_x"], i ,"Even", c_cell["AR"])    


        # append results
        comp_simple_odd.append(filtered_image_odd)
        comp_simple_even.append(filtered_image_even)
        
        # Compute and store the response of the complex cell
        complex_cell.append(np.sqrt(filtered_image_odd ** 2 + filtered_image_odd ** 2))


    # Convert to numpy array
    simple_odd = np.array(simple_odd) 
    simple_even = np.array(simple_even)
    comp_simple_odd = np.array(comp_simple_odd) 
    comp_simple_even = np.array(comp_simple_even)
    complex_cell = np.array(complex_cell)


    # Extend the response of the complex cell according to the size of the overlap
    comp_copy = complex_cell.copy()
    comp_copy_st = complex_cell.copy()
    complex_cell = []
    complex_cell_st = []

    # 曲率セル用 (同じ方位のsimple cell と complex cell を組み合わせる)
    over_st = np.array(overlap_dic["st"])

    # 符号セル用
    over_horizontality = np.array(overlap_dic["side"])
    over_verticality = np.array(overlap_dic["verticality"])

    # Compare over_horizontality and over_verticality and set the larger of the two to overlap
    overlap = np.where(over_horizontality > over_verticality , over_horizontality , over_verticality)
    
    for i in range(len(theta)):
        complex_cell_st.append(np.pad(comp_copy_st[i],(over_st), mode= 'constant', constant_values = 0))
        complex_cell.append(np.pad(comp_copy[i], ((overlap[1], overlap[1]), (overlap[0], overlap[0])), mode='constant', constant_values=0))
        
    complex_cell_st = np.array(complex_cell_st)
    complex_cell = np.array(complex_cell)
    
    #===================================================================================
    """ Create end stop cell response 

        simple cell.shape (theta , y , x)
        complex cell.shape (theta , y + 2 * overlap["y"] , x + 2 * overlap["x"])
        complex_cell_st.shape (rheta , y + 2 * overlap["st"] , x + 2 * overlap["st"])

        The complex cell is expanding the image by the size of the overlap!!

    """
    
    # 直線
    end_stop_st0 = simple_even[0, :, :] -  end_stop_Params["gain"] * \
        (complex_cell_st[0, over_st + img_size , :img_size] + complex_cell[0, -over_st -img_size, :img_size] )
        
    end_stop_st90 = simple_even[2,:,:] - end_stop_Params["gain"] * \
        (complex_cell_st[2, :img_size , over_st + img_size] + complex_cell[2 , :img_size , over_st + img_size])
        

    #　end-stopping cell indicating response to right-pointing arc (右向きの弧に応答を示すend-stop) → Neurons responding to the shape of "("
    end_stop_right = simple_even[0,:,:] - \
        end_stop_Params["gain"] * \
            (complex_cell[1, over_horizontality[1]:over_horizontality[1]+img_size , -img_size:]  + complex_cell[3 , -over_horizontality[1]-img_size:-over_horizontality[1] ,-img_size:])
        
    # end-stopping cell indicating response to left-pointing arc (左向きの弧に応答を示すend-stop) → Neurons responding to the shape of ")"
    end_stop_left = simple_even[0,:,:] - \
        end_stop_Params["gain"] * \
            (complex_cell[3 , over_horizontality[1]:over_horizontality[1]+img_size ,:img_size]  + complex_cell[1 , -over_horizontality[1]-img_size:-over_horizontality[1] , :img_size])


    # end-stopping cell indicating response to upward arc (上向きの弧に応答を示すend-stop)
    end_stop_top = simple_even[2,:,:] - \
        end_stop_Params["gain"] * \
            (complex_cell[1 , :img_size , over_verticality[0]:img_size+over_verticality[0]]  + complex_cell[3 , :img_size , over_verticality[0]:img_size+over_verticality[0]])
        

    # end-stopping cell indicating response to downward arc (下向きの弧に応答を示すend-stop)
    end_stop_bottom = simple_even[2,:,:] - \
        end_stop_Params["gain"] * \
            (complex_cell[3 , -img_size: , over_verticality[0]:img_size+over_verticality[0]]  + complex_cell[1 , -img_size: , -over_verticality[0]-img_size:-over_verticality[0]])
    

    end_stop = []
    end_stop.append(end_stop_right) 
    end_stop.append(end_stop_left)
    end_stop.append(end_stop_top) 
    end_stop.append(end_stop_bottom)

    end_stop_st = []
    end_stop_st.append(end_stop_st0)
    end_stop_st.append(end_stop_st90)

    filter_dic = {
        "gabor_s_odd" : simple_odd,
        "gabor_s_even" : simple_even,
        "gabor_c_odd" : comp_simple_odd,
        "gabor_c_even" : comp_simple_even,
        "complex" : comp_copy
    }

    return end_stop , end_stop_st , filter_dic

#============================================================================#
#                    Function to display filtering results
#============================================================================#
def filter_result(filter_dic):
    theta = [0,45,90,135]
    fig, axes = plt.subplots(2,4 , figsize=(15, 5))

    fig1, axes1 = plt.subplots(3,4 , figsize=(15, 5))

    for i in range(len(theta)):
        # 各フィルターと結果の表示
        axes[0,i].imshow(filter_dic["gabor_s_odd"][i] , cmap='gray')
        axes[0,i].set_title(str(i * 45) + '_odd_simple')
        axes[0,i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes[1,i].imshow(filter_dic["gabor_s_even"][i] , cmap='gray')
        axes[1,i].set_title(str(i * 45) + '_even_simple')
        axes[1,i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes1[0,i].imshow(filter_dic["gabor_c_odd"][i] , cmap='gray')
        axes1[0,i].set_title(str(i * 45) + '_odd_comp')
        axes1[0,i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes1[1,i].imshow(filter_dic["gabor_c_even"][i] , cmap='gray')
        axes1[1,i].set_title(str(i * 45) + '_even_comp')
        axes1[1,i].set_xticks([]) ; axes[0, i].set_yticks([])

        axes1[2,i].imshow(filter_dic["complex"][i] , cmap='gray')
        axes1[2,i].set_title(str(i * 45) + '_complex')
        axes1[2,i].set_xticks([]) ; axes[0, i].set_yticks([])
    return

if __name__ == "__main__":
    # Example: Draw a circle with a diameter of 50 on an image with img_size=200
    img_size = 200

    img = cv2.imread("circle/circle_3.png", cv2.IMREAD_GRAYSCALE)

    # Resize img to img_size x img_size
    img = cv2.resize(img, (img_size, img_size))

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
        'AR' : 2, #aspect ratio
        'sigma_x' : 3, #width?
    }

    complex_cell_params = {
        'AR' : 4,
        'sigma_x' : 3,
    }

    end_stop_Params = {
        'gain' : 2.5 ,
        'overlap_x' : 3, #no use
        'overlap_y' : 1 #no use
    }

    overlap_dic = {
        "side" : [2,1],
        "verticality" : [1,2],
        "st" : 1
    }

    end_stop , end_stop_st , filter_result_dic = End_Stopped_cell(img , img_size , complex_cell_params , simple_cell_Params , end_stop_Params , overlap_dic)
    
    filter_result(filter_result_dic)
    
    fig, axes = plt.subplots(1, 5 , figsize=(15, 5))

    # end stop Display cell response
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
    axes[4].set_title('endstop_input')
    
    plt.show()

