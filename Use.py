import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from end_stop_cell_ver2 import *
from funcs import *

plt.rcParams['font.family'] = 'Meiryo'

#================================================================#
#                       end_stop_cellの実装                      #
#================================================================#

# 使用例: size=200の画像で直径50の円を描く
img_size = 200

img = cv2.imread("circle/circle_1.png", cv2.IMREAD_GRAYSCALE)
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
    'gain' : 2.5 ,
    'overlap_x' : 3,
    'overlap_y' : 1
}

#overlapの大きさを決定する([x,y])
overlap_dic = {
    "side" : [2,1],
    "verticality" : [1,2],
}

end_stop , filter_result_dic = End_Stopped_cell(img , img_size , complex_cell_params , simple_cell_Params , end_stop_Params , overlap_dic)

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


