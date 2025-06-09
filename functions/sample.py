import cv2
import numpy as np
import matplotlib.pyplot as plt

from functions.GaborFilter import FeatureExtraction


def main():
    img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
    
    kernel = np.zeros((41, 41))
    kernel[20, 20] = 1
    
    sigma = 3 # 2~10
    # 2: 2.201,
    # 3: 3.128,
    # 4: 4.467,
    # 5: 6.322,
    # 6: 8.971,
    # 7: 12.694,
    # 8: 17.964,
    # 9: 25.419,
    # 10: 35.941
    
    theta = 45 # 0, 45, 90, 135
    gabor_type = "Odd" # "Odd" or "Even"
    sAspect = -2 # -3～3, 0のときは縦横比1:1
    
    imOut = FeatureExtraction(img, sigma, theta, gabor_type, sAspect=sAspect)
    imOut = np.clip(imOut + 128, 0, 255)
    imKernel = FeatureExtraction(kernel, sigma, theta, gabor_type, sAspect=sAspect)
    imKernelRange = np.max(np.abs(imKernel))

    fig, ax = plt.subplots(1, 2)
    ax = ax.flatten()
    ax[0].set_title("Kernel")
    ax[0].imshow(imKernel, cmap="gray", vmin=-imKernelRange, vmax=imKernelRange)
    ax[0].axis("off")
    ax[1].set_title("Output")
    ax[1].imshow(imOut, cmap="gray", vmin=0, vmax=255)
    ax[1].axis("off")
    plt.show()

if __name__ == "__main__":
    main()
