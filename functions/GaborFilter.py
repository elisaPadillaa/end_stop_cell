import cv2
import numpy as np
from matplotlib import pyplot as plt

# from functions.Gauss_GaborFunc_v3 import GaborFilter
from functions.Gauss_GaborFunc_v4 import GaborFilter
from functions.Gaussian_func import Gaussian_func, Gaussian_func_new


def main():
    " Settings for viewing filter shape "
    imInput = np.zeros((201, 201))
    imInput[100, 100] = 1

    " Input image settings "
    # imInput = np.float64(cv2.cvtColor(cv2.imread("./img/Mandrill.bmp"), cv2.COLOR_BGR2GRAY))

    sigma = 5  # (2~10)
    theta = 0   # (0, 45, 90, 135)
    gaborType = "Even"   # ("Odd" or "Even")

    imOutput = FeatureExtraction(imInput, sigma, theta, gaborType)

    # imOutput[imOutput < 0] = -imOutput[imOutput < 0]
    # imOutput = np.abs(imOutput)

    plus = np.sum(imOutput[imOutput > 0])
    minus = np.sum(imOutput[imOutput < 0])
    print(plus/-minus)

    plt.figure()
    plt.imshow(imOutput)
    plt.show()


def FeatureExtraction(img: np.ndarray, sigma: int, theta: int, gaborType: str, sGain: float = 1.0, sAspect: int=0):
    """
    Feature extraction using Gabor filter

    Parameters
    ----------
    img: Array (Height, Width, 1 channel)
        Input image

    sigma: int (Values between 2 and 10)
        Standard deviation of Gaussian (how wide the curve is)
        We define the value of the standard deviation as follows
        Set value: Actual value
          2: 2.201,
          3: 3.128,
          4: 4.467,
          5: 6.322,
          6: 8.971,
          7: 12.694,
          8: 17.964,
          9: 25.419,
         10: 35.941

    theta: int (One of the values 0,45,90,135)
        Orientation of the line to be extracted

    gaborType: list(string) (Either "Odd", "Even")
        Change the phase (even: cos wave, odd: sin wave)
        even (cos) dectect lines or bars (thin stripe dark or light surrounded by opposite)
        odd (sin) detect edges (transitions between light and dark)

    Return
    ----------
    imOutput: Array (Height, Width, 1 channel)
        Image after feature extraction
        
        
    """
    
    
    # Gaussianを新しい関数に変更
    imOutput = Gaussian_func_new(
        np.squeeze(
            GaborFilter(
                img[np.newaxis, :, :], #newaxis just adds another dimension 
                theta, gaborType, sigma, sGain=sGain, sAspect=sAspect 
            ),
            0,
        ),
        sigma
    )

    return imOutput

if __name__ == "__main__":
    main()
