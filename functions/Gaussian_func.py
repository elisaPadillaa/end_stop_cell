import numpy as np
import time

from functions.utils import conv2DWith1D


# 画像の読み込み関数
def Gaussian_func(imInput, sNumOfFilter):
    ### フィルタ用配列準備 ############################################################
    sImageHeight = imInput.shape[0]
    sImageWidth = imInput.shape[1]
    imTempFLOAT = np.zeros(
        (sNumOfFilter * 2 + 1, sImageHeight, sImageWidth), dtype=np.float64)
    imTempFLOAT[0, :, :] = imInput

    ### フィルタ重み設定 ##############################################################
    sWeight00 = 10
    sWeight01 = 3
    sWeight10 = 8
    sWeight11 = 4

    ### フィルタ実行 ##################################################################
    for sFilter in range(sNumOfFilter):
        sBaseNumber = int(sFilter / 2) + 1
        if sFilter % 2 == 0:
            sSizeOfFilter = 2 ** sBaseNumber
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight01
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight00
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight01
        else:
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight11
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight10
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight11

        for sPositionX in range(sImageWidth):
            sCenter = (
                imTempFLOAT[sFilter * 2, :, sPositionX]
                * vWeight[(2 ** (sBaseNumber - 1))]
            )

            sLatter = (
                imTempFLOAT[
                    sFilter * 2, :, max(0, sPositionX - (2 ** (sBaseNumber - 1)))
                ]
                * vWeight[0]
            )

            sFormer = (
                imTempFLOAT[
                    sFilter * 2,
                    :,
                    min(sPositionX + (2 ** (sBaseNumber - 1)), sImageWidth - 1),
                ]
                * vWeight[(2 ** (sBaseNumber - 1)) * 2]
            )

            imTempFLOAT[sFilter * 2 + 1, :,
                        sPositionX] = sFormer + sCenter + sLatter

        imTempFLOAT[sFilter * 2 + 1, :,
                    :] = imTempFLOAT[sFilter * 2 + 1, :, :] / 16

        for sPositionY in range(sImageHeight):
            sCenter = (
                imTempFLOAT[sFilter * 2 + 1, sPositionY, :]
                * vWeight[(2 ** (sBaseNumber - 1))]
            )

            sLatter = (
                imTempFLOAT[
                    sFilter * 2 + 1, max(sPositionY - (2 ** (sBaseNumber - 1)), 0), :
                ]
                * vWeight[0]
            )

            sFormer = (
                imTempFLOAT[
                    sFilter * 2 + 1,
                    min(sPositionY + (2 ** (sBaseNumber - 1)), sImageHeight - 1),
                    :,
                ]
                * vWeight[(2 ** (sBaseNumber - 1)) * 2]
            )

            imTempFLOAT[sFilter * 2 + 2, sPositionY,
                        :] = sFormer + sCenter + sLatter

        imTempFLOAT[sFilter * 2 + 2, :,
                    :] = imTempFLOAT[sFilter * 2 + 2, :, :] / 16

    imOutputFLOAT = imTempFLOAT[sNumOfFilter * 2, :, :]

    return imOutputFLOAT


def Gaussian_func_new(imInput, sNumOfFilter):
    imTempFloat64 = imInput.copy().astype(np.float64)

    sWeight00 = 10
    sWeight01 = 3
    sWeight10 = 8
    sWeight11 = 4

    for sFilter in range(sNumOfFilter):
        sBaseNumber = int(sFilter / 2) + 1
        if sFilter % 2 == 0:
            sSizeOfFilter = 2 ** sBaseNumber
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight01
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight00
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight01
        else:
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight11
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight10
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight11

        imPadded = np.pad(imTempFloat64, 2 **
                          (sBaseNumber - 1), mode='edge')
        imConvX = conv2DWith1D(imPadded, vWeight, sAxis=0) / 16
        imConvY = conv2DWith1D(imConvX, vWeight, sAxis=1) / 16
        imTempFloat64 = imConvY
    return imTempFloat64


def Gaussian_func_new_X(imInput, sNumOfFilter):
    imTempFloat64 = imInput.copy().astype(np.float64)

    sWeight00 = 10
    sWeight01 = 3
    sWeight10 = 8
    sWeight11 = 4

    for sFilter in range(sNumOfFilter):
        sBaseNumber = int(sFilter / 2) + 1
        if sFilter % 2 == 0:
            sSizeOfFilter = 2 ** sBaseNumber
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight01
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight00
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight01
        else:
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight11
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight10
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight11

        imPadded = np.pad(imTempFloat64, 2 **
                          (sBaseNumber - 1), mode='edge')
        imConvX = conv2DWith1D(imPadded, vWeight, sAxis=0) / 16
        imConvX = imConvX[2 ** (sBaseNumber - 1): -2 ** (sBaseNumber - 1)][:]
        imTempFloat64 = imConvX
    return imTempFloat64


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    n = 100
    sSigma = 5
    imInput = cv2.imread("./functions/sample.png", cv2.IMREAD_GRAYSCALE)

    elapsedTimeSum = [0] * 2
    for i in range(n):
        start = time.time()
        imOutput1 = Gaussian_func(imInput, sSigma)
        elapsedTimeSum[0] += time.time() - start
        start = time.time()
        imOutput2 = Gaussian_func_new(imInput, sSigma)
        elapsedTimeSum[1] += time.time() - start
        if i == 0:
            print(np.all(imOutput1 == imOutput2))

    print("Elapsed time (original):", elapsedTimeSum[0] / n)
    print("Elapsed time (new):", elapsedTimeSum[1] / n)

    # if (elapsedTimeSum[0] / n) < (elapsedTimeSum[1] / n):
    #     print(f"Original is faster({
    #           elapsedTimeSum[1] / n - elapsedTimeSum[0] / n})")
    # else:
    #     print(f"New is faster({
    #           elapsedTimeSum[0] / n - elapsedTimeSum[1] / n})")

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(imOutput1, cmap="gray")
    ax1.set_title("Original")
    ax2.imshow(imOutput2, cmap="gray")
    ax2.set_title("New")
    plt.show()
