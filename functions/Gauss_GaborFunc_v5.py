import numpy as np
import yaml

# データの型を指定
np_dtype = np.float64

with open("./parameter_gauss_ver2.yml", "r") as f:
    parameters = yaml.load(f, Loader=yaml.SafeLoader)


def Extend_Image(Image, Extend_px, sOrientation, sOrthogonal=1):
    sFrame, sHeight, sWidth = Image.shape

    Result = np.zeros(
        (sFrame, sHeight + 2 * Extend_px, sWidth + 2 * Extend_px),
        dtype=np_dtype,
    )
    Result[:, Extend_px:-Extend_px, Extend_px:-
           Extend_px] = Image.copy()  # float64

    if sOrientation == 0 or sOrientation == 90:
        # float64
        Result = Result.transpose(1, 0, 2)
        Result[0:Extend_px, :, :] = Result[Extend_px, :, :]
        Result[-Extend_px:, :, :] = Result[-Extend_px - 1, :, :]
        Result = Result.transpose(1, 0, 2)
        Result = Result.transpose(2, 1, 0)
        Result[0:Extend_px, :, :] = Result[Extend_px, :, :]
        Result[-Extend_px:, :, :] = Result[-Extend_px - 1, :, :]
        Result = Result.transpose(2, 1, 0)
    elif sOrientation == 45 or sOrientation == 135:
        if sOrientation == 45:
            sS_direct = -1 * sOrthogonal
        elif sOrientation == 135:
            sS_direct = 1 * sOrthogonal

        for s in range(1, Extend_px + 1):
            # float64
            Result = Result.transpose(2, 1, 0)
            Result[sWidth + Extend_px + s - 1] = np.roll(
                Result[sWidth + Extend_px - 1], shift=-s * sS_direct, axis=0
            )
            Result[Extend_px - s] = np.roll(
                Result[Extend_px], shift=s * sS_direct, axis=0
            )
            Result = Result.transpose(2, 1, 0)

            Result = Result.transpose(1, 2, 0)
            Result[sHeight + Extend_px + s - 1] = np.roll(
                Result[sHeight + Extend_px - 1], shift=-s * sS_direct, axis=0
            )
            Result[Extend_px - s] = np.roll(
                Result[Extend_px], shift=s * sS_direct, axis=0
            )
            Result = Result.transpose(2, 0, 1)
    return Result


def Extend_Image_new(imInput, sExtendPx, sOrientation, sOrthogonal=1):
    sFrame, sHeight, sWidth = imInput.shape

    imOutput = np.zeros(
        (sFrame, sHeight + 2 * sExtendPx, sWidth + 2 * sExtendPx),
        dtype=np_dtype,
    )
    imOutput[:, sExtendPx:-sExtendPx, sExtendPx:-
             sExtendPx] = imInput.copy()  # float64

    if sOrientation == 0 or sOrientation == 90:
        # float64
        imOutput = np.pad(imInput[0], sExtendPx, mode='edge')
    elif sOrientation == 45 or sOrientation == 135:
        if sOrientation == 135:
            imInput = np.flip(imInput, axis=1)
            imOutput = np.flip(imOutput, axis=1)
        for k in range(-sHeight+1, sWidth-1):
            imDiag1D = np.diag(imInput[0], k=k)
            imPaddedDiag1D = np.pad(imDiag1D, sExtendPx, mode='edge')
            if k > 0:
                np.fill_diagonal(imOutput[0][:, k:], imPaddedDiag1D)
            else:
                np.fill_diagonal(imOutput[0][-k:, :], imPaddedDiag1D)
        if sOrientation == 135:
            imOutput = np.flip(imOutput, axis=1)
    return imOutput


def GaborFilter(hmDoG, sOrientation, chrGaborType, sSigma, sGain=1, sAspect=0):
    # パラメータの取得===========================================================================
    sFrame = hmDoG.shape[0]
    sHeight = hmDoG.shape[1]
    sWidth = hmDoG.shape[2]

    vSigma = [sSigma, sSigma + 2]
    vSigma_tmp = vSigma.copy()
    if sOrientation == 45 or sOrientation == 135:
        key = "aslant"
        if vSigma[0] == 2:
            vSigma[0] -= 1
        elif vSigma[0] == 3:
            vSigma[0] -= 2
        elif vSigma[0] == 4:
            vSigma[0] -= 3
            # vSigma[1] -= 1

        if sAspect > 0:
            vSigma[1] += sAspect
        elif sAspect < 0:
            if vSigma[1] + sAspect * 1.5 < vSigma[0]:
                vSigma[0] = vSigma[1]
            else:
                vSigma[1] += int(sAspect * 1.5)

    else:
        key = "straight"
        if vSigma[0] == 1:
            # vSigma[0] += 0
            vSigma[1] += 1
        # elif vSigma[0] == 4:
        # vSigma[0] -= 2
        else:
            vSigma[1] += 1

        if sAspect > 0:
            vSigma[1] += sAspect
        elif sAspect < 0:
            if vSigma[1] + sAspect < vSigma[0]:
                vSigma[0] = vSigma[1]
            else:
                vSigma[1] += sAspect

    # print(sOrientation, chrGaborType, sSigma, [
    #       i for i in range(vSigma[0], vSigma[1])])

    # f = open("./functions/parameter_gauss_ver2.yml", "r")
    # parameters = yaml.load(f, Loader=yaml.SafeLoader)
    Shift = parameters[key][chrGaborType][
        str(vSigma_tmp[0]) + "_" + str(vSigma_tmp[1])
    ]["Shift"]
    vCoeff = parameters[key][chrGaborType][
        str(vSigma_tmp[0]) + "_" + str(vSigma_tmp[1])
    ]["Coeff"]
    sPhase_Coff = parameters[key][chrGaborType]["sPhase_Coff"]

    _, _, _, _, _, Shift_Max = Shift
    # ======================================================================================
    # ======================================================================================
    # DoG画像を拡張
    hmDoG_cp = Extend_Image(hmDoG, Shift_Max, sOrientation)  # float64

    if sOrientation == 0:
        sOrientation = 90
        sR_shift = 0
        sWeight_axis = 2
    elif sOrientation == 45:
        sR_shift = 1
        sWeight_axis = 2
    elif sOrientation == 90:
        sOrientation = 0
        sR_shift = 0
        sWeight_axis = 1
    elif sOrientation == 135:
        sR_shift = -1
        sWeight_axis = 2

    hmH_Gabor_b = vCoeff[0] * hmDoG_cp * sGain  # float64

    for i in range(6):
        # float64
        tmp = np.roll(hmDoG_cp, -Shift[i] * sR_shift, axis=1)
        hmH_Gabor_b += (
            sPhase_Coff
            * vCoeff[i + 1]
            * sGain
            * np.roll(tmp, -Shift[i], axis=sWeight_axis)
        )
        tmp = np.roll(hmDoG_cp, Shift[i] * sR_shift, axis=1)
        hmH_Gabor_b += vCoeff[i + 1] * sGain * \
            np.roll(tmp, Shift[i], axis=sWeight_axis)

    # 元サイズに戻す
    hmH_Gabor = hmH_Gabor_b[:, Shift_Max:-
                            Shift_Max, Shift_Max:-Shift_Max]  # float64
    # ======================================================================================
    # =======================================================================================
    sV_Shift_Max = 2 ** (int(vSigma_tmp[1] / 2))
    # float64
    hmTempFLOAT = np.zeros(
        (
            sFrame,
            (vSigma[1]) * 2 + 1,
            sHeight + sV_Shift_Max * 2,
            sWidth + sV_Shift_Max * 2,
        ),
        dtype=np_dtype,
    )
    # float64
    hmTempFLOAT[:, (vSigma[0]) * 2, :, :] = Extend_Image(
        hmH_Gabor, sV_Shift_Max, sOrientation, -1
    )

    # =========================================================================================
    # ==========================================================================================
    # フィルタ重み設定
    sWeight00 = 10
    sWeight01 = 3
    sWeight10 = 8
    sWeight11 = 4

    # ##シグマの設定値だけまわす (仮に5で0~4)# +1
    for sFilter in range(vSigma[0], vSigma[1]):
        sBaseNumber = int(sFilter / 2) + 1  # ##112233…

        # ##σに合わせた重みの設定
        sSizeOfFilter = 2 ** sBaseNumber
        vWeight = np.zeros(sSizeOfFilter + 1, dtype=np_dtype)  # float64

        if sFilter % 2 == 0:  # ##sFilterで分岐 0,2,4,…偶のとき
            # float64
            vWeight[0] = sWeight01
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight00
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight01
        else:  # ##sFilterで分岐 1,3,…奇のとき
            # float64
            vWeight[0] = sWeight11
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight10
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight11

        sShift = 2 ** (sBaseNumber - 1)
        # 0→2→4→6→
        Center = hmTempFLOAT[:, sFilter * 2, :, :] * \
            vWeight[(2 ** (sBaseNumber - 1))]  # float64

        if sOrientation == 0:
            vShift = [(0, -sShift), (0, sShift)]
        elif sOrientation == 90:
            vShift = [(-sShift, 0), (sShift, 0)]
        elif sOrientation == 45:
            vShift = [(-sShift, sShift), (sShift, -sShift)]
        elif sOrientation == 135:
            vShift = [(sShift, sShift), (-sShift, -sShift)]

        tmp = Center.copy()  # float64
        for count, (shift_ax1, shift_ax2) in enumerate(vShift):
            # float64
            Periphery = (
                hmTempFLOAT[:, sFilter * 2, :, :] *
                vWeight[sShift * 2 * (1 - count)]
            )
            Periphery = np.roll(Periphery, shift_ax1, axis=1)
            Periphery = np.roll(Periphery, shift_ax2, axis=2)
            Periphery = Periphery.transpose(1, 0, 2)
            if shift_ax1 > 0:
                Periphery[0:shift_ax1, :,
                          :] = Periphery[shift_ax1, :, :]  # float64
            elif shift_ax1 < 0:
                Periphery[shift_ax1:, :,
                          :] = Periphery[shift_ax1 - 1, :, :]  # float64
            # float64
            Periphery = Periphery.transpose(1, 0, 2)
            Periphery = Periphery.transpose(2, 1, 0)
            if shift_ax2 > 0:
                Periphery[0:shift_ax2, :,
                          :] = Periphery[shift_ax2, :, :]  # float64
            elif shift_ax2 < 0:
                Periphery[shift_ax2:, :,
                          :] = Periphery[shift_ax2 - 1, :, :]  # float64
            Periphery = Periphery.transpose(2, 1, 0)  # float64

            tmp += Periphery

        hmTempFLOAT[:, sFilter * 2 + 2, :, :] = tmp / 16

    return hmTempFLOAT[
        :,
        (vSigma[1]) * 2,
        sV_Shift_Max:-sV_Shift_Max,
        sV_Shift_Max:-sV_Shift_Max,
    ]


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import time

    n = 100
    sExtendPixel = 32
    sOrientation = 135
    sOrthogonal = 1
    imInput = cv2.imread("./functions/sample.png", cv2.IMREAD_GRAYSCALE)

    elapsedTimeSum = [0] * 2
    for i in range(n):
        start = time.time()
        imOutput1 = Extend_Image(
            imInput[np.newaxis, :, :], sExtendPixel, sOrientation)
        elapsedTimeSum[0] += time.time() - start
        start = time.time()
        imOutput2 = Extend_Image_new(
            imInput[np.newaxis, :, :], sExtendPixel, sOrientation)
        elapsedTimeSum[1] += time.time() - start
        if i == 0:
            print(np.all(imOutput1 == imOutput2))

    print("Elapsed time (original):", elapsedTimeSum[0] / n)
    print("Elapsed time (new):", elapsedTimeSum[1] / n)

    if (elapsedTimeSum[0] / n) < (elapsedTimeSum[1] / n):
        print(
            f"Original is faster ({elapsedTimeSum[1] / n - elapsedTimeSum[0] / n})")
    else:
        print(
            f"New is faster({elapsedTimeSum[0] / n - elapsedTimeSum[1] / n})")

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(imOutput1.squeeze(), cmap="gray")
    ax1.set_title("Original")
    ax2.imshow(imOutput2.squeeze(), cmap="gray")
    ax2.set_title("New")
    plt.show()
