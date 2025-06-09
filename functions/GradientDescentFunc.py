import numpy as np


# Gabor関数 シフト幅は23行目からのslistShiftValue参照
def Odd(vGL, slistShiftValue, slistCoeff, sPattern="normalize"):
    vOdd = np.linspace(-46, 47, 187)
    vOdd[:] = 0

    for i in range(len(slistShiftValue)):
        vOdd += (slistCoeff[i] * np.roll(vGL, shift=slistShiftValue[i], axis=0)) - (
            slistCoeff[i] * np.roll(vGL, shift=-slistShiftValue[i], axis=0)
        )

    if sPattern == "normalize":
        vOdd = vOdd / np.max(vOdd)

    return vOdd


def Odd2(vDoG_lab, slistHalfShiftValue, slistCoeff):
    vOdd_2 = np.linspace(-46, 47, 94)
    vOdd_2[:] = 0

    # Odd関数からスライド幅*0.5
    for i in range(len(slistHalfShiftValue)):
        vOdd_2 += (
            slistCoeff[i] * np.roll(vDoG_lab, shift=slistHalfShiftValue[i], axis=0)
        ) - (slistCoeff[i] * np.roll(vDoG_lab, shift=-slistHalfShiftValue[i], axis=0))

    print("シフト数：", slistHalfShiftValue)

    return vOdd_2


def Even(vGL, slistShiftValue, slistCoeff, sPattern="normalize"):
    vEven = np.linspace(-46, 47, 187)
    vEven[:] = 0

    vEven += slistCoeff[0] * vGL

    for i in range(1, len(slistShiftValue)):
        vEven += (slistCoeff[i] * np.roll(vGL, shift=slistShiftValue[i], axis=0)) + (
            slistCoeff[i] * np.roll(vGL, shift=-slistShiftValue[i], axis=0)
        )

    if sPattern == "normalize":
        vEven = vEven / np.max(vEven)

    return vEven


def Even2(vDoG_lab, slistHalfShiftValue, slistCoeff):
    vEven2 = np.linspace(-46, 47, 94)
    vEven2[:] = 0

    vEven2 += slistCoeff[0] * vDoG_lab

    for i in range(1, len(slistHalfShiftValue)):
        vEven2 += (
            slistCoeff[i] * np.roll(vDoG_lab, shift=slistHalfShiftValue[i], axis=0)
        ) + (slistCoeff[i] * np.roll(vDoG_lab, shift=-slistHalfShiftValue[i], axis=0))

    print("シフト数：", slistHalfShiftValue)

    return vEven2


# 誤差関数
def Errorrate(vGabor, vEven, sDC):
    # 二乗誤差値算出 (できるだけ小さくしたい)
    Errorvalue = np.sum((vGabor - vEven) ** 2)

    # DC成分の誤差(1にしたい)
    if sDC == 1:
        # Errorvalue = 0
        # sTargetPositive = 0
        # sTargetNegative = 0
        # sPositive = 0
        # sNegative = 0
        # for i in range(187):
        #     if vGabor[i] > 0:
        #         sTargetPositive += vGabor[i]
        #     else:
        #         sTargetNegative -= vGabor[i]

        #     if vEven[i] > 0:
        #         sPositive += vEven[i]
        #     else:
        #         sNegative -= vEven[i]
        # Errorvalue += ((sTargetPositive - sPositive) ** 2) ** 0.5
        # Errorvalue += ((sTargetNegative - sNegative) ** 2) ** 0.5

        sPositive = 0
        sNegative = 0
        for i in range(187):
            if vEven[i] > 0:
                sPositive += vEven[i]
            else:
                sNegative -= vEven[i]
        Errorvalue += ((sPositive - sNegative) ** 2) ** 0.5

    return Errorvalue


# 微分用
def Numerical_diff_Odd(vGabor, vGL, slistShiftValue, slistCoeff, sNumber, sDC):
    h = 1e-4  # 1e-4
    tmplistCoeff = slistCoeff.copy()
    print(tmplistCoeff)
    tmplistCoeff[sNumber] += h

    # ###各パラメータ学習##################################################################
    Error = (
        Errorrate(
            vGabor,
            Odd(vGL, slistShiftValue, tmplistCoeff),
            sDC,
        )
        - Errorrate(
            vGabor,
            Odd(vGL, slistShiftValue, slistCoeff),
            sDC,
        )
    ) / h

    return Error


def Numerical_diff_Even(vGabor, vGL, slistShiftValue, slistCoeff, sNumber, sDC):
    h = 1e-4  # 1e-4
    tmplistCoeff = slistCoeff.copy()
    tmplistCoeff[sNumber] += h

    # ###各パラメータ学習##################################################################
    Error = (
        Errorrate(
            vGabor,
            Even(vGL, slistShiftValue, tmplistCoeff),
            sDC,
        )
        - Errorrate(
            vGabor,
            Even(vGL, slistShiftValue, slistCoeff),
            sDC,
        )
    ) / h

    return Error
