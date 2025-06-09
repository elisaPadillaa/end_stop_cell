import numpy as np
import yaml


def loadGaborlikeWeight(sOri, sSigma, cGaborType, cWeightPath: str = "./functions/parameter_gauss_ver2.yml"):
    with open(cWeightPath, "r") as f:
        parameters = yaml.load(f, Loader=yaml.SafeLoader)
    if sOri == 0 or sOri == 90:
        key = "straight"
    elif sOri == 45 or sOri == 135:
        key = "aslant"
    else:
        raise ValueError("Orientation must be 0, 45, 90, or 135.")
    if cGaborType not in ["Even", "Odd"]:
        raise ValueError("Gabor type must be 'Even' or 'Odd'.")
    vShift = parameters[key][cGaborType][f"{sSigma}_{sSigma+2}"]["Shift"]
    vCoeff = parameters[key][cGaborType][f"{sSigma}_{sSigma+2}"]["Coeff"]
    sPhaseCoff = parameters[key][cGaborType]["sPhase_Coff"]
    return {"vShift": vShift, "vCoeff": vCoeff, "sPhaseCoff": sPhaseCoff}


def makeGaborlikeDiscreteWeight(vShift, vCoeff, sPhaseCoff, sWidth: int = 31):
    sWeightHalfIdx = sWidth // 2
    print(sWeightHalfIdx)
    vWeightHalf = np.zeros(sWeightHalfIdx + 1)
    vWeightHalf[0] = vCoeff[0]
    for sShift, sCoeff in zip(vShift, vCoeff[1:]):
        vWeightHalf[sShift] = sCoeff
    return np.concatenate([np.flip(vWeightHalf[1:])*sPhaseCoff, vWeightHalf[:]])


# axis = 0: horizontal, 1: vertical
def conv2DWith1D(imInput: np.ndarray, vKernel: np.ndarray, sAxis: int = 0) -> np.ndarray:
    sKernelSize = len(vKernel)
    vStrides = imInput.strides
    if sAxis == 0:
        vOutputShape = (
            imInput.shape[0], imInput.shape[1] - sKernelSize + 1, sKernelSize)
        vOutputStrides = (vStrides[0], vStrides[1], vStrides[1])
    else:
        vOutputShape = (
            imInput.shape[0] - sKernelSize + 1, imInput.shape[1], sKernelSize)
        vOutputStrides = (vStrides[0], vStrides[1], vStrides[0])
    vWindows = np.lib.stride_tricks.as_strided(
        imInput, vOutputShape,
        strides=vOutputStrides)
    return np.tensordot(vWindows, vKernel, axes=((2), (0)))


def PositiveRectify(imInput: np.ndarray) -> np.ndarray:
    return np.maximum(0, imInput)


def NegativeRectify(imInput: np.ndarray) -> np.ndarray:
    return np.minimum(0, imInput)


def ClipRange(imInput: np.ndarray, sMin: int = 0, sMax: int = 255) -> np.ndarray:
    return np.maximum(sMin, np.minimum(sMax, imInput))
