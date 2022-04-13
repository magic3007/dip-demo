import dip
import cv2
import numpy as np


def MyMethod(img, mode):
    # 增加的改进方法
    # img是要处理的BGR格式图片
    # outputdir是处理后的文件路径
    # mode是模式，‘gray’：黑白图片；‘color’是彩色图片

    if mode == 'gray':
        img_laplace = dip.laplace_transform(img)
        img_sharp = np.clip(img - img_laplace, a_min=0, a_max=255)
        img_laplace = np.clip(img_laplace, a_min=0, a_max=255)
        return img, img_laplace.astype(np.uint8), img_sharp.astype(np.uint8)
    elif mode == 'color':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_laplace = dip.laplace_transform(img_gray)
        img_sharp = np.clip(img_gray - img_laplace, a_min=0, a_max=255)
        img_laplace = np.clip(img_laplace, a_min=0, a_max=255)
        return (img,
                cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_laplace.astype(np.uint8), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_sharp.astype(np.uint8), cv2.COLOR_GRAY2BGR))
