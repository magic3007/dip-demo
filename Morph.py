import dip
import cv2
import numpy as np

def MyMethod(img, mode):
    # 增加的改进方法
    # img是要处理的BGR格式图片
    # outputdir是处理后的文件路径
    # mode是模式，‘gray’：黑白图片；‘color’是彩色图片

    if mode == 'gray':
        img_erose = dip.erosion(img, 3)
        img_dilate = dip.dilation(img, 3)
        img_open = dip.opening(img, 3)
        img_close = dip.closing(img, 3)
        img_grad = dip.morph_grad(img, 3)
        img_tophat = dip.top_hat(img, 3)
        img_blackhat = dip.black_hat(img, 3)
        return img, img_erose, img_dilate, img_open, img_close, img_grad, img_tophat, img_blackhat
    elif mode == 'color':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_erose = dip.erosion(img_gray, 3)
        img_dilate = dip.dilation(img_gray, 3)
        img_open = dip.opening(img_gray, 3)
        img_close = dip.closing(img_gray, 3)
        img_grad = dip.morph_grad(img_gray, 3)
        img_tophat = dip.top_hat(img_gray, 3)
        img_blackhat = dip.black_hat(img_gray, 3)
        return (img,
                cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_erose, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_dilate, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_open, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_close, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_grad, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_tophat, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_blackhat, cv2.COLOR_GRAY2BGR))
