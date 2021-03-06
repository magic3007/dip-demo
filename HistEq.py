import cv2
import dip
import numpy as np


def MyMethod(img, mode):
    # 增加的改进方法
    # img是要处理的BGR格式图片
    # outputdir是处理后的文件路径
    # mode是模式，‘gray’：黑白图片；‘color’是彩色图片

    if mode == 'gray':
        # dst1 = cv2.equalizeHist(img)
        dst1 = dip.hist_eq(img)
        return dst1
    elif mode == 'color':
        img_BGR = img
        img_HLS = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HLS)
        # equalizeHist in BGR
        img_BGR_eq = img_BGR.copy()
        # img_BGR_eq[:, :, 0] = cv2.equalizeHist(img_BGR_eq[:, :, 0])
        # img_BGR_eq[:, :, 1] = cv2.equalizeHist(img_BGR_eq[:, :, 1])
        # img_BGR_eq[:, :, 2] = cv2.equalizeHist(img_BGR_eq[:, :, 2])
        for i in range(3):
            img_BGR_eq[:, :, i] = dip.hist_eq(img_BGR_eq[:, :, i])

        # equlizeHist in lightness
        img_HLS_eql = img_HLS.copy()
        # img_HLS_eql[:,:,1]=cv2.equalizeHist(img_HLS_eql[:,:,1])
        img_HLS_eql[:, :, 1] = dip.hist_eq(img_HLS_eql[:, :, 1])

        # equlizeHist in lightness ; equlizeHist and exp in saturation
        # img_HLS[:, :, 1] = cv2.equalizeHist(img_HLS[:, :, 1])
        # img_HLS[:, :, 2] = cv2.equalizeHist(img_HLS[:, :, 2])
        img_HLS[:, :, 1] = dip.hist_eq(img_HLS[:, :, 1])
        img_HLS[:, :, 2] = dip.hist_eq(img_HLS[:, :, 2])
        img_HLS[:, :, 2] = np.uint8(
            np.clip(img_HLS[:, :, 2]/255.*160+50., 0, 255))

        return img_BGR, img_BGR_eq, cv2.cvtColor(img_HLS_eql, cv2.COLOR_HLS2BGR), cv2.cvtColor(img_HLS, cv2.COLOR_HLS2BGR)
