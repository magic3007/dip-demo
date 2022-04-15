#!/usr/bin/env python3
# %%
import sys
import cv2
import os
from pathlib import Path
if True:
    sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
    import HistEq as he
    import dip

# %%
gray_paths = [
    "../../2022-ImagesSet/histeq1.jpg",
    "../../2022-ImagesSet/histeq2.jpg",
    "../../2022-ImagesSet/histeq3.jpg",
    "../../2022-ImagesSet/histeq4.jpg",
]
gray_dirs = [
    "../../2022-ImagesSet/BSD68/",
]
color_paths = [
    "../../2022-ImagesSet/histeqColor.jpg",
]
color_dirs = [
    "../../2022-ImagesSet/CBSD68/",
]

gray_result_dir = "./HistEq_test/gray"
color_result_dir = "./HistEq_test/color"

# %%
for gray_dir in gray_dirs:
    for f in os.listdir(gray_dir):
        f = os.path.join(gray_dir, f)
        if os.path.isfile(f) and os.path.splitext(f)[-1] in {'.png', '.jpg'}:
            gray_paths.append(f)

for color_dir in color_dirs:
    for f in os.listdir(color_dir):
        f = os.path.join(color_dir, f)
        if os.path.isfile(f) and os.path.splitext(f)[-1] in {'.png', '.jpg'}:
            color_paths.append(f)


if not os.path.isdir(gray_result_dir):
    os.makedirs(gray_result_dir)

if not os.path.isdir(color_result_dir):
    os.makedirs(color_result_dir)

#%%
for gray_path in gray_paths:
    try:
        img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        img_eq = dip.hist_eq(img)
        file_name = Path(gray_path).stem
        ext_name = os.path.splitext(gray_path)[-1]
        result_path = os.path.join(gray_result_dir, file_name + "_eq" + ext_name)
        cv2.imwrite(result_path, img_eq)
    except Exception as e:
        print("Image {} failed: {}".format(gray_path, e))
# %%
for color_path in color_paths:
    try:
        img = cv2.imread(color_path, cv2.IMREAD_COLOR)
        _, img_BGR_eq, img_HLS_eql, img_HLS = he.MyMethod(img, 'color')
        file_name = Path(color_path).stem
        ext_name = os.path.splitext(color_path)[-1]
        cv2.imwrite(os.path.join(color_result_dir, file_name + "_bgr_eq" + ext_name), img_BGR_eq)
        cv2.imwrite(os.path.join(color_result_dir, file_name + "_hls_eq" + ext_name), img_HLS_eql)
        cv2.imwrite(os.path.join(color_result_dir, file_name + "_hls" + ext_name), img_HLS)
    except Exception as e:
        print("Image {} failed: {}".format(color_path, e))

# %%
