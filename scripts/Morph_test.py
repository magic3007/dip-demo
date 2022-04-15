#!/usr/bin/env python3
# %%
import sys
import cv2
import os
from pathlib import Path
if True:
    sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
    import Morph as mo

# %%
gray_paths = [
    "../../2022-ImagesSet/word_bw.bmp",
]
gray_dirs = [
    "../../2022-ImagesSet/BSD68/",
]

gray_result_dir = "./Morph_test/gray"

# %%
for gray_dir in gray_dirs:
    for f in os.listdir(gray_dir):
        f = os.path.join(gray_dir, f)
        if os.path.isfile(f) and os.path.splitext(f)[-1] in {'.png', '.jpg'}:
            gray_paths.append(f)

if not os.path.isdir(gray_result_dir):
    os.makedirs(gray_result_dir)

# %%
for gray_path in gray_paths:
    try:
        img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        _, img_open, img_close, img_erose, img_dilate = mo.MyMethod(
            img, mode='gray')
        file_name = Path(gray_path).stem
        ext_name = os.path.splitext(gray_path)[-1]
        cv2.imwrite(os.path.join(gray_result_dir,
                                 file_name + "_open" + ext_name), img_open)
        cv2.imwrite(os.path.join(gray_result_dir, file_name +
                                 "_close" + ext_name), img_close)
        cv2.imwrite(os.path.join(gray_result_dir, file_name +
                                 "_erose" + ext_name), img_erose)
        cv2.imwrite(os.path.join(gray_result_dir, file_name +
                                 "_dilate" + ext_name), img_dilate)
    except Exception as e:
        print("Image {} failed: {}".format(gray_path, e))
# %%
