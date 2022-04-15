#!/usr/bin/env python3
# %%
import sys
import cv2
import os
from pathlib import Path
if True:
    sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
    import Laplace as la

# %%
gray_paths = [
    "../../2022-ImagesSet/moon.tif",
]
gray_dirs = [
    "../../2022-ImagesSet/BSD68/",
]

gray_result_dir = "./Laplace_test/gray"

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
        _, img_laplace, img_sharp = la.MyMethod(img, mode='gray')
        file_name = Path(gray_path).stem
        ext_name = os.path.splitext(gray_path)[-1]
        cv2.imwrite(os.path.join(gray_result_dir, file_name +
                                 "_laplace" + ext_name), img_laplace)
        cv2.imwrite(os.path.join(gray_result_dir, file_name +
                                 "_sharp" + ext_name), img_sharp)
    except Exception as e:
        print("Image {} failed: {}".format(gray_path, e))
# %%
