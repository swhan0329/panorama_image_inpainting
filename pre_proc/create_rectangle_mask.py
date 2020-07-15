"""
Rectangular mask generator

created by Seo Woo Han
2020.05.07
"""
import os
import sys
import cv2
import glob
import argparse
import numpy as np
from random import randint, seed


def make_save_random_mask(inp_path, mask_shape, output_dir=None):
    mask = create_rectangle_mask(mask_shape)
    base_name, _ = os.path.splitext(os.path.basename(inp_path))
    cv2.imwrite(output_dir + "/" + base_name + ".png", mask)


def create_rectangle_mask(shape):
    """
    Generates a random irregular mask with lines, circles and elipses
    """

    mask = np.zeros((shape[1], shape[0], 3), np.uint8)
    mask_h, mask_w, _ = mask.shape

    make_mask = True
    # Draw random rectangle
    while make_mask:
        x1, x2 = randint(1, mask_w), randint(1, mask_w)
        y1, y2 = randint(50, mask_h-50), randint(50, mask_h-50)
        if abs(y2 - y1) > mask_h/8:
            make_mask = True
        elif abs(y2 - y1) < mask_h/10:
            make_mask = True
        elif abs(x2 - x1) < mask_w/10:
            make_mask = True
        elif abs(x2 - x1) > mask_w/8:
            make_mask = True
        else:
            make_mask = False
    cv2.rectangle(mask, (x1, y1), (x2, y2), (1, 1, 1), -1)

    mask = mask * 255.

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="\
        path to equi-rectangular image dataset directory.")
    parser.add_argument("-mask_shape_w", type=int, help="\
        mask shape width", default=1024)
    parser.add_argument("-mask_shape_h", type=int, help="\
        mask shape height", default=512)
    parser.add_argument("-o", default="./mask")
    args = parser.parse_args()

    inp_paths = glob.glob(os.path.join(args.i, "*.png"))
    mask_shape = (args.mask_shape_w, args.mask_shape_h)
    out_dir = args.o

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    i = 0
    for inp_path in inp_paths:
        if i % 1000 == 0:
            print(i, "/", len(inp_paths))
        make_save_random_mask(inp_path=inp_path, mask_shape=mask_shape, output_dir=out_dir)
        i = i + 1
