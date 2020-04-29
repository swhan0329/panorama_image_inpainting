"""
Random mask generator

This code is based on https://github.com/MathiasGruber/PConv-Keras.git
"""
import os
import sys
import cv2
import glob
import argparse
import numpy as np
from random import randint, seed

def make_save_random_mask(inp_path, mask_shape, output_dir=None):

    mask = create_random_mask(mask_shape)
    base_name, _ = os.path.splitext(os.path.basename(inp_path))
    cv2.imwrite(output_dir+"/"+base_name+".png", mask)

def create_random_mask(shape):
    """
    Generates a random irregular mask with lines, circles and elipses
    """

    mask = np.zeros((shape[1], shape[0], 3), np.uint8)
    mask_h, mask_w, _ = mask.shape

    # Set size scale
    size = int((mask_w + mask_h) * 0.01)
    if mask_w < 64 or mask_h < 64:
        raise Exception("Width and Height of mask must be at least 64!")
        
    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, mask_w), randint(1, mask_w)
        y1, y2 = randint(1, mask_h), randint(1, mask_h)
        thickness = randint(3, size)
        cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)
            
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, mask_w), randint(1, mask_h)
        radius = randint(3, size)
        cv2.circle(mask,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, mask_w), randint(1, mask_h)
        s1, s2 = randint(1, mask_w), randint(1, mask_h)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(mask, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)

    mask = mask * 255.

    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="\
        path to equi-rectangular image dataset directory.")
    parser.add_argument("-mask_shape_w", type=int, help="\
        mask shape width", default=1920)
    parser.add_argument("-mask_shape_h", type=int, help="\
        mask shape height", default=1080)
    parser.add_argument("-o", default="./mask")
    args = parser.parse_args()

    inp_paths = glob.glob(os.path.join(args.i, "*.png"))
    mask_shape = (args.mask_shape_w, args.mask_shape_h)
    out_dir = args.o

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    i = 0
    for inp_path in inp_paths:
        if i % 1000 ==0:
            print(i,"/",len(inp_paths))
        make_save_random_mask(inp_path=inp_path, mask_shape=mask_shape,output_dir=out_dir)
        i = i + 1