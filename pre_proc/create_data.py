from __future__ import absolute_import, division, print_function

import os
import io
import glob
import base64
import json
import argparse
#from multiprocessing import Process

from tqdm import tqdm
import numpy as np
from PIL import Image

#from pre_proc.equi_to_cube import e2c
from equi_to_cube import e2c


def ndarr2b64utf8(img):
    img_t = Image.fromarray(img)
    with io.BytesIO() as output:
        img_t.save(output, format="PNG")
        content = output.getvalue()
        b64_barr = base64.b64encode(content)
        b_string = b64_barr.decode('utf-8')
        return b_string

def b64utf82ndarr(b_string):
    b64_barr = b_string.encode('utf-8')
    content = base64.b64decode(b64_barr)
    img = Image.open(io.BytesIO(content))
    inp_np = np.asarray(img)
    return inp_np


def create_name_pair(inp_paths):
    pair_dict = dict()
    for inp_path in inp_paths:
        base_name, _ = os.path.splitext(os.path.basename(inp_path))
        pair_dict[base_name] = inp_path
    return pair_dict


def save_erp_image(inp_path, output_dir, mask_path=None):
    img = Image.open(inp_path, "r")
    inp_np = np.asarray(img)

    base_name, _ = os.path.splitext(os.path.basename(inp_path))

    out_dict = dict()
    out_dict['type'] = 'erp'
    out_dict['f_name'] = inp_path
    out_dict['mask_flag'] = False

    if mask_path is not None:
        img_mask = Image.open(mask_path, "r")
        inp_np_mask = np.asarray(img_mask)
        out_dict['mask_flag'] = True

    erp_imgs = dict()
    b_string = ndarr2b64utf8(inp_np)
    if mask_path is None:
        erp_imgs['erp'] = [b_string]
    else:
        b_string_mask = ndarr2b64utf8(inp_np_mask)
        erp_imgs['erp'] = [b_string, b_string_mask]
    
    out_dict['imgs'] = erp_imgs
    with open(os.path.join(output_dir, base_name + ".json"), "w") as f:
        json.dump(out_dict, f)


def save_cubemap_image(inp_path, output_dir, mask_path=None, face_w=256):
    img = Image.open(inp_path, "r")
    inp_np = np.asarray(img)
    cm, cl = e2c(inp_np, face_w=face_w)

    base_name, _ = os.path.splitext(os.path.basename(inp_path))
    face_list = ['f', 'r', 'b', 'l', 't', 'd']

    out_dict = dict()
    out_dict['type'] = 'cube'
    out_dict['f_name'] = inp_path
    out_dict['mask_flag'] = False

    if mask_path is not None:
        img_mask = Image.open(mask_path, "r")
        inp_np_mask = np.asarray(img_mask)
        cm_mask, cl_mask = e2c(inp_np_mask, face_w=face_w)
        out_dict['mask_flag'] = True
    
    cube_imgs = dict()

    for idx, face in enumerate(cl):
        b_string = ndarr2b64utf8(face)
        if mask_path is None:
            cube_imgs[face_list[idx]] = [b_string]
        else:
            b_string_mask = ndarr2b64utf8(cl_mask[idx])
            cube_imgs[face_list[idx]] = [b_string, b_string_mask]
    
    out_dict['imgs'] = cube_imgs
    with open(os.path.join(output_dir, base_name + ".json"), "w") as f:
        json.dump(out_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="\
        path to input directory. (<input dir>/*.png)")
    parser.add_argument("-m", default=None, type=str, help="\
        path to mask directory. \
        The file name of mask have to be matched with the input image. \
        (<input dir>/*.png)")
    parser.add_argument("-fmt", default="cube", choices=["erp", "cube"])
    #parser.add_argument("-cores", default=8, help="number of cores for pre-processing.")
    parser.add_argument("-face_w", default=256)
    parser.add_argument("-o", default="output_dir")
    args = parser.parse_args()

    inp_paths = glob.glob(os.path.join(args.i, "*.png"))
    out_dir = args.o
    mask_dir = args.m
    face_w = args.face_w
    #cores = args.cores
    is_mask_pair = True if mask_dir is not None else False
    

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


    if is_mask_pair:
        mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))

        # crate pair
        name_path_pair = create_name_pair(inp_paths)
        name_path_pair_mask = create_name_pair(mask_paths)

        pair = dict()
        for k, v in name_path_pair.items():
            if k in name_path_pair_mask:
                pair[k] = {'img': v, 'mask': name_path_pair_mask[k]}

        if args.fmt == "erp":
            print("erp with mask pair")
            for k, v in tqdm(pair.items(), desc="img mask pair"):
                save_erp_image(v['img'], out_dir, mask_path=v['mask'])
        else:
            print("cubemap with mask pair")
            out_json = dict()
            for k, v in tqdm(pair.items(), desc="img mask pair"):
                save_cubemap_image(v['img'], out_dir, mask_path=v['mask'], face_w=face_w)
    else:
        if args.fmt == "erp":
            print("erp wo/ mask pair")
            for inp_path in tqdm(inp_paths, desc="imgs"):
                save_erp_image(inp_path, out_dir)
        else:
            print("cubemap wo/ mask pair")
            out_json = dict()
            for inp_path in tqdm(inp_paths, desc="imgs"):
                save_cubemap_image(inp_path, out_dir, face_w=face_w)








