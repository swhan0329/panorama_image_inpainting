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

from create_data import b64utf82ndarr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required="True", type=str, help="\
        path to input directory. (<input dir>/*.json)")
    parser.add_argument("-o", default="output_dir")
    args = parser.parse_args()

    inp_paths = glob.glob(os.path.join(args.i, "*.json"))
    out_dir = args.o

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


    for inp_path in tqdm(inp_paths, desc="decode"):
        base_name, _ = os.path.splitext(os.path.basename(inp_path))
        with open(inp_path, "r") as f:
            in_json = json.load(f)
            for k, v in in_json['pano'].items():
                img_dnarr = b64utf82ndarr(v[0])
                img = Image.fromarray(img_dnarr)
                img.save(os.path.join(out_dir, "{}_{}.png".format(base_name,k)))
            for k, v in in_json['cube'].items():
                img_dnarr = b64utf82ndarr(v[0])
                img = Image.fromarray(img_dnarr)
                img.save(os.path.join(out_dir, "{}_{}.png".format(base_name, k)))








