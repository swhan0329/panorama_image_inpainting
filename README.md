# erp_inpainting

## pre-processing
1. create data as json format
```bash
    python create_data.py \
        -i 'path to input directory' \ #(<input dir>/*.png)
        -o 'path to output directory' \
        -m 'path to mask directory' \ # opt value.
        -face_w 256 \ #width and height of face of cube
        -fmt 'cube' # erp or cube
```
2. create random mask
```bash
    python create_random_mask.py \
        -i 'path to input directory' \ #(<input dir>/*.json)
        -mask_shape_w 1920 \ # width of mask
        -mask_shape_h 1080 \ # height of mask
        -o 'path to output directory'
```
