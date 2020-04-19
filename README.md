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
2. create tf_records if you use tensorflow.
```bash
    python create_tf_record.py
```
