# PIGAN
This is an official PyTorch code of a 360-degree panoramic image inpainting network v2 paper.
paper: https://arxiv.org/abs/2010.16003

## Title
PIGAN: A 360-degree Panoramic Image Inpainting Network using a Cube Map

## Abstract
Inpainting has been continuously studied in the field of computer vision. As artificial intelligence technology developed, deep learning technology was introduced in inpainting research, helping to improve performance. Currently, the input target of an inpainting algorithm using deep learning has been studied from a single image to a video. However, deep learning-based inpainting technology for panoramic images has not been actively studied. We propose a 360-degree panoramic image inpainting method using generative adversarial networks (GANs). The proposed network inputs a 360-degree equirectangular format panoramic image converts it into a cube map format, which has relatively little distortion and uses it as a training network. Since the cube map format is used, the correlation of the six sides of the cube map should be considered. Therefore, all faces of the cube map are used as input for the whole discriminative network, and each face of the cube map is used as input for the slice discriminative network to determine the authenticity of the generated image. The proposed network performed qualitatively better than existing single-image inpainting algorithms and baseline algorithms.

## Step for using this code
1. Download this repository in your local computer.
2. Run download.sh file for downloading 360-degree panoramic image dataset. (Dataset paper link: https://cgv.cs.nthu.edu.tw/projects/360SP)
```bash
    python download.sh
```
### pre-processing
2-1. create data as json format
```bash
    python create_data.py \
        -i 'path to input directory' \ #(<input dir>/*.png)
        -o 'path to output directory' \
        -m 'path to mask directory' \ # opt value.
        -face_w 256 \ #width and height of face of cube
        -fmt 'cube' # erp or cube
```
2-2. create random mask
```bash
    python create_random_mask.py \
        -i 'path to input directory' \ #(<input dir>/*.json)
        -mask_shape_w 1920 \ # width of mask
        -mask_shape_h 1080 \ # height of mask
        -o 'path to output directory'
```
3. Set some hyperparameters following your machine(e.g. The number of gpus, batch_size) in main.py.
4. Run the main.py file.
```
python main.py
```
5. After some epochs, you can find checkpoints, log in your folder.
