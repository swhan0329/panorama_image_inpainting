# echo "create data test w/o mask"
# python create_data.py \
# -i '/home/sw/360VR+Inpainting/code/erp_inpainting/images/temp_image' \
# -o '/home/sw/360VR+Inpainting/code/erp_inpainting/images/out_temp_image' \
# -fmt 'cube'

#echo "create data test w/ mask"
#python create_data.py \
#-i '../images/temp_image' \
#-m '../images/temp_mask' \
#-o '../images/out_temp_image' \

echo "create data test w/ mask"
python create_data.py \
-i '../../../dataset/panoramas_dataset/train' \
-m '../../../dataset/random_mask_dataset/train' \
-o './images/json_image_train' \

# echo "data load test"
# python data_load_test.py \
# -i '/home/sw/360VR+Inpainting/code/erp_inpainting/images/out_temp_image' \
# -o '/home/sw/360VR+Inpainting/code/erp_inpainting/images/out_temp'

# echo "make random mask test"
# python create_random_mask.py \
# -i '/home/sw/360VR+Inpainting/data/panoramas_dataset/train' \
# -mask_shape_w 1920 \
# -mask_shape_h 1080 \
# -o '/home/sw/360VR+Inpainting/data/random_mask_dataset/train'