echo "create data test"
python create_data.py \
-i '/home/sw/360VR+Inpainting/code/erp_inpainting/data/temp' \
-o '/home/sw/360VR+Inpainting/code/erp_inpainting/data/out_temp' \
-fmt 'cube'

# echo "data load test"
# python data_load_test.py \
# -i '/home/sw/360VR+Inpainting/code/erp_inpainting/data/out_temp' \
# -o '/home/sw/360VR+Inpainting/code/erp_inpainting/data/out_temp'