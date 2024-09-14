# Aachen Day-Night-v1.1 images to a maximum size of 1600 pixels 
# Cambridge landmarks images to 1024 pixels. 
# 7Scenes, we retain the original resolution of 640x480 pixels.

python batch_matching.py \
--dataset_name 7scenes \
--dataset_path /Titan/dataset/7scenes/ \
--pair_path /Titan/dataset/airhloc/7scenes/ \
--k_retrieve 30

python batch_matching.py \
--dataset_name cambridge \
--dataset_path /Titan/dataset/cambridge/ \
--pair_path /Titan/dataset/airhloc/cambridge/ \
--k_retrieve 30
