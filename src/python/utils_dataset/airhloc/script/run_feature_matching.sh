# Aachen Day-Night-v1.1 images to a maximum size of 1600 pixels 
# Cambridge landmarks images to 1024 pixels. 
# 7Scenes, we retain the original resolution of 640x480 pixels.

python batch_feature_matching.py \
--dataset_name 7scenes \
--dataset_path /Titan/dataset/7scenes/ \
--pair_path /Titan/dataset/airhloc/7scenes/ \
--k_retrieve 30 \
--debug

python batch_feature_matching.py \
--dataset_name cambridge \
--dataset_path /Titan/dataset/cambridge/ \
--pair_path /Titan/dataset/airhloc/cambridge/ \
--k_retrieve 30 \
--debug

python batch_feature_matching.py \
--dataset_name aachen_v1_1 \
--dataset_path /Titan/dataset/aachen_v1_1/ \
--pair_path /Titan/dataset/airhloc/aachen_v1_1/ \
--k_retrieve 30 \
--debug
