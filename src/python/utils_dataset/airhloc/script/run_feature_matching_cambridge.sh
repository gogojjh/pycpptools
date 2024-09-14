# Aachen Day-Night-v1.1 images to a maximum size of 1600 pixels 
# Cambridge landmarks images to 1024 pixels. 
# 7Scenes, we retain the original resolution of 640x480 pixels.
python batch_feature_matching.py \
--dataset_name cambridge \
--dataset_path /Rocket_ssd/dataset/cambridge/ \
--pair_path /Rocket_ssd/dataset/airhloc/cambridge/ \
--k_retrieve 1