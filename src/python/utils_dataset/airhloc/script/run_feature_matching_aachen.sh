# Aachen Day-Night-v1.1 images to a maximum size of 1600 pixels 
# Cambridge landmarks images to 1024 pixels. 
# 7Scenes, we retain the original resolution of 640x480 pixels.
python batch_feature_matching.py \
--dataset_name aachen_v1_1 \
--dataset_path /Rocket_ssd/dataset/aachen_v1_1/ \
--pair_path /Rocket_ssd/dataset/airhloc/aachen_v1_1/ \
--k_retrieve 1