import os
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../image_matching_models"))
from matching import get_matcher
import time
import numpy as np
import cv2

device = 'cuda' # 'cpu'
matcher = get_matcher('superglue', device=device)  # superpoint+superglue
img_size = 512

def get_pair_dict(pair_path):
	file_dict = {}
	with open(pair_path, 'r') as file:
		for line in file:
			line = line.strip()
			seq2, _ = line.split(' ')
			file_dict[seq2] = []
	with open(pair_path, 'r') as file:
		for line in file:
			line = line.strip()
			seq2, seq1 = line.split(' ')
			file_dict[seq2].append(seq1)
	return file_dict

def list_folders(path):
	return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def perform_matching(matcher, path_img1, path_img2, resize):
	str1, str2 = path_img1.split('/')[-1], path_img2.split('/')[-1]
	# print(f'Matching between {str1} and {str2}')
	img0 = matcher.load_image(path_img1, resize=resize)
	img1 = matcher.load_image(path_img2, resize=resize)
	start_time = time.time()
	result = matcher(img0, img1)
	comp_time = time.time() - start_time
	# num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['inliers0'], result['inliers1']
	# print(f'Number of inliers: {num_inliers}')
	return comp_time * 1000

def main():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--dataset_name', type=str, help='Dataset name: 7scenes, aachen, cambridge')
	parser.add_argument('--dataset_path', type=str, help='Path_to_dataset')
	parser.add_argument('--pair_path', type=str, help='Path_to_pairs.txt')
	parser.add_argument('--k_retrieve', type=int, default=1, help='Number of items to retrieve: 1, 2, 3, ..., 30')
	parser.add_argument('--debug', action='store_true', help='Debug mode')
	args = parser.parse_args()

	scenes = [d for d in os.listdir(args.pair_path) if os.path.isdir(os.path.join(args.pair_path, d))]
	scenes.sort()
	print(f'Scenes in dataset: {scenes}')

	# Load pairs
	total_time_k_pairs = {}
	for i in range(31): total_time_k_pairs[i] = []

	for scene in scenes:
		pair_path = os.path.join(args.pair_path, scene, 'pairs-query-netvlad30.txt')
		print(f'Pair path: {pair_path}')
		scene_path = os.path.join(args.dataset_path, scene)
		print(f'Scene path: {scene_path}')
		if os.path.exists(pair_path):
			pair_dict = get_pair_dict(pair_path)
			print(f'Number of Reference: {len(pair_dict.keys())}')

		# Get the size of the image
		img_path = os.path.join(scene_path, pair_dict[list(pair_dict.keys())[0]][0])
		img = cv2.imread(img_path)
		img_width, img_height = img.shape[1], img.shape[0]
		print(f'Image size: {img_width} x {img_height}')
		if args.dataset_name == '7scenes':
			resize = (img_height, img_width) # HxW
		elif args.dataset_name == 'aachen':
			resize = (int(img_height * img_width / 1024), 1024)
		elif args.dataset_name == 'cambridge':
			resize = (int(img_height * img_width / 1600), 1600)

		# Perform matching on every k pairs w.r.t. every reference image
		for seq2 in pair_dict.keys():
			ref_img_path = os.path.join(scene_path, seq2)
			seq1_list = pair_dict[seq2]
			# Perform matching w.r.t. a specific reference image
			comp_times = 0
			for i in range(args.k_retrieve):
				tar_img_path = os.path.join(scene_path, seq1_list[i])
				comp_times += perform_matching(matcher, ref_img_path, tar_img_path, resize)
				total_time_k_pairs[i + 1].append(comp_times)
			if args.debug: break
		if args.debug: break

	for k, v in total_time_k_pairs.items():
		if len(v) == 0: continue
		if k != 1 and k != 2 and k != 3 and k !=4 and k != 5 and k != 10 and k != 20 and k != 30: continue
		v = v[1:] # Remove the first element (outlier)
		avg_time = sum(v) / len(v)
		print(f'K={k}: {avg_time}ms')
		np.savetxt(os.path.join(args.pair_path, f'total_matching_time_{k}.txt'), 
				   np.array(v), fmt='%.5f')
		np.savetxt(os.path.join(args.pair_path, f'average_matching_time_{k}.txt'), 
				   np.array([avg_time]), fmt='%.5f')
		print(f'Averaged matching time: {avg_time}ms')

if __name__ == "__main__":
	main()