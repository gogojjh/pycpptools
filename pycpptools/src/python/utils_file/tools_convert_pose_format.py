import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix
from utils_dataset.map_free_reloc.data_loader import read_poses, read_timestamps

def convert_g2o_to_mapfree(input_file, output_file):
	poses = np.empty((0, 8), dtype=object)
	with open(input_file, 'r') as f:
		for line in f:
			if line.startswith('VERTEX_SE3:QUAT'):
				data = line.strip().split(' ')
				node_id = int(data[1])
				tx, ty, tz = map(float, data[2:5])
				qx, qy, qz, qw = map(float, data[5:])
				Tw2c = convert_vec_to_matrix(np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]), 'xyzw')
				Tc2w = np.linalg.inv(Tw2c)
				trans, quat = convert_matrix_to_vec(Tc2w, 'wxyz')
				img_name = f'seq/{node_id:06}.color.jpg'
				vec = np.empty((1, 8), dtype=object)
				vec[0, 0], vec[0, 1:5], vec[0, 5:] = img_name, quat, trans
				poses = np.vstack((poses, vec))
	np.savetxt(output_file, poses, fmt='%s ' + '%.6f ' * (poses.shape[1] - 1))
	print(f"Finish converting {input_file} to {output_file}")

def convert_g2o_to_tum(input_file, output_file):
	# poses = np.empty((0, 8), dtype=object)
	# with open(input_file, 'r') as f:
	#     for line in f:
	#         if line.startswith('VERTEX_SE3:QUAT'):
	#             data = line.strip().split(' ')
	#             node_id = int(data[1])
	#             tx, ty, tz = map(float, data[2:5])
	#             qx, qy, qz, qw = map(float, data[5:])
	#             Tw2c = convert_vec_to_matrix(np.array([qx, qy, qz, qw], np.array([tx, ty, tz])), 'xyzw')
	#             Tw2c = np.linalg.inv(Tw2c)
	#             trans, quat = convert_matrix_to_vec(Tw2c, 'wxyz')
	#             img_name = f'seq/{node_id:06}.color.jpg'
	#             vec = np.empty((1, 8), dtype=object)
	#             vec[0, 0], vec[0, 1:4], vec[0, 4:] = img_name, quat, trans
	#             poses = np.vstack((poses, vec))
	# np.savetxt(output_file, poses, fmt='%s ' + '%.6f' * (poses.shape[1] - 1))
	pass

def convert_mapfree_to_tum(input_pose_file, input_time_file, output_file):
	poses_dict = read_poses(input_pose_file)
	times_dict = read_timestamps(input_time_file)
	poses = np.zeros((0, 8), dtype=np.float64)
	for key in poses_dict.keys():
		if key in times_dict.keys():
			trans, quat = poses_dict[key][4:], poses_dict[key][0:4]
			Tc2w = convert_vec_to_matrix(trans, quat, 'wxyz')
			Tw2c = np.linalg.inv(Tc2w)
			trans, quat = convert_matrix_to_vec(Tw2c, 'xyzw')
			vec = np.zeros((1, 8), dtype=np.float64)
			vec[0, 0], vec[0, 1:4], vec[0, 4:] = times_dict[key], trans, quat
			poses = np.vstack((poses, vec))
	np.savetxt(output_file, poses, fmt='%.6f ' * poses.shape[1])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert pose files')
	parser.add_argument('--input_pose_file', type=str, help='Input pose file')
	parser.add_argument('--input_time_file', type=str, help='Input time file')
	parser.add_argument('--output_pose_file', type=str, help='Output pose file')

	parser.add_argument('--input_pose_type', type=str, help='Input pose file type: g2o, mapfree, tum')
	parser.add_argument('--output_pose_type', type=str, help='Output pose file type: g2o, mapfree, tum')
	args = parser.parse_args()

	if args.input_pose_type == 'g2o' and args.output_pose_type == 'mapfree':
		convert_g2o_to_mapfree(args.input_pose_file, args.output_pose_file)
	elif args.input_pose_type == 'g2o' and args.output_pose_type == 'tum':
		convert_g2o_to_tum(args.input_pose_file, args.output_pose_file)
	elif args.input_pose_type == 'mapfree' and args.output_pose_type == 'tum':
		convert_mapfree_to_tum(args.input_pose_file, args.input_time_file, args.output_pose_file)
