import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix

def convert_mapfree_to_tum(input_file, output_file):
	# if not os.path.exists(input_file):
	# 	print(f"Poses not found in {input_file}")
	# 	return None

	# poses = dict()
	# with open(input_file, 'r') as f:
	# 	for line_id, line in enumerate(f):
	# 		if line.startswith('#'): 
	# 			continue
	# 		if line.startswith('seq'):
	# 			img_name = line.strip().split(' ')[0]
	# 			data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, qw, qx, qy, tx, ty, tz
	# 		else:
	# 			img_name = f'seq/{line_id:06}.color.jpg'
	# 			data = [float(p) for p in line.strip().split(' ')] # Each row: qw, qx, qy, tx, ty, tz
	# 		poses[img_name] = np.array(data)
    pass

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
                Tw2c = np.linalg.inv(Tw2c)
                trans, quat = convert_matrix_to_vec(Tw2c, 'wxyz')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pose files')
    parser.add_argument('--input', type=str, help='Input pose file')
    parser.add_argument('--output', type=str, help='Output pose file')
    parser.add_argument('--input_pose_type', type=str, help='Input pose file type: g2o, mapfree, tum')
    parser.add_argument('--output_pose_type', type=str, help='Output pose file type: g2o, mapfree, tum')
    args = parser.parse_args()

    if args.input_pose_type == 'g2o' and args.output_pose_type == 'mapfree':
        convert_g2o_to_mapfree(args.input, args.output)
    elif args.input_pose_type == 'g2o' and args.output_pose_type == 'tum':
        convert_g2o_to_tum(args.input, args.output)
