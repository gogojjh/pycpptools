import numpy as np

def read_poses(input_pose_file):
	poses_dict = dict()
	with open(input_pose_file, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, qw, qx, qy, tx, ty, tz
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				data = [float(p) for p in line.strip().split(' ')] # Each row: qw, qx, qy, tx, ty, tz
			poses_dict[img_name] = np.array(data)
	return poses_dict
	
def read_timestamps(input_time_file):
	times_dict = dict()
	with open(input_time_file, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				times_dict[img_name] = float(line.strip().split(' ')[1]) # Each row: image_name, timestamp
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				times_dict[img_name] = float(line.strip().split(' ')[1])
	return times_dict