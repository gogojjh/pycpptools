import numpy as np

if __name__ == '__main__':
    poses = np.loadtxt('/Rocket_ssd/dataset/data_topo_loc/vloc_eval_data/groundtruth/traj/EDJbREhghzL_seq2.txt')
    
    dt = abs(poses[0, 0] - poses[-1, 0])
    
    total_dis = 0
    for i in range(1, poses.shape[0]):
        total_dis += np.linalg.norm(poses[i, 1:3] - poses[i-1, 1:3])
    
    print(f'Time: {dt:.3f}s, Total distance: {total_dis:.3f}m')
