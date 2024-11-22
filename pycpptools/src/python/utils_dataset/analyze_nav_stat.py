import argparse
import rosbag
import math
import numpy as np

def analyze_nav_stat(poses):
    """Calculate navigation path length and time."""
    start_time, end_time = 0, 0
    start_idx, end_idx = 0, 0
    for i in range(poses.shape[0]):
        dis = np.linalg.norm(poses[i, 1:3] - poses[0, 1:3])
        if dis > 0.5:
            start_time = poses[i, 0]
            start_idx = i
            break
    for i in range(poses.shape[0]-1, -1, -1):
        dis = np.linalg.norm(poses[i, 1:3] - poses[-1, 1:3])
        if dis > 0.5:
            end_time = poses[i, 0]
            end_idx = i
            break
    end_time = 1725542253.52 + 374
    nav_time = end_time - start_time
    nav_dis = 0
    for i in range(start_idx, end_idx):
        nav_dis += np.linalg.norm(poses[i, 1:3] - poses[i-1, 1:3])
    return nav_time, nav_dis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate path lengths from odometry data in ROS bag files.")
    parser.add_argument("--odom_nav_1", type=str, help="Path to the first navigation file containing gt odometry data.")
    parser.add_argument("--odom_nav_2", type=str, help="Path to the second navigation file containing gt odometry data.")
    args = parser.parse_args()

    poses_nav_ref = np.loadtxt(args.odom_nav_1)
    poses_nav_vloc = np.loadtxt(args.odom_nav_2)

    # Calculate path length
    dt = abs(poses_nav_ref[0, 0] - poses_nav_ref[-1, 0])
    total_dis = 0
    for i in range(1, poses_nav_ref.shape[0]):
        total_dis += np.linalg.norm(poses_nav_ref[i, 1:3] - poses_nav_ref[i-1, 1:3])
    print(f'Time: {dt:.3f}s, Total distance: {total_dis:.3f}m')

    # Calculate navigation path length
    nav_time_ref, nav_dis_ref = analyze_nav_stat(poses_nav_ref)
    print(f'[Ref] Navigation time: {nav_time_ref:.2f}s, Navigation distance: {nav_dis_ref:.2f}m')
    print(f'[Ref] {nav_time_ref:.2f}s, {nav_dis_ref:.2f}m')

    nav_time_vloc, nav_dis_vloc = analyze_nav_stat(poses_nav_vloc)
    print(f'[Vloc] Navigation time: {nav_time_vloc:.2f}s, Navigation distance: {nav_dis_vloc:.2f}m')
    print(f'[Vloc] {nav_time_vloc:.2f}s, {nav_dis_vloc:.2f}m')
    time_vloc_ratio = nav_time_vloc / nav_time_ref
    dis_vloc_ratio = nav_dis_vloc / nav_dis_ref
    print(f'[Vloc - Ref] Time ratio: {time_vloc_ratio:.2f}, Distance ratio: {dis_vloc_ratio:.2f}')
    print(f'[Vloc - Ref] {time_vloc_ratio:.2f}, {dis_vloc_ratio:.2f}')