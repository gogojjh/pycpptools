from matplotlib import pylab, colors
import os
import sys
from tools_color_setting import PALLETE

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    import matplotlib
    matplotlib.use('Agg')  # set the backend before importing pyplot

    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib import pylab, colors

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import init, Fore
import plot_utils as pu

init(autoreset=True)
rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 8})
rc('text', usetex=True)
FORMAT = '.pdf'
params = {'axes.titlesize': 10, 'legend.fontsize': 8, 'legend.numpoints': 1}
pylab.rcParams.update(params)

import bisect
class StampedPoses:
    def __init__(self):
        self.data = []  # List to store (time, pose) tuples in ascending order of time

    def __len__(self):
        return len(self.data)

    def get_item(self, idx):
        if idx < 0 or idx >= len(self.data):
            return None
        return idx, self.data[idx]

    def time_exists(self, query_time):
        idx = bisect.bisect_left(self.data, (query_time,))
        if idx != len(self.data) and self.data[idx][0] == query_time:
            return True
        return False

    def add(self, time, pose):
        """
        :param time: timestamp
        :param pose: gtsam.Pose3, gtsam.Pose2, numpy array (np.ndarray)
        """        
        if self.time_exists(time):
            bisect.insort(self.data, (time + 1e-6, pose))
        else:
            bisect.insort(self.data, (time, pose))

    def find_closest(self, query_time):
        if not self.data:
            return None, None

        # Find the position where the query_time would be inserted
        idx = bisect.bisect_left(self.data, (query_time,))

        # Check the closest time
        if idx == 0:
            return idx, self.data[0]
        if idx == len(self.data):
            return idx, self.data[-1]

        before = self.data[idx - 1]
        after = self.data[idx]

        # Compare which one is closer
        if query_time - before[0] <= after[0] - query_time:
            return idx-1, before
        else:
            return idx, after
        
    def to_numpy(self):
        if not isinstance(self.data[0][1], np.ndarray):
            print('Not support conversion to numpy for non-numpy poses')
            return None
        time_numpy = np.array([data[0] for data in self.data])
        pose_numpy = np.array([data[1] for data in self.data])
        combined_numpy = np.hstack((time_numpy.reshape(-1, 1), pose_numpy))
        return combined_numpy

def convert_tum_to_stamped_pose(tum_poses):
    stamped_poses = StampedPoses()
    for pose in tum_poses:
        stamped_poses.add(pose[0], pose[1:])
    return stamped_poses

def plot_trajectories(args, sgt, slo, svl, spf):
    sgt_data, slo_data, svl_data, spf_data = sgt.to_numpy(), slo.to_numpy(), svl.to_numpy(), spf.to_numpy()
    spf_data_w_vloc_trig = np.zeros((spf_data.shape[0], spf_data.shape[1]))
    for i in range(len(spf_data)):
        timestamp = spf_data[i, 0]
        idx, closest = svl.find_closest(timestamp)
        if closest is None: continue
        if abs(closest[0] - timestamp) < 0.01:
            n_check = 5
            if i + n_check >= len(spf_data): continue
            dis = 0
            for j in range(i + 1, i + n_check):
                dis += np.linalg.norm(spf_data[j, 1:4] - spf_data[j-1, 1:4])
            if dis / n_check > 0.05:
                spf_data_w_vloc_trig = np.vstack((spf_data_w_vloc_trig, spf_data[i, :])) # trigger large correction

    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111,
                        aspect='equal',
                        xlabel='X [m]',
                        ylabel='Y [m]')
    ax.set_title('Estimated Trajectories on xxx dataset')
    
    pu.plot_trajectory_top(ax, sgt_data[:, 1:4], PALLETE[0], 'Groundtruth', 1.0, linestyle='-.', linewidth=3.0)
    pu.plot_trajectory_top(ax, slo_data[:, 1:4], PALLETE[3], 'DepthReg', 0.85, linestyle='-', linewidth=3.0)
    pu.plot_trajectory_top(ax, svl_data[:, 1:4], PALLETE[1], 'VLoc', 0.85, linestyle='--', linewidth=3.0)
    pu.plot_trajectory_top(ax, spf_data[:, 1:4], PALLETE[2], 'PoseFusion', 0.85, linestyle='-', linewidth=3.0)
    pu.plot_trajectory_top_spot(ax, sgt_data[0, 1:4].reshape(1, -1), PALLETE[4], 'Start Point', 1.0, marker='*', markersize=7.0, zorder=10)
    pu.plot_trajectory_top_spot(ax, sgt_data[-1, 1:4].reshape(1, -1), PALLETE[4], 'End Point', 1.0, marker='^', markersize=5.5, zorder=10)
    pu.plot_trajectory_top_spot(ax, spf_data_w_vloc_trig[:, 1:4], PALLETE[1], None, 1.0, marker='+', markersize=9.0, zorder=0)
    # pu.plot_trajectory_top_spot(ax, spf_data_w_vloc_trig[:, 1:4], PALLETE[1], None, 1.0, marker='x', markersize=9.0, zorder=0)

    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(args.out_figure + FORMAT, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print('Save figure to {}'.format(args.out_figure + FORMAT))

def main():
    parser = argparse.ArgumentParser(description='Visualize the odometry trajectories')
    parser.add_argument('--odom_gt', type=str, default=None, help='Path to gt.txt file')
    parser.add_argument('--odom_local_odom', type=str, default=None, help='Path to local_odom.txt file')
    parser.add_argument('--odom_vloc', type=str, default=None, help='Path to vloc.txt file')
    parser.add_argument('--odom_pose_fusion', type=str, default=None, help='Path to pose_fusion.txt file')
    parser.add_argument('--out_figure', type=str, default=None, help='Path to save the output')
    args = parser.parse_args()

    if args.odom_gt is None or args.odom_local_odom is None or args.odom_vloc is None or args.odom_pose_fusion is None:
        print('Please provide all the paths to the odometry files')
        return
    
    poses_gt = np.loadtxt(args.odom_gt)
    stamped_poses_gt = convert_tum_to_stamped_pose(poses_gt)
    poses_local_odom = np.loadtxt(args.odom_local_odom)
    stamped_poses_local_odom = convert_tum_to_stamped_pose(poses_local_odom)
    poses_vloc = np.loadtxt(args.odom_vloc)
    stamped_poses_vloc = convert_tum_to_stamped_pose(poses_vloc)
    poses_fusion = np.loadtxt(args.odom_pose_fusion)
    stamped_poses_fusion = convert_tum_to_stamped_pose(poses_fusion)
    plot_trajectories(args, stamped_poses_gt, stamped_poses_local_odom, stamped_poses_vloc, stamped_poses_fusion)

if __name__ == '__main__':
    main()