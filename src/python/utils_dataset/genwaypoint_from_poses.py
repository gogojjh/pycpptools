import argparse
import numpy as np

def poses_to_waypoints(poses, sample=1):
    waypoints = poses[::sample, 1:4]
    return waypoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process waypoints and poses.')
    parser.add_argument('--input_pose', type=str, required=True, help='Path to the poses file')
    parser.add_argument('--output_waypoint', type=str, required=True, help='Path to the waypoints file')
    args = parser.parse_args()

    poses = np.loadtxt(args.input_pose)
    print(f"Pose shape: {poses.shape}")
    waypoints = poses_to_waypoints(poses, sample=100)

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {waypoints.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]

    with open(args.output_waypoint, 'w') as f:
        for line in lines:
            f.write(line + '\n')
        for waypoint in waypoints:
            f.write(f"{waypoint[0]:.5f} {waypoint[1]:.5f} {waypoint[2]:.5f}\n")