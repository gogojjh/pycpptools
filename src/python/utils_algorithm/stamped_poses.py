from typing import Union
import numpy as np
import bisect
import gtsam

class StampedPoses:
    def __init__(self):
        self.data = []  # List to store (time, pose) tuples in ascending order of time

    def __len__(self):
        return len(self.data)

    def get_item(self, idx):
        if idx < 0 or idx >= len(self.data):
            return None
        return idx, self.data[idx]

    def add(self, time, pose: Union[gtsam.Pose3, gtsam.Pose2, np.ndarray]):
        # Insert the (time, pose) tuple in the correct position to maintain order
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

if __name__ == "__main__":
    poses = StampedPoses()
    poses.add(5.0, np.eye(4))
    poses.add(1.0, np.eye(4))
    poses.add(3.0, np.eye(4))
    print(len(poses))

    idx, closest = poses.find_closest(0.0)
    print(idx, closest)
    idx, closest = poses.find_closest(1.0)
    print(idx, closest)
    idx, closest = poses.find_closest(1.5)
    print(idx, closest)
    idx, closest = poses.find_closest(4.0)
    print(idx, closest)
    idx, closest = poses.find_closest(5.0)
    print(idx, closest)
    idx, closest = poses.find_closest(6.0)
    print(idx, closest)