import os
import shutil
import numpy as np
import re

def samplefile(dir1, dir2):
    # Ensure the destination directory exists
    os.makedirs(dir2, exist_ok=True)
    # List all files in the source directory
    files = sorted(os.listdir(dir1))
    # Iterate over files and copy every 50th file
    for i in range(0, len(files), 50):
        file_path = os.path.join(dir1, files[i * 3])
        if os.path.isfile(file_path):  # Ensure it's a file and not a directory
            shutil.copy(file_path, dir2)
        print(file_path)
        
        # file_path = os.path.join(dir1, files[i * 3 + 1])
        # if os.path.isfile(file_path):  # Ensure it's a file and not a directory
        #     shutil.copy(file_path, dir2)
        # print(file_path)
        
        # file_path = os.path.join(dir1, files[i * 3 + 2])
        # if os.path.isfile(file_path):  # Ensure it's a file and not a directory
        #     shutil.copy(file_path, dir2)
        # print(file_path)

    print(f"Copied every 50th file from {dir1} to {dir2}.")

def rearrangefile(dir):
    os.makedirs(os.path.join(dir, 'new_seq1'), exist_ok=True)

    poses = np.loadtxt(os.path.join(dir, 'poses.txt'), dtype=object)
    intrinsics = np.loadtxt(os.path.join(dir, 'intrinsics.txt'), dtype=object)
    
    new_poses = np.empty((1, 8), dtype=object)
    new_poses[0, :] = poses[0, :]
    new_intrinsics = np.empty((1, 7), dtype=object)
    new_intrinsics[0, :] = intrinsics[0, :]

    new_id = 0
    img_filenames = os.listdir(os.path.join(dir, 'seq1'))
    img_filenames.sort()
    for img_filename in img_filenames:
        if 'jpg' in img_filename:
            match = re.search(r'\d+', img_filename)
            if match:
                id = int(match.group())
            else:
                continue

            rgb_img_filename = img_filename
            depth_img_filename = img_filename.replace('jpg', 'zed.png')

            new_img_filename = f'frame_{new_id:05d}.jpg'
            new_img_path = os.path.join(dir, 'new_seq1', new_img_filename)
            shutil.copy(os.path.join(dir,'seq1', rgb_img_filename), new_img_path)

            new_depth_img_filename = f'frame_{new_id:05d}.zed.png'
            new_depth_img_path = os.path.join(dir, 'new_seq1', new_depth_img_filename)
            shutil.copy(os.path.join(dir,'seq1', depth_img_filename), new_depth_img_path)

            vec = poses[id + 1, :].reshape(1, 8)
            vec[0, 0] = f'seq1/frame_{new_id:05d}.jpg'
            new_poses = np.vstack((new_poses, vec))

            vec = intrinsics[id + 1, :].reshape(1, 7)
            vec[0, 0] = f'seq1/frame_{new_id:05d}.jpg'
            new_intrinsics = np.vstack((new_intrinsics, vec))

            print(f'Copied image id {id} to {new_id}')
            new_id += 1
    np.savetxt(os.path.join(dir, 'new_intrinsics.txt'), new_intrinsics, fmt='%s %s %s %s %s %s %s')
    np.savetxt(os.path.join(dir, 'new_poses.txt'), new_poses, fmt='%s %s %s %s %s %s %s %s')

if __name__ == "__main__":
    # Define the source and destination directories
    # dir1 = '/Rocket_ssd/dataset/data_anymal/data_generation_20240731/out_general/seq'
    # dir2 = '/Rocket_ssd/dataset/data_anymal/data_generation_20240731/out_general/seq_sample'
    # samplefile(dir1, dir2)

    dir = '/Rocket_ssd/dataset/data_anymal/data_generation_20240731/test/s00009'
    rearrangefile(dir)