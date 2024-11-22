import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils_ros.tools_ros_msg_conversion import convert_pts_to_rospts, convert_rospts_to_pts
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix
import argparse

"""Initialization"""
def get_T_map_odom(args):
    if 'anymal_lab_upstair_20240726' in args.input_odom_path:
        T_ini = np.array([-0.768525481224,  0.614494562149, -0.178227856755, -4.841787815094,
                        -0.567707598209, -0.783390223980, -0.252997875214,  2.182228803635,
                        -0.295087754726, -0.093254014850,  0.950908482075, -1.624489426613,
                        0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]).reshape(4, 4)
        quat_opt = np.array([-0.15136214, -0.03806806, 0.92908281, -0.33532888])
        trans_opt = np.array([-5.11131202, 2.24829277, -0.43381437])
    elif 'anymal_lab_upstair_20240722_0' in args.input_odom_path:
        T_ini = np.array([-0.750229,  0.630141, -0.200197, -5.839335,
                        -0.595961, -0.775612, -0.207983,  3.704699,
                        -0.286334, -0.036725,  0.957426, -1.205188,
                        0.000000,  0.000000,  0.000000,  1.000000]).reshape(4, 4)
        quat_opt = np.array([-0.15522467, -0.02857271,  0.92712253, -0.33990103])
        trans_opt = np.array([-6.08708462, 3.78580191, -0.43436535])
    elif 'anymal_lab_upstair_20240722_1' in args.input_odom_path:
        T_ini = np.array([-0.445381, 0.884470, -0.139101, -6.629053,
                        -0.825982, -0.465846, -0.317398, 4.481498,
                        -0.345528, -0.026468, 0.938035, -1.348357,
                        0.000000, 0.000000, 0.000000, 1.000000]).reshape(4, 4)
        quat_opt = np.array([-0.14449567, -0.05527009, 0.83518574, -0.52776036])
        trans_opt = np.array([-6.8743248, 4.74509394, -0.43336767])
    elif 'anymal_lab_upstair_20240722_2' in args.input_odom_path:
        T_ini = np.array([-0.647574, 0.737584, -0.191358, -6.644090,
                        -0.682026, -0.673030, -0.286130, 4.379685,
                        -0.339834, -0.054779, 0.938889, -0.486524,
                        0.000000, 0.000000, 0.000000, 1.000000]).reshape(4, 4)
        quat_opt = np.array([-0.1479165, -0.04143926, 0.89792602, -0.41247104])
        trans_opt = np.array([-7.0990504, 4.49911226, -0.44070144])
    elif 'anymal_ops_msg_20240722_0' in args.input_odom_path:
        T_ini = np.array([0.714914, -0.662595, 0.223307, -6.758866,
                        0.649282, 0.747616, 0.139652, 5.002020,
                        -0.259480, 0.045150, 0.964692, -1.727461,
                        0.000000, 0.000000, 0.000000, 1.000000]).reshape(4, 4)
        quat_opt = np.array([-0.04252619,   0.14530231,  0.35316788,  0.92322869])
        trans_opt = np.array([-6.83056196,  5.00640565, -0.50530686])
    elif 'anymal_ops_msg_20240726' in args.input_odom_path:
        T_ini = np.array([0.757572, -0.606689, 0.240860, -6.709962,
                        0.585685, 0.794683, 0.159539, 5.117188,
                        -0.288198, 0.020206, 0.957358, 0.707165,
                        0.000000, 0.000000, 0.000000, 1.000000]).reshape(4, 4)
        quat_opt = np.array([-0.02938097, 0.15062856, 0.3208862,  0.93460143])
        trans_opt = np.array([-6.54708896, 5.15270756, -0.55152046])
    # return T_opt
    T_opt = convert_vec_to_matrix(trans_opt, quat_opt)
    return T_opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake_lidar_localization: convert local odomery to global.')
    parser.add_argument('--input_odom_path', help='path_to_odometry', default='/tmp/odometry.txt')
    parser.add_argument('--output_global_odom_path', help='path_to_global_odometry', default='/tmp/odometry_global.txt')
    args = parser.parse_args()

    T_map_odom = get_T_map_odom(args)

    global_odom_list = []
    local_odom = np.loadtxt(args.input_odom_path)
    print('Number of local odometry: ', len(local_odom))
    for odom in local_odom:
        timestamp = odom[0]
        trans = odom[1:4]
        quat = odom[4:]
        T_global = T_map_odom @ convert_vec_to_matrix(trans, quat)
        trans_global, quat_global = convert_matrix_to_vec(T_global)
        global_odom_list.append([timestamp, *trans_global, *quat_global])
    np.savetxt(args.output_global_odom_path, np.array(global_odom_list))



