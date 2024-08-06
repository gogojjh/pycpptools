import open3d as o3d
import numpy as np

def read_rgbd_images(color_path, depth_path):
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
    return rgbd_image

def compute_odometry(source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init, jacobian):
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init, jacobian, o3d.pipelines.odometry.OdometryOption())
    return success, trans

def visualize_odometry(target_pcd, source_rgbd_image, pinhole_camera_intrinsic, trans):
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
    source_pcd.transform(trans)
    o3d.visualization.draw_geometries([target_pcd, source_pcd], zoom=0.48, front=[0.0999, -0.1787, -0.9788], lookat=[0.0345, -0.0937, 1.8033], up=[-0.0067, -0.9838, 0.1790])

if __name__ == "__main__":
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(redwood_rgbd.camera_intrinsic_path)
    print(pinhole_camera_intrinsic.intrinsic_matrix)

    source_rgbd_image = read_rgbd_images(redwood_rgbd.color_paths[0], redwood_rgbd.depth_paths[0])
    target_rgbd_image = read_rgbd_images(redwood_rgbd.color_paths[4], redwood_rgbd.depth_paths[4])
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)

    odo_init = np.identity(4)

    # success_color_term, trans_color_term = compute_odometry(source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init, 
    #                                                         o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm())
    # if success_color_term:
    #     print("Using RGB-D Odometry")
    #     print(trans_color_term)
    #     source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         source_rgbd_image, pinhole_camera_intrinsic)
    #     source_pcd_color_term.transform(trans_color_term)
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term],
    #                                     zoom=0.48,
    #                                     front=[0.0999, -0.1787, -0.9788],
    #                                     lookat=[0.0345, -0.0937, 1.8033],
    #                                     up=[-0.0067, -0.9838, 0.1790])

    success_hybrid_term, trans_hybrid_term = compute_odometry(source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init, 
                                                              o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())        
    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term],
                                        zoom=0.48,
                                        front=[0.0999, -0.1787, -0.9788],
                                        lookat=[0.0345, -0.0937, 1.8033],
                                        up=[-0.0067, -0.9838, 0.1790])