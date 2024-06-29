import numpy as np
import sensor_msgs.point_cloud2 as pc2

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

##### Sensor data
def convert_cvimg_to_rosimg(image, encoding, header, compressed=False):
		bridge = CvBridge()
		img_msg = None
		image_16UC1 = image.astype(np.uint16)[:, :, 0] if encoding == 'mono16' else image

		if not compressed:
				img_msg = bridge.cv2_to_imgmsg(image_16UC1, encoding=encoding, header=header)
		else:
				img_msg = bridge.cv2_to_compressed_imgmsg(image_16UC1)
				img_msg.header = header

		return img_msg

def convert_rosimg_to_cvimg(img_msg, compressed=False):
		bridge = CvBridge()
		cv_image = None
		if compressed:
			cv_image = bridge.compressedimgmsg_to_cv2(img_msg, "passthrough")
		else:
			cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
		return cv_image

def convert_pts_to_rospts(header, pts, intensity=None, color=None, label=None):
		msg = PointCloud2()
		msg.header = header
		# Define the fields
		fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='rgb', offset=16, datatype=pc2.PointField.UINT32, count=1),
				pc2.PointField(name='label', offset=20, datatype=pc2.PointField.FLOAT32, count=1)
		]
		msg.fields.extend(fields)
		# Prepare points array
		dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4"), ("rgba", "u4"), ("label", "f4")])
		pointsColor = np.zeros(pts.shape[0], dtype=dtype)
		pointsColor["x"], pointsColor["y"], pointsColor["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
		if intensity is not None:
				pointsColor["intensity"] = intensity
		if color is not None:
				pointsColor["rgba"] = color.view('uint32')
		if label is not None:
				pointsColor["label"] = label
		msg.data = pointsColor.tobytes()
		msg.point_step = 24
		msg.height = 1
		msg.width = pts.shape[0]
		msg.row_step = msg.point_step * msg.width
		msg.is_bigendian = False
		return msg

##### Odometry
def convert_rosodom_to_vec(odom):
		"""trans: np.array([x y z]); quat: np.array([qw qx qy qz]) """
		trans = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
		quat = np.array([odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z])
		return trans, quat

def convert_vec_to_rosodom(trans, quat, header, child_frame_id):
		"""trans: np.array([x y z]); quat: np.array([qw qx qy qz]) """
		odom = Odometry()
		odom.header = header
		odom.child_frame_id = child_frame_id
		odom.pose.pose.position.x = trans[0]
		odom.pose.pose.position.y = trans[1]
		odom.pose.pose.position.z = trans[2]
		odom.pose.pose.orientation.x = quat[1]
		odom.pose.pose.orientation.y = quat[2]
		odom.pose.pose.orientation.z = quat[3]
		odom.pose.pose.orientation.w = quat[0]
		return odom

def convert_vec_to_rosodom(tx, ty, tz, qx, qy, qz, qw, header, child_frame_id):
		odom = Odometry()
		odom.header = header
		odom.child_frame_id = child_frame_id
		odom.pose.pose.position.x = tx
		odom.pose.pose.position.y = ty
		odom.pose.pose.position.z = tz
		odom.pose.pose.orientation.x = qx
		odom.pose.pose.orientation.y = qy
		odom.pose.pose.orientation.z = qz
		odom.pose.pose.orientation.w = qw
		return odom

def convert_vec_to_rospose(tx, ty, tz, qx, qy, qz, qw, header):
		pose = PoseStamped()
		pose.header = header
		pose.pose.position.x = tx
		pose.pose.position.y = ty
		pose.pose.position.z = tz
		pose.pose.orientation.x = qx
		pose.pose.orientation.y = qy
		pose.pose.orientation.z = qz
		pose.pose.orientation.w = qw
		return pose

def convert_vec_to_rostf(tx, ty, tz, qx, qy, qz, qw, header, child_frame_id):
		tf_msg = TFMessage()
		tf_data = TransformStamped()
		tf_data.header = header
		tf_data.child_frame_id = child_frame_id
		tf_data.transform.translation.x = tx
		tf_data.transform.translation.y = ty
		tf_data.transform.translation.z = tz
		tf_data.transform.rotation.x = qx
		tf_data.transform.rotation.y = qy
		tf_data.transform.rotation.z = qz
		tf_data.transform.rotation.w = qw
		tf_msg.transforms.append(tf_data)
		return tf_msg

##### Visualization message
def get_ros_marker_camera_frustum(header, position, orientation, length=10.0):
		marker = Marker()
		marker.header = header
		marker.ns = "frustum"
		marker.type = Marker.LINE_LIST
		marker.action = Marker.ADD
		marker.id = 0
		marker.pose.position.x = position[0]
		marker.pose.position.y = position[1]
		marker.pose.position.z = position[2]
		marker.pose.orientation.x = orientation[0]
		marker.pose.orientation.y = orientation[1]
		marker.pose.orientation.z = orientation[2]
		marker.pose.orientation.w = orientation[3]
		marker.scale.x = 0.25  # width
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		marker.color.a = 1.0
		# Define frustum points
		points = [
				[-length/2, -length/2, length/2], [length/2, -length/2, length/2],
				[-length/2, -length/2, length/2], [-length/2, length/2, length/2],
				[length/2, length/2, length/2], [-length/2, length/2, length/2],
				[length/2, length/2, length/2], [length/2, -length/2, length/2],
				[-length/2, -length/2, length/2], [0.0, 0.0, 0.0],
				[-length/2, -length/2, length/2], [0.0, 0.0, 0.0],
				[length/2, length/2, length/2], [0.0, 0.0, 0.0],
				[length/2, length/2, length/2], [0.0, 0.0, 0.0]
		]
		marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
		return marker
