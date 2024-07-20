import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from . import planning_utils


class SimulatorBridge:
    def __init__(self, args):
        self.cv_bridge = CvBridge()
        self.rgb_measurement = None
        self.depth_measurement = None
        self.semantic_measurment = None
        self.rgb_noise = args.rgb_noise
        self.depth_noise = args.depth_noise

        self.get_simulator_camera_info()

        self.pose_pub = rospy.Publisher(
            "/set_camera_pose", Pose, queue_size=1, latch=True
        )

        self.rgb_sub = rospy.Subscriber("/rgbd/image", Image, self.update_rgb)
        self.depth_sub = rospy.Subscriber("/rgbd/depth_image", Image, self.update_depth)
        self.semantic_sub = rospy.Subscriber(
            "/semantic/labels_map", Image, self.update_semantic
        )

    def get_simulator_camera_info(self):
        camera_info_raw = rospy.wait_for_message(f"/rgbd/camera_info", CameraInfo)
        K = camera_info_raw.K  # intrinsic matrix
        H = int(camera_info_raw.height)  # image height
        W = int(camera_info_raw.width)  # image width
        self.camera_info = {
            "image_resolution": [H, W],
            "c": [K[2], K[5]],
            "focal": [K[0], K[4]],
        }

    def move_camera(self, pose):
        translation = pose[:3, -1]
        quaternion = planning_utils.rotation_2_quaternion(pose[:3, :3])

        camera_pose_msg = Pose()
        camera_pose_msg.position.x = translation[0]
        camera_pose_msg.position.y = translation[1]
        camera_pose_msg.position.z = translation[2]
        camera_pose_msg.orientation.x = quaternion[0]
        camera_pose_msg.orientation.y = quaternion[1]
        camera_pose_msg.orientation.z = quaternion[2]
        camera_pose_msg.orientation.w = quaternion[3]

        self.pose_pub.publish(camera_pose_msg)

    def update_rgb(self, data):
        self.rgb_measurement = data

    def update_depth(self, data):
        self.depth_measurement = data

    def update_semantic(self, data):
        self.semantic_measurement = data

    def get_measurement(self):
        rgb = self.cv_bridge.imgmsg_to_cv2(self.rgb_measurement, "rgb8")
        rgb = np.array(rgb, dtype=float)

        depth = self.cv_bridge.imgmsg_to_cv2(self.depth_measurement, "32FC1")
        depth = np.array(depth, dtype=float)

        if self.rgb_noise != 0:
            rgb_noise = np.random.normal(0.0, self.rgb_noise, rgb.shape)
            rgb += rgb_noise

        if self.depth_noise != 0:
            depth_noise = np.random.normal(0.0, self.depth_noise, depth.shape)
            depth += depth_noise

        semantic = self.cv_bridge.imgmsg_to_cv2(self.semantic_measurement, "mono8")
        semantic = np.array(semantic)
        return rgb, depth, semantic
