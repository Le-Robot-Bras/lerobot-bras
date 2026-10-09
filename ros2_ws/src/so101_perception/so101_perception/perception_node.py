#!/usr/bin/env python3
"""perception_node.py — golden ball detection and 3D registration.

TODO:
  - Subscribe to /external_cam/image_raw and /external_cam/camera_info
  - Publish /object_position (5-10 Hz)
  - Segment the ball in HSV (yellow/gold)
  - Find the contour center (u, v)
  - Project pixel -> 3D via K^{-1} and the camera->world TF
  - Intersect with the z=0 plane (table) to estimate the depth
  - Get cam_K / cam_T from the driver (params or CameraInfo)
"""
import threading

import cv2
import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo, Image


# -- HSV ranges for the red cube -------------------------------------------------
# Measured on the MuJoCo scene (OpenCV HSV: H in 0-179): the cube is H ~168.
# The low-hue red range (H 0-10) only catches the brown drop box, so it is unused.
# Tune according to the lighting of the scene (and of the real camera).
HSV_MIN = np.array([160, 120, 70], dtype=np.uint8)
HSV_MAX = np.array([180, 255, 255], dtype=np.uint8)

# The cube has holes: close them so the mask is one blob, ignore specks below this.
MIN_BLOB_AREA_PX = 30
CLOSE_KERNEL = np.ones((5, 5), np.uint8)

# Height (m) of the tracked point: centre of the 5 cm cube resting on the table.
TRACKED_Z = 0.025

# cam_T comes from MuJoCo (x right, y up, z backwards); the pinhole ray below is
# OpenCV (x right, y down, z forward). This flips y and z.
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


class PerceptionNode(rclpy.node.Node):
    """Detects the golden ball and publishes its 3D position."""

    def __init__(self):
        super().__init__("so101_perception")

    def _cb_camera_info(self, msg):

        raise NotImplementedError("TO DO")

    def _cb_image(self, msg):
        raise NotImplementedError("TO DO")

    def _detect_object(self, cv_image):
        """Finds the red cube in a BGR image.

        Returns ((u, v), mask): the pixel center of the cube and the binary mask,
        or (None, mask) if nothing big enough is visible.
        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        blob = max(contours, key=cv2.contourArea)
        if cv2.contourArea(blob) < MIN_BLOB_AREA_PX:
            return None, mask

        m = cv2.moments(blob)
        return (m["m10"] / m["m00"], m["m01"] / m["m00"]), mask

    def _project_to_3d(self, u, v):
        """Pixel (u, v) -> 3D point in the robot frame, on the plane z = TRACKED_Z.

        Returns None if the ray is (almost) horizontal or points behind the camera.
        """
        x = (u - self.cam_K[0, 2]) / self.cam_K[0, 0]
        y = (v - self.cam_K[1, 2]) / self.cam_K[1, 1]

        direction_cam = np.array([x, y, 1.0])
        direction_world = self.cam_T[:3, :3] @ MUJOCO_TO_OPENCV @ direction_cam
        origin_world = self.cam_T[:3, 3]

        if abs(direction_world[2]) < 1e-6:
            return None
        t = (TRACKED_Z - origin_world[2]) / direction_world[2]
        if t < 0:
            return None

        return origin_world + t * direction_world

    def _publish_result(self):
        raise NotImplementedError("TO DO")

    def _fetch_cam_params(self):
        raise NotImplementedError("TO DO")


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
