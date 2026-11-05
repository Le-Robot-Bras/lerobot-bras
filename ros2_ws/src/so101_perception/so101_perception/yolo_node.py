#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        self.bridge = CvBridge()

        # Loads YOLOv8 nano (automatically downloads ~6MB weights on first run)
        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        self.get_logger().info("YOLO model loaded!")

        # Subscribe to the webcam stream from driver_node
        self.sub = self.create_subscription(
            Image, "/external_cam/image_raw", self.image_callback, 10
        )

        # Publish the image with bounding boxes drawn on it
        self.pub = self.create_publisher(Image, "/yolo/image_annotated", 10)

    def image_callback(self, msg: Image):
        # 1. ROS Image -> OpenCV BGR frame
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 2. Run YOLO inference
        results = self.model(frame, verbose=False)

        # 3. Draw boxes & labels onto the frame
        annotated_frame = results[0].plot()

        # 4. OpenCV BGR frame -> ROS Image and publish
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()