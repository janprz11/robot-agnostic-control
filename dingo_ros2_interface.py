# -*- coding: ascii -*-

import os
import sys

# ensure ROS2 Python packages are discoverable even when launched
# outside of a shell that has sourced /opt/ros/humble/setup.zsh
_ROS2_PYTHON_PATHS = [
    "/opt/ros/humble/lib/python3.10/site-packages",
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
]
for _p in _ROS2_PYTHON_PATHS:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from color_print import cprint, BLUE, GRAY
import numpy as np
from PIL import Image
import threading
import time
from typing import Any


class DingoROS2Interface:

    def __init__(self, config: dict[str, Any]) -> None:
        import rclpy
        from rclpy.node import Node

        self.config = config
        self.task_completed = False
        self.current_images: dict[str, Image.Image] = {}
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frames: dict[str, np.ndarray] = {}

        # read configurable parameters from config
        self.cmd_vel_topic = config.get("ros2_cmd_vel_topic", "/cmd_vel")
        self.linear_speed = config.get("ros2_linear_speed", 0.3)
        self.angular_speed = config.get("ros2_angular_speed", 0.3)
        self.action_duration = config.get("ros2_action_duration", 2.0)
        self.camera_topics = config.get("ros2_camera_topics", {})

        # initialize ROS2
        rclpy.init()
        self.node = rclpy.create_node("dingo_mindstorm")

        # set up velocity publisher (TwistStamped for Clearpath Dingo)
        from geometry_msgs.msg import TwistStamped

        self._twist_stamped_cls = TwistStamped
        self.vel_pub = self.node.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self._frame_id = config.get("ros2_frame_id", "base_link")
        cprint(
            f"[ROS2] Publishing velocity on {self.cmd_vel_topic}",
            color=BLUE,
            flush=True,
        )

        # set up camera subscribers
        self._setup_camera_subscribers()

        # start a spin thread so callbacks fire
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        # wait for first camera frame and populate current_images
        self._wait_for_cameras()
        self.get_camera_images()

        print("Dingo ROS2 interface initialized", flush=True)

    def _compressed_image_callback(self, msg: Any, cam_name: str) -> None:
        import cv2

        # decode compressed image buffer and convert BGR to RGB
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._frame_lock:
                self._latest_frames[cam_name] = frame

    def _make_twist_stamped(self, linear_x: float, angular_z: float) -> Any:
        msg = self._twist_stamped_cls()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        return msg

    def _raw_image_callback(self, msg: Any, cam_name: str) -> None:
        # determine dtype and channels from encoding
        encoding = msg.encoding.lower()
        if encoding in ("rgb8",):
            channels = 3
        elif encoding in ("bgr8",):
            channels = 3
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding in ("mono8",):
            channels = 1
        else:
            channels = 3  # fallback

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels
        )

        # convert BGR variants to RGB
        if "bgr" in encoding:
            import cv2

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2RGB if channels == 4 else cv2.COLOR_BGR2RGB,
            )
        elif channels == 4:
            frame = frame[:, :, :3]  # drop alpha

        with self._frame_lock:
            self._latest_frames[cam_name] = frame

    def _setup_camera_subscribers(self) -> None:
        from sensor_msgs.msg import CompressedImage as CompressedImageMsg
        from sensor_msgs.msg import Image as ImageMsg

        # create a ROS2 subscriber for each configured camera topic
        for cam_name, topic in self.camera_topics.items():
            # auto-detect compressed vs raw based on topic name
            if "compressed" in topic.lower():
                self.node.create_subscription(
                    CompressedImageMsg,
                    topic,
                    lambda msg, name=cam_name: self._compressed_image_callback(
                        msg, name
                    ),
                    10,
                )
                cprint(
                    f"[ROS2] Subscribed to compressed image: {topic} as '{cam_name}'",
                    color=GRAY,
                    flush=True,
                )
            else:
                self.node.create_subscription(
                    ImageMsg,
                    topic,
                    lambda msg, name=cam_name: self._raw_image_callback(msg, name),
                    10,
                )
                cprint(
                    f"[ROS2] Subscribed to raw image: {topic} as '{cam_name}'",
                    color=GRAY,
                    flush=True,
                )

    def _spin_loop(self) -> None:
        import rclpy

        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _wait_for_cameras(self) -> None:
        if not self.camera_topics:
            print("Warning: no camera topics configured", flush=True)
            return

        timeout = 15.0
        start = time.time()
        while time.time() - start < timeout:
            with self._frame_lock:
                got = set(self._latest_frames.keys())
            if got >= set(self.camera_topics.keys()):
                print(
                    f"All {len(self.camera_topics)} camera(s) receiving frames",
                    flush=True,
                )
                return
            time.sleep(0.2)
        missing = set(self.camera_topics.keys()) - got
        print(
            f"Warning: timed out waiting for cameras: {missing}",
            flush=True,
        )

    def close(self) -> None:
        print("Closing Dingo ROS2 interface...", flush=True)
        self._running = False

        # send zero velocity to stop the robot
        try:
            self.vel_pub.publish(self._make_twist_stamped(0.0, 0.0))
        except Exception as e:
            print(f"Warning: failed to send stop command: {e}", flush=True)

        # shut down ROS2
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            import rclpy

            rclpy.shutdown()
        except Exception:
            pass

        print("Dingo ROS2 interface closed", flush=True)

    def execute_command(self, command: str) -> str:
        if command == "task_complete":
            cprint(
                "[ROS2] Task complete signaled by VLM",
                color=BLUE,
                flush=True,
            )
            self.task_completed = True
            return "Task complete"

        # map command string to linear and angular velocity
        linear_x = 0.0
        angular_z = 0.0

        if command == "move_forward":
            linear_x = self.linear_speed
            label = "Moving forward"
        elif command == "move_backward":
            linear_x = -self.linear_speed
            label = "Moving backward"
        elif command == "rotate_left":
            angular_z = self.angular_speed
            label = "Rotating left"
        elif command == "rotate_right":
            angular_z = -self.angular_speed
            label = "Rotating right"
        else:
            cprint(
                f"[ROS2] Unknown command: {command}",
                color=BLUE,
                flush=True,
            )
            return f"Unknown command: {command}"

        cprint(
            f"[ROS2] {label} (duration={self.action_duration}s)",
            color=BLUE,
            flush=True,
        )
        cprint(
            f"[ROS2] Publishing TwistStamped on '{self.cmd_vel_topic}': "
            f"linear.x={linear_x}, angular.z={angular_z}",
            color=BLUE,
            flush=True,
        )

        # publish velocity for the configured duration
        rate_hz = 20
        interval = 1.0 / rate_hz
        n_published = 0
        end_time = time.time() + self.action_duration
        while time.time() < end_time:
            msg = self._make_twist_stamped(linear_x, angular_z)
            self.vel_pub.publish(msg)
            n_published += 1
            time.sleep(interval)

        # stop the robot
        stop_msg = self._make_twist_stamped(0.0, 0.0)
        self.vel_pub.publish(stop_msg)
        cprint(
            f"[ROS2] Published {n_published} msgs, then stop (linear.x=0, angular.z=0)",
            color=BLUE,
            flush=True,
        )

        # clear cached frames so next get_camera_images returns a
        # frame captured after the robot has stopped moving
        with self._frame_lock:
            self._latest_frames.clear()

        # wait for fresh frames to arrive
        self._wait_for_cameras()

        return label

    def get_camera_images(self) -> dict[str, Image.Image]:
        # convert latest numpy frames to PIL images
        images = {}
        with self._frame_lock:
            for cam_name, frame in self._latest_frames.items():
                images[cam_name] = Image.fromarray(frame, "RGB")

        # fill in placeholders for any cameras with no frame yet
        for cam_name in self.camera_topics:
            if cam_name not in images:
                images[cam_name] = Image.new("RGB", (640, 480), (128, 128, 128))
                cprint(
                    f"[ROS2] No frame from '{cam_name}' yet",
                    color=GRAY,
                    flush=True,
                )

        self.current_images = images
        return images
