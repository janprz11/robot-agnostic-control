# -*- coding: ascii -*-

from color_print import cprint, BLUE
import cv2
import numpy as np
from PIL import Image
from pymavlink import mavutil
import threading
import time
from typing import Any

# ardusub flight modes
DEPTH_HOLD_MODE = 2
MANUAL_MODE = 19
STABILIZE_MODE = 0

# manual control value range
THRUST_MAX = 1000
THRUST_MIN = -1000
THRUST_NEUTRAL = 0

# camera key used by the monitor system
CAMERA_NAME = "rov_fw_camera"


class BlueROVInterface:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.task_completed = False
        self.current_images: dict[str, Image.Image] = {}
        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._active_command = False
        self._command_lock = threading.Lock()

        # read configurable parameters from config
        self.connection_string = config.get("rov_connection", "udpin:0.0.0.0:14550")
        self.camera_port = config.get("rov_camera_port", 5600)
        self.command_duration = config.get("rov_command_duration", 2.0)
        self.command_rate = config.get("rov_command_rate", 10)
        self.color_correct = config.get("rov_color_correct", False)
        self.color_correct_blend = config.get("rov_color_correct_blend", 0.5)

        # load per-movement thrust intensities with defaults
        default_thrust = config.get("rov_thrust", 500)
        intensities = config.get("rov_thrust_intensities", {})
        self.thrust_backward = intensities.get("move_backward", default_thrust)
        self.thrust_down = intensities.get("move_down", default_thrust // 2)
        self.thrust_forward = intensities.get("move_forward", default_thrust)
        self.thrust_rotate_left = intensities.get("rotate_left", default_thrust)
        self.thrust_rotate_right = intensities.get("rotate_right", default_thrust)
        self.thrust_up = intensities.get("move_up", default_thrust)

        # connect to ROV, start background threads, arm, and start camera
        self._connect()
        self._start_heartbeat()
        self._start_idle_loop()
        self._set_flight_mode()
        self._arm()
        self._start_camera()

        print("BlueROV interface initialized", flush=True)

    def _arm(self) -> None:
        self.conn.arducopter_arm()
        self.conn.motors_armed_wait()
        cprint("ROV armed", color=BLUE, flush=True)

    def _camera_loop(self) -> None:
        # launch ffmpeg subprocess and continuously read decoded frames
        import os
        import subprocess

        sdp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "bluerov_stream.sdp"
        )
        width, height = 512, 512
        frame_bytes = width * height * 3

        cmd = [
            "ffmpeg",
            "-protocol_whitelist",
            "file,udp,rtp,crypto,data",
            "-i",
            sdp_path,
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "-sn",
            "-",
        ]
        print(f"Starting ffmpeg capture: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=frame_bytes * 2,
        )

        while self._running:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._frame_lock:
                self._latest_frame = rgb

        proc.terminate()
        proc.wait()

    @staticmethod
    def _color_correct(img: Image.Image, blend: float = 0.5) -> Image.Image:
        # apply gray-world white balance, blend with original, then CLAHE
        arr = np.asarray(img, dtype=np.float32)
        means = arr.mean(axis=(0, 1))
        overall = means.mean()
        scale = overall / np.maximum(means, 1e-6)
        scale[0] = min(scale[0], 2.0)  # cap red to avoid pink overshoot
        corrected = np.clip(arr * scale, 0, 255)
        blended = np.clip(blend * corrected + (1 - blend) * arr, 0, 255).astype(
            np.uint8
        )
        lab = cv2.cvtColor(blended, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(out, "RGB")

    def _connect(self) -> None:
        print(f"Connecting to ROV at {self.connection_string}...", flush=True)
        self.conn = mavutil.mavlink_connection(self.connection_string)
        self.conn.wait_heartbeat()
        print(
            f"Connected (system {self.conn.target_system}, "
            f"component {self.conn.target_component})",
            flush=True,
        )

    def _disarm(self) -> None:
        self.conn.arducopter_disarm()
        self.conn.motors_disarmed_wait()
        cprint("ROV disarmed", color=BLUE, flush=True)

    def _heartbeat_loop(self) -> None:
        while self._running:
            self.conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )
            time.sleep(0.5)

    def _idle_loop(self) -> None:
        while self._running:
            with self._command_lock:
                active = self._active_command
            if not active:
                self.conn.mav.manual_control_send(
                    self.conn.target_system,
                    0,
                    0,
                    500,  # neutral vertical
                    0,
                    0,
                )
            time.sleep(0.5)

    def _send_manual_control(
        self, x: int, y: int, z: int, r: int, duration: float
    ) -> None:
        # send manual control messages at the configured rate for the duration
        with self._command_lock:
            self._active_command = True

        try:
            interval = 1.0 / self.command_rate
            end_time = time.time() + duration

            while time.time() < end_time:
                self.conn.mav.manual_control_send(
                    self.conn.target_system,
                    x,
                    y,
                    z,
                    r,
                    0,
                )
                time.sleep(interval)

            # send zero thrust to stop
            self.conn.mav.manual_control_send(
                self.conn.target_system,
                0,
                0,
                500,  # neutral vertical in stabilize
                0,
                0,
            )
        finally:
            with self._command_lock:
                self._active_command = False

    def _set_flight_mode(self) -> None:
        self.conn.set_mode(DEPTH_HOLD_MODE)
        cprint("Flight mode set to Depth Hold", color=BLUE, flush=True)

    def _start_camera(self) -> None:
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        print(
            f"Camera capture started on UDP port {self.camera_port}",
            flush=True,
        )
        # wait for the first frame before proceeding
        timeout = 10.0
        start = time.time()
        while time.time() - start < timeout:
            with self._frame_lock:
                if self._latest_frame is not None:
                    print("First camera frame received", flush=True)
                    return
            time.sleep(0.1)
        print("Warning: timed out waiting for first camera frame", flush=True)

    def _start_heartbeat(self) -> None:
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()
        print("Heartbeat thread started", flush=True)

    def _start_idle_loop(self) -> None:
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()
        print("Idle control loop started", flush=True)

    def close(self) -> None:
        print("Closing BlueROV interface...", flush=True)
        self._running = False
        try:
            self._disarm()
        except Exception as e:
            print(f"Warning: disarm failed: {e}", flush=True)
        print("BlueROV interface closed", flush=True)

    def execute_command(self, command: str) -> str:
        duration = self.command_duration

        if command == "move_backward":
            thrust = self.thrust_backward
            cprint(
                f"[ROV] Moving backward (thrust={thrust}, duration={duration}s)",
                color=BLUE,
                flush=True,
            )
            self._send_manual_control(-thrust, 0, 500, 0, duration)
            return "Moving backward"
        elif command == "move_down":
            thrust = self.thrust_down
            cprint(
                f"[ROV] Moving down (thrust={thrust}, duration={duration}s)",
                color=BLUE,
                flush=True,
            )
            self._send_manual_control(0, 0, 500 - thrust, 0, duration)
            return "Moving down"
        elif command == "move_forward":
            duration = duration * 2
            thrust = self.thrust_forward
            cprint(
                f"[ROV] Moving forward (thrust={thrust}, duration={duration}s)",
                color=BLUE,
                flush=True,
            )
            self._send_manual_control(thrust, 0, 500, 0, duration)
            return "Moving forward"
        elif command == "move_up":
            thrust = self.thrust_up
            cprint(
                f"[ROV] Moving up (thrust={thrust}, duration={duration}s)",
                color=BLUE,
                flush=True,
            )
            self._send_manual_control(0, 0, 500 + thrust, 0, duration)
            return "Moving up"
        elif command == "rotate_left":
            thrust = self.thrust_rotate_left
            cprint(
                f"[ROV] Rotating left (thrust={thrust}, duration={duration}s)",
                color=BLUE,
                flush=True,
            )
            self._send_manual_control(0, 0, 500, -thrust, duration)
            return "Rotating left"
        elif command == "rotate_right":
            thrust = self.thrust_rotate_right
            cprint(
                f"[ROV] Rotating right (thrust={thrust}, duration={duration}s)",
                color=BLUE,
                flush=True,
            )
            self._send_manual_control(0, 0, 500, thrust, duration)
            return "Rotating right"
        else:
            cprint(f"[ROV] Unknown command: {command}", color=BLUE, flush=True)
            return f"Unknown command: {command}"

    def get_camera_images(self) -> dict[str, Image.Image]:
        # grab the latest frame, rotate, and optionally apply color correction
        with self._frame_lock:
            frame = self._latest_frame

        if frame is not None:
            img = Image.fromarray(frame, "RGB").rotate(90, expand=True)
        else:
            img = Image.new("RGB", (512, 512), (128, 128, 128))
            cprint(
                "[ROV] No camera frame available yet",
                color=BLUE,
                flush=True,
            )

        if self.color_correct:
            img = self._color_correct(img, blend=self.color_correct_blend)

        self.current_images = {CAMERA_NAME: img}
        return self.current_images
