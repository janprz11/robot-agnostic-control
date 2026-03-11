# -*- coding: ascii -*-

from brain import str2mac, mac2str
from brain.brain_buffer import BrainBuffer
from brain.brain_interface import BrainInterface
from color_print import cprint, BLUE, GRAY
from datetime import datetime
from io import BytesIO
import logging
from numpy import clip
from PIL import Image, UnidentifiedImageError
from queue import Empty, Queue
import time
from typing import Any


class AlhakamLimbInterface:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.task_completed = False
        self.movement_intensity = 0.5
        self.current_images: dict[str, Image.Image] = {}

        # servo position tracking
        self.servo_positions: dict[str, float] = {}
        self.inner_lr_servo = ""
        self.outer_lr_servo = ""
        self.updown_servo = ""
        self.finger_servo = ""

        # assign servo roles from config ordering
        if "servos" in config:
            for mac in self.config["servos"].values():
                self.servo_positions[mac] = 0.0
            servos_list = list(self.config["servos"].values())
            if len(servos_list) >= 4:
                self.inner_lr_servo = servos_list[1]
                self.outer_lr_servo = servos_list[3]
                self.updown_servo = servos_list[2]
                self.finger_servo = servos_list[0]

        # robot position tracking (soft tracker for system prompt)
        self.robot_position = {
            "x_axis": 0.0,
            "y_axis": 0.0,
            "z_rotation": 0.0,
            "finger_position": 0.0,
        }

        # load robot position template
        with open(self.config["robot_position_template_file"], "r") as infile:
            self.robot_position_template = infile.read()

        # tool usage tracking
        self.tools_used: list[dict[str, Any]] = list()

        # mac registration tracking
        self._registered_macs: list[str] = list()

        # initialize BrainBuffer
        self.brain_buffer = BrainBuffer()
        self.brain_buffer.queue_size = self.config["queue_size"]
        self.brain_buffer.external_new_mac_callback = self._new_mac

        # initialize BrainInterface
        self.brain_interface = BrainInterface()
        self.brain_interface.config["queue_size"] = self.config["queue_size"]
        self.brain_interface.new_mac_callback = self.brain_buffer.new_mac_callback
        self.brain_interface.initialize(config=config)

        # suppress brain/routing loggers that configure themselves during
        # initialize(); lazy import means we have to silence them here
        for _name in list(logging.root.manager.loggerDict):
            if _name.startswith(("brain", "routing_utilities")):
                _lg = logging.getLogger(_name)
                _lg.setLevel(logging.CRITICAL)
                _lg.disabled = True
                for _h in _lg.handlers[:]:
                    _lg.removeHandler(_h)

        # pre-register dummy MAC addresses if any
        if "dummy_mac_addresses" in self.config:
            for mac in self.config["dummy_mac_addresses"]:
                mac_byte = str2mac(mac)
                print(f"Pre-registering dummy MAC address: {mac}")
                if mac_byte not in self.brain_buffer.received_data:
                    self.brain_buffer.received_data[mac_byte] = Queue(
                        maxsize=self.brain_buffer.queue_size
                    )
                self._new_mac(mac_byte, None)

        # wait for all MAC addresses to register
        print("Waiting for MAC addresses to register...")
        while len(self._registered_macs) != len(
            set(
                list(self.config["servos"].values())
                + list(self.config["cameras"].values())
            )
        ):
            time.sleep(0.01)
        time.sleep(5)
        print("All MAC addresses registered successfully")

        # re-suppress brain loggers; the receiver thread adds its own
        # handlers after initialize()
        for _name in list(logging.root.manager.loggerDict):
            if _name.startswith(("brain", "routing_utilities")):
                _lg = logging.getLogger(_name)
                _lg.setLevel(logging.CRITICAL)
                _lg.disabled = True
                for _h in _lg.handlers[:]:
                    _lg.removeHandler(_h)

        self.zero_servos()

    def _log_tool_usage(
        self, tool_name: str, parameters: dict[str, Any] | None = None
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # build tool record
        tool_record = {
            "timestamp": timestamp,
            "tool_name": tool_name,
            "parameters": parameters or {},
        }
        self.tools_used.append(tool_record)

    def _new_mac(self, mac: bytes, queue: object) -> None:
        mac_str = mac2str(mac)
        # register the MAC if it matches a known servo or camera
        if (
            mac_str in self.config["cameras"].values()
            or mac_str in self.config["servos"].values()
        ):
            self._registered_macs.append(mac_str)
            if self.config.get("set_slew_rate", False):
                self.brain_interface.configure(
                    mac_str, slew_rate=self.config["slew_rate"]
                )

    def _update_robot_position(self, command: str) -> None:
        default_intensity = 0.5
        position_limit = 1.0
        command_lower = command.lower().strip()

        # parse movement command and clamp position to limits
        if "move_left" in command_lower:
            new = self.robot_position["x_axis"] - default_intensity
            self.robot_position["x_axis"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "move_right" in command_lower:
            new = self.robot_position["x_axis"] + default_intensity
            self.robot_position["x_axis"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "move_up" in command_lower:
            new = self.robot_position["y_axis"] + default_intensity
            self.robot_position["y_axis"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "move_down" in command_lower:
            new = self.robot_position["y_axis"] - default_intensity
            self.robot_position["y_axis"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "rotate_left" in command_lower:
            new = self.robot_position["z_rotation"] - default_intensity
            self.robot_position["z_rotation"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "rotate_right" in command_lower:
            new = self.robot_position["z_rotation"] + default_intensity
            self.robot_position["z_rotation"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "bend_finger" in command_lower:
            new = self.robot_position["finger_position"] - default_intensity
            self.robot_position["finger_position"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "extend_finger" in command_lower:
            new = self.robot_position["finger_position"] + default_intensity
            self.robot_position["finger_position"] = max(
                -position_limit, min(position_limit, new)
            )
        elif "reset_position" in command_lower:
            self.robot_position = {
                "x_axis": 0.0,
                "y_axis": 0.0,
                "z_rotation": 0.0,
                "finger_position": 0.0,
            }

        cprint(
            f"Position updated after command '{command}': {self.robot_position}",
            color=GRAY,
        )

    def _update_servo_position(self, mac_address: str, value: float) -> None:
        if not self.brain_interface:
            logging.warning("Brain interface not set, cannot control servos")
            return

        if mac_address not in self.config["limits"]:
            logging.error(f"Invalid servo MAC address: {mac_address}")
            return

        # clamp value to configured limits and send to servo
        min_val = self.config["limits"][mac_address]["min"]
        max_val = self.config["limits"][mac_address]["max"]
        new_position = clip(value, min_val, max_val)

        self.servo_positions[mac_address] = new_position

        if not self.config.get("dummy_drive", False):
            self.brain_interface.drive(mac_address, new_position)
            logging.debug(f"Servo {mac_address} moved to position {new_position}")

        time.sleep(self.config.get("sleep", 0.1))

    def bend_finger(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("bend_finger", {"intensity": intensity})

        new_finger_position = self.robot_position["finger_position"] - intensity
        if new_finger_position < -1.0:
            print(
                f"Warning: Cannot bend finger - would exceed position limit (-1.0). "
                f"Current finger: {self.robot_position['finger_position']:.2f}"
            )
            return "Cannot bend finger - position limit reached"

        self.robot_position["finger_position"] = max(-1.0, new_finger_position)

        current_pos = self.servo_positions.get(self.finger_servo, 0)
        self._update_servo_position(
            self.finger_servo,
            current_pos
            + intensity
            * (
                self.config["limits"][self.finger_servo]["max"]
                - self.config["limits"][self.finger_servo]["min"]
            )
            / 2,
        )
        return "Bending finger"

    def close(self) -> None:
        print("Closing Alhakami limb interface...")
        try:
            self.zero_servos()
        except Exception as e:
            print(f"Warning: zero servos failed: {e}")
        try:
            self.brain_interface.stop()
        except Exception as e:
            print(f"Warning: brain interface stop failed: {e}")
        print("Alhakami limb interface closed")

    def execute_command(self, command: str) -> str:
        command = command.strip()
        # dispatch command to the corresponding movement method
        if command == "bend_finger":
            return self.bend_finger()
        elif command == "extend_finger":
            return self.extend_finger()
        elif command == "move_down":
            return self.move_down()
        elif command == "move_left":
            return self.move_left()
        elif command == "move_right":
            return self.move_right()
        elif command == "move_up":
            return self.move_up()
        elif command == "reset_position":
            return self.reset_position()
        elif command == "rotate_left":
            return self.rotate_left()
        elif command == "rotate_right":
            return self.rotate_right()
        elif command == "task_complete":
            return self.task_complete()
        else:
            return f"Unknown command: {command}"

    def extend_finger(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("extend_finger", {"intensity": intensity})

        new_finger_position = self.robot_position["finger_position"] + intensity
        if new_finger_position > 1.0:
            print(
                f"Warning: Cannot extend finger - would exceed position limit (1.0). "
                f"Current finger: {self.robot_position['finger_position']:.2f}"
            )
            return "Cannot extend finger - position limit reached"

        self.robot_position["finger_position"] = min(1.0, new_finger_position)

        current_pos = self.servo_positions.get(self.finger_servo, 0)
        self._update_servo_position(
            self.finger_servo,
            current_pos
            - intensity
            * (
                self.config["limits"][self.finger_servo]["max"]
                - self.config["limits"][self.finger_servo]["min"]
            )
            / 2,
        )
        return "Extending finger"

    def format_robot_position(self) -> str:
        return self.robot_position_template.format(**self.robot_position)

    def get_camera_images(self) -> dict[str, Image.Image]:
        # drain each camera buffer and keep only the latest image
        for camera_name, mac_str in self.config["cameras"].items():
            mac_byte = str2mac(mac_str)
            if mac_byte in self.brain_buffer.received_data:
                buffer = self.brain_buffer.received_data[mac_byte]
                latest_image = None
                while buffer.qsize() > 0:
                    try:
                        item = buffer.get(block=False)
                        try:
                            latest_image = Image.open(BytesIO(item["data"]))
                        except (TypeError, UnidentifiedImageError):
                            pass
                    except Empty:
                        pass
                if latest_image is not None:
                    self.current_images[camera_name] = latest_image
        # drain the tx_result_queue
        while self.brain_interface.tx_result_queue.qsize() > 0:
            self.brain_interface.tx_result_queue.get(block=False)
        return self.current_images

    def get_robot_position_str(self) -> str:
        return self.format_robot_position()

    def move_down(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("move_down", {"intensity": intensity})

        new_y_position = self.robot_position["y_axis"] - intensity
        if new_y_position < -1.0:
            print(
                f"Warning: Cannot move down - would exceed position limit (-1.0). "
                f"Current y: {self.robot_position['y_axis']:.2f}"
            )
            return "Cannot move down - position limit reached"

        self.robot_position["y_axis"] = max(-1.0, new_y_position)

        current_pos = self.servo_positions.get(self.updown_servo, 0)
        self._update_servo_position(
            self.updown_servo,
            current_pos
            - intensity
            * (
                self.config["limits"][self.updown_servo]["max"]
                - self.config["limits"][self.updown_servo]["min"]
            )
            / 2,
        )
        return "Moving down"

    def move_left(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("move_left", {"intensity": intensity})

        new_x_position = self.robot_position["x_axis"] - intensity
        if new_x_position < -1.0:
            print(
                f"Warning: Cannot move left - would exceed position limit (-1.0). "
                f"Current x: {self.robot_position['x_axis']:.2f}"
            )
            return "Cannot move left - position limit reached"

        self.robot_position["x_axis"] = max(-1.0, new_x_position)

        current_outer = self.servo_positions.get(self.outer_lr_servo, 0)
        self._update_servo_position(
            self.outer_lr_servo,
            current_outer
            + intensity
            * (
                self.config["limits"][self.outer_lr_servo]["max"]
                - self.config["limits"][self.outer_lr_servo]["min"]
            )
            / 2,
        )
        return "Moving left"

    def move_right(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("move_right", {"intensity": intensity})

        new_x_position = self.robot_position["x_axis"] + intensity
        if new_x_position > 1.0:
            print(
                f"Warning: Cannot move right - would exceed position limit (1.0). "
                f"Current x: {self.robot_position['x_axis']:.2f}"
            )
            return "Cannot move right - position limit reached"

        self.robot_position["x_axis"] = min(1.0, new_x_position)

        current_outer = self.servo_positions.get(self.outer_lr_servo, 0)
        self._update_servo_position(
            self.outer_lr_servo,
            current_outer
            - intensity
            * (
                self.config["limits"][self.outer_lr_servo]["max"]
                - self.config["limits"][self.outer_lr_servo]["min"]
            )
            / 2,
        )
        return "Moving right"

    def move_up(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("move_up", {"intensity": intensity})

        new_y_position = self.robot_position["y_axis"] + intensity
        if new_y_position > 1.0:
            print(
                f"Warning: Cannot move up - would exceed position limit (1.0). "
                f"Current y: {self.robot_position['y_axis']:.2f}"
            )
            return "Cannot move up - position limit reached"

        self.robot_position["y_axis"] = min(1.0, new_y_position)

        current_pos = self.servo_positions.get(self.updown_servo, 0)
        self._update_servo_position(
            self.updown_servo,
            current_pos
            + intensity
            * (
                self.config["limits"][self.updown_servo]["max"]
                - self.config["limits"][self.updown_servo]["min"]
            )
            / 2,
        )
        return "Moving up"

    def reset_position(self) -> str:
        self._log_tool_usage("reset_position")
        self.robot_position = {
            "x_axis": 0.0,
            "y_axis": 0.0,
            "z_rotation": 0.0,
            "finger_position": 0.0,
        }
        if "servos" in self.config:
            for mac in self.config["servos"].values():
                self._update_servo_position(mac, 0.0)
        return "Reset to zero position"

    def rotate_left(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("rotate_left", {"intensity": intensity})

        new_z_rotation = self.robot_position["z_rotation"] + intensity
        if new_z_rotation > 1.0:
            print(
                f"Warning: Cannot rotate left - would exceed position limit (1.0). "
                f"Current z: {self.robot_position['z_rotation']:.2f}"
            )
            return "Cannot rotate left - position limit reached"

        self.robot_position["z_rotation"] = min(1.0, new_z_rotation)

        current_inner = self.servo_positions.get(self.inner_lr_servo, 0)
        self._update_servo_position(
            self.inner_lr_servo,
            current_inner
            - intensity
            * (
                self.config["limits"][self.inner_lr_servo]["max"]
                - self.config["limits"][self.inner_lr_servo]["min"]
            )
            / 2,
        )
        return "Rotating left"

    def rotate_right(self, intensity: float | None = None) -> str:
        intensity = intensity if intensity is not None else self.movement_intensity
        self._log_tool_usage("rotate_right", {"intensity": intensity})

        new_z_rotation = self.robot_position["z_rotation"] - intensity
        if new_z_rotation < -1.0:
            print(
                f"Warning: Cannot rotate right - would exceed position limit (-1.0). "
                f"Current z: {self.robot_position['z_rotation']:.2f}"
            )
            return "Cannot rotate right - position limit reached"

        self.robot_position["z_rotation"] = max(-1.0, new_z_rotation)

        current_inner = self.servo_positions.get(self.inner_lr_servo, 0)
        self._update_servo_position(
            self.inner_lr_servo,
            current_inner
            + intensity
            * (
                self.config["limits"][self.inner_lr_servo]["max"]
                - self.config["limits"][self.inner_lr_servo]["min"]
            )
            / 2,
        )
        return "Rotating right"

    def task_complete(self) -> str:
        logging.info("Task completion signal received")
        self._log_tool_usage("task_complete")
        logging.info("Resetting robot position to origin before task completion")
        self.task_completed = True
        return "Task completed successfully - robot position reset to origin - terminating program"

    def zero_servos(self) -> None:
        # drive all servos to zero and reset position state
        for mac in self.config["servos"].values():
            assert mac in self.config["limits"]
            self.brain_interface.drive(mac, 0)
            time.sleep(0.1)
        time.sleep(self.config.get("sleep", 1))
        self.servo_positions = {mac: 0.0 for mac in self.config["servos"].values()}
        self.robot_position = {
            "x_axis": 0.0,
            "y_axis": 0.0,
            "z_rotation": 0.0,
            "finger_position": 0.0,
        }
        cprint("\n Zeroed servos", color=BLUE)
