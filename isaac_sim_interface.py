# -*- coding: ascii -*-
# must be imported before any other omni imports;
# simulation app must be initialized first

from color_print import cprint, BLUE, GRAY
import numpy as np
from PIL import Image
from typing import Any


class IsaacSimInterface:

    def __init__(self, config: dict[str, Any]) -> None:
        # initialize simulation state and references
        self.config = config
        self.simulation_app = None
        self.world = None
        self.stage = None
        self.timeline = None
        self.current_images = {}
        self.camera_paths = {}
        self.cameras = {}

        self.task_completed = False
        self.max_instructions = config.get("max_instructions", 25)
        self._step_count = 0
        self._distances = list()

        self._initialize_simulation()

        # record initial distance before any instructions
        d = self.get_distance_to_target()
        if d is not None:
            self._distances.append(d)

    def _initialize_simulation(self) -> None:
        from isaacsim import SimulationApp

        # build the SimulationApp launch configuration from config
        sim_config = {
            "width": self.config.get("sim_width", 1280),
            "height": self.config.get("sim_height", 720),
            "window_width": self.config.get("sim_window_width", 1920),
            "window_height": self.config.get("sim_window_height", 1080),
            "headless": self.config.get("sim_headless", False),
            "hide_ui": self.config.get("sim_hide_ui", False),
            "renderer": self.config.get("sim_renderer", "RaytracedLighting"),
            "anti_aliasing": 0,
            "extra_args": [
                "--/renderer/multiGpu/enabled=false",
                "--/renderer/activeGpu=0",
            ],
        }

        print("Initializing Isaac Sim...", flush=True)
        self.simulation_app = SimulationApp(launch_config=sim_config)
        print("SimulationApp created", flush=True)

        # enable async rendering so multiple render products resolve in parallel
        import carb

        settings = carb.settings.get_settings()
        settings.set("/app/asyncRendering", True)
        settings.set("/app/asyncRenderingLowLatency", True)
        settings.set("/log/level", "error")
        settings.set("/log/fileLogLevel", "error")
        settings.set("/log/outputStreamLevel", "error")
        print("Async rendering enabled", flush=True)

        # import omni modules after SimulationApp is created
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.stage import is_stage_loading
        from pxr import Sdf, UsdLux

        # enable livestream if configured
        if self.config.get("sim_enable_livestream", False):
            enable_extension("omni.kit.livestream.webrtc")
            self.simulation_app.set_setting("/app/window/drawMouse", True)
            print("Livestream extension enabled", flush=True)

        # load the USD stage
        usd_path = self.config.get("sim_usd_path", "")
        if usd_path:
            print(f"Loading USD stage: {usd_path}", flush=True)
            omni.usd.get_context().open_stage(usd_path)
            while is_stage_loading():
                self.simulation_app.update()
            print("Stage loaded successfully", flush=True)

        self.stage = omni.usd.get_context().get_stage()

        # add light source if not present
        if not self.stage.GetPrimAtPath("/DistantLight"):
            distantLight = UsdLux.DistantLight.Define(
                self.stage, Sdf.Path("/DistantLight")
            )
            distantLight.CreateIntensityAttr(500)

        self.world = World(stage_units_in_meters=1.0)

        import omni.timeline

        self.timeline = omni.timeline.get_timeline_interface()

        self.world.play()
        print(f"Timeline playing: {self.timeline.is_playing()}", flush=True)
        self._setup_robot()
        self._setup_proximity_check()

        print("Isaac Sim initialized successfully", flush=True)

    def _move_backward(self) -> str:
        return self._run_action(
            self.var_backward, "Moving backward", self.action_steps * 2
        )

    def _move_forward(self) -> str:
        return self._run_action(
            self.var_forward, "Moving forward", self.action_steps * 2
        )

    def _rotate_left(self) -> str:
        return self._run_action(self.var_left, "Rotating left", self.action_steps)

    def _rotate_right(self) -> str:
        return self._run_action(self.var_right, "Rotating right", self.action_steps)

    def _run_action(self, var: Any, label: str, steps: int) -> str:
        # activate the action variable, step the simulation, then deactivate
        cprint(f"[SIM] {label} for {steps} steps", color=BLUE, flush=True)
        self._set_variable(var, True)
        for _ in range(steps):
            self.step()
        self._set_variable(var, False)
        return label

    def _set_variable(self, var: Any, state: bool) -> None:
        if var is not None:
            var.set(self.graph_context, state)

    def _setup_cameras(self) -> None:
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        camera_names = self.config.get("sim_cameras", [])
        self.cam_width = self.config.get("sim_cam_width", 640)
        self.cam_height = self.config.get("sim_cam_height", 480)

        print(
            f"Setting up {len(camera_names)} simulation cameras "
            f"at {self.cam_width}x{self.cam_height}...",
            flush=True,
        )

        # search the stage for each named camera prim and initialize a sensor
        for cam_name in camera_names:
            camera_prim = None
            for prim in self.stage.Traverse():
                if prim.GetName() == cam_name and prim.IsA(UsdGeom.Camera):
                    camera_prim = prim
                    break

            if camera_prim is None:
                print(
                    f"Warning: Camera '{cam_name}' not found in stage",
                    flush=True,
                )
                continue

            camera_path = str(camera_prim.GetPath())
            self.camera_paths[cam_name] = camera_path

            cam_sensor = Camera(
                prim_path=camera_path,
                resolution=(self.cam_width, self.cam_height),
                name=cam_name,
            )
            cam_sensor.initialize()
            self.cameras[cam_name] = cam_sensor
            print(
                f"Camera sensor initialized: {cam_name} at {camera_path}",
                flush=True,
            )

        # warmup all cameras
        print("Warming up simulation cameras...", flush=True)
        for i in range(120):
            self.world.step(render=True)
            if i % 20 == 0:
                print(f"  Warmup frame {i}...", flush=True)

        print(
            f"Camera setup complete. {len(self.cameras)} cameras ready.",
            flush=True,
        )

    def _setup_proximity_check(self) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics

        robot_name = self.config.get("sim_robot_prim_name", "Dingo")
        target_name = self.config.get("sim_target_prim_name", "SM_FireExtinguisher_02")
        self.proximity_threshold = self.config.get("sim_proximity_threshold", 0.1)

        # traverse the stage to find robot and target prims by name
        robot_root = None
        self.robot_body_path = None
        self.target_prim_path = None

        for prim in self.stage.Traverse():
            if prim.GetName() == robot_name and prim.IsA(UsdGeom.Xformable):
                robot_root = prim
                print(
                    f"Proximity check: found robot root '{robot_name}' at {prim.GetPath()}",
                    flush=True,
                )
            if prim.GetName() == target_name and prim.IsA(UsdGeom.Xformable):
                self.target_prim_path = str(prim.GetPath())
                print(
                    f"Proximity check: found target '{target_name}' at {prim.GetPath()}",
                    flush=True,
                )

        if robot_root is not None:
            for child in Usd.PrimRange(robot_root):
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    self.robot_body_path = str(child.GetPath())
                    print(
                        f"Proximity check: found robot rigid body at {self.robot_body_path}",
                        flush=True,
                    )
                    break
            if self.robot_body_path is None:
                print(
                    f"Warning: no rigid body found under '{robot_name}' -- proximity check disabled",
                    flush=True,
                )
        else:
            print(
                f"Warning: robot prim '{robot_name}' not found -- proximity check disabled",
                flush=True,
            )

        if self.target_prim_path is None:
            print(
                f"Warning: target prim '{target_name}' not found -- proximity check disabled",
                flush=True,
            )

        # compute static target world position once
        self.target_world_pos = None
        if self.target_prim_path is not None:
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            target_xform = xform_cache.GetLocalToWorldTransform(
                self.stage.GetPrimAtPath(self.target_prim_path)
            )
            pos = target_xform.ExtractTranslation()
            self.target_world_pos = np.array([pos[0], pos[1], pos[2]])
            print(
                f"Proximity check: target world position = {self.target_world_pos}",
                flush=True,
            )

        self.proximity_enabled = (
            self.robot_body_path is not None and self.target_world_pos is not None
        )
        if self.proximity_enabled:
            print("Proximity check: enabled", flush=True)

    def _setup_robot(self) -> None:
        # look up the action graph and bind movement variables
        import omni.graph.core as og

        graph_path = self.config.get(
            "sim_graph_path", "/Graphs/differential_controller"
        )
        graph = og.get_graph_by_path(graph_path)
        self.graph_context = graph.get_default_graph_context()
        self.var_left = graph.find_variable("Left")
        self.var_right = graph.find_variable("Right")
        self.var_forward = graph.find_variable("Forward")
        self.var_backward = graph.find_variable("Backward")

        self.action_steps = self.config.get("sim_action_steps", 60)

        print(f"Robot graph loaded: {graph_path}", flush=True)

        self._setup_cameras()

    def _task_complete(self) -> str:
        cprint("[SIM] Task complete -- shutting down", color=BLUE, flush=True)
        self.close()
        import sys

        sys.exit(0)

    def check_proximity(self) -> bool:
        distance = self.get_distance_to_target()
        if distance is None:
            return False
        return distance <= self.proximity_threshold

    def close(self) -> None:
        print("Closing Isaac Sim...", flush=True)
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None
        print("Isaac Sim closed", flush=True)

    def execute_command(self, command: str) -> str:
        if command == "move_forward":
            result = self._move_forward()
        elif command == "move_backward":
            result = self._move_backward()
        elif command == "rotate_left":
            result = self._rotate_left()
        elif command == "rotate_right":
            result = self._rotate_right()
        else:
            return f"Unknown command: {command}"

        self._step_count += 1

        # record distance and check proximity in a single call
        d = self.get_distance_to_target()
        if d is not None:
            self._distances.append(d)
            if not self.task_completed and d <= self.proximity_threshold:
                cprint(
                    "[SIM] Robot is within proximity threshold of target -- stopping",
                    color=BLUE,
                    flush=True,
                )
                self.task_completed = True

        if not self.task_completed and self._step_count >= self.max_instructions:
            cprint(
                f"[SIM] Reached max instructions ({self.max_instructions}) -- stopping",
                color=BLUE,
                flush=True,
            )
            self.task_completed = True

        return result

    def get_camera_images(self) -> dict[str, Image.Image]:
        images = {}

        # step multiple frames so all render products resolve
        for _ in range(len(self.cameras) + 1):
            self.step()

        for cam_name, cam_sensor in self.cameras.items():
            try:
                rgba = cam_sensor.get_rgba()

                if rgba is not None and isinstance(rgba, np.ndarray) and rgba.size > 0:
                    rgb = rgba[:, :, :3]
                    cprint(
                        f"[DEBUG] {cam_name}: shape={rgb.shape}, "
                        f"dtype={rgb.dtype}, "
                        f"min={rgb.min()}, max={rgb.max()}",
                        color=GRAY,
                        flush=True,
                    )
                    images[cam_name] = Image.fromarray(rgb.astype(np.uint8), "RGB")
                else:
                    cprint(
                        f"[DEBUG] {cam_name}: no data from sensor",
                        color=GRAY,
                        flush=True,
                    )
                    images[cam_name] = Image.new(
                        "RGB",
                        (self.cam_width, self.cam_height),
                        (128, 128, 128),
                    )

            except Exception as e:
                cprint(
                    f"Error capturing image from {cam_name}: {e}",
                    color=GRAY,
                    flush=True,
                )
                import traceback

                traceback.print_exc()
                images[cam_name] = Image.new(
                    "RGB",
                    (self.cam_width, self.cam_height),
                    (128, 128, 128),
                )

        self.current_images = images

        # save individual camera images for debugging
        for cam_name, img in images.items():
            debug_path = f"debug_{cam_name}.png"
            img.save(debug_path)
            cprint(f"[DEBUG] Saved {debug_path}", color=GRAY, flush=True)

        return images

    def get_distance_to_target(self) -> float | None:
        if not self.proximity_enabled:
            return None

        from pxr import UsdGeom

        # extract robot world position from the translate xform op
        robot_prim = self.stage.GetPrimAtPath(self.robot_body_path)
        xformable = UsdGeom.Xformable(robot_prim)

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get()
                robot_world_pos = np.array([t[0], t[1], t[2]])
                distance = float(
                    np.linalg.norm(robot_world_pos - self.target_world_pos)
                )
                cprint(
                    f"[PROXIMITY] robot={robot_world_pos}, distance={distance:.3f}m (threshold={self.proximity_threshold}m)",
                    color=GRAY,
                    flush=True,
                )
                return distance

        cprint(
            "[PROXIMITY] could not read robot translate op",
            color=GRAY,
            flush=True,
        )
        return None

    def is_running(self) -> bool:
        if self.simulation_app is None:
            return False
        return self.simulation_app.is_running() and not self.simulation_app.is_exiting()

    def step(self) -> None:
        if self.world is not None:
            self.world.step(render=True)

    def write_distance_log(self, log_file: str, max_instructions: int) -> None:
        if not self._distances:
            return
        expected_len = max_instructions + 1
        last_distance = self._distances[-1]
        while len(self._distances) < expected_len:
            self._distances.append(last_distance)
        with open(log_file, "a") as f:
            f.write(",".join(f"{d:.3f}" for d in self._distances) + "\n")
        cprint(f"Distance log appended to {log_file}", color=GRAY, flush=True)
