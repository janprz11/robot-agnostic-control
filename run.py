#!/usr/bin/env python
# -*- coding: ascii -*-

from color_print import cprint, BLUE, GRAY, YELLOW
import csv
from datetime import datetime
import hydra
from image_upscale import (
    load_model,
    upscale_image,
    MODEL_4X,
    MODEL_2X,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
)
import json
import logging
from mindstorm import Mindstorm
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
from PIL import Image
import pyapriltags
import random
import subprocess
import time
import torch
from typing import Any
import wandb
import warnings

TAG_ID = 29

# disable all logging
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
    logging.getLogger(name).disabled = True
for name in (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "brain",
    "brain.brain_buffer",
    "brain.brain_interface",
    "brain.brain_receiver",
    "brain.__main__",
    "routing_utilities",
    "routing_utilities.router",
    "routing_utilities.arbitrator",
    "routing_utilities.test_service",
):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).disabled = True
warnings.filterwarnings("ignore")


class Brain:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.current_images = dict()

        self.at_detector = pyapriltags.Detector(families="tag16h5", quad_decimate=1.0)

        # instantiate the robot interface based on config
        interface_name = self.config.get("interface")
        if interface_name == "alhakami_limb":
            print("Initializing Brain in ALHAKAMI LIMB mode")
            from Alhakami_limb import AlhakamLimbInterface

            self.interface = AlhakamLimbInterface(config)
        elif interface_name == "blackjack":
            print("Initializing Brain in BLACKJACK mode")
            from blackjack_interface import BlackjackInterface

            self.interface = BlackjackInterface(config)
        elif interface_name == "bluerov":
            print("Initializing Brain in BLUEROV mode")
            from bluerov_interface import BlueROVInterface

            self.interface = BlueROVInterface(config)
        elif interface_name == "dingo_ros2":
            print("Initializing Brain in DINGO ROS2 mode")
            from dingo_ros2_interface import DingoROS2Interface

            self.interface = DingoROS2Interface(config)
        elif interface_name == "isaac_sim":
            print("Initializing Brain in ISAAC SIM mode")
            from isaac_sim_interface import IsaacSimInterface

            self.interface = IsaacSimInterface(config)
        else:
            raise ValueError(
                f"Unknown interface: {interface_name}. " "Set 'interface' in config."
            )

    def _can_see_tag(self) -> bool:
        # stitch camera images and detect april tags in the grayscale frame
        camera_states = [1, 1, 1, 1]
        camera_tensor = torch.tensor(camera_states, dtype=torch.float32)
        camera_tensor /= camera_tensor.sum()
        self.update_images()
        stitched_img = self.stich_images(upscale=False)

        stitched_img.save("debug_stitched_image.png")
        cprint("Saved stitched image to debug_stitched_image.png", color=GRAY)

        grayscale_img = stitched_img.convert("L")
        tags = self.at_detector.detect(
            np.asarray(grayscale_img),
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )
        return TAG_ID in [tag.tag_id for tag in tags]

    def _get_upscale_models(self) -> tuple:
        # lazily load and cache the super-resolution models
        if not hasattr(self, "_upscale_models"):
            sr_device = self.config.get("upscale_device", "cuda:1")
            model_4x, processor_4x = load_model(MODEL_4X, device=sr_device)
            model_2x, processor_2x = load_model(MODEL_2X, device=sr_device)
            self._upscale_models = (
                model_4x,
                processor_4x,
                model_2x,
                processor_2x,
            )
        return self._upscale_models

    def close(self) -> None:
        self.interface.close()

    def run_mindstorm(self) -> None:
        # initialize mindstorm and create per-run log directory
        mindstorm = Mindstorm(self.config)

        global_step = 0
        step_counts = list()
        start_time = time.time()

        # create per-run JSONL log directory
        save_logs = self.config.get("save_logs", False)
        run_log_dir = None
        jsonl_path = None
        if save_logs:
            run_timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
            run_log_dir = os.path.join("logs", run_timestamp)
            os.makedirs(run_log_dir, exist_ok=True)
            jsonl_path = os.path.join(run_log_dir, "run.jsonl")
            cprint(f"JSONL run log: {jsonl_path}", color=GRAY, flush=True)

        # main control loop; runs until task completes or runtime expires
        while (
            not self.interface.task_completed
            and time.time() - start_time < 60 * self.config["runtime"]
        ):

            global_step += 1

            time.sleep(self.config.get("sleep", 0))

            # update images and create debug stitched image before mindstorm decision
            self.update_images()
            stitched_img = self.stich_images()
            stitched_img.save("debug_stitched_image.png")
            cprint(
                "Updated debug_stitched_image.png before mindstorm decision",
                color=GRAY,
            )

            # save log image immediately so it is on disk even if the program is interrupted
            if save_logs:
                iter_image_name = f"iter_{global_step:03d}.png"
                stitched_img.save(os.path.join(run_log_dir, iter_image_name))

            mindstorm.set_simulation_images(self.current_images)

            # update robot position string and run the mindstorm pipeline
            get_pos = getattr(self.interface, "get_robot_position_str", None)
            if get_pos is not None:
                mindstorm.robot_position_str = get_pos()

            response = mindstorm()
            command = response.split()[-1]
            result = self.interface.execute_command(command)

            # forward pending interface events to mindstorm memory
            get_events = getattr(self.interface, "get_pending_events", None)
            if get_events is not None:
                for event in get_events():
                    mindstorm.update_memory_with_event(event)

            execution_info = {
                "type": "command_execution",
                "command": command,
                "result": result if result else "Command executed",
            }

            # write iteration data to the JSONL run log
            if save_logs:
                entry = {
                    "iteration": global_step,
                    "timestamp": datetime.now().isoformat(),
                    "command": command,
                    "execution_result": result if result else "Command executed",
                    "image": iter_image_name,
                }
                entry.update(mindstorm.last_interaction)
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            if self.interface.task_completed:
                logging.info("Task completion detected - terminating program")
                break

            time.sleep(self.config.get("sleep", 0))

        # write distance log if interface supports it
        if save_logs:
            write_dist = getattr(self.interface, "write_distance_log", None)
            if write_dist is not None:
                log_file = self.config.get("distance_log_file", "distance_log.csv")
                write_dist(log_file, self.config.get("max_instructions", 25))

        # check for april tag in camera frames when task completes
        april_tag_seen = False
        if self.interface.task_completed:
            try:
                april_tag_seen = self._can_see_tag()
                logging.info(f"AprilTag detection result: {april_tag_seen}")
                cprint(f"AprilTag detected: {april_tag_seen}", color=BLUE)
            except Exception as e:
                logging.warning(f"AprilTag detection failed: {e}")
                cprint(f"AprilTag detection failed: {e}", color=GRAY)

        # log run outcome to csv
        elapsed = time.time() - start_time
        outcome = "completed" if self.interface.task_completed else "timeout"
        if save_logs:
            run_log_file = self.config.get(
                "run_log_file",
                "results/V2_localisation/mindstorm_run_log.csv",
            )
            csv_log_dir = os.path.dirname(run_log_file)
            if csv_log_dir:
                os.makedirs(csv_log_dir, exist_ok=True)
            file_exists = os.path.isfile(run_log_file)
            with open(run_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "steps",
                            "outcome",
                            "elapsed_s",
                            "april_tag_seen",
                        ]
                    )
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        global_step,
                        outcome,
                        f"{elapsed:.1f}",
                        april_tag_seen,
                    ]
                )
            cprint(
                f"Run log appended to {run_log_file}: {global_step} steps, "
                f"{outcome}, april_tag_seen={april_tag_seen}",
                color=GRAY,
            )

            summary = {
                "type": "summary",
                "total_iterations": global_step,
                "outcome": outcome,
                "elapsed_s": round(elapsed, 1),
            }
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(summary) + "\n")
            cprint(
                f"JSONL run log finalised: {jsonl_path}",
                color=GRAY,
                flush=True,
            )

        # save final stitched images with run number
        result_dir = "results/V2_localisation"
        os.makedirs(result_dir, exist_ok=True)
        existing = [
            f
            for f in os.listdir(result_dir)
            if f.startswith("debug_stitched_image_run_") and f.endswith(".png")
        ]
        run_number = len(existing) // 2 + 1
        raw_path = os.path.join(
            result_dir, f"debug_stitched_image_run_{run_number}.png"
        )
        self.stich_images(upscale=False).save(raw_path)
        cprint(f"Saved raw stitched image to {raw_path}", color=GRAY)
        upscaled_path = os.path.join(
            result_dir,
            f"debug_stitched_image_run_{run_number}_upscaled.png",
        )
        self.stich_images(upscale=True).save(upscaled_path)
        cprint(f"Saved upscaled stitched image to {upscaled_path}", color=GRAY)

        if self.config["save_step_counts"]:
            np.save(self.config["step_counts_outfile"], step_counts)

    def run_mindstorm_random(self) -> None:
        # initialize mindstorm and load available tool names
        mindstorm = Mindstorm(self.config)

        # load available tools (exclude task_complete)
        with open(self.config["tools_definition_file"], "r") as f:
            tools = json.load(f)
        tool_names = [t["name"] for t in tools if t["name"] != "task_complete"]

        max_steps = self.config.get("max_steps", 25)
        save_logs = self.config.get("save_logs", True)
        start_time = time.time()
        task_completed = False

        run_log_dir = None
        jsonl_path = None
        if save_logs:
            run_timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
            run_log_dir = os.path.join("logs", run_timestamp)
            os.makedirs(run_log_dir, exist_ok=True)
            jsonl_path = os.path.join(run_log_dir, "run.jsonl")
            cprint(f"JSONL run log: {jsonl_path}", color=GRAY, flush=True)

        # execute random commands for up to max_steps iterations
        global_step = 0
        for step in range(1, max_steps + 1):
            global_step = step

            time.sleep(1)

            self.update_images()
            stitched_img = self.stich_images()
            stitched_img.save("debug_stitched_image.png")

            command = random.choice(tool_names)
            cprint(
                f"[RANDOM] step {step}/{max_steps}: {command}",
                color=YELLOW,
                flush=True,
            )
            result = self.interface.execute_command(command)

            mindstorm.set_simulation_images(self.current_images)
            get_pos = getattr(self.interface, "get_robot_position_str", None)
            if get_pos is not None:
                mindstorm.robot_position_str = get_pos()

            # run full LLM pipeline (monitors + controller)
            controller_response = mindstorm()
            cprint(f"Controller response: {controller_response}", color=GRAY)

            if save_logs:
                iter_image_name = f"iter_{step:03d}.png"
                stitched_img.save(os.path.join(run_log_dir, iter_image_name))
                entry = {
                    "iteration": step,
                    "timestamp": datetime.now().isoformat(),
                    "random_command": command,
                    "execution_result": result if result else "Command executed",
                    "controller_response": controller_response,
                    "image": iter_image_name,
                }
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            if "task_complete" in controller_response.lower():
                task_completed = True
                cprint(
                    "Controller issued task_complete - terminating",
                    color=BLUE,
                )
                break

            time.sleep(self.config["sleep"])

        # compute elapsed time, detect april tag, and save final results
        elapsed = time.time() - start_time
        outcome = "completed" if task_completed else "max_steps"

        april_tag_seen = self._can_see_tag()

        # save final stitched images
        result_dir = "results/V2_random"
        os.makedirs(result_dir, exist_ok=True)
        existing = [
            f
            for f in os.listdir(result_dir)
            if f.startswith("random_stitched_run_") and f.endswith(".png")
        ]
        run_number = len(existing) // 2 + 1
        raw_path = os.path.join(result_dir, f"random_stitched_run_{run_number}.png")
        self.stich_images(upscale=False).save(raw_path)
        cprint(f"Saved raw stitched image to {raw_path}", color=GRAY)
        upscaled_path = os.path.join(
            result_dir,
            f"random_stitched_run_{run_number}_upscaled.png",
        )
        self.stich_images(upscale=True).save(upscaled_path)
        cprint(f"Saved upscaled stitched image to {upscaled_path}", color=GRAY)

        if save_logs:
            run_log_file = self.config.get(
                "run_log_file", "results/V2_random/random_run_log.csv"
            )
            csv_log_dir = os.path.dirname(run_log_file)
            if csv_log_dir:
                os.makedirs(csv_log_dir, exist_ok=True)
            file_exists = os.path.isfile(run_log_file)
            with open(run_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "steps",
                            "outcome",
                            "elapsed_s",
                            "april_tag_seen",
                        ]
                    )
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        global_step,
                        outcome,
                        f"{elapsed:.1f}",
                        april_tag_seen,
                    ]
                )
            cprint(
                f"Run log appended to {run_log_file}: {global_step} steps, {outcome}",
                color=GRAY,
            )

            summary = {
                "type": "summary",
                "total_iterations": global_step,
                "outcome": outcome,
                "elapsed_s": round(elapsed, 1),
                "april_tag_seen": april_tag_seen,
            }
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(summary) + "\n")
            cprint(
                f"JSONL run log finalised: {jsonl_path}",
                color=GRAY,
                flush=True,
            )

    def stich_images(self, upscale: bool = True) -> Image.Image:
        # determine tile dimensions based on interface type and upscale setting
        interface_name = self.config.get("interface")
        do_upscale = (
            upscale
            and self.config.get("upscale_images", False)
            and interface_name != "isaac_sim"
        )

        if interface_name == "isaac_sim":
            sim_cameras = self.config.get("sim_cameras", [])
            num_cameras = len(sim_cameras) if sim_cameras else len(self.current_images)
        else:
            num_cameras = len(self.current_images)

        do_crop = self.config.get("crop_images", False)

        # upscaled or cropped images are 512x512, otherwise use native resolution
        if do_upscale or do_crop:
            tile_h, tile_w = 512, 512
        else:
            tile_h, tile_w = 480, 640
            for img in self.current_images.values():
                if isinstance(img, Image.Image):
                    tile_w, tile_h = img.size
                    break
                elif isinstance(img, np.ndarray):
                    tile_h, tile_w = img.shape[0], img.shape[1]
                    break

        stiched = np.zeros((tile_h * max(num_cameras, 1), tile_w, 3), dtype=np.uint8)

        for i, cam_name in enumerate(sorted(self.current_images.keys())):
            try:
                img = self.current_images[cam_name]
                if isinstance(img, Image.Image):
                    img = np.array(img)

                # crop to 512x512 from the top-center of the image
                if do_crop:
                    h, w = img.shape[0], img.shape[1]
                    left = (w - tile_w) // 2
                    img = img[:tile_h, left : left + tile_w]

                if do_upscale:
                    m4x, p4x, m2x, p2x = self._get_upscale_models()
                    pil_img = Image.fromarray(img)
                    sr_input_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
                    if pil_img.size != sr_input_size:
                        pil_img = pil_img.resize(
                            sr_input_size, Image.Resampling.LANCZOS
                        )
                    pil_img = upscale_image(pil_img, m4x, p4x, m2x, p2x)
                    img = np.array(pil_img)
                np.copyto(stiched[i * tile_h : (i + 1) * tile_h, :, :], img)
            except (KeyError, OSError, ValueError) as e:
                cprint(
                    f"Error stitching image for {cam_name}: {e}",
                    color=GRAY,
                )
        return Image.fromarray(stiched, "RGB")

    def update_images(self) -> None:
        self.current_images = self.interface.get_camera_images()


@hydra.main(config_name="test", config_path="configs", version_base=None)
def main(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    if config["log_level"] == "DEBUG":
        logging.root.setLevel(logging.DEBUG)
    elif config["log_level"] == "INFO":
        logging.root.setLevel(logging.INFO)
    elif config["log_level"] == "WARNING":
        logging.root.setLevel(logging.WARNING)
    elif config["log_level"] == "ERROR":
        logging.root.setLevel(logging.ERROR)
    elif config["log_level"] == "CRITICAL":
        logging.root.setLevel(logging.CRITICAL)

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
        config.update(
            {
                "branch": branch,
                "commit_hash": commit_hash,
            }
        )
    except subprocess.SubprocessError:
        pass

    if config["use_wandb"]:
        wandb.login()
        wandb.init(config=config, project="robot-agnostic", save_code=True)

    for filename in [
        "episode_returns_outfile",
        "model_outfile",
        "step_counts_outfile",
    ]:
        if filename in config:
            if os.path.exists(config[filename]):
                logging.warning(f"Overwriting {config[filename]}")
            if "/" in config[filename]:
                try:
                    os.makedirs("/".join(config[filename].split("/")[:-1]))
                except OSError:
                    pass

    brain = Brain(config)

    # run the selected algorithm and ensure cleanup on exit
    try:
        if config["algorithm"] == "mindstorm_random":
            brain.run_mindstorm_random()
        else:
            assert config["algorithm"] == "mindstorm"
            brain.run_mindstorm()
    finally:
        brain.close()


if __name__ == "__main__":
    main()
