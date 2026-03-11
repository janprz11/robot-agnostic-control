# -*- coding: ascii -*-

from backbone import *
from color_print import cprint, CYAN, GRAY, MAGENTA, YELLOW
from image_upscale import load_model, upscale_image, MODEL_4X, MODEL_2X
import os
from parse_robot_description import parse_sdf, format_for_llm
from PIL import Image, ImageFile
import re
import time
from typing import Any

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Mindstorm:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        self.n_monitor_runs = self.config.get("n_monitor_runs", 1)

        # read prompts
        with open(self.config["tools_definition_file"], "r") as infile:
            self.tools_definition = infile.read()
        with open(self.config["controller_system_prompt_file"], "r") as infile:
            self.controller_system_prompt = infile.read()
        with open(self.config["monitor_system_prompt_file"], "r") as infile:
            self.monitor_system_prompt = infile.read()
        with open(self.config["controller_to_monitor_prompt_file"], "r") as infile:
            self.controller_to_monitor_prompt = infile.read()
        with open(self.config["controller_to_response_prompt_file"], "r") as infile:
            self.controller_to_response_prompt = infile.read()
        with open(self.config["default_query_file"], "r") as infile:
            self.default_query = infile.read()
        with open(self.config["memory_update_prompt_file"], "r") as infile:
            self.memory_update_prompt = infile.read()
        with open(self.config["meta_analysis_prompt_file"], "r") as infile:
            self.meta_analysis_prompt = infile.read()
        with open(self.config["command_execution_event_file"], "r") as infile:
            self.command_execution_event_template = infile.read()
        with open(self.config["object_detection_event_file"], "r") as infile:
            self.object_detection_event_template = infile.read()

        # build pipelines

        # select the model class based on config
        if config["model"] == "ChatGPT":
            model = ChatGPT41
        elif config["model"] == "Llama11B":
            model = Llama11B
        else:
            assert config["model"] == "SmolVLM2"
            model = SmolVLM2

        self.pipeline = model(
            use_history=False,
        )

        # set up the memory curator pipeline if configured
        if "memory_curator_prompt_file" in self.config:
            with open(self.config["memory_curator_prompt_file"], "r") as infile:
                self.memory_curator_prompt = infile.read()
            self.memory_pipeline = self.pipeline.clone()
            self.memory_pipeline.reset(
                system_prompt=self.memory_curator_prompt,
                use_history=False,
            )

        # generate a simplified robot description from the SDF file
        self.robot_description = self._generate_robot_description(
            pipeline=self.pipeline.clone()
        )

        self.environment_memory = "Unknown environment."
        self.react_history = "0. Reset position.\n"
        self.react_step = 0
        self.full_history = []

        # set externally by the interface via run.py
        self.robot_position_str = "Position tracking not available."

        # stores structured data from the most recent __call__
        self.last_interaction = {}

        # format and set the initial controller system prompt
        system_prompt = self.controller_system_prompt.format(
            robot_description=self.robot_description,
            tools_definition=self.tools_definition,
            environment_memory=self.environment_memory,
            robot_position=self.robot_position_str,
            react_history=self.react_history,
            user_query="",
        )
        self.pipeline.reset(
            system_prompt=self.controller_system_prompt,
            use_history=True,
        )

        self.monitors = dict()
        self.simulation_images = {}

        # load AI upscale models if enabled
        upscale_models = None
        if self.config.get("upscale_images", False):
            cprint("Loading super-resolution models...", color=CYAN)
            model_4x, processor_4x = load_model(MODEL_4X)
            model_2x, processor_2x = load_model(MODEL_2X)
            upscale_models = (model_4x, processor_4x, model_2x, processor_2x)

        # determine camera names based on interface type
        interface = self.config.get("interface")
        if interface == "isaac_sim":
            camera_names = self.config.get("sim_cameras", [])
        elif interface == "bluerov":
            from bluerov_interface import CAMERA_NAME

            camera_names = [CAMERA_NAME]
        elif interface == "dingo_ros2":
            camera_names = list(self.config.get("ros2_camera_topics", {}).keys())
        elif interface == "alhakami_limb":
            camera_names = list(self.config.get("cameras", {}).keys())
        elif interface == "blackjack":
            camera_names = ["blackjack_table"]
        else:
            camera_names = []

        for camera_name in camera_names:
            self.monitors[camera_name] = Monitor(
                camera_name,
                self.pipeline.clone(),
                self.monitor_system_prompt,
                meta_analysis_prompt=self.meta_analysis_prompt,
                upscale_models=upscale_models,
            )

    def __call__(self, query: str | None = None) -> str:
        if query is None:
            query = self.default_query

        # rebuild the system prompt with current memory and context
        updated_system_prompt = self.controller_system_prompt.format(
            robot_description=self.robot_description,
            tools_definition=self.tools_definition,
            environment_memory=self.environment_memory,
            robot_position=self.robot_position_str,
            react_history=self.react_history,
            user_query=query,
        )
        self.pipeline.reset(system_prompt=updated_system_prompt, use_history=True)

        # generate a visual query and collect responses from each monitor
        monitor_prompt = self.pipeline(
            self.controller_to_monitor_prompt, max_new_tokens=8192
        )
        monitor_responses = list()
        for name, monitor in self.monitors.items():
            sim_image = self.simulation_images.get(name)
            if sim_image is None and monitor.current_image is None:
                cprint(
                    f"Skipping monitor {name} - no image available yet",
                    color=GRAY,
                )
                continue
            response = monitor(
                monitor_prompt, image=sim_image, n_runs=self.n_monitor_runs
            )
            monitor_responses.append(f"{name}: " + response)

        # aggregate responses and generate final output
        monitor_responses_str = "\n".join(monitor_responses)
        response_prompt = self.controller_to_response_prompt.format(
            query_for_monitors=monitor_prompt,
            monitor_responses=monitor_responses_str,
        )
        cprint(f"Monitor responses: {monitor_responses_str}\n", color=CYAN)

        full_response = self.pipeline(response_prompt, max_new_tokens=8192)
        thinking_content = self._extract_thinking_content(full_response)

        # extract the command from <result> tags in the response
        result_match = re.search(r"<result>(.*?)</result>", full_response, re.DOTALL)
        if not result_match:
            result_match = re.search(
                r"<result>(.*?)</result>", full_response, re.DOTALL
            )
            if not result_match:
                raise ValueError("No <result> or <result> tags found in the response.")
        response = result_match.group(1).strip()
        if thinking_content:
            cprint(f"Controller reasoning: {thinking_content}", color=YELLOW)
        cprint(f"Controller response: {response}", color=YELLOW)

        # append interaction to history and increment the react step
        current_item = (
            f"\nUser: {query}\n Visual Query: {monitor_prompt} \n "
            f"Monitor Responses: {monitor_responses_str} \n "
            f"Control Command: {response}\n"
        )
        self.full_history.append(current_item)

        self.react_step += 1
        self.react_history = self.react_history + f"{self.react_step}. {response}\n"

        self._update_memory(current_item)
        cprint("================================================", color=MAGENTA)
        cprint(
            "Environment Memory Updated:",
            self.environment_memory,
            color=MAGENTA,
        )

        self.last_interaction = {
            "query": query,
            "monitor_prompt": monitor_prompt,
            "monitor_responses": monitor_responses,
            "thinking_content": thinking_content,
            "command": response,
            "environment_memory": self.environment_memory,
        }

        return response

    def _extract_thinking_content(self, full_response: str) -> str:
        think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return ""

    def _generate_robot_description(self, pipeline: Any) -> str | None:
        sdf_file_path = self.config.get("robot_sdf_file")
        if not sdf_file_path or not os.path.exists(sdf_file_path):
            return "Unknown robot."

        robot_data = parse_sdf(sdf_file_path)
        if not robot_data:
            return "Unknown robot."

        detailed_description = format_for_llm(robot_data)

        temp_pipeline = pipeline
        temp_pipeline.reset(
            system_prompt="You are a helpful assistant that creates simplified descriptions of robots based on their technical specifications",
            use_history=False,
        )

        with open(self.config["parser_prompt_file"], "r") as infile:
            parser_prompt = infile.read()

        formatted_prompt = parser_prompt.format(description=detailed_description)
        simplified_description = temp_pipeline(formatted_prompt)

        return simplified_description

    def _update_memory(self, current_item: str) -> None:
        # update environment memory via the curator LLM or raw append
        if not hasattr(self, "memory_curator_prompt"):
            return

        # skip LLM curation, append raw input directly
        if not self.config.get("curate_memory", True):
            self.environment_memory += current_item
            return

        input_prompt = self.memory_update_prompt.format(
            current_memory=self.environment_memory, current_item=current_item
        )
        updated_memory = self.memory_pipeline(input_prompt, max_new_tokens=8192)
        self.environment_memory = updated_memory

    def get_all_current_images(self) -> list[Image.Image | None]:
        return [monitor.get_current_image() for monitor in self.monitors.values()]

    def set_simulation_images(self, images: dict[str, Image.Image]) -> None:
        self.simulation_images = images

    def update_memory_with_event(self, event_info: dict) -> None:
        if not self.config.get("enable_memory_updates", True) or not hasattr(
            self, "memory_curator_prompt"
        ):
            return

        event_type = event_info.get("type", "unknown")
        event_message = ""

        if event_type == "command_execution":
            command = event_info.get("command", "unknown command")
            result = event_info.get("result", "No result returned")

            result_text = str(result)
            if len(result_text) > 100:
                result_summary = result_text[:97] + "..."
            else:
                result_summary = result_text

            event_message = self.command_execution_event_template.format(
                command=command, result_summary=result_summary
            )

        elif event_type == "object_detection":
            object_name = event_info.get("object_name", "unknown object")
            camera_name = event_info.get("camera_name", "unknown camera")
            last_movement = event_info.get("last_movement", "none")

            event_message = self.object_detection_event_template.format(
                object_name=object_name,
                camera_name=camera_name,
                last_movement=last_movement,
            )

        else:
            event_message = f"Event ({event_type}):\n"
            for key, value in event_info.items():
                if key != "type":
                    event_message += f"- {key}: {value}\n"

        self._update_memory(f"\n{event_message}\n")
        cprint(f"Memory updated with {event_type} event", color=MAGENTA)


class Monitor:

    def __init__(
        self,
        camera_name: str,
        pipeline: Any,
        monitor_system_prompt: str,
        meta_analysis_prompt: str,
        upscale_models: tuple | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.pipeline = pipeline
        self.upscale_models = upscale_models
        self.monitor_system_prompt = monitor_system_prompt
        self.meta_analysis_prompt = meta_analysis_prompt
        self.pipeline.reset(
            system_prompt=self.monitor_system_prompt,
            use_history=False,
        )
        self.current_image = None

    def __call__(
        self,
        prompt: str | None = None,
        image: Image.Image | None = None,
        n_runs: int = 3,
    ) -> str:
        time.sleep(0.5)
        self._update_image(image=image)

        # upscale the final image once before sending to the VLM
        if self.upscale_models is not None and self.current_image is not None:
            model_4x, processor_4x, model_2x, processor_2x = self.upscale_models
            self.current_image = upscale_image(
                self.current_image,
                model_4x,
                processor_4x,
                model_2x,
                processor_2x,
            )

        responses = []
        for i in range(n_runs):
            response = self.pipeline(prompt, self.current_image, max_new_tokens=8192)
            responses.append(response)

        # aggregate multiple responses via a meta-analysis prompt
        if n_runs > 1:
            cprint(
                f"  Performing meta-analysis on {n_runs} responses...",
                color=CYAN,
            )
            analyses = "\n".join(
                [f"Analysis {i+1}: {resp}" for i, resp in enumerate(responses)]
            )
            meta_prompt = self.meta_analysis_prompt.format(
                n_runs=n_runs, analyses=analyses
            )
            meta_result = self.pipeline(meta_prompt, max_new_tokens=8192)
            cprint("  Meta-analysis completed", color=CYAN)
            return meta_result
        else:
            return responses[0]

    def _update_image(self, image: Image.Image | None = None) -> None:
        if image is not None:
            self.set_current_image(image)

    def get_current_image(self) -> Image.Image | None:
        if self.current_image is not None:
            return self.current_image.copy()
        else:
            return None

    def set_current_image(self, image: Image.Image) -> None:
        self.current_image = image
