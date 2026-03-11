#!/usr/bin/env python
# -*- coding: ascii -*-

import argparse
import os
from typing import Any
import xml.etree.ElementTree as ET


def format_for_llm(robot_data: dict | None) -> str:
    if not robot_data:
        return "Could not parse robot data."

    # build the links section
    description = f"Robot Model Description: {robot_data['model_name']}\n\n"
    description += "--- Links (Rigid Bodies) ---\n"
    if not robot_data["links"]:
        description += "No links found.\n"
    else:
        for name, data in robot_data["links"].items():
            description += f"- Link Name: '{name}'\n"
            if data.get("pose_relative_to_model"):
                description += f"  - Pose relative to model origin: {data['pose_relative_to_model']}\n"
            if data.get("sensors"):
                description += f"  - Attached Sensors: {', '.join(data['sensors'])}\n"

    # build the joints section
    description += "\n--- Joints (Connections & Potential Actuation Points) ---\n"
    if not robot_data["joints"]:
        description += "No joints found.\n"
    else:
        for name, data in robot_data["joints"].items():
            description += f"- Joint Name: '{name}'\n"
            description += f"  - Type: {data['type']}\n"
            description += (
                f"  - Connects: Link '{data['parent']}' to Link '{data['child']}'\n"
            )
            if data.get("pose_relative_to_child"):
                description += f"  - Pose relative to child link ('{data['child']}'): {data['pose_relative_to_child']}\n"
            if data.get("axis"):
                description += f"  - Axis of Motion (relative to joint frame): {data['axis']['xyz']}\n"
                if data["axis"].get("normalize"):
                    description += "    - Axis vector should be normalized.\n"
                if data["axis"].get("use_parent_model_frame"):
                    description += (
                        "    - Axis is expressed in the parent model's frame.\n"
                    )
            if data.get("limits"):
                limits = data["limits"]
                description += f"  - Limits: Lower={limits['lower']}, Upper={limits['upper']}, Effort={limits['effort']}, Velocity={limits['velocity']}\n"
            if data["type"].lower() in ["revolute", "prismatic"]:
                description += "  - Note: This is a movable joint, likely actuated by a servo/motor. Its *current* angle/position is dynamic and must be read from the robot's state.\n"
            elif data["type"].lower() == "fixed":
                description += "  - Note: This is a fixed connection, not actuated.\n"

    # build the cameras section
    description += "\n--- Cameras ---\n"
    if not robot_data["cameras"]:
        description += "No cameras found.\n"
    else:
        for name, data in robot_data["cameras"].items():
            description += f"- Camera Name: '{name}'\n"
            description += f"  - Attached to: Link '{data['parent_link']}'\n"
            if data.get("pose_relative_to_link"):
                description += f"  - Pose relative to parent link: {data['pose_relative_to_link']}\n"
            description += f"  - Update Rate: {data['update_rate']} Hz\n"
            params = data.get("camera_params", {})
            description += f"  - Image Size: {params.get('image_width')}x{params.get('image_height')}\n"
            description += (
                f"  - Horizontal FOV: {params.get('horizontal_fov')} radians\n"
            )
            description += f"  - Format: {params.get('image_format')}\n"
            description += f"  - Clipping: Near={params.get('clip_near')}, Far={params.get('clip_far')}\n"

    # build the summary with counts
    description += "\n--- Summary ---\n"
    description += (
        "This description outlines the physical structure based on the SDF file.\n"
    )
    description += "To control the robot, you need this structure AND the real-time state (current joint angles/positions) from the robot's sensors/API.\n"
    num_movable_joints = sum(
        1
        for j in robot_data["joints"].values()
        if j["type"].lower() in ["revolute", "prismatic"]
    )
    description += f"Identified {len(robot_data['links'])} links, {len(robot_data['joints'])} joints ({num_movable_joints} potentially movable), and {len(robot_data['cameras'])} cameras.\n"

    return description


def main(args: dict[str, Any] = dict()) -> None:
    sdf_file = args.get("sdf_file", "")
    parsed_data = parse_sdf(sdf_file)
    if parsed_data:
        llm_description = format_for_llm(parsed_data)
        print("\n--- LLM-Friendly Description ---")
        print(llm_description)


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="parse an SDF robot description file and print an LLM-friendly summary"
    )
    parser.add_argument(
        "sdf_file",
        type=str,
        help="path to the SDF robot description file",
    )
    return vars(parser.parse_args())


def parse_pose(pose_element: ET.Element | None) -> dict | None:
    # split pose text into position and orientation components
    if pose_element is not None and pose_element.text:
        parts = pose_element.text.split()
        if len(parts) == 6:
            return {
                "x": float(parts[0]),
                "y": float(parts[1]),
                "z": float(parts[2]),
                "roll": float(parts[3]),
                "pitch": float(parts[4]),
                "yaw": float(parts[5]),
            }
    return None


def parse_sdf(sdf_file_path: str) -> dict | None:
    if not os.path.exists(sdf_file_path):
        print(f"ERROR: SDF file not found at {sdf_file_path}")
        return None

    # parse the XML tree and locate the model element
    try:
        tree = ET.parse(sdf_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"ERROR: parsing SDF file: {e}")
        return None

    model = root.find(".//model")
    if model is None:
        print("ERROR: No <model> tag found in the SDF file.")
        return None

    # initialize the robot data structure
    robot_data = {
        "model_name": model.get("name", "unnamed_model"),
        "links": {},
        "joints": {},
        "cameras": {},
    }

    # extract links
    for link in model.findall("link"):
        link_name = link.get("name")
        if link_name:
            link_data = {"name": link_name, "sensors": list()}
            pose_elem = link.find("pose")
            link_data["pose_relative_to_model"] = parse_pose(pose_elem)
            robot_data["links"][link_name] = link_data

    # extract joints
    for joint in model.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type", "unknown")
        if joint_name:
            joint_data = {
                "name": joint_name,
                "type": joint_type,
                "parent": joint.findtext("parent"),
                "child": joint.findtext("child"),
                "axis": None,
                "limits": None,
            }

            axis_elem = joint.find("axis/xyz")
            if axis_elem is not None and axis_elem.text:
                normalize = axis_elem.get("normalize_axis", "false").lower() == "true"
                joint_data["axis"] = {
                    "xyz": axis_elem.text.strip(),
                    "normalize": normalize,
                }
                use_parent_frame = (
                    axis_elem.get("use_parent_model_frame", "false").lower() == "true"
                )
                joint_data["axis"]["use_parent_model_frame"] = use_parent_frame

            limit_elem = joint.find("axis/limit")
            if limit_elem is not None:
                joint_data["limits"] = {
                    "lower": limit_elem.findtext("lower", "-inf"),
                    "upper": limit_elem.findtext("upper", "inf"),
                    "effort": limit_elem.findtext("effort", "inf"),
                    "velocity": limit_elem.findtext("velocity", "inf"),
                }
                for key in ["lower", "upper", "effort", "velocity"]:
                    try:
                        joint_data["limits"][key] = float(joint_data["limits"][key])
                    except (ValueError, TypeError):
                        pass

            pose_elem = joint.find("pose")
            joint_data["pose_relative_to_child"] = parse_pose(pose_elem)
            robot_data["joints"][joint_name] = joint_data

    # extract sensors (specifically cameras)
    for link_name, link_data in robot_data["links"].items():
        link_element = model.find(f".//link[@name='{link_name}']")
        if link_element is not None:
            for sensor in link_element.findall("sensor"):
                sensor_name = sensor.get("name")
                sensor_type = sensor.get("type")

                if sensor_name and sensor_type == "camera":
                    camera_data = {
                        "name": sensor_name,
                        "parent_link": link_name,
                        "type": sensor_type,
                        "update_rate": sensor.findtext("update_rate", "unknown"),
                        "pose_relative_to_link": None,
                        "camera_params": {},
                    }

                    pose_elem = sensor.find("pose")
                    camera_data["pose_relative_to_link"] = parse_pose(pose_elem)

                    camera_elem = sensor.find("camera")
                    if camera_elem is not None:
                        camera_data["camera_params"] = {
                            "horizontal_fov": camera_elem.findtext(
                                "horizontal_fov", "unknown"
                            ),
                            "image_width": camera_elem.findtext(
                                "image/width", "unknown"
                            ),
                            "image_height": camera_elem.findtext(
                                "image/height", "unknown"
                            ),
                            "image_format": camera_elem.findtext(
                                "image/format", "unknown"
                            ),
                            "clip_near": camera_elem.findtext("clip/near", "unknown"),
                            "clip_far": camera_elem.findtext("clip/far", "unknown"),
                        }
                        for key in ["horizontal_fov", "clip_near", "clip_far"]:
                            try:
                                camera_data["camera_params"][key] = float(
                                    camera_data["camera_params"][key]
                                )
                            except (ValueError, TypeError):
                                pass
                        for key in ["image_width", "image_height"]:
                            try:
                                camera_data["camera_params"][key] = int(
                                    camera_data["camera_params"][key]
                                )
                            except (ValueError, TypeError):
                                pass

                    robot_data["cameras"][sensor_name] = camera_data
                    robot_data["links"][link_name]["sensors"].append(sensor_name)

    return robot_data


if __name__ == "__main__":
    main(parse_args())
