# RACAS: Controlling Diverse Robots With a Single Agentic System

This repository contains the source code for the RACAS framework described in [our paper](https://arxiv.org/abs/2603.05621).

RACAS is a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. The system eliminates the need for retraining when moving between different robot embodiments, requiring only descriptions of robots, action definitions, and task specifications.

## Supported Platforms

The system has been validated across structurally diverse platforms:

- **Alhakami et al. Limb** -- a multi-jointed robotic limb
- **Clearpath Dingo** -- a wheeled ground robot (ROS 2 and NVIDIA Isaac Sim)
- **BlueROV2** -- an underwater vehicle
- **Blackjack** -- a Gymnasium card game environment used for ablation studies

## Project Structure

```
.
├── run.py                       # main entry point
├── mindstorm.py                 # MINDSTORM algorithm implementation
├── backbone.py                  # LLM/VLM backbone wrappers
├── execute.py                   # controller execution loop
├── Alhakami_limb.py             # Alhakami limb controller
├── Alhakami_limb_interface.py   # Alhakami limb hardware interface
├── blackjack_interface.py       # Blackjack environment interface
├── blackjack_wrapper.py         # Gymnasium wrapper for Blackjack
├── blackjack_modified_goal.py   # modified Blackjack environment
├── bluerov_interface.py         # BlueROV2 interface (pymavlink)
├── dingo_ros2_interface.py      # Dingo ROS 2 interface
├── isaac_sim_interface.py       # NVIDIA Isaac Sim interface
├── image_upscale.py             # camera image super-resolution
├── parse_robot_description.py   # robot description parser
├── color_print.py               # terminal output utilities
├── configs/                     # Hydra configuration files
├── prompts/                     # prompt templates and per-robot prompts
└── robots/                      # robot description files
```

## Installation

```bash
pip install -r requirements.txt
```

The Alhakami et al. Limb interface additionally requires the brain package, which must be installed manually.

Platform-specific dependencies not included in `requirements.txt`:
- **Dingo ROS 2**: requires a ROS 2 installation with `rclpy`, `geometry_msgs`, and `sensor_msgs`
- **Isaac Sim**: requires NVIDIA Isaac Sim with its Python environment

## Usage

Experiments are launched via Hydra:

```bash
python run.py --config-name <config>
```

where `<config>` is one of the YAML files in `configs/` (without the `.yaml` extension). For example:

```bash
python run.py --config-name Dingo_simulation
python run.py --config-name Alhakami_limb
python run.py --config-name BlueROV
python run.py --config-name Blackjack42
```

## Citation

```bibtex
@article{ashley2025racas,
  title={RACAS: Controlling Diverse Robots With a Single Agentic System},
  author={Ashley, Dylan R. and Przepi{\'o}ra, Jan and Chen, Yimeng and Abualsaud, Ali and Yesmagambet, Nurzhan and Park, Shinkyu and Feron, Eric and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:2603.05621},
  year={2025}
}
```
