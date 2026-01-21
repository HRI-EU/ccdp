"""
Configuration and constants for the door manipulation pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def project_root() -> Path:
    """
    Resolve the repository root based on this file location.

    The file lives in ``Main Codes/`` so its parent is the project root.
    """
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DoorConfig:
    # Simulation / environment
    dt: float = 0.02
    framerate: int = 10
    n_dof: int = 4
    render_size: int = 480
    gravity: tuple[int, int, int] = (0, 0, 0)
    timestep: float = 0.01

    # Trajectory timing
    total_time: float = 13.0
    t_pre_reach: float = 4.0
    t_reach: float = 6.0
    t_grasp: float = 7.0
    t_final_reach: float = 12.0

    # Data generation
    n_demos: int = 2500
    seed: int = 0
    preload_dataset: bool = False
    save_generated_dataset: bool = True
    dataset_cache_name: str = "door_demos.pkl"
    avoidance_dataset_name: str = "door_demos_avoidance.pt"
    avoidance_preload: bool = True
    avoidance_save: bool = True
    avoidance_preload_model: bool = True
    avoidance_n_envs: int = 1000
    avoidance_samples_per_env: int = 20
    avoidance_action_limit: float = 0.5
    avoidance_distance_threshold: float = 0.3

    # Diffusion
    n_diffusion_steps: int = 200
    batch_size: int = 32
    batch_size_avoidance: int = 256
    n_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    preload_uncond_model: bool = True
    preload_state_model: bool = True
    default_camera: str = "closeup_xyz"
    default_stride: int = 1

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def traj_length(self) -> int:
        return int(self.total_time / self.dt)

    @property
    def project_dir(self) -> Path:
        return project_root()

    @property
    def model_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def dataset_cache_path(self) -> Path:
        return self.model_dir / self.dataset_cache_name

    @property
    def uncond_ckpt_path(self) -> Path:
        return self.model_dir / "door_uncond_model.pth"

    @property
    def state_ckpt_path(self) -> Path:
        return self.model_dir / "door_state_model.pth"

    @property
    def avoidance_dataset_path(self) -> Path:
        return self.model_dir / self.avoidance_dataset_name

    @property
    def avoidance_ckpt_path(self) -> Path:
        return self.model_dir / "door_avoidance_model.pth"

    @property
    def xml_path(self) -> Path:
        return self.project_dir / "xml" / "my_env_hand.xml"

    @property
    def xml_path_vis(self) -> Path:
        return self.project_dir / "xml" / "my_env_multi_hand.xml"

    @property
    def slide_mode_labels(self) -> tuple[str, ...]:
        return (
            "hinge about z (lift)",
            "hinge about y (pull -pi/2..0)",
            "hinge about x (pull -pi/2..0)",
            "hinge about x (push 0..pi/2)",
        )
