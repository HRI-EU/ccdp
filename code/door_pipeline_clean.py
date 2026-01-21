"""
Cleaned door manipulation pipeline based on `Door_Final_revised.ipynb`.

This script keeps the original functionality (trajectory synthesis, diffusion
model training, and conditional sampling) but organizes it into reusable
functions with clear boundaries between simulation logic and learning code.

High-level flow:
1) Build a Mujoco environment and synthesize demonstrations.
2) Normalize trajectories and create dataloaders.
3) Train an unconditional diffusion model and a state-conditioned one.
4) Sample key-frame proposals conditioned on the handle position.

Nothing runs on import; call `main()` or individual functions from your own
scripts/notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import fmin_slsqp
from torch.utils.data import DataLoader
from tqdm import trange

from door_config import DoorConfig
from door_models import (
    ConditionalUnet1D,
    Normalizer,
    TrajectoryDataset,
    cosine_beta_schedule,
    forward_diffusion,
)


@dataclass
class DemoTrajectories:
    ee: np.ndarray
    q: np.ndarray
    obj: np.ndarray
    gripper: np.ndarray
    all_states: np.ndarray
    failed: np.ndarray
    x_des: np.ndarray
    key_states: np.ndarray
    cabinet_pos: np.ndarray
    slide_mode: np.ndarray
    start_state: np.ndarray


@dataclass(frozen=True)
class DemoContext:
    """
    Minimal environment state needed to replay a stored demonstration.
    """

    cabinet_pos: np.ndarray
    slide_mode: int
    start_state: np.ndarray
    x_des: np.ndarray
    door_angle: float


def save_demos(path: Path, demos: DemoTrajectories) -> None:
    """
    Persist demos to disk for reuse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ee": demos.ee,
        "q": demos.q,
        "obj": demos.obj,
        "gripper": demos.gripper,
        "all_states": demos.all_states,
        "failed": demos.failed,
        "x_des": demos.x_des,
        "key_states": demos.key_states,
        "cabinet_pos": demos.cabinet_pos,
        "slide_mode": demos.slide_mode,
        "start_state": demos.start_state,
    }
    torch.save(payload, path)


def load_demos(path: Path) -> DemoTrajectories:
    """
    Load previously saved demos from disk.
    """
    payload = torch.load(path, map_location="cpu")
    n_demos = payload["q"].shape[0] if payload["q"].ndim == 3 else 1
    if "slide_mode" not in payload:
        payload["slide_mode"] = np.zeros(n_demos, dtype=np.int64)
    if "start_state" not in payload:
        all_states = payload["all_states"]
        if all_states.ndim == 2:
            payload["start_state"] = all_states[0].copy()
        else:
            payload["start_state"] = all_states[:, 0].copy()
    return DemoTrajectories(**payload)


def build_environment(cfg: DoorConfig):
    """
    Initialize Mujoco models, data buffers, and renderers.
    """
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    model_vis = mujoco.MjModel.from_xml_path(str(cfg.xml_path_vis))

    data = mujoco.MjData(model)
    data_aux = mujoco.MjData(model)  # temporary buffer for IK
    data_vis = mujoco.MjData(model_vis)

    model.opt.timestep = cfg.timestep
    model.opt.gravity = cfg.gravity

    mujoco.mj_resetDataKeyframe(model_vis, data_vis, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=cfg.render_size, width=cfg.render_size)
    renderer_vis = mujoco.Renderer(model_vis, height=cfg.render_size, width=cfg.render_size)
    return model, model_vis, data, data_aux, data_vis, renderer, renderer_vis


def update_joint_model(model: mujoco.MjModel, slide_mode: int) -> mujoco.MjModel:
    """
    Adjust hinge parameters depending on the door opening mode.
    """
    joint = model.joint("boxdoorhinge")
    if slide_mode == 0: #lift
        joint.axis = np.array([0, 0, 1])
        joint.type = np.array([2])
        joint.range = np.array([-np.pi / 2, 0])
    elif slide_mode == 1:#pull y
        joint.axis = np.array([0, 1, 0])
        joint.type = np.array([2])
        joint.range = np.array([-np.pi / 2, 0])
    elif slide_mode == 2:#pull x
        joint.axis = np.array([1, 0, 0])
        joint.type = np.array([2])
        joint.range = np.array([-np.pi / 2, 0])
    elif slide_mode == 3:#push x
        joint.axis = np.array([1, 0, 0])
        joint.type = np.array([2])
        joint.range = np.array([0, np.pi / 2])
    elif slide_mode == -1:# fixed
        joint.axis = np.array([0, 1, 0])
        joint.type = np.array([3])
        joint.range = np.array([0, 0])
    return model


def slide_mode_label(slide_mode: int, cfg: Optional[DoorConfig] = None) -> str:
    """
    Human-readable label for the sampled door opening direction.
    """
    if cfg is not None and 0 <= slide_mode < len(cfg.slide_mode_labels):
        return cfg.slide_mode_labels[slide_mode]
    return {
        0: "hinge about z (lift)",
        1: "hinge about y (pull -pi/2..0)",
        2: "hinge about x (pull -pi/2..0)",
        3: "hinge about x (push 0..pi/2)",
    }.get(slide_mode, f"mode {slide_mode}")


def extract_demo_context(demos: DemoTrajectories, demo_idx: int) -> DemoContext:
    """
    Gather the environment information associated with a given demonstration.
    """
    n_demos = demos.all_states.shape[0]
    if demo_idx < 0 or demo_idx >= n_demos:
        raise IndexError(f"Demo index {demo_idx} out of range [0, {n_demos})")

    slide_mode_raw = np.asarray(demos.slide_mode[demo_idx]).item()
    return DemoContext(
        cabinet_pos=demos.cabinet_pos[demo_idx],
        slide_mode=int(slide_mode_raw),
        start_state=demos.start_state[demo_idx],
        x_des=demos.x_des[demo_idx],
        door_angle=float(demos.key_states[demo_idx, 3, -2]),
    )


def ik_objective(
    q: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x_des_point: np.ndarray,
    x_des_orient: np.ndarray,
    z_des_orient: np.ndarray,
    offset: float,
    offset_orient_z: float,
    q_ref: np.ndarray,
    n_dof: int,
) -> float:
    """
    IK objective: blend position, orientation, and regularization.
    """
    data.qpos[:n_dof] = q
    mujoco.mj_forward(model, data)
    cur_pos = (data.body("right_silicone_pad").xpos + data.body("left_silicone_pad").xpos) / 2
    cur_x = data.body("base").xmat.reshape([3, 3])[:, 0]
    cur_z = data.body("base").xmat.reshape([3, 3])[:, 2]

    theta = abs(-cur_x @ x_des_orient)
    theta_2 = max(0.0, offset_orient_z - cur_z @ z_des_orient)
    pos_res = cur_pos - (x_des_point - cur_z * offset)
    error = 10 * np.linalg.norm(pos_res) - theta + theta_2 + 0.01 * np.linalg.norm(q - q_ref)
    return float(error)


def step_trajectory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_traj: np.ndarray,
    n_dof: int,
    gripper: float = 1.0,
    render: bool = False,
    renderer: Optional[mujoco.Renderer] = None,
    framerate: int = 10,
) -> Tuple[mujoco.MjData, List[np.ndarray], Dict[str, np.ndarray]]:
    """
    Execute a joint trajectory and optionally render frames.
    """
    n_steps = q_traj.shape[0]
    frames: List[np.ndarray] = []
    q_hist = np.zeros([n_steps, n_dof])
    ee_hist = np.zeros([n_steps, 3])
    obj_hist = np.zeros([n_steps])
    gripper_hist = np.zeros([n_steps])
    full_state = np.zeros([n_steps, model.nq])
    force_hist = np.zeros([n_steps, 6])
    passive_force = np.zeros([n_steps, 1])

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)

    time_0 = data.time
    for i in range(n_steps):
        data.ctrl[:n_dof] = q_traj[i, :]
        data.ctrl[n_dof] = int(gripper * 255)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        mujoco.mj_rnePostConstraint(model, data)

        q_hist[i, :] = data.qpos[:n_dof]
        ee_hist[i, :] = (data.body("right_silicone_pad").xpos + data.body("left_silicone_pad").xpos) / 2
        obj_hist[i] = data.qpos[-1]
        gripper_hist[i] = data.ctrl[n_dof] > 0
        full_state[i, :] = data.qpos
        force_hist[i, :] = data.cfrc_ext[17, :]
        passive_force[i, 0] = data.qfrc_passive[-1]

        if render and renderer is not None:
            if len(frames) < (data.time - time_0) * framerate:
                renderer.update_scene(data, camera="closeup_xyz")
                frames.append(renderer.render())

    traj = {
        "q": q_hist,
        "ee": ee_hist,
        "Object": obj_hist,
        "gripper": gripper_hist,
        "All": full_state,
        "force": force_hist,
        "q_force": passive_force,
    }
    return data, frames, traj


def set_robot_phase(data: mujoco.MjData, via_points: np.ndarray, phase: float, n_dof: int) -> None:
    """
    Place the robot at a point along the via-point sequence, where phase is in [0, 1].
    """
    phase_clamped = np.clip(phase, 0.0, 1.0)
    idx = int(round(phase_clamped * (via_points.shape[0] - 1)))
    data.qpos[:n_dof] = via_points[idx]
    mujoco.mj_forward(data.model, data)


def pid_follow_via_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    via_points: np.ndarray,
    phase: float,
    phase_lookahead: float,
    kp: float,
    kd: float,
    gripper_closed: bool,
    gripper_close_threshold: float,
) -> Tuple[np.ndarray, bool]:
    """
    Compute a PD control command that tracks a via point ahead of the current phase.

    - phase: current task phase in [0, 1].
    - phase_lookahead: how far ahead to target, also in [0, 1]; e.g., 0.05 means 5% ahead.
    - gripper_closed is sticky; it is latched once the hand is close to the door handle.
    """
    n_points = via_points.shape[0]
    n_dof = model.nu - 1  # last ctrl is gripper

    target_phase = np.clip(phase + phase_lookahead, 0.0, 1.0)
    target_idx = int(round(target_phase * (n_points - 1)))
    print("target_idx:", target_idx, "phase:", phase, "lookahead:", phase_lookahead)
    q_des = via_points[target_idx]

    q_err = q_des - data.qpos[:n_dof]
    qd_err = -data.qvel[:n_dof]
    print("q_err:", q_err, "qd_err:", qd_err)
    ctrl = kp * q_err + kd * qd_err

    # Gripper latch: once close, keep closed.
    if not gripper_closed:
        ee_pos = (data.body("right_silicone_pad").xpos + data.body("left_silicone_pad").xpos) / 2
        handle_pos = data.geom("door_handle").xpos
        gripper_closed = np.linalg.norm(ee_pos - handle_pos) <= gripper_close_threshold

    full_ctrl = np.zeros(model.nu)
    full_ctrl[:n_dof] = q_des
    full_ctrl[n_dof] = 255 if gripper_closed else 0
    return full_ctrl, gripper_closed


def estimate_phase_from_state(
    cfg: DoorConfig,
    data: mujoco.MjData,
    via_points: np.ndarray,
    n_dof: int,
    gripper_closed: bool,
    interp_points: int = 10,
) -> Tuple[float, int, np.ndarray, np.ndarray]:
    """
    Estimate the current phase in [0, 1] by projecting the robot joint state onto the
    via points, accounting for non-uniform timing in keyframes. The path is optionally
    densified with linear interpolation between consecutive via points to make phase
    estimation smoother.

    Returns (phase, closest_index, via_points_dense, phases_dense).
    """
    if via_points.shape[0] < 2:
        return 0.0, 0, via_points, np.array([0.0])

    keyframe_indices = [
        int(cfg.t_pre_reach / 3 / cfg.dt),
        int(cfg.t_pre_reach * 2 / 3 / cfg.dt),
        int(cfg.t_pre_reach * 3 / 3 / cfg.dt),
        int((cfg.t_pre_reach + cfg.t_reach) / 2 / cfg.dt),
        int(cfg.t_reach / cfg.dt),
        int((cfg.t_reach + (cfg.t_final_reach - cfg.t_grasp) / 3) / cfg.dt),
        int((cfg.t_reach + 2 * (cfg.t_final_reach - cfg.t_grasp) / 3) / cfg.dt),
        int(cfg.t_final_reach / cfg.dt),
    ]

    if via_points.shape[0] == len(keyframe_indices):
        keyframe_indices_arr = np.array(keyframe_indices, dtype=float)
        keyframe_phases = (keyframe_indices_arr - keyframe_indices_arr[0]) / (
            keyframe_indices_arr[-1] - keyframe_indices_arr[0]
        )
        reach_idx = int(
            np.argmin(np.abs(keyframe_indices_arr - int(cfg.t_reach / cfg.dt)))
        )
        reach_phase = float(keyframe_phases[reach_idx])
    else:
        keyframe_phases = np.linspace(0.0, 1.0, via_points.shape[0])
        reach_phase = float(
            np.clip(cfg.t_reach / cfg.t_final_reach, 0.0, 1.0)
            if cfg.t_final_reach > 0
            else 0.0
        )

    if interp_points > 0:
        interp_segments: List[np.ndarray] = []
        phase_segments: List[np.ndarray] = []
        for i in range(via_points.shape[0] - 1):
            segment = np.linspace(
                via_points[i],
                via_points[i + 1],
                num=interp_points + 2,  # include endpoints, drop last to avoid duplicates
                endpoint=True,
            )[:-1]
            interp_segments.append(segment)
            phase_segment = np.linspace(
                keyframe_phases[i],
                keyframe_phases[i + 1],
                num=interp_points + 2,
                endpoint=True,
            )[:-1]
            phase_segments.append(phase_segment)
        interp_segments.append(via_points[-1][None, :])  # final endpoint
        phase_segments.append(np.array([keyframe_phases[-1]]))
        via_points_dense = np.concatenate(interp_segments, axis=0)
        phases_dense = np.concatenate(phase_segments, axis=0)
    else:
        via_points_dense = via_points
        phases_dense = keyframe_phases

    if gripper_closed:
        mask = phases_dense >= reach_phase
    else:
        mask = phases_dense <= reach_phase

    if not np.any(mask):
        return 0.0, 0, via_points_dense, phases_dense

    des_via_points = via_points_dense[mask]
    des_phases = phases_dense[mask]
    des_indices = np.flatnonzero(mask)

    if des_via_points.shape[0] < 2:
        return 0.0, 0, via_points_dense, phases_dense

    q_cur = data.qpos[:n_dof]
    dists = np.linalg.norm(des_via_points[:, :n_dof] - q_cur[None, :], axis=1)
    idx_local = int(np.argmin(dists))
    idx = int(des_indices[idx_local])
    phase = float(des_phases[idx_local])
    return phase, idx, via_points_dense, phases_dense


def step_along_via_points(
    cfg: DoorConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    via_points: np.ndarray,
    phase_lookahead: float,
    kp: float,
    kd: float,
    gripper_closed: bool,
    gripper_close_threshold: float,
    interp_points: int = 10,
    new_traj: bool = False,
) -> Tuple[mujoco.MjData, bool, float]:
    """
    One control step that:
    1) Estimates current phase from joint positions.
    2) Computes a PD command toward a future via point (phase + lookahead).
    3) Steps the simulation and returns updated data, gripper latch, and phase.
    """
    n_dof = model.nu - 1

    phase, idx, via_points_dense, phases_dense = estimate_phase_from_state(
        cfg=cfg,
        data=data,
        via_points=via_points,
        n_dof=n_dof,
        gripper_closed=gripper_closed,
        interp_points=interp_points,
    )
    desired_phase = min(phase + phase_lookahead, 1.0)
    # print(desired_phase)
    if new_traj:
        desired_phase = min(desired_phase, 0.15)
    desired_idx = int(np.argmin(np.abs(phases_dense - desired_phase)))
    # print("Estimated phase:", phase)
    # print("Closest index:", idx, "desired idx:", desired_idx )    
    ctrl = via_points_dense[desired_idx, :n_dof]
    # print(via_points)
    # print("Control:", data.ctrl)
    data.ctrl[:n_dof] = ctrl
    data.ctrl[n_dof] = 255 if gripper_closed else 0
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return data, phase


# def visualize_key_frames(
#     model_vis: mujoco.MjModel,
#     data_vis: mujoco.MjData,
#     renderer_vis: mujoco.Renderer,
#     key_frames: np.ndarray,
#     n_dof: int,
# ) -> np.ndarray:
#     """
#     Render three canonical camera views for a few key configurations.
#     """
#     mujoco.mj_resetDataKeyframe(model_vis, data_vis, 0)
#     data_vis.qpos[:n_dof] = key_frames[0, :n_dof]
#     data_vis.qpos[12:16] = key_frames[4, :n_dof]
#     data_vis.qpos[24:28] = key_frames[-1, :n_dof]
#     mujoco.mj_forward(model_vis, data_vis)

#     renderer_vis.update_scene(data_vis, camera="closeup_xyz")
#     image_xyz = renderer_vis.render()

#     renderer_vis.update_scene(data_vis, camera="closeup_xz")
#     image_xz = renderer_vis.render()

#     renderer_vis.update_scene(data_vis, camera="closeup_xy")
#     image_xy = renderer_vis.render()
#     return np.concatenate([image_xyz, image_xz, image_xy], axis=1).astype(np.uint8)


def visualize_key_frames(
    cfg: DoorConfig,
    x_des: np.ndarray,
    key_frames: np.ndarray,
) -> np.ndarray:
    """
    Replay a stored demonstration along with its environment context.
    """
    _, model_vis, _, _, data_vis, _, renderer_vis = build_environment(cfg)
    model_vis.body("cabinet").pos = x_des
    mujoco.mj_forward(model_vis, data_vis)

    # mujoco.mj_resetDataKeyframe(model_vis, data_vis, 0)
    data_vis.qpos[:4] = key_frames[0, :4]
    data_vis.qpos[12:16] = key_frames[4, :4]
    data_vis.qpos[24:28] = key_frames[-1, :4]
    mujoco.mj_forward(model_vis, data_vis)

    renderer_vis.update_scene(data_vis, camera="closeup_xyz")
    image_xyz = renderer_vis.render()

    renderer_vis.update_scene(data_vis, camera="closeup_xz")
    image_xz = renderer_vis.render()

    renderer_vis.update_scene(data_vis, camera="closeup_xy")
    image_xy = renderer_vis.render()
    return np.concatenate([image_xyz, image_xz, image_xy], axis=1).astype(np.uint8)

@dataclass
class DemoPlayback:
    """
    Convenience bundle for demo playback results.
    """

    frames: List[np.ndarray]
    context: DemoContext
    trajectories: Dict[str, np.ndarray]


def visualize_demo_with_context(
    cfg: DoorConfig,
    demos: DemoTrajectories,
    demo_idx: int,
    camera: str = "closeup_xyz",
    stride: int = 1,
) -> DemoPlayback:
    """
    Replay a stored demonstration along with its environment context.
    """
    context = extract_demo_context(demos, demo_idx)
    model, _, data, _, _, renderer, _ = build_environment(cfg)
    model = update_joint_model(model, context.slide_mode)
    model.body("cabinet").pos = context.cabinet_pos
    mujoco.mj_forward(model, data)

    frames: List[np.ndarray] = []
    all_states = demos.all_states[demo_idx]
    t_idx = 0
    for step in range(0, all_states.shape[0], stride):
        # print(data.qpos.shape, data.qpos.shape, all_states[step].shape)
        data.qpos[:] = all_states[step]
        mujoco.mj_forward(model, data)
        t_idx += model.opt.timestep # Manually added due to lack of step function (only visualization)
        if len(frames) < (t_idx) * cfg.framerate:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    trajectories = {
        "q": demos.q[demo_idx],
        "ee": demos.ee[demo_idx],
        "obj": demos.obj[demo_idx],
        "gripper": demos.gripper[demo_idx],
    }
    return DemoPlayback(frames=frames, context=context, trajectories=trajectories)


def visualize_demo(
    cfg: DoorConfig,
    demos: DemoTrajectories,
    demo_idx: int,
    camera: str = "closeup_xyz",
    stride: int = 1,
) -> List[np.ndarray]:
    """
    Render a full demonstration by replaying stored joint states.

    The cabinet pose is restored from the stored demo to keep visuals aligned
    with the random environment configuration that was used for generation.
    """
    return visualize_demo_with_context(cfg, demos, demo_idx, camera=camera, stride=stride).frames


def generate_single_demo(
    cfg: DoorConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    data_aux: mujoco.MjData,
    renderer: mujoco.Renderer,
    rng: np.random.Generator,
) -> DemoTrajectories:
    """
    Generate one demonstration by sampling a door pose and a motion mode.
    """
    dt = cfg.dt
    traj_length = cfg.traj_length
    n_dof = cfg.n_dof

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_resetDataKeyframe(model, data_aux, 0)

    # Randomize cabinet pose
    cabinet_pos = np.array([rng.random() + 0.8, rng.random(), rng.random() + 0.25])
    model.body("cabinet").pos = cabinet_pos

    mujoco.mj_forward(model, data)
    x_des = data.geom("door_handle").xpos.copy()
    x_des_orient = data.geom("door_handle").xmat.reshape([3, 3])[2, :].copy()
    hinge_location = data.body("hingedoor").xpos
    L = x_des[1] - hinge_location[1]
    x_move_traj = np.linspace(0, L, int((cfg.t_final_reach - cfg.t_grasp) / dt))

    slide_mode = int(rng.random() * 4)
    model = update_joint_model(model, slide_mode)

    traj_ee_pos_x = x_des[0] * np.ones([int((cfg.t_final_reach - cfg.t_grasp) / dt)])
    traj_ee_pos_y = x_des[1] * np.ones_like(traj_ee_pos_x)
    traj_ee_pos_z = x_des[2] * np.ones_like(traj_ee_pos_x)
    obj_traj_id = 0
    if slide_mode == 0: # lift
        traj_ee_pos_z = np.linspace(x_des[2], x_des[2] + 0.2, traj_ee_pos_x.shape[0])
        obj_traj_id = 2
    elif slide_mode == 1: # pull out
        traj_ee_pos_x = x_des[0] - x_move_traj
        obj_traj_id = 0
    elif slide_mode == 2: # slide right
        traj_ee_pos_y = x_des[1] - x_move_traj
        obj_traj_id = 1
    elif slide_mode == 3: # slide left
        traj_ee_pos_y = x_des[1] + x_move_traj
        obj_traj_id = 1

    q_init = data.qpos[:n_dof].copy()

    normal_alignment = 0.5 + 0.5 * rng.random()
    q_des_1 = fmin_slsqp(
        ik_objective,
        q_init,
        args=(model, data_aux, x_des, x_des_orient, np.array([1, 0, 0]), 0.13, normal_alignment, q_init, n_dof),
        iprint=False,
    )
    data_aux.qpos[:n_dof] = q_des_1
    mujoco.mj_forward(model, data_aux)
    x_des_1 = (data_aux.body("right_silicone_pad").xpos + data_aux.body("left_silicone_pad").xpos) / 2
    des_z_orient = data_aux.body("base").xmat.reshape([3, 3])[:, 2]

    traj_pre = np.linspace(q_init, q_des_1, int(cfg.t_pre_reach / dt))

    traj_reach = np.zeros([int((cfg.t_reach - cfg.t_pre_reach) / dt), n_dof])
    ee_pos_des = np.linspace(x_des_1, x_des, traj_reach.shape[0])
    q_prev = q_des_1.copy()
    for j in range(traj_reach.shape[0]):
        x_des_i = ee_pos_des[j]
        q_des_2 = fmin_slsqp(
            ik_objective,
            q_prev,
            args=(model, data_aux, x_des_i, x_des_orient, des_z_orient, 0.0, 1.0, q_prev, n_dof),
            iprint=False,
        )
        data_aux.qpos[:n_dof] = q_des_2
        mujoco.mj_forward(model, data_aux)
        des_z_orient = data_aux.body("base").xmat.reshape([3, 3])[:, 2]
        traj_reach[j, :] = q_des_2
        q_prev = q_des_2.copy()

    traj_grasp = np.linspace(traj_reach[-1], traj_reach[-1], int((cfg.t_grasp - cfg.t_reach) / dt))

    data, _, traj_part1 = step_trajectory(
        model, data, traj_pre, n_dof, gripper=0.0, render=False, renderer=renderer, framerate=cfg.framerate
    )
    data, _, traj_part2 = step_trajectory(
        model, data, traj_reach, n_dof, gripper=0.0, render=False, renderer=renderer, framerate=cfg.framerate
    )
    data, _, traj_part3 = step_trajectory(
        model, data, traj_grasp, n_dof, gripper=1.0, render=False, renderer=renderer, framerate=cfg.framerate
    )

    opening_traj: List[np.ndarray] = []
    q_init_open = data.qpos[:n_dof].copy()
    for j in range(traj_ee_pos_y.shape[0]):
        x_des_i = np.array([traj_ee_pos_x[j], traj_ee_pos_y[j], traj_ee_pos_z[j]])
        q_des_i = fmin_slsqp(
            ik_objective,
            q_init_open,
            args=(model, data_aux, x_des_i, x_des_orient, des_z_orient, 0.0, 0.98, q_init_open, n_dof),
            iprint=False,
        )
        data_aux.qpos[:n_dof] = q_des_i
        mujoco.mj_forward(model, data_aux)
        des_z_orient = data_aux.body("base").xmat.reshape([3, 3])[:, 2]
        q_init_open = q_des_i.copy()
        opening_traj.append(q_des_i)

    opening_traj.extend([opening_traj[-1]] * int((cfg.total_time - cfg.t_final_reach) / dt))
    opening_array = np.array(opening_traj)
    data, _, traj_open = step_trajectory(
        model, data, opening_array, n_dof, gripper=1.0, render=False, renderer=renderer, framerate=cfg.framerate
    )

    ee_traj = np.zeros([traj_length, 3])
    q_traj = np.zeros([traj_length, n_dof])
    obj_traj = np.zeros([traj_length, 3])
    gripper_traj = np.zeros([traj_length])
    all_states = np.zeros([traj_length, model.nq])
    force_traj = np.zeros([traj_length, 6])
    q_forces = np.zeros([traj_length, 1])
    failure_traj = np.zeros([traj_length, 4])

    idx_pre = slice(0, int(cfg.t_pre_reach / dt))
    idx_reach = slice(int(cfg.t_pre_reach / dt), int(cfg.t_reach / dt))
    idx_grasp = slice(int(cfg.t_reach / dt), int(cfg.t_grasp / dt))
    idx_open = slice(int(cfg.t_grasp / dt), traj_length)

    ee_traj[idx_pre, :] = traj_part1["ee"]
    ee_traj[idx_reach, :] = traj_part2["ee"]
    ee_traj[idx_grasp, :] = traj_part3["ee"]
    ee_traj[idx_open, :] = traj_open["ee"]

    q_traj[idx_pre, :] = traj_part1["q"]
    q_traj[idx_reach, :] = traj_part2["q"]
    q_traj[idx_grasp, :] = traj_part3["q"]
    q_traj[idx_open, :] = traj_open["q"]
    obj_traj[idx_open, obj_traj_id] = traj_open["Object"]

    gripper_traj[idx_pre] = traj_part1["gripper"]
    gripper_traj[idx_reach] = traj_part2["gripper"]
    gripper_traj[idx_grasp] = traj_part3["gripper"]
    gripper_traj[idx_open] = traj_open["gripper"]

    all_states[idx_pre, :] = traj_part1["All"]
    all_states[idx_reach, :] = traj_part2["All"]
    all_states[idx_grasp, :] = traj_part3["All"]
    all_states[idx_open, :] = traj_open["All"]

    force_traj[idx_pre, :] = traj_part1["force"]
    force_traj[idx_reach, :] = traj_part2["force"]
    force_traj[idx_grasp, :] = traj_part3["force"]
    force_traj[idx_open, :] = traj_open["force"]

    q_forces[idx_pre, :] = traj_part1["q_force"]
    q_forces[idx_reach, :] = traj_part2["q_force"]
    q_forces[idx_grasp, :] = traj_part3["q_force"]
    q_forces[idx_open, :] = traj_open["q_force"]

    for mode in range(4):
        if mode != slide_mode:
            failure_traj[idx_open, mode] = 1

    key_states = np.zeros([4, n_dof + 2])
    key_states[0, :n_dof] = q_init
    key_states[0, -1] = slide_mode
    key_states[1, :n_dof] = traj_part1["q"][-1]
    key_states[2, :n_dof] = traj_part2["q"][-1]
    key_states[3, :n_dof] = traj_open["q"][-1]
    key_states[3, -2] = data.qpos[-1]

    demo = DemoTrajectories(
        ee=ee_traj,
        q=q_traj,
        obj=obj_traj,
        gripper=gripper_traj,
        all_states=all_states,
        failed=failure_traj,
        x_des=x_des,
        key_states=key_states,
        cabinet_pos=cabinet_pos,
        slide_mode=np.array(slide_mode, dtype=np.int64),
        start_state=all_states[0].copy(),
    )
    return demo


def generate_dataset(cfg: DoorConfig, seed: int = 0) -> DemoTrajectories:
    """
    Generate multiple demonstrations and stack them together.
    """
    rng = np.random.default_rng(seed)
    model, _, data, data_aux, _, renderer, _ = build_environment(cfg)

    demos: List[DemoTrajectories] = []
    for idx in trange(cfg.n_demos, desc="generating demos"):
        demo = generate_single_demo(cfg, model, data, data_aux, renderer, rng)
        demos.append(demo)

    def stack(field: str) -> np.ndarray:
        return np.stack([getattr(d, field) for d in demos], axis=0)

    return DemoTrajectories(
        ee=stack("ee"),
        q=stack("q"),
        obj=stack("obj"),
        gripper=stack("gripper"),
        all_states=stack("all_states"),
        failed=stack("failed"),
        x_des=stack("x_des"),
        key_states=stack("key_states"),
        cabinet_pos=stack("cabinet_pos"),
        slide_mode=stack("slide_mode"),
        start_state=stack("start_state"),
    )


def load_or_generate_dataset(cfg: DoorConfig, seed: int = 0) -> DemoTrajectories:
    """
    Load demos from disk if requested, otherwise generate and optionally cache them.
    """
    if cfg.preload_dataset and cfg.dataset_cache_path.exists():
        print(f"Loading cached demos from {cfg.dataset_cache_path}")
        return load_demos(cfg.dataset_cache_path)

    demos = generate_dataset(cfg, seed=seed)
    if cfg.save_generated_dataset:
        save_demos(cfg.dataset_cache_path, demos)
        print(f"Saved demos to {cfg.dataset_cache_path}")
    return demos


def build_dataloaders(cfg: DoorConfig, demos: DemoTrajectories):
    """
    Create normalized datasets and dataloaders for unconditional and state-conditioned training.
    """
    n_demos, traj_len, _ = demos.q.shape
    keyframe_indices = [
        int(cfg.t_pre_reach / 3 / cfg.dt),
        int(cfg.t_pre_reach * 2 / 3 / cfg.dt),
        int(cfg.t_pre_reach * 3 / 3 / cfg.dt),
        int((cfg.t_pre_reach + cfg.t_reach) / 2 / cfg.dt),
        int(cfg.t_reach / cfg.dt),
        int((cfg.t_reach + (cfg.t_final_reach - cfg.t_grasp) / 3) / cfg.dt),
        int((cfg.t_reach + 2 * (cfg.t_final_reach - cfg.t_grasp) / 3) / cfg.dt),
        traj_len - 1,
    ]

    full_traj = np.zeros([n_demos, len(keyframe_indices), cfg.n_dof])
    for idx, k in enumerate(keyframe_indices):
        full_traj[:, idx, :] = demos.all_states[:, k, : cfg.n_dof]

    x_des_traj = torch.from_numpy(demos.x_des).to(cfg.device)
    actions = torch.from_numpy(full_traj).to(cfg.device)

    state_norm = Normalizer()
    state_norm.fit(x_des_traj)
    action_norm = Normalizer()
    action_norm.fit(actions)

    states_normalized = state_norm.normalize(x_des_traj)
    actions_normalized = action_norm.normalize(actions)

    uncond_dataset = TrajectoryDataset(actions_normalized)
    state_dataset = TrajectoryDataset(actions_normalized, states_normalized)

    uncond_loader = DataLoader(uncond_dataset, batch_size=cfg.batch_size, shuffle=True)
    state_loader = DataLoader(state_dataset, batch_size=cfg.batch_size, shuffle=True)

    return (
        action_norm,
        state_norm,
        uncond_loader,
        state_loader,
    )


def train_diffusion_model(
    model: ConditionalUnet1D,
    dataloader: DataLoader,
    cfg: DoorConfig,
    conditioning: bool,
    save_path: Optional[Path] = None,
) -> List[float]:
    """
    Generic diffusion training loop used for both unconditional and conditioned models.
    """
    device = cfg.device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * cfg.n_epochs)

    beta_schedule = cosine_beta_schedule(cfg.n_diffusion_steps).to(device)
    alpha = 1.0 - beta_schedule
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    epoch_losses: List[float] = []
    for epoch in trange(cfg.n_epochs, desc="train", leave=False):
        total_loss = 0.0
        for batch in dataloader:
            if conditioning:
                actions, labels = batch[0].to(device), batch[1].to(device)
            else:
                actions = batch.to(device)
                labels = None
            n_batch = actions.shape[0]
            noise = torch.randn_like(actions, device=device)
            t = torch.randint(0, cfg.n_diffusion_steps, (n_batch,), device=device)
            noisy, noise = forward_diffusion(actions, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod)
            pred_noise = model(noisy, t, labels) if conditioning else model(noisy, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.detach().item()
        epoch_losses.append(total_loss / len(dataloader))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return epoch_losses


def sample_conditioned(
    uncond_model: ConditionalUnet1D,
    state_model: ConditionalUnet1D,
    actions_norm: Normalizer,
    state_norm: Normalizer,
    label: torch.Tensor,
    cfg: DoorConfig,
    n_samples: int = 30,
) -> torch.Tensor:
    """
    Sample trajectories given a state label using classifier-free guidance.
    """
    device = cfg.device
    beta_schedule = cosine_beta_schedule(cfg.n_diffusion_steps).to(device)
    alpha = 1.0 - beta_schedule
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    label_scaled = state_norm.normalize(label.reshape(1, -1)).to(torch.float32).to(device).repeat(n_samples, 1)
    x = torch.randn(n_samples, 8, cfg.n_dof, device=device)

    with torch.no_grad():
        for t in reversed(range(cfg.n_diffusion_steps)):
            beta_t = beta_schedule[t]
            alpha_t = alpha[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cump_t = sqrt_one_minus_alpha_cumprod[t]
            noise = torch.randn_like(x, device=device) if t > 0 else 0
            pred_noise_state = state_model(x, torch.full((x.size(0),), t, device=device, dtype=torch.long), label_scaled)
            dx_state = (x - beta_t / sqrt_one_minus_alpha_cump_t * pred_noise_state) / sqrt_alpha_t - x
            pred_noise_uncond = uncond_model(x, torch.full((x.size(0),), t, device=device, dtype=torch.long))
            dx_uncond = (x - beta_t / sqrt_one_minus_alpha_cump_t * pred_noise_uncond) / sqrt_alpha_t - x
            dx = dx_uncond + 0.8 * (dx_state - dx_uncond)
            x = x + dx + noise * beta_t**0.5
    return actions_norm.unnormalize(x)


def sample_dp_explorative(
    uncond_model: ConditionalUnet1D,
    state_model: ConditionalUnet1D,
    actions_norm: Normalizer,
    state_norm: Normalizer,
    label: torch.Tensor,
    cfg: DoorConfig,
    n_samples: int = 30,
) -> torch.Tensor:
    """
    Wrapper around the conditioned sampler to mirror the original notebook helper.
    """
    return sample_conditioned(
        uncond_model=uncond_model,
        state_model=state_model,
        actions_norm=actions_norm,
        state_norm=state_norm,
        label=label,
        cfg=cfg,
        n_samples=n_samples,
    )


def synthesize_avoidance_dataset(
    cfg: DoorConfig,
    uncond_model: ConditionalUnet1D,
    state_model: ConditionalUnet1D,
    actions_norm: Normalizer,
    state_norm: Normalizer,
    n_envs: Optional[int] = None,
    samples_per_env: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate avoidance samples: [x_des(3), failure_action(32), success_action(32)].
    """
    device = cfg.device
    n_envs = n_envs or cfg.avoidance_n_envs
    samples_per_env = samples_per_env or cfg.avoidance_samples_per_env

    model, _, data, _, _, _, _ = build_environment(cfg)
    label_dim = data.geom("door_handle").xpos.shape[0]
    action_dim = 8 * cfg.n_dof
    avoidance_samples = torch.zeros((0, label_dim + 2 * action_dim), device=device)

    uncond_model = uncond_model.to(device).eval()
    state_model = state_model.to(device).eval()

    with torch.no_grad():
        for _ in trange(n_envs, desc="avoidance data"):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            cabinet_pos = np.array([np.random.rand() + 0.8, np.random.rand(), np.random.rand() + 0.25])
            model.body("cabinet").pos = cabinet_pos
            mujoco.mj_forward(model, data)

            x_des = torch.from_numpy(data.geom("door_handle").xpos.copy()).to(device)
            label = x_des.reshape(1, -1)
            samples = sample_dp_explorative(
                uncond_model, state_model, actions_norm, state_norm, label.squeeze(0), cfg, n_samples=samples_per_env
            )

            samples_norm = actions_norm.normalize(samples)
            samples_flat = samples_norm.reshape(samples_norm.shape[0], -1)
            samples_max = samples_flat.max(dim=1).values
            samples_min = samples_flat.min(dim=1).values
            feasible_mask = (samples_max <= cfg.avoidance_action_limit) & (samples_min >= -cfg.avoidance_action_limit)
            if feasible_mask.sum().item() < 2:
                continue

            feasible_actions = samples[feasible_mask]
            flat_actions = feasible_actions.reshape(feasible_actions.shape[0], -1)
            cdist = torch.cdist(flat_actions, flat_actions)
            pair_idx = torch.nonzero(cdist > cfg.avoidance_distance_threshold, as_tuple=False)
            if pair_idx.numel() == 0:
                continue

            point_main = flat_actions[pair_idx[:, 0]]
            point_failure = flat_actions[pair_idx[:, 1]]
            labels_aug = label.repeat(point_main.shape[0], 1)
            avoidance_batch = torch.cat([labels_aug, point_failure, point_main], dim=1)
            avoidance_samples = torch.cat([avoidance_samples, avoidance_batch], dim=0)
    return avoidance_samples


def load_or_generate_avoidance(
    cfg: DoorConfig,
    uncond_model: ConditionalUnet1D,
    state_model: ConditionalUnet1D,
    actions_norm: Normalizer,
    state_norm: Normalizer,
) -> Optional[torch.Tensor]:
    """
    Load avoidance dataset from disk or synthesize it if configured.
    """
    if cfg.avoidance_preload and cfg.avoidance_dataset_path.exists():
        data = torch.load(cfg.avoidance_dataset_path, map_location=cfg.device)
        print(f"Loaded avoidance dataset from {cfg.avoidance_dataset_path} with shape {tuple(data.shape)}")
        return data

    if cfg.avoidance_generate:
        avoidance_samples = synthesize_avoidance_dataset(cfg, uncond_model, state_model, actions_norm, state_norm)
        if cfg.avoidance_save:
            cfg.avoidance_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(avoidance_samples.cpu(), cfg.avoidance_dataset_path)
            print(f"Saved avoidance dataset to {cfg.avoidance_dataset_path} with shape {tuple(avoidance_samples.shape)}")
        return avoidance_samples

    print("Avoidance dataset not loaded or generated (disabled in config).")
    return None


def load_model_if_available(model: ConditionalUnet1D, ckpt_path: Path, device: torch.device) -> ConditionalUnet1D:
    """
    Load model weights if the checkpoint exists; otherwise return the model unchanged.
    """
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}, training from scratch.")
    return model.to(device)


def build_avoidance_dataloader(
    avoidance_samples: torch.Tensor,
    actions_norm: Normalizer,
    state_norm: Normalizer,
    cfg: DoorConfig,
):
    """
    Prepare dataloader and label dimension for avoidance training.
    """
    avoidance_samples = avoidance_samples.to(cfg.device)
    avoidance_states = avoidance_samples[:, :3]
    avoidance_actions_1 = avoidance_samples[:, 3:35].reshape(-1, 8, cfg.n_dof)
    avoidance_actions_2 = avoidance_samples[:, 35:].reshape(-1, 8, cfg.n_dof)

    avoidance_states_normalized = state_norm.normalize(avoidance_states)
    avoidance_actions_1_normalized = actions_norm.normalize(avoidance_actions_1)
    avoidance_actions_2_normalized = actions_norm.normalize(avoidance_actions_2)

    avoidance_labels = torch.cat([avoidance_states_normalized, avoidance_actions_1_normalized.reshape(-1, 8 * cfg.n_dof)], dim=1)
    dataset = TrajectoryDataset(avoidance_actions_2_normalized, avoidance_labels)
    loader = DataLoader(dataset, batch_size=cfg.batch_size_avoidance, shuffle=True)
    return loader, avoidance_labels.shape[1]


def sample_with_avoidance(
    uncond_model: ConditionalUnet1D,
    state_model: ConditionalUnet1D,
    avoidance_model: ConditionalUnet1D,
    actions_norm: Normalizer,
    state_norm: Normalizer,
    label_state: torch.Tensor,
    avoid_actions: Optional[List[torch.Tensor]],
    cfg: DoorConfig,
    n_samples: int = 1,
    guidance_state: float = 1.0,
    guidance_avoid: float = 2.0,
) -> torch.Tensor:
    """
    Sample trajectories while steering away from avoidance actions.
    """
    device = cfg.device
    beta_schedule = cosine_beta_schedule(cfg.n_diffusion_steps).to(device)
    alpha = 1.0 - beta_schedule
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    label_state_norm = state_norm.normalize(label_state.reshape(1, -1)).to(torch.float32).to(device)
    x = torch.randn(n_samples, 8, cfg.n_dof, device=device)

    avoid_labels: List[torch.Tensor] = []
    for avoid in avoid_actions or []:
        avoid = avoid.reshape(1, 8, cfg.n_dof).to(device)
        avoid_norm = actions_norm.normalize(avoid).reshape(1, -1).to(torch.float32)
        avoid_labels.append(torch.cat([label_state_norm, avoid_norm], dim=1))

    with torch.no_grad():
        for t in reversed(range(cfg.n_diffusion_steps)):
            beta_t = beta_schedule[t]
            alpha_t = alpha[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cump_t = sqrt_one_minus_alpha_cumprod[t]
            noise = torch.randn_like(x, device=device) if t > 0 else 0

            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            pred_uncond = uncond_model(x, t_tensor)
            dx_uncond = (x - beta_t / sqrt_one_minus_alpha_cump_t * pred_uncond) / sqrt_alpha_t - x

            pred_state = state_model(x, t_tensor, label_state_norm.repeat(x.size(0), 1))
            dx_state = (x - beta_t / sqrt_one_minus_alpha_cump_t * pred_state) / sqrt_alpha_t - x

            dx_avoid = torch.zeros_like(dx_state)
            for lbl in avoid_labels:
                pred_avoid = avoidance_model(x, t_tensor, lbl.repeat(x.size(0), 1))
                dx_i = (x - beta_t / sqrt_one_minus_alpha_cump_t * pred_avoid) / sqrt_alpha_t - x
                dx_avoid += dx_i - dx_uncond

            dx = dx_uncond + guidance_state * (dx_state - dx_uncond) + guidance_avoid * dx_avoid
            x = x + dx + noise * beta_t**0.5
    return actions_norm.unnormalize(x)


def visualize_key_states(cfg: DoorConfig, x_des: np.ndarray, key_frames: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper to render sampled key states for the current cabinet pose.
    """
    return visualize_key_frames(cfg, x_des, key_frames)


def sample_and_visualize_avoidance(
    cfg: DoorConfig,
    uncond_model: ConditionalUnet1D,
    state_model: ConditionalUnet1D,
    avoidance_model: ConditionalUnet1D,
    actions_norm: Normalizer,
    state_norm: Normalizer,
    x_des: np.ndarray,
    avoid_actions: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Sample a trajectory with avoidance guidance and render the resulting key states.
    """
    label_state = torch.from_numpy(x_des).to(cfg.device)
    samples = sample_with_avoidance(
        uncond_model,
        state_model,
        avoidance_model,
        actions_norm,
        state_norm,
        label_state,
        avoid_actions,
        cfg,
        n_samples=1,
    )
    key_states = samples[0].cpu().numpy()
    image = visualize_key_states(cfg, x_des, key_states)
    return samples, image


def main():
    """
    Example entrypoint that:
    - Generates demonstrations
    - Prepares dataloaders
    - Trains unconditional + conditional diffusion models
    - Saves checkpoints in the models directory

    Training is kept in the main guard so importing this file will not start
    heavy jobs in environments without GPUs.
    """
    cfg = DoorConfig()
    demos = load_or_generate_dataset(cfg, seed=cfg.seed)
    action_norm, state_norm, uncond_loader, state_loader = build_dataloaders(cfg, demos)

    uncond_model = ConditionalUnet1D(input_dim=cfg.n_dof, global_cond_dim=0, diffusion_step_embed_dim=256)
    if cfg.preload_uncond_model and cfg.uncond_ckpt_path.exists():
        load_model_if_available(uncond_model, cfg.uncond_ckpt_path, cfg.device)
    else:
        train_diffusion_model(uncond_model, uncond_loader, cfg, conditioning=False, save_path=cfg.uncond_ckpt_path)

    state_model = ConditionalUnet1D(input_dim=cfg.n_dof, global_cond_dim=demos.x_des.shape[1], diffusion_step_embed_dim=256)
    if cfg.preload_state_model and cfg.state_ckpt_path.exists():
        load_model_if_available(state_model, cfg.state_ckpt_path, cfg.device)
    else:
        train_diffusion_model(state_model, state_loader, cfg, conditioning=True, save_path=cfg.state_ckpt_path)

    label = torch.from_numpy(demos.x_des[0]).to(cfg.device)
    samples = sample_conditioned(uncond_model, state_model, action_norm, state_norm, label, cfg, n_samples=4)
    print("Sampled trajectories shape:", samples.shape)

    avoidance_samples = load_or_generate_avoidance(cfg, uncond_model, state_model, action_norm, state_norm)
    if avoidance_samples is not None:
        avoidance_loader, label_dim = build_avoidance_dataloader(avoidance_samples, action_norm, state_norm, cfg)
        avoidance_model = ConditionalUnet1D(input_dim=cfg.n_dof, global_cond_dim=label_dim, diffusion_step_embed_dim=256)
        if cfg.avoidance_preload_model and cfg.avoidance_ckpt_path.exists():
            load_model_if_available(avoidance_model, cfg.avoidance_ckpt_path, cfg.device)
        else:
            train_diffusion_model(avoidance_model, avoidance_loader, cfg, conditioning=True, save_path=cfg.avoidance_ckpt_path)


if __name__ == "__main__":
    main()
