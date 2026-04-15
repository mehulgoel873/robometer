#!/usr/bin/env python3
"""
RoboMimic dataset loader for Robometer fine-tuning.
Loads HDF5-format RoboMimic trajectories with image observations.
Supports all tasks (can, lift, square, tool_hang, transport) by auto-discovering
from directory structure: base_path/<task>/<split>/image_v15.hdf5
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from dataset_upload.helpers import generate_unique_id


TASK_DESCRIPTIONS = {
    "can": "pick up the can and place it in the target location",
    "lift": "lift the cube above the table",
    "square": "pick up the square nut and place it on the peg",
    "tool_hang": "hang the tool on the hook",
    "transport": "transport the object between robots",
}

VALID_SPLITS = {"ph", "mh"}

CAMERA_FALLBACK_ORDER = [
    "agentview_image",
    "sideview_image",
    "shouldercamera0_image",
]


class RoboMimicFrameLoader:
    """Pickle-able loader that reads RoboMimic frames from an HDF5 dataset on demand."""

    def __init__(self, hdf5_path: str, dataset_path: str):
        self.hdf5_path = hdf5_path
        self.dataset_path = dataset_path

    def __call__(self) -> np.ndarray:
        """Load frames from HDF5. Returns np.ndarray (T, H, W, 3) uint8."""
        with h5py.File(self.hdf5_path, "r") as f:
            if self.dataset_path not in f:
                raise KeyError(
                    f"Dataset path '{self.dataset_path}' not found in {self.hdf5_path}"
                )
            frames = f[self.dataset_path][:]

        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Unexpected frames shape for {self.dataset_path} in {self.hdf5_path}: "
                f"{getattr(frames, 'shape', None)}"
            )

        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        return frames


def _discover_tasks(base_path: Path) -> List[Tuple[str, str, Path]]:
    """Discover all task/split combinations with image_v15.hdf5.

    Returns:
        List of (task_name, split, hdf5_path) tuples.
    """
    results = []
    for task_dir in sorted(base_path.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for split_dir in sorted(task_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            assert split in VALID_SPLITS, (
                f"Unexpected split '{split}' in {task_dir}. Expected one of {VALID_SPLITS}"
            )
            hdf5_path = split_dir / "image_v15.hdf5"
            if hdf5_path.exists():
                results.append((task_name, split, hdf5_path))
            else:
                print(f"  Skipping {task_name}/{split}: no image_v15.hdf5")
    return results


def _find_camera_key(
    obs_group: h5py.Group, camera: Optional[str] = None
) -> Optional[str]:
    """Find the camera observation key in an obs group.

    Args:
        obs_group: HDF5 group for a demo's observations (data/demo_X/obs)
        camera: Optional camera key override. If provided and exists, used directly.

    Returns:
        Camera key string, or None if no image observation found.
    """
    if camera and camera in obs_group:
        return camera

    for key in CAMERA_FALLBACK_ORDER:
        if key in obs_group:
            return key

    return None


def load_robomimic_dataset(
    base_path: str,
    max_trajectories: Optional[int] = None,
    camera: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Load RoboMimic dataset organized by task and split.

    Args:
        base_path: Path to the RoboMimic dataset root (e.g. /data/.../robomimic/)
        max_trajectories: Maximum total trajectories to load (None = all)
        camera: Camera observation key override. If None, auto-detects.

    Returns:
        Dictionary mapping "{task}_{split}" keys to lists of trajectory dicts.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"RoboMimic dataset path not found: {base_path}")

    print(f"Loading RoboMimic dataset from: {base_path}")
    print("=" * 80)

    tasks = _discover_tasks(base_path)
    print(f"Discovered {len(tasks)} task/split combinations with image data")

    task_data: Dict[str, List[Dict]] = {}
    total_loaded = 0

    for task_name, split, hdf5_path in tasks:
        if max_trajectories and total_loaded >= max_trajectories:
            break

        optimal = "optimal" if split == "ph" else "suboptimal"
        task_desc = TASK_DESCRIPTIONS.get(task_name, task_name.replace("_", " "))
        group_key = f"{task_name}_{split}"

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"  Skipping {group_key}: no 'data' group in HDF5")
                continue

            data_group = f["data"]
            demo_keys = sorted(data_group.keys())

            # Detect camera key from first demo
            first_obs = data_group[demo_keys[0]]["obs"]
            camera_key = _find_camera_key(first_obs, camera)
            if camera_key is None:
                print(f"  Skipping {group_key}: no image observation found")
                continue

            trajectories = []
            for demo_key in tqdm(demo_keys, desc=f"  {group_key}", leave=False):
                if max_trajectories and total_loaded >= max_trajectories:
                    break

                demo = data_group[demo_key]
                obs = demo["obs"]

                if camera_key not in obs:
                    continue

                ep_length = obs[camera_key].shape[0]
                dataset_path = f"data/{demo_key}/obs/{camera_key}"

                trajectory = {
                    "frames": RoboMimicFrameLoader(str(hdf5_path), dataset_path),
                    "actions": np.zeros((ep_length, 7), dtype=np.float32),
                    "is_robot": False,
                    "task": task_desc,
                    "optimal": optimal,
                    "id": generate_unique_id(),
                    "quality_label": "successful",
                    "data_source": "robomimic",
                    "partial_success": 1.0,
                }
                trajectories.append(trajectory)
                total_loaded += 1

        if trajectories:
            task_data[group_key] = trajectories
            print(
                f"  {group_key}: {len(trajectories)} trajectories "
                f"(camera={camera_key}, optimal={optimal})"
            )

    total = sum(len(v) for v in task_data.values())
    print(f"\nTotal: {total} trajectories from {len(task_data)} task/split groups")
    return task_data
