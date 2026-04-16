#!/usr/bin/env python3
"""
Split a RoboMimic dataset root into train / eval dataset roots.

The script walks an input directory containing RoboMimic files such as:

    <input_root>/<task>/<split>/image_v15.hdf5

and writes:

    <output_root>/train/<task>/<split>/image_v15.hdf5
    <output_root>/eval/<task>/<split>/image_v15.hdf5

Each HDF5 file is split independently by randomly selecting a fraction of its
trajectory groups (``data/demo_*``) into eval and keeping the rest in train.
Non-HDF5 files under the input root are copied into both outputs so companion
metadata stays available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing RoboMimic task folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root that will contain train/ and eval/ subdirectories.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Fraction of trajectories from each HDF5 file to place into eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Each file gets a deterministic per-file seed derived from this value.",
    )
    parser.add_argument(
        "--train-dir-name",
        type=str,
        default="train",
        help="Directory name to use under output-root for the train split.",
    )
    parser.add_argument(
        "--eval-dir-name",
        type=str,
        default="eval",
        help="Directory name to use under output-root for the eval split.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="split_manifest.json",
        help="Manifest filename to write under output-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned split without writing output files.",
    )
    return parser.parse_args()


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def discover_hdf5_files(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("image_v15.hdf5") if path.is_file())


def file_rng(seed: int, relative_path: Path) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{relative_path.as_posix()}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="big", signed=False))


def compute_eval_count(num_demos: int, eval_ratio: float) -> int:
    if num_demos <= 1:
        return 0

    eval_count = int(round(num_demos * eval_ratio))
    eval_count = max(1, eval_count)
    eval_count = min(eval_count, num_demos - 1)
    return eval_count


def split_demo_names(
    demo_names: Sequence[str],
    eval_ratio: float,
    seed: int,
    relative_path: Path,
) -> tuple[list[str], list[str]]:
    if not demo_names:
        raise ValueError(f"No demo groups found in {relative_path}")

    eval_count = compute_eval_count(len(demo_names), eval_ratio)
    if eval_count == 0:
        return list(demo_names), []

    rng = file_rng(seed=seed, relative_path=relative_path)
    eval_indices = set(rng.choice(len(demo_names), size=eval_count, replace=False).tolist())
    train_demos = [name for index, name in enumerate(demo_names) if index not in eval_indices]
    eval_demos = [name for index, name in enumerate(demo_names) if index in eval_indices]
    return train_demos, eval_demos


def total_num_samples(data_group: h5py.Group, demo_names: Sequence[str]) -> int:
    total = 0
    for demo_name in demo_names:
        demo_group = data_group[demo_name]
        if "num_samples" in demo_group.attrs:
            total += int(demo_group.attrs["num_samples"])
        elif "actions" in demo_group:
            total += int(demo_group["actions"].shape[0])
        else:
            raise KeyError(f"Could not infer num_samples for demo '{demo_name}'")
    return total


def filter_mask_values(values: np.ndarray, selected_demo_names: Sequence[str]) -> np.ndarray:
    if values.ndim != 1:
        return values

    if values.dtype.kind == "S":
        selected = np.asarray([name.encode("utf-8") for name in selected_demo_names], dtype=values.dtype)
        return values[np.isin(values, selected)]
    if values.dtype.kind == "U":
        selected = np.asarray(selected_demo_names, dtype=values.dtype)
        return values[np.isin(values, selected)]
    if values.dtype.kind != "O":
        return values

    filtered_items: list[Any] = []
    selected_set = set(selected_demo_names)
    for item in values.tolist():
        decoded = item.decode("utf-8") if isinstance(item, bytes) else item
        if decoded in selected_set:
            filtered_items.append(item)
    return np.asarray(filtered_items, dtype=values.dtype)


def copy_mask_tree(src_obj: h5py.Group | h5py.Dataset, dst_parent: h5py.Group, selected_demo_names: Sequence[str]) -> None:
    if isinstance(src_obj, h5py.Group):
        dst_group = dst_parent.create_group(src_obj.name.rsplit("/", 1)[-1])
        copy_attrs(src_obj.attrs, dst_group.attrs)
        for child_name in src_obj.keys():
            copy_mask_tree(src_obj[child_name], dst_group, selected_demo_names)
        return

    if src_obj.ndim == 1 and src_obj.dtype.kind in {"S", "U", "O"}:
        filtered = filter_mask_values(src_obj[()], selected_demo_names)
        dst_dataset = dst_parent.create_dataset(src_obj.name.rsplit("/", 1)[-1], data=filtered, dtype=src_obj.dtype)
        copy_attrs(src_obj.attrs, dst_dataset.attrs)
        return

    src_obj.file.copy(src_obj, dst_parent, name=src_obj.name.rsplit("/", 1)[-1])


def write_split_hdf5(
    src_path: Path,
    dst_path: Path,
    selected_demo_names: Sequence[str],
    split_name: str,
    eval_ratio: float,
    seed: int,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src_file, h5py.File(dst_path, "w") as dst_file:
        copy_attrs(src_file.attrs, dst_file.attrs)
        dst_file.attrs["robometer_split_source"] = str(src_path)
        dst_file.attrs["robometer_split_role"] = split_name
        dst_file.attrs["robometer_split_seed"] = seed
        dst_file.attrs["robometer_split_eval_ratio"] = eval_ratio

        for top_level_name in src_file.keys():
            if top_level_name in {"data", "mask"}:
                continue
            src_file.copy(src_file[top_level_name], dst_file, name=top_level_name)

        src_data = src_file["data"]
        dst_data = dst_file.create_group("data")
        copy_attrs(src_data.attrs, dst_data.attrs)
        for demo_name in selected_demo_names:
            src_file.copy(src_data[demo_name], dst_data, name=demo_name)
        dst_data.attrs["total"] = total_num_samples(src_data, selected_demo_names)

        if "mask" in src_file:
            dst_mask = dst_file.create_group("mask")
            copy_attrs(src_file["mask"].attrs, dst_mask.attrs)
            for child_name in src_file["mask"].keys():
                copy_mask_tree(src_file["mask"][child_name], dst_mask, selected_demo_names)


def copy_shared_files(input_root: Path, outputs: Iterable[Path]) -> int:
    copied = 0
    output_paths = [path.resolve() for path in outputs]
    for src_path in sorted(path for path in input_root.rglob("*") if path.is_file() and path.name != "image_v15.hdf5"):
        resolved_src = src_path.resolve()
        if any(out == resolved_src or out in resolved_src.parents for out in output_paths):
            continue

        relative_path = src_path.relative_to(input_root)
        for output_root in outputs:
            dst_path = output_root / relative_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        copied += 1
    return copied


def ensure_safe_output_layout(input_root: Path, output_root: Path, split_roots: Sequence[Path]) -> None:
    resolved_input = input_root.resolve()
    resolved_output = output_root.resolve()

    if resolved_input == resolved_output or resolved_input in resolved_output.parents:
        raise ValueError("output-root must not be the same as, or nested inside, input-root")

    for split_root in split_roots:
        if split_root.exists() and any(split_root.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty directory: {split_root}")


def main() -> None:
    args = parse_args()

    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError(f"--eval-ratio must be strictly between 0 and 1, got {args.eval_ratio}")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    train_root = output_root / args.train_dir_name
    eval_root = output_root / args.eval_dir_name

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    ensure_safe_output_layout(input_root=input_root, output_root=output_root, split_roots=[train_root, eval_root])

    hdf5_paths = discover_hdf5_files(input_root)
    if not hdf5_paths:
        raise FileNotFoundError(f"No image_v15.hdf5 files found under: {input_root}")

    manifest: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "train_root": str(train_root),
        "eval_root": str(eval_root),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "files": [],
    }

    print(f"Discovered {len(hdf5_paths)} image_v15.hdf5 file(s) under {input_root}")
    for src_path in hdf5_paths:
        relative_path = src_path.relative_to(input_root)
        with h5py.File(src_path, "r") as src_file:
            if "data" not in src_file:
                raise KeyError(f"Missing 'data' group in {src_path}")
            demo_names = sorted(src_file["data"].keys())

        train_demos, eval_demos = split_demo_names(
            demo_names=demo_names,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            relative_path=relative_path,
        )

        file_summary = {
            "relative_path": relative_path.as_posix(),
            "num_demos": len(demo_names),
            "num_train": len(train_demos),
            "num_eval": len(eval_demos),
            "train_demos": train_demos,
            "eval_demos": eval_demos,
        }
        manifest["files"].append(file_summary)

        print(
            f"{relative_path}: total={len(demo_names)} train={len(train_demos)} eval={len(eval_demos)}"
        )

        if args.dry_run:
            continue

        write_split_hdf5(
            src_path=src_path,
            dst_path=train_root / relative_path,
            selected_demo_names=train_demos,
            split_name=args.train_dir_name,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )
        write_split_hdf5(
            src_path=src_path,
            dst_path=eval_root / relative_path,
            selected_demo_names=eval_demos,
            split_name=args.eval_dir_name,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return

    train_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    copied_files = copy_shared_files(input_root=input_root, outputs=[train_root, eval_root])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total_demos = sum(item["num_demos"] for item in manifest["files"])
    total_train = sum(item["num_train"] for item in manifest["files"])
    total_eval = sum(item["num_eval"] for item in manifest["files"])

    print()
    print(f"Wrote train split to: {train_root}")
    print(f"Wrote eval split to: {eval_root}")
    print(f"Copied {copied_files} non-HDF5 file(s) into both outputs")
    print(f"Manifest: {manifest_path}")
    print(f"Trajectory totals: total={total_demos} train={total_train} eval={total_eval}")


if __name__ == "__main__":
    main()
