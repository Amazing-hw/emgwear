# s01_data_split.py
# -*- coding: utf-8 -*-

"""
步骤1：数据整理与切分（适配 6-ch PPG + EMG + ACC）

数据格式：
  - ppg: (6, N) @ 100Hz  6通道PPG
  - emg: (2, N) @ 1000Hz 2通道肌电
  - acc: (3, N) @ 100Hz  3通道加速度

使用 EMG 模态作为数据集。
CLI 兼容旧版。
"""

import os
import sys
import glob
import json
import argparse
import re
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py

# Linux/macOS 默认 fork 模式多进程读 H5 可能死锁，强制 spawn
if sys.platform != "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def has_emg_two_channels(shape):
    if len(shape) < 1:
        return False
    if len(shape) == 3:
        return shape[1] == 2 or shape[2] == 2
    return shape[0] == 2 or (len(shape) >= 2 and shape[1] == 2)


_WINDOW_NAME_RE = re.compile(r"^(?P<base>.+)_w(?P<index>\d+)_(?P<label>-?\d+)$")


def parse_window_group_name(name):
    m = _WINDOW_NAME_RE.match(str(name))
    if not m:
        return None
    return {
        "base": m.group("base"),
        "window_index": int(m.group("index")),
        "target": int(m.group("label")),
    }


def _scan_nested_window_sample(h5_file, sample_name, parent_group, filtered):
    windows = []
    targets = set()
    for window_name in parent_group.keys():
        child = parent_group[window_name]
        if not isinstance(child, h5py.Group):
            continue
        parsed = parse_window_group_name(window_name)
        if parsed is None:
            filtered["invalid_target"] += 1
            continue
        if "emg" not in child:
            filtered["no_emg"] += 1
            continue
        shape = child["emg"].shape
        if not has_emg_two_channels(shape):
            filtered["emg_channel_count"] += 1
            continue
        targets.add(parsed["target"])
        windows.append((parsed["window_index"], window_name, shape))

    if not windows:
        return None
    if len(targets) != 1:
        filtered["invalid_target"] += len(windows)
        return None

    windows.sort(key=lambda item: item[0])
    windows = windows[3:]
    if not windows:
        return None

    return {
        "sample_name": sample_name,
        "h5_file": h5_file,
        "target": int(next(iter(targets))),
        "window_names": [name for _idx, name, _shape in windows],
        "window_indices": [int(idx) for idx, _name, _shape in windows],
        "emg_shape": list(windows[0][2]),
    }


def find_h5_files(dataset_dir):
    h5_files = glob.glob(os.path.join(dataset_dir, "*.h5"))
    if len(h5_files) == 0:
        h5_files = glob.glob(os.path.join("..", dataset_dir, "*.h5"))
    return sorted(h5_files)


def _scan_one_h5(h5_file):
    """单文件扫描。返回 (samples_list, filtered_counts_dict)。"""
    samples = []
    filtered = {"emg_channel_count": 0, "no_emg": 0, "invalid_target": 0}
    try:
        with h5py.File(h5_file, "r") as f:
            for sample_name in f.keys():
                grp = f[sample_name]
                if "emg" not in grp:
                    nested = _scan_nested_window_sample(h5_file, sample_name, grp, filtered)
                    if nested is not None:
                        samples.append(nested)
                    else:
                        filtered["no_emg"] += 1
                    continue
                if "emg" not in grp:
                    continue
                try:
                    label = int(sample_name.split("_")[-1])
                except (TypeError, ValueError, IndexError):
                    filtered["invalid_target"] += 1
                    continue

                shape = grp["emg"].shape

                if not has_emg_two_channels(shape):
                    filtered["emg_channel_count"] += 1
                    continue

                samples.append({
                    "sample_name": sample_name,
                    "h5_file": h5_file,
                    "target": int(label),
                    "emg_shape": list(shape),
                })
    except (OSError, h5py.HDF5DecodeError) as e:
        print(f"读取 {h5_file} 失败: {e}")
    except Exception as e:
        print(f"读取 {h5_file} 异常: {e}")
    return samples, filtered


def scan_h5_samples(dataset_dir, n_workers=None):
    """并行扫描 H5（n_workers=None 自动，=1 单进程）。"""
    h5_files = find_h5_files(dataset_dir)
    print(f"找到 {len(h5_files)} 个 H5 文件")
    print(f"[生产] 使用全部H5文件: {h5_files}")

    if n_workers is None:
        n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    n_workers = max(1, int(n_workers))

    samples = []
    filtered_count = {"emg_channel_count": 0, "no_emg": 0, "invalid_target": 0}

    if n_workers == 1 or len(h5_files) <= 1:
        for h5_file in h5_files:
            s, fc = _scan_one_h5(h5_file)
            samples.extend(s)
            for k in filtered_count:
                filtered_count[k] += fc.get(k, 0)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for s, fc in ex.map(_scan_one_h5, h5_files):
                samples.extend(s)
                for k in filtered_count:
                    filtered_count[k] += fc.get(k, 0)

    if any(v > 0 for v in filtered_count.values()):
        print(f"过滤样本: {filtered_count}")

    samples.sort(key=lambda s: (s["h5_file"], s["sample_name"]))
    return samples


def _stable_sample_key(sample):
    h5_name = os.path.basename(str(sample.get("h5_file", "")))
    return f"{h5_name}::{sample.get('sample_name', '')}"


def _stable_h5_key(sample):
    return os.path.basename(str(sample.get("h5_file", "")))


def _hash_fraction(text, seed=42):
    payload = f"{seed}::{text}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, "big", signed=False)
    return value / float(1 << 64)


def split_samples(samples, valid_size=0.15, test_size=0.15, random_state=42):
    """Stable per-H5 hash-bucket split; adding samples does not reshuffle existing samples."""
    valid_size = float(valid_size)
    test_size = float(test_size)
    if valid_size < 0 or test_size < 0 or valid_size + test_size >= 1.0:
        raise ValueError("valid_size and test_size must be non-negative and sum to less than 1")

    split = {"train": [], "valid": [], "test": []}
    by_h5 = {}
    for sample in samples:
        by_h5.setdefault(_stable_h5_key(sample), []).append(sample)

    for h5_key in sorted(by_h5):
        group = sorted(
            by_h5[h5_key],
            key=_stable_sample_key,
        )
        for sample in group:
            score = _hash_fraction(_stable_sample_key(sample), seed=random_state)
            if score < test_size:
                split["test"].append(sample)
            elif score < test_size + valid_size:
                split["valid"].append(sample)
            else:
                split["train"].append(sample)
    return split


def summarize_split(split):
    for part in ["train", "valid", "test"]:
        arr = split[part]
        n0 = sum(1 for s in arr if s["target"] == 0)
        n1 = sum(1 for s in arr if s["target"] == 1)
        print(f"{part}: total={len(arr)}, target0={n0}, target1={n1}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    samples = scan_h5_samples(args.dataset_dir, n_workers=args.n_workers)

    print(f"总样本数: {len(samples)}")
    print(f"target=0: {sum(1 for s in samples if s['target'] == 0)}")
    print(f"target=1: {sum(1 for s in samples if s['target'] == 1)}")

    split = split_samples(
        samples,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state
    )
    summarize_split(split)

    out_path = os.path.join(args.artifact_dir, "splits.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"切分结果已保存: {out_path}")


if __name__ == "__main__":
    main()
