# s02_ir_dc_threshold.py
# -*- coding: utf-8 -*-

"""
步骤2：PPG DC 阈值筛选（适配 6-ch PPG 取平均）

与旧版主要差异：
1. 读取 6-ch PPG，先对 6 通道取平均得到单通道 PPG，保持 100Hz 原始采样率
2. 其余 DC/ACDC 阈值逻辑、多阶段搜索、输出 schema 完全不变

公共接口 / CLI / 输出 schema 完全不变。
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor

from s03_extract_feature_pool import _load_named_windows, normalize_sensor_array


STAGE1_FS = 100
STAGE1_WINDOW_SEC = 3.0
STAGE1_STRIDE_SEC = 1.0
FIXED_STAGE1_DC_THRESHOLD = 2.2e6
FIXED_STAGE1_AC_DC_THRESHOLD = 0.35
FIXED_STAGE1_SEARCH_SOURCE = "fixed_engineering_threshold"

# Linux/macOS 默认 fork 模式多进程读 H5 可能死锁，强制 spawn
if sys.platform != "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def load_ppg(sample):
    """Read 6-ch PPG and return the 100Hz six-channel mean signal."""
    named = _load_named_windows(sample, "ppg", 6)
    if named is not None:
        return np.mean(named, axis=2).astype(np.float64)
    with h5py.File(sample["h5_file"], "r") as f:
        ppg = normalize_sensor_array(f[sample["sample_name"]]["ppg"][:], 6)
    if ppg.ndim == 1:
        return ppg.astype(np.float64)
    if ppg.ndim == 3:
        return np.mean(ppg, axis=2).astype(np.float64)
    return np.mean(ppg, axis=1).astype(np.float64)



# =========================================================
# 单样本 → windows（worker 复用）
# =========================================================

def extract_dc_acdc_features(ppg, fs=STAGE1_FS, window_indices=None):
    win = int(STAGE1_WINDOW_SEC * fs)
    stride = int(STAGE1_STRIDE_SEC * fs)
    ppg = np.asarray(ppg, dtype=np.float64)
    if ppg.ndim == 2:
        rows = []
        for idx, x in enumerate(ppg):
            if len(x) >= 2:
                neighbor_mean = (x[:-1] + x[1:]) / 2.0
                dc = float(np.min(neighbor_mean))
                ac = float(np.median(np.abs(np.diff(x))))
            else:
                dc = float(np.mean(x))
                ac = 0.0
            if window_indices is not None and idx < len(window_indices):
                start_100hz = int(window_indices[idx] * stride)
            else:
                start_100hz = int(idx * win)
            rows.append({
                "start_100hz": start_100hz,
                "dc": dc,
                "ac": ac,
                "ac_dc_ratio": float(ac / (np.abs(dc) + 1e-12)),
            })
        return rows
    rows = []
    for i in range(0, len(ppg) - win + 1, stride):
        x = ppg[i:i + win]
        if len(x) >= 2:
            neighbor_mean = (x[:-1] + x[1:]) / 2.0
            dc = float(np.min(neighbor_mean))
            ac = float(np.median(np.abs(np.diff(x))))
        else:
            dc = float(np.mean(x))
            ac = 0.0
        rows.append({
            "start_100hz": int(i),
            "dc": dc,
            "ac": ac,
            "ac_dc_ratio": float(ac / (np.abs(dc) + 1e-12)),
        })
    return rows


def _extract_windows_from_sample(sample, min_duration):
    try:
        ppg = load_ppg(sample)
    except Exception as e:
        print(f"read failed {sample.get('sample_name')}: {e}")
        return []

    if np.asarray(ppg).ndim < 2 and len(ppg) < min_duration:
        return []

    rows = []
    for item in extract_dc_acdc_features(
            ppg, fs=STAGE1_FS, window_indices=sample.get("window_indices")):
        rows.append({
            "sample_name": sample["sample_name"],
            "h5_file": sample["h5_file"],
            "target": int(sample["target"]),
            **item,
        })
    return rows


def _worker_extract_sample(args_tuple):
    try:
        sample, min_duration = args_tuple
        return _extract_windows_from_sample(sample, min_duration)
    except Exception as e:
        sample_name = args_tuple[0].get("sample_name", "?") if args_tuple else "?"
        print(f"\n[worker error] {sample_name}: {e}")
        return []


def extract_stage1_windows(samples, min_duration_sec=STAGE1_WINDOW_SEC, n_workers=None):
    """样本级并行（n_workers=None 自动选 cpu-1；=1 单进程）。"""
    min_duration = int(min_duration_sec * 100)

    if n_workers is None:
        n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    n_workers = max(1, int(n_workers))

    args_list = [(s, min_duration) for s in samples]
    all_rows = []
    if n_workers == 1 or len(samples) <= 2:
        for a in args_list:
            all_rows.extend(_worker_extract_sample(a))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for rows in ex.map(_worker_extract_sample, args_list, chunksize=1):
                all_rows.extend(rows)
    return pd.DataFrame(all_rows)


# =========================================================
# 阈值评分：向量化
# =========================================================

def _prepare_arrays(df):
    """从 df 一次性提取评分所需的 numpy 数组。"""
    if len(df) == 0:
        return None

    dc = df["dc"].values.astype(np.float64)
    acdc = df["ac_dc_ratio"].values.astype(np.float64)
    target = df["target"].values.astype(np.int8)

    names = df["sample_name"].values
    uniq, inv = np.unique(names, return_inverse=True)
    n_samples = len(uniq)

    sample_target = np.zeros(n_samples, dtype=np.int8)
    seen = np.zeros(n_samples, dtype=bool)
    for i, code in enumerate(inv):
        if not seen[code]:
            sample_target[code] = target[i]
            seen[code] = True

    return {
        "dc": dc,
        "acdc": acdc,
        "target": target,
        "sample_idx": inv.astype(np.int64),
        "n_samples": int(n_samples),
        "sample_target": sample_target,
        "num_windows": int(len(df)),
    }


def _fast_eval_threshold(arrs, dc_th, acdc_th):
    """直接基于 arrs 的快速版本，返回与 eval_threshold 同结构 dict。"""
    if arrs is None:
        return {
            "dc_threshold": float(dc_th),
            "ac_dc_threshold": float(acdc_th),
            "window_target1_pass_rate": 0.0,
            "window_target0_pass_rate": 0.0,
            "window_target0_reject_rate": 0.0,
            "sample_target1_pass_rate": 0.0,
            "sample_target0_pass_rate": 0.0,
            "sample_target0_reject_rate": 0.0,
            "num_windows": 0,
            "num_samples": 0,
            "num_target1_samples": 0,
            "num_target0_samples": 0,
        }

    dc = arrs["dc"]
    acdc = arrs["acdc"]
    target = arrs["target"]
    sample_idx = arrs["sample_idx"]
    n_samples = arrs["n_samples"]
    sample_target = arrs["sample_target"]

    p_win = (dc > dc_th) & (acdc < acdc_th)

    t1_mask = target == 1
    t0_mask = target == 0
    n_t1 = int(t1_mask.sum())
    n_t0 = int(t0_mask.sum())
    w1 = float(p_win[t1_mask].mean()) if n_t1 > 0 else 0.0
    w0 = float(p_win[t0_mask].mean()) if n_t0 > 0 else 0.0

    sample_pass = np.zeros(n_samples, dtype=bool)
    np.logical_or.at(sample_pass, sample_idx, p_win)

    n_s_t1 = int((sample_target == 1).sum())
    n_s_t0 = int((sample_target == 0).sum())
    s1 = float(sample_pass[sample_target == 1].mean()) if n_s_t1 > 0 else 0.0
    s0 = float(sample_pass[sample_target == 0].mean()) if n_s_t0 > 0 else 0.0

    return {
        "dc_threshold": float(dc_th),
        "ac_dc_threshold": float(acdc_th),
        "window_target1_pass_rate": float(w1),
        "window_target0_pass_rate": float(w0),
        "window_target0_reject_rate": float(1.0 - w0),
        "sample_target1_pass_rate": float(s1),
        "sample_target0_pass_rate": float(s0),
        "sample_target0_reject_rate": float(1.0 - s0),
        "num_windows": int(arrs["num_windows"]),
        "num_samples": int(n_samples),
        "num_target1_samples": int(n_s_t1),
        "num_target0_samples": int(n_s_t0),
    }


def eval_threshold(df, dc_th, acdc_th):
    """向后兼容：单次调用时内部组好 arrays 再算。"""
    arrs = _prepare_arrays(df)
    return _fast_eval_threshold(arrs, dc_th, acdc_th)


# =========================================================
# 网格生成
# =========================================================

def _build_grids(df, n_points=501, acdc_extra_uniform=True):
    dc_values = df["dc"].replace([np.inf, -np.inf], np.nan).dropna().values
    acdc_values = df["ac_dc_ratio"].replace([np.inf, -np.inf], np.nan).dropna().values

    if len(dc_values) == 0 or len(acdc_values) == 0:
        raise RuntimeError("Stage1 数据为空，无法搜索阈值。")

    dc_min, dc_max = float(np.min(dc_values)), float(np.max(dc_values))
    dc_span = max(dc_max - dc_min, 1.0)
    dc_grid = np.unique(np.concatenate([
        np.percentile(dc_values, np.linspace(0, 100, n_points)),
        np.linspace(dc_min - 0.10 * dc_span, dc_max, n_points),
    ]))
    dc_grid = np.sort(dc_grid)

    acdc_min, acdc_max = float(np.min(acdc_values)), float(np.max(acdc_values))
    acdc_span = max(acdc_max - acdc_min, 1e-6)

    pieces = [
        np.percentile(acdc_values, np.linspace(0, 100, n_points)),
        np.linspace(max(0.0, acdc_min - 0.05 * acdc_span), acdc_max + 0.20 * acdc_span, n_points),
    ]
    if acdc_extra_uniform:
        pieces.append(np.linspace(0.01, 1.50, max(n_points - 100, 100)))
    acdc_grid = np.sort(np.unique(np.concatenate(pieces)))
    return dc_grid, acdc_grid


def make_search_grids(df):
    return _build_grids(df, n_points=501, acdc_extra_uniform=True)


def make_coarse_grids(df, n_points=50):
    return _build_grids(df, n_points=n_points, acdc_extra_uniform=True)


def make_fine_grids(dc_center, acdc_center, dc_span_factor=0.2, acdc_span_factor=0.2, n_points=30):
    dc_grid = np.linspace(
        dc_center * (1 - dc_span_factor),
        dc_center * (1 + dc_span_factor),
        n_points
    )
    acdc_grid = np.linspace(
        acdc_center * (1 - acdc_span_factor),
        acdc_center * (1 + acdc_span_factor),
        n_points
    )
    return dc_grid, acdc_grid


# =========================================================
# 阈值比较
# =========================================================

def compare_thresholds(candidate, best):
    if candidate is None:
        return False
    if best is None:
        return True
    if candidate["sample_target1_pass_rate"] < best["sample_target1_pass_rate"]:
        return False
    if best["sample_target1_pass_rate"] < candidate["sample_target1_pass_rate"]:
        return True
    if candidate["sample_target0_reject_rate"] > best["sample_target0_reject_rate"] + 1e-12:
        return True
    if candidate["sample_target0_reject_rate"] < best["sample_target0_reject_rate"] - 1e-12:
        return False
    if candidate["window_target0_reject_rate"] > best["window_target0_reject_rate"] + 1e-12:
        return True
    if candidate["window_target0_reject_rate"] < best["window_target0_reject_rate"] - 1e-12:
        return False
    if candidate["dc_threshold"] < best["dc_threshold"] - 1e-12:
        return True
    if candidate["dc_threshold"] > best["dc_threshold"] + 1e-12:
        return False
    if candidate["ac_dc_threshold"] > best["ac_dc_threshold"] + 1e-12:
        return True
    return False


# =========================================================
# 并行网格评估
# =========================================================

_WORKER_ARRS = None


def _init_grid_worker(arrs_pickle):
    """子进程初始化：反序列化 arrays 一次。"""
    global _WORKER_ARRS
    _WORKER_ARRS = pickle.loads(arrs_pickle)


def _grid_eval_one(args_tuple):
    dc_th, acdc_th, target_pos_pass_rate = args_tuple
    m = _fast_eval_threshold(_WORKER_ARRS, dc_th, acdc_th)
    feasible = m["sample_target1_pass_rate"] + 1e-12 >= target_pos_pass_rate
    return m, bool(feasible)


def _grid_search(arrs, dc_grid, acdc_grid, target_pos_pass_rate, n_workers):
    """并行评估网格 (dc × acdc)，返回 (best, feasible_count)。"""
    tasks = [(float(dc), float(acdc), float(target_pos_pass_rate))
             for dc in dc_grid for acdc in acdc_grid]

    if n_workers is None:
        n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    n_workers = max(1, int(n_workers))

    best, feasible_count = None, 0
    if n_workers == 1 or len(tasks) <= 64:
        for dc_th, acdc_th, tpr in tasks:
            m = _fast_eval_threshold(arrs, dc_th, acdc_th)
            if m["sample_target1_pass_rate"] + 1e-12 < tpr:
                continue
            feasible_count += 1
            if compare_thresholds(m, best):
                best = m
        return best, feasible_count

    arrs_pickle = pickle.dumps(arrs, protocol=pickle.HIGHEST_PROTOCOL)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_grid_worker,
                             initargs=(arrs_pickle,)) as ex:
        chunk = max(64, len(tasks) // (n_workers * 8))
        for m, feasible in ex.map(_grid_eval_one, tasks, chunksize=chunk):
            if not feasible:
                continue
            feasible_count += 1
            if compare_thresholds(m, best):
                best = m
    return best, feasible_count


# =========================================================
# 全量搜索 / 多阶段搜索
# =========================================================

def search_deploy_threshold(df, target_pos_pass_rate=1.0, n_workers=None):
    arrs = _prepare_arrays(df)
    dc_grid, acdc_grid = _build_grids(df, n_points=501)
    best, feasible_count = _grid_search(arrs, dc_grid, acdc_grid,
                                        target_pos_pass_rate, n_workers)
    if best is not None:
        best["feasible_count"] = int(feasible_count)
    return best


def search_deploy_threshold_multi_stage(df, target_pos_pass_rate=1.0, n_workers=None):
    """
    多阶段搜索：
    Stage 1 粗网格 → Stage 2 局部细网格 → Stage 3 differential_evolution。
    """
    import time
    start_time = time.time()

    arrs = _prepare_arrays(df)

    # ===== Stage 1: 粗网格 =====
    print("\n[Stage 1] 粗网格搜索 (50x50)...")
    stage1_start = time.time()
    dc_grid_coarse, acdc_grid_coarse = _build_grids(df, n_points=50)
    best, feasible_count = _grid_search(
        arrs, dc_grid_coarse, acdc_grid_coarse, target_pos_pass_rate, n_workers
    )
    stage1_time = time.time() - stage1_start
    print(f"  Stage 1 完成: {feasible_count}个可行解, 耗时 {stage1_time:.2f}秒")
    if best is not None:
        print(f"  当前最优: dc={best['dc_threshold']:.4e}, acdc={best['ac_dc_threshold']:.4f}")

    if best is None:
        print("[警告] Stage 1未找到可行解，尝试放宽搜索...")
        best = None
        dc_grid_coarse2, acdc_grid_coarse2 = _build_grids(df, n_points=30)
        best, fc2 = _grid_search(
            arrs, dc_grid_coarse2, acdc_grid_coarse2, target_pos_pass_rate, n_workers
        )
        feasible_count += fc2
        if best is None:
            return None

    # ===== Stage 2: 细网格 =====
    print("\n[Stage 2] 局部精细搜索 (30x30)...")
    stage2_start = time.time()
    dc_center = best["dc_threshold"]
    acdc_center = best["ac_dc_threshold"]
    dc_fine_grid, acdc_fine_grid = make_fine_grids(
        dc_center, acdc_center,
        dc_span_factor=0.2, acdc_span_factor=0.2, n_points=30
    )
    stage2_best, stage2_feasible = _grid_search(
        arrs, dc_fine_grid, acdc_fine_grid, target_pos_pass_rate, n_workers
    )
    stage2_time = time.time() - stage2_start
    print(f"  Stage 2 完成: {stage2_feasible}个可行解, 耗时 {stage2_time:.2f}秒")
    if stage2_best is not None and compare_thresholds(stage2_best, best):
        best = stage2_best
        print(f"  当前最优: dc={best['dc_threshold']:.4e}, acdc={best['ac_dc_threshold']:.4f}")

    # ===== Stage 3: differential evolution =====
    print("\n[Stage 3] 差分进化微调优化...")
    stage3_start = time.time()
    try:
        from scipy.optimize import differential_evolution

        def objective(x):
            dc_th, acdc_th = x
            m = _fast_eval_threshold(arrs, dc_th, acdc_th)
            if m["sample_target1_pass_rate"] + 1e-12 < target_pos_pass_rate:
                return 1.0
            return -m["sample_target0_reject_rate"] - 0.1 * m["window_target0_reject_rate"]

        dc_values = df["dc"].replace([np.inf, -np.inf], np.nan).dropna().values
        acdc_values = df["ac_dc_ratio"].replace([np.inf, -np.inf], np.nan).dropna().values
        dc_min, dc_max = float(np.min(dc_values)), float(np.max(dc_values))
        acdc_min, acdc_max = float(np.min(acdc_values)), float(np.max(acdc_values))
        dc_range = max(dc_max - dc_min, 1.0) * 0.3
        acdc_range = max(acdc_max - acdc_min, 1e-6) * 0.3
        bounds = [
            (max(best["dc_threshold"] - dc_range, dc_min),
             min(best["dc_threshold"] + dc_range, dc_max)),
            (max(best["ac_dc_threshold"] - acdc_range, acdc_min),
             min(best["ac_dc_threshold"] + acdc_range, acdc_max)),
        ]

        result = differential_evolution(
            objective, bounds,
            maxiter=50, popsize=10, tol=1e-6,
            mutation=(0.5, 1.0), recombination=0.7,
            polish=True,
            workers=1,
        )
        if result.success:
            stage3_best = _fast_eval_threshold(arrs, result.x[0], result.x[1])
            if (stage3_best["sample_target1_pass_rate"] + 1e-12 >= target_pos_pass_rate
                    and compare_thresholds(stage3_best, best)):
                best = stage3_best
                print(f"  Stage 3 优化成功: dc={best['dc_threshold']:.4e}, "
                      f"acdc={best['ac_dc_threshold']:.4f}")
    except Exception as e:
        print(f"  Stage 3 差分进化优化跳过: {e}")

    stage3_time = time.time() - stage3_start
    print(f"  Stage 3 耗时 {stage3_time:.2f}秒")

    total_time = time.time() - start_time
    if best is not None:
        best["feasible_count"] = int(feasible_count + stage2_feasible)
        best["stage1_time"] = round(stage1_time, 2)
        best["stage2_time"] = round(stage2_time, 2)
        best["stage3_time"] = round(stage3_time, 2)
        best["total_search_time"] = round(total_time, 2)
        best["search_method"] = "multi_stage"
        best["total_evaluations"] = 50 * 50 + 30 * 30 + 500

    print(f"\n[完成] 多阶段搜索总耗时: {total_time:.2f}秒")
    print(f"  最优阈值: dc={best['dc_threshold']:.4e}, acdc={best['ac_dc_threshold']:.4f}")
    print(f"  target=1样本通过率: {best['sample_target1_pass_rate']*100:.1f}%")
    print(f"  target=0样本拒绝率: {best['sample_target0_reject_rate']*100:.1f}%")
    return best


def make_train_threshold(deploy_dc, deploy_acdc, train_dc_ratio=0.90, train_acdc_margin=0.10):
    train_dc = float(deploy_dc * train_dc_ratio)
    train_acdc = float(deploy_acdc + train_acdc_margin)
    return {
        "dc_threshold": train_dc,
        "ac_dc_threshold": train_acdc,
        "rule": "宽松 Stage1，仅用于 second-stage 特征提取/筛选/训练，不用于最终部署评估",
        "train_dc_ratio": float(train_dc_ratio),
        "train_acdc_margin": float(train_acdc_margin),
    }


def build_fixed_stage1_threshold_artifact(
        df_train,
        df_valid,
        fixed_dc_threshold=FIXED_STAGE1_DC_THRESHOLD,
        fixed_ac_dc_threshold=FIXED_STAGE1_AC_DC_THRESHOLD,
        train_dc_ratio=0.90,
        train_acdc_margin=0.10):
    deploy_dc = float(fixed_dc_threshold)
    deploy_acdc = float(fixed_ac_dc_threshold)
    train_threshold = make_train_threshold(
        deploy_dc, deploy_acdc,
        train_dc_ratio=train_dc_ratio,
        train_acdc_margin=train_acdc_margin,
    )
    return {
        "threshold_search_enabled": False,
        "threshold_search_note": (
            "Stage1 uses a fixed engineering threshold. "
            "DC threshold search is intentionally disabled."
        ),
        "dc_threshold": deploy_dc,
        "ac_dc_threshold": deploy_acdc,

        "deploy_stage1_threshold": {
            "dc_threshold": deploy_dc,
            "ac_dc_threshold": deploy_acdc,
            "fs": int(STAGE1_FS),
            "window_sec": float(STAGE1_WINDOW_SEC),
            "stride_sec": float(STAGE1_STRIDE_SEC),
            "points_100hz": int(STAGE1_WINDOW_SEC * STAGE1_FS),
            "rule": "dc > dc_threshold and ac_dc_ratio < ac_dc_threshold",
            "sample_rule": "sample_pass = any(window_pass)",
            "target_requirement": "fixed engineering gate; no train/valid threshold search",
            "search_source": FIXED_STAGE1_SEARCH_SOURCE,
        },

        "train_stage1_threshold": train_threshold,

        "deploy_train_metrics": eval_threshold(df_train, deploy_dc, deploy_acdc),
        "deploy_valid_metrics": eval_threshold(df_valid, deploy_dc, deploy_acdc),
        "train_gate_metrics_on_train": eval_threshold(
            df_train, train_threshold["dc_threshold"], train_threshold["ac_dc_threshold"]
        ),
        "train_gate_metrics_on_valid": eval_threshold(
            df_valid, train_threshold["dc_threshold"], train_threshold["ac_dc_threshold"]
        ),

        "notes": [
            "deploy_stage1_threshold is fixed at dc=2.2e6 and ac_dc=0.35.",
            "Stage1 train/valid/test all use the same deployment gate.",
            "train_stage1_threshold remains a relaxed gate only for Stage2 train/valid feature extraction.",
            "test does not participate in threshold search, feature selection, or model threshold selection.",
        ]
    }


# =========================================================
# 可视化
# =========================================================

def plot_stage1_scatter(df_train, df_valid, deploy_dc, deploy_acdc, out_path):
    """画 DC vs AC/DC 散点图，不同 target 不同颜色，阈值虚线叠加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, df, title in [(ax1, df_train, "Train"), (ax2, df_valid, "Valid")]:
        if len(df) == 0:
            ax.set_title(f"{title} (no data)")
            continue

        t0 = df[df["target"] == 0]
        t1 = df[df["target"] == 1]

        ax.scatter(t0["dc"], t0["ac_dc_ratio"], c="red", s=8, alpha=0.4,
                   label="target=0 (not-worn)", edgecolors="none")
        ax.scatter(t1["dc"], t1["ac_dc_ratio"], c="green", s=8, alpha=0.6,
                   label="target=1 (worn)", edgecolors="none")

        ax.axvline(x=deploy_dc, color="blue", linestyle="--", linewidth=1.5,
                   label=f"dc={deploy_dc:.1e}")
        ax.axhline(y=deploy_acdc, color="blue", linestyle="--", linewidth=1.5,
                   label=f"ac/dc={deploy_acdc:.4f}")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between([deploy_dc, xlim[1]], 0, deploy_acdc,
                        alpha=0.05, color="blue")
        ax.text(deploy_dc * 1.05, deploy_acdc * 0.5, "PASS",
                fontsize=14, color="blue", alpha=0.3, weight="bold")

        ax.set_xlabel("DC")
        ax.set_ylabel("AC/DC Ratio")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(f"Stage1 PPG Threshold: DC > {deploy_dc:.1e},  AC/DC < {deploy_acdc:.4f}",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"散点图已保存: {out_path}")


# =========================================================
# main
# =========================================================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--search_method", choices=["fixed", "grid", "multi_stage"],
                        default="fixed",
                        help="阈值确定方式：fixed=用 --fixed_* 给定值；"
                             "grid=在 train 上单阶段网格搜索；"
                             "multi_stage=粗网格+细网格+differential_evolution")
    parser.add_argument("--target_pos_pass_rate", type=float, default=1.0,
                        help="search_method=grid/multi_stage 时要求 target=1 样本通过率下限")
    parser.add_argument("--fixed_dc_threshold", type=float, default=FIXED_STAGE1_DC_THRESHOLD)
    parser.add_argument("--fixed_ac_dc_threshold", type=float, default=FIXED_STAGE1_AC_DC_THRESHOLD)
    parser._option_string_actions["--search_method"].help = (
        "Legacy compatibility only; ignored because Stage1 threshold search is disabled."
    )
    parser._option_string_actions["--target_pos_pass_rate"].help = (
        "Legacy compatibility only; ignored because Stage1 threshold search is disabled."
    )
    parser.add_argument("--min_duration_sec", type=float, default=STAGE1_WINDOW_SEC)
    parser.add_argument("--train_dc_ratio", type=float, default=0.90)
    parser.add_argument("--train_acdc_margin", type=float, default=0.10)
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()
    elif not isinstance(args, argparse.Namespace):
        args = parser.parse_args(args)

    split_path = os.path.join(args.artifact_dir, "splits.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    os.makedirs(args.artifact_dir, exist_ok=True)

    print("=" * 80)
    print(f"提取 Stage1 train/valid 窗口 (n_workers={args.n_workers})")
    print("=" * 80)

    df_train = extract_stage1_windows(split["train"], min_duration_sec=args.min_duration_sec,
                                       n_workers=args.n_workers)
    df_valid = extract_stage1_windows(split["valid"], min_duration_sec=args.min_duration_sec,
                                       n_workers=args.n_workers)

    df_train.to_csv(os.path.join(args.artifact_dir, "stage1_train_windows.csv"), index=False)
    df_valid.to_csv(os.path.join(args.artifact_dir, "stage1_valid_windows.csv"), index=False)

    print(f"train windows: {len(df_train)}")
    print(f"valid windows: {len(df_valid)}")

    requested_search_method = args.search_method
    if requested_search_method != "fixed":
        print(
            f"[stage1] --search_method {requested_search_method!r} is ignored; "
            "Stage1 DC threshold search is disabled."
        )
    args.search_method = "fixed"

    if len(df_train) == 0:
        raise RuntimeError("Stage1 train 窗口为空。")

    # 阈值确定：fixed / grid / multi_stage
    if args.search_method == "fixed":
        deploy_dc = float(args.fixed_dc_threshold)
        deploy_acdc = float(args.fixed_ac_dc_threshold)
        search_source = FIXED_STAGE1_SEARCH_SOURCE
    elif args.search_method == "grid":
        print(f"\n[search] 在 train 上单阶段网格搜索 (target_pos_pass_rate={args.target_pos_pass_rate})")
        best = search_deploy_threshold(
            df_train, target_pos_pass_rate=args.target_pos_pass_rate,
            n_workers=args.n_workers)
        deploy_dc = float(best["dc_threshold"])
        deploy_acdc = float(best["ac_dc_threshold"])
        search_source = "grid"
    elif args.search_method == "multi_stage":
        print(f"\n[search] 多阶段搜索: 粗网格+细网格+DE (target_pos_pass_rate={args.target_pos_pass_rate})")
        best = search_deploy_threshold_multi_stage(
            df_train, target_pos_pass_rate=args.target_pos_pass_rate,
            n_workers=args.n_workers)
        deploy_dc = float(best["dc_threshold"])
        deploy_acdc = float(best["ac_dc_threshold"])
        search_source = "multi_stage"
    else:
        raise ValueError(f"未知 search_method: {args.search_method}")
    print(f"[search] 最终阈值: dc={deploy_dc:.3e}, ac/dc={deploy_acdc:.4f} (source={search_source})")

    train_metrics = eval_threshold(df_train, deploy_dc, deploy_acdc)
    valid_metrics = eval_threshold(df_valid, deploy_dc, deploy_acdc)

    train_threshold = make_train_threshold(
        deploy_dc, deploy_acdc,
        train_dc_ratio=args.train_dc_ratio,
        train_acdc_margin=args.train_acdc_margin
    )

    train_gate_metrics_on_train = eval_threshold(
        df_train, train_threshold["dc_threshold"], train_threshold["ac_dc_threshold"]
    )
    train_gate_metrics_on_valid = eval_threshold(
        df_valid, train_threshold["dc_threshold"], train_threshold["ac_dc_threshold"]
    )

    result = {
        "threshold_search_enabled": False,
        "threshold_search_note": "Stage1 uses a fixed engineering threshold. DC threshold search is disabled.",
        "dc_threshold": float(deploy_dc),
        "ac_dc_threshold": float(deploy_acdc),

        "deploy_stage1_threshold": {
            "dc_threshold": float(deploy_dc),
            "ac_dc_threshold": float(deploy_acdc),
            "fs": int(STAGE1_FS),
            "window_sec": float(STAGE1_WINDOW_SEC),
            "stride_sec": float(STAGE1_STRIDE_SEC),
            "points_100hz": int(STAGE1_WINDOW_SEC * STAGE1_FS),
            "rule": "dc > dc_threshold and ac_dc_ratio < ac_dc_threshold",
            "sample_rule": "sample_pass = any(window_pass)",
            "target_requirement": "target=1 sample pass rate must be 100%",
            "search_source": search_source,
        },

        "train_stage1_threshold": train_threshold,

        "deploy_train_metrics": train_metrics,
        "deploy_valid_metrics": valid_metrics,
        "train_gate_metrics_on_train": train_gate_metrics_on_train,
        "train_gate_metrics_on_valid": train_gate_metrics_on_valid,

        "notes": [
            "deploy_stage1_threshold 用于最终部署和端到端 test 评估",
            "train_stage1_threshold 用于第二阶段 train/valid 特征提取、特征筛选、模型训练",
            "test 不参与任何阈值搜索、特征筛选、模型阈值选择"
        ]
    }

    print("\n" + "=" * 80)
    print("Stage1 阈值结果")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    plot_path = os.path.join(args.artifact_dir, "stage1_scatter.png")
    plot_stage1_scatter(df_train, df_valid, deploy_dc, deploy_acdc, plot_path)

    out_path = os.path.join(args.artifact_dir, "stage1_threshold.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
