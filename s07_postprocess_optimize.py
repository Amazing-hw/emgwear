# s07_postprocess_optimize.py
# -*- coding: utf-8 -*-

"""Search postprocess parameters from per-sample window NPZ caches."""

import argparse
import json
import os
import pickle
import sys
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from s06_deploy_eval import apply_postprocess

# Module-level storage for worker processes (ProcessPoolExecutor initializer pattern)
_SCORE_DATA = None


def _looks_like_float_csv(value):
    if not isinstance(value, str) or not value.startswith("-"):
        return False
    try:
        for part in value.split(","):
            float(part.strip())
    except ValueError:
        return False
    return True


def _normalize_negative_csv_options(argv, option_names):
    normalized = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in option_names and i + 1 < len(argv) and _looks_like_float_csv(argv[i + 1]):
            normalized.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def _init_score_worker(payload_bytes):
    global _SCORE_DATA
    _SCORE_DATA = pickle.loads(payload_bytes)


def _score_grid_point(params):
    prepared_caches, fp_cost = _SCORE_DATA
    metrics = evaluate_params_fast(prepared_caches, params)
    score = score_metrics(metrics, fp_cost=fp_cost)
    return {**params, **metrics, "score": float(score)}


def _format_elapsed(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _progress_due(done, total, interval, last_report_done):
    if done >= total:
        return True
    interval = max(1, int(interval or 1))
    return done == 1 or (done - last_report_done) >= interval


def _print_search_progress(done, total, start_time, best_score, workers):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    print(
        "[s07] progress "
        f"{done}/{total} ({done / total:.1%}) "
        f"elapsed={_format_elapsed(elapsed)} eta={_format_elapsed(remaining)} "
        f"workers={workers} best_score={best_score:.6f}",
        flush=True,
    )


REQUIRED_KEYS = (
    "sample_name",
    "target",
    "prob_raw",
    "stage1_enabled",
    "quality",
    "model_threshold",
    "window_sec",
    "stride_sec",
    "model_fingerprint_json",
    "feature_names_json",
)


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


def load_window_cache_npz(path):
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path} missing required cache keys: {missing}")

        cache = {k: data[k].copy() for k in data.files}

    prob = np.asarray(cache["prob_raw"], dtype=np.float64)
    stage1 = np.asarray(cache["stage1_enabled"], dtype=np.int8)
    quality = np.asarray(cache["quality"], dtype=np.float64)
    if not (len(prob) == len(stage1) == len(quality)):
        raise ValueError(f"{path} has inconsistent window array lengths")

    cache["path"] = str(path)
    cache["sample_name"] = str(_scalar(cache["sample_name"]))
    cache["target"] = int(_scalar(cache["target"]))
    cache["prob_raw"] = prob
    cache["stage1_enabled"] = stage1
    cache["quality"] = quality
    cache["model_threshold"] = float(_scalar(cache["model_threshold"]))
    cache["window_sec"] = float(_scalar(cache["window_sec"]))
    cache["stride_sec"] = float(_scalar(cache["stride_sec"]))
    cache["model_fingerprint_json"] = str(_scalar(cache["model_fingerprint_json"]))
    cache["feature_names_json"] = str(_scalar(cache["feature_names_json"]))
    n = len(prob)
    for key in ("ood_rate", "stage1_dc", "stage1_acdc", "stage1_dc_margin", "stage1_acdc_margin"):
        if key in cache:
            arr = np.asarray(cache[key], dtype=np.float64)
            if len(arr) != n:
                # Pad/trim to match window count; use NaN padding (not np.resize which repeats)
                padded = np.full(n, np.nan, dtype=np.float64)
                copy_len = min(len(arr), n)
                padded[:copy_len] = arr[:copy_len]
                arr = padded
            cache[key] = arr
        else:
            cache[key] = np.full(n, np.nan, dtype=np.float64)
    return cache


def _window_slice(cache, skip_initial_windows=0):
    start = max(0, int(skip_initial_windows))
    n = len(cache["prob_raw"])
    return slice(min(start, n), n)


def run_postprocess_on_cache(cache, params, model_threshold=None, skip_initial_windows=0):
    threshold = cache["model_threshold"] if model_threshold is None else float(model_threshold)
    sl = _window_slice(cache, skip_initial_windows)
    quality_metas = [{"quality": float(q)} for q in np.asarray(cache["quality"], dtype=float)[sl]]
    return apply_postprocess(
        cache["prob_raw"][sl],
        quality_metas,
        method="state_machine",
        cfg=params,
        model_threshold=threshold,
        stride_sec=cache["stride_sec"],
        stage1_frames=cache["stage1_enabled"][sl].astype(bool).tolist(),
    )


def load_cache_dir(cache_dir):
    paths = sorted(p for p in os.listdir(cache_dir) if p.endswith(".npz"))
    return [load_window_cache_npz(os.path.join(cache_dir, p)) for p in paths]


def parse_csv_floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_splits(text):
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def resolve_cache_dir(artifact_dir, cache_root, split):
    cache_dir = os.path.join(artifact_dir, cache_root, split)
    if not os.path.isdir(cache_dir) and cache_root == "window_outputs":
        legacy = os.path.join(artifact_dir, "window_cache", split)
        if os.path.isdir(legacy):
            cache_dir = legacy
    return cache_dir


def load_cache_splits(artifact_dir, cache_root, splits):
    caches = []
    cache_dirs = {}
    for split in splits:
        cache_dir = resolve_cache_dir(artifact_dir, cache_root, split)
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"window cache directory not found: {cache_dir}")
        split_caches = load_cache_dir(cache_dir)
        for cache in split_caches:
            cache["split"] = split
        caches.extend(split_caches)
        cache_dirs[split] = cache_dir
    return caches, cache_dirs


def threshold_candidates_for_cache(cache, offsets, min_threshold=0.02, max_threshold=0.98):
    base = float(cache["model_threshold"])
    values = []
    for offset in offsets:
        value = float(np.clip(base + float(offset), min_threshold, max_threshold))
        if value not in values:
            values.append(value)
    return values


def sample_window_status(cache, skip_initial_windows=0):
    sl = _window_slice(cache, skip_initial_windows)
    probs = np.asarray(cache["prob_raw"], dtype=float)[sl]
    enabled = np.asarray(cache["stage1_enabled"], dtype=np.int8)[sl].astype(bool)
    if len(probs) == 0:
        return "no_windows"
    threshold = float(cache["model_threshold"])
    preds = ((probs >= threshold) & enabled).astype(int)
    target = int(cache["target"])
    return "all_correct" if bool(np.all(preds == target)) else "hard"


def filter_hard_samples(caches, skip_initial_windows=0):
    selected = []
    counts = {
        "total_samples": int(len(caches)),
        "hard_samples": 0,
        "all_correct_samples": 0,
        "no_window_samples": 0,
    }
    for cache in caches:
        status = sample_window_status(cache, skip_initial_windows=skip_initial_windows)
        cache["window_status"] = status
        if status == "hard":
            selected.append(cache)
            counts["hard_samples"] += 1
        elif status == "all_correct":
            counts["all_correct_samples"] += 1
        else:
            counts["no_window_samples"] += 1
    return selected, counts


def iter_param_grid(threshold_offsets=None, include_threshold=True):
    offsets = list(threshold_offsets if threshold_offsets is not None else [0.0])
    alphas = [0.25, 0.4, 0.6]
    t_ons = [0.55, 0.65, 0.75]
    t_offs = [0.20, 0.35, 0.45]
    k_ons = [1, 2, 3, 5]
    k_offs = [1, 2, 3, 5]
    cooldowns = [0, 2, 5]
    median_ks = [1, 3, 5]
    offset_values = offsets if include_threshold else [0.0]
    for threshold_offset, alpha, t_on, t_off, k_on, k_off, cooldown, median_k in product(
        offset_values, alphas, t_ons, t_offs, k_ons, k_offs, cooldowns, median_ks
    ):
        if t_off >= t_on:
            continue
        yield {
            "alpha": alpha,
            "T_on": t_on,
            "T_off": t_off,
            "K_on": k_on,
            "K_off": k_off,
            "cooldown_sec": cooldown,
            "median_k": median_k,
            "threshold_offset": float(threshold_offset),
        }


def _unique_param_values(grid, key):
    values = []
    for params in grid:
        value = params[key]
        if value not in values:
            values.append(value)
    return values


def _freeze_params(params):
    return tuple(sorted(params.items()))


def _postprocess_grid_priority(params):
    return (
        abs(float(params.get("threshold_offset", 0.0))),
        int(params.get("K_on", 1)),
        -int(params.get("K_off", 1)),
        float(params.get("T_on", 0.0)) - float(params.get("T_off", 0.0)),
        int(params.get("median_k", 1)),
        float(params.get("cooldown_sec", 0.0)),
        float(params.get("alpha", 0.0)),
    )


def select_postprocess_search_grid(grid, search_budget=None):
    grid = list(grid)
    if search_budget is None:
        return grid
    budget = int(search_budget)
    if budget <= 0 or budget >= len(grid):
        return grid

    selected = []
    seen = set()

    def add(params):
        key = _freeze_params(params)
        if key in seen or len(selected) >= budget:
            return
        selected.append(params)
        seen.add(key)

    if grid:
        add(grid[0])
        add(grid[len(grid) // 2])
        add(grid[-1])

    for params in sorted(grid, key=_postprocess_grid_priority):
        add(params)
        if len(selected) >= budget:
            break
    return selected


def _causal_median_filter_fast(values, k):
    k = int(k or 1)
    arr = np.asarray(values, dtype=np.float64)
    if k <= 1 or arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.size):
        start = max(0, i - k + 1)
        out[i] = float(np.median(arr[start:i + 1]))
    return out


def prepare_fast_search_caches(caches, threshold_offsets, median_ks, skip_initial_windows=0):
    fast_caches = []
    offsets = [float(x) for x in threshold_offsets]
    median_ks = [int(x) for x in median_ks]
    for cache in caches:
        sl = _window_slice(cache, skip_initial_windows)
        raw = np.asarray(cache["prob_raw"], dtype=np.float64)[sl]
        enabled = np.asarray(cache["stage1_enabled"], dtype=np.int8)[sl].astype(bool)
        quality = np.clip(np.asarray(cache["quality"], dtype=np.float64)[sl], 0.0, 1.0)
        model_threshold = float(cache["model_threshold"])

        series = {}
        window_preds = {}
        for offset in offsets:
            adjusted_threshold = float(np.clip(model_threshold + offset, 0.02, 0.98))
            probs = np.clip(raw - adjusted_threshold + 0.5, 0.0, 1.0)
            probs = np.where(enabled, probs, 0.0)
            for median_k in median_ks:
                filtered = _causal_median_filter_fast(probs, median_k)
                key = (float(offset), int(median_k))
                series[key] = filtered
                window_preds[key] = (filtered >= 0.5).astype(np.int8)

        skipped = min(max(0, int(skip_initial_windows)), len(cache["prob_raw"]))
        fast_caches.append({
            "sample_name": cache.get("sample_name"),
            "target": int(cache["target"]),
            "stride_sec": float(cache["stride_sec"]),
            "quality": quality,
            "series": series,
            "window_preds": window_preds,
            "skipped_initial_windows": int(skipped),
        })
    return fast_caches


def _run_fast_state_machine(prepared_cache, params):
    key = (float(params.get("threshold_offset", 0.0)), int(params.get("median_k", 1)))
    probs = prepared_cache["series"][key]
    quality = prepared_cache["quality"]
    if probs.size == 0:
        return 0, np.asarray([], dtype=np.int8)

    alpha = float(params.get("alpha", 0.4))
    t_on = float(params.get("T_on", 0.75))
    t_off = float(params.get("T_off", 0.35))
    k_on = int(params.get("K_on", 5))
    k_off = int(params.get("K_off", 5))
    cooldown_sec = float(params.get("cooldown_sec", 5))
    stride_sec = float(prepared_cache["stride_sec"])
    cooldown_steps = int(cooldown_sec / stride_sec) if stride_sec > 0 else int(cooldown_sec)

    state = 0
    score = 0.0
    on_count = 0
    off_count = 0
    steps_since_flip = 999
    states = np.empty(probs.size, dtype=np.int8)

    for i, p in enumerate(probs):
        eff_alpha = alpha * float(quality[i]) if i < quality.size else alpha
        score = eff_alpha * float(p) + (1.0 - eff_alpha) * score
        steps_since_flip += 1

        if state == 0:
            if score > t_on:
                on_count += 1
            else:
                on_count = max(0, on_count - 1)
            if on_count >= k_on and steps_since_flip >= cooldown_steps:
                state = 1
                on_count = 0
                off_count = 0
                steps_since_flip = 0
        else:
            if score < t_off:
                off_count += 1
            else:
                off_count = max(0, off_count - 1)
            if off_count >= k_off and steps_since_flip >= cooldown_steps:
                state = 0
                on_count = 0
                off_count = 0
                steps_since_flip = 0
        states[i] = state

    return int(states[-1]), states


def evaluate_params(caches, params, skip_initial_windows=0):
    n_samples = 0
    sample_correct = 0
    tp = fp = fn = 0
    neg = sample_fp = 0
    n_windows = 0
    window_correct = 0
    for cache in caches:
        pred, states, _window_preds, _scores = run_postprocess_on_cache(
            cache, params, skip_initial_windows=skip_initial_windows
        )
        target = int(cache["target"])
        pred = int(pred)
        n_samples += 1
        sample_correct += int(pred == target)
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        if target == 0:
            neg += 1
            sample_fp += int(pred == 1)
        for state in states:
            n_windows += 1
            window_correct += int(int(state) == target)

    if n_samples == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "window_accuracy": 0.0,
            "sample_fp_rate": 0.0,
        }

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fp_rate = float(sample_fp / neg) if neg else 0.0
    skipped = sum(min(max(0, int(skip_initial_windows)), len(c["prob_raw"])) for c in caches)
    return {
        "accuracy": float(sample_correct / n_samples),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "window_accuracy": float(window_correct / n_windows) if n_windows else 0.0,
        "sample_fp_rate": fp_rate,
        "skipped_initial_windows": int(skipped),
    }


def evaluate_params_fast(prepared_caches, params):
    n_samples = 0
    sample_correct = 0
    tp = fp = fn = 0
    neg = sample_fp = 0
    n_windows = 0
    window_correct = 0
    skipped = 0
    key = (float(params.get("threshold_offset", 0.0)), int(params.get("median_k", 1)))

    for cache in prepared_caches:
        target = int(cache["target"])
        pred, states = _run_fast_state_machine(cache, params)
        n_samples += 1
        sample_correct += int(pred == target)
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        if target == 0:
            neg += 1
            sample_fp += int(pred == 1)

        n_windows += int(states.size)
        if states.size:
            window_correct += int(np.sum(states == target))
        skipped += int(cache.get("skipped_initial_windows", 0))

    if n_samples == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "window_accuracy": 0.0,
            "sample_fp_rate": 0.0,
        }

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fp_rate = float(sample_fp / neg) if neg else 0.0
    return {
        "accuracy": float(sample_correct / n_samples),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "window_accuracy": float(window_correct / n_windows) if n_windows else 0.0,
        "sample_fp_rate": fp_rate,
        "skipped_initial_windows": int(skipped),
    }


def score_metrics(metrics, fp_cost=1.5):
    return (
        metrics["accuracy"]
        + 0.35 * metrics["recall"]
        + 0.15 * metrics["window_accuracy"]
        - float(fp_cost) * metrics["sample_fp_rate"]
    )


def search_postprocess(caches, fp_cost=1.5, skip_initial_windows=0, n_workers=None,
                       threshold_offsets=None, max_candidates=None, progress_interval=200,
                       search_budget=None):
    full_grid = list(iter_param_grid(threshold_offsets=threshold_offsets))
    budget = search_budget if search_budget is not None else max_candidates
    grid = select_postprocess_search_grid(full_grid, search_budget=budget)
    n_workers = max(1, int(n_workers or 1))
    total = len(grid)
    if total == 0:
        empty = pd.DataFrame()
        return None, empty
    median_ks = _unique_param_values(grid, "median_k")
    offsets = _unique_param_values(grid, "threshold_offset")
    prepared_caches = prepare_fast_search_caches(
        caches,
        threshold_offsets=offsets,
        median_ks=median_ks,
        skip_initial_windows=skip_initial_windows,
    )

    print(
        "[s07] 搜参开始: "
        f"candidates={total}, full_grid={len(full_grid)}, search_budget={budget}, "
        f"samples={len(caches)}, workers={n_workers}, "
        f"precomputed_series={len(offsets) * len(median_ks)}, "
        f"progress_interval={max(1, int(progress_interval or 1))}",
        flush=True,
    )
    start_time = time.time()
    last_report_done = 0

    if n_workers <= 1 or len(grid) <= 4:
        rows = []
        best = None
        for done, params in enumerate(grid, start=1):
            metrics = evaluate_params_fast(prepared_caches, params)
            score = score_metrics(metrics, fp_cost=fp_cost)
            row = {**params, **metrics, "score": float(score)}
            rows.append(row)
            if best is None or score > best["score"]:
                best = row
            if _progress_due(done, total, progress_interval, last_report_done):
                _print_search_progress(done, total, start_time, float(best["score"]), n_workers)
                last_report_done = done
        print(f"[s07] 搜参完成: best_score={float(best['score']):.6f}", flush=True)
        return best, pd.DataFrame(rows).sort_values("score", ascending=False)

    payload = (prepared_caches, float(fp_cost))
    payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    chunksize = max(1, len(grid) // (n_workers * 4))

    rows = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_score_worker,
        initargs=(payload_bytes,),
    ) as executor:
        best = None
        for done, row in enumerate(executor.map(_score_grid_point, grid, chunksize=chunksize), start=1):
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
            if _progress_due(done, total, progress_interval, last_report_done):
                _print_search_progress(done, total, start_time, float(best["score"]), n_workers)
                last_report_done = done

    print(f"[s07] 搜参完成: best_score={float(best['score']):.6f}", flush=True)
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def scan_window_thresholds(caches, thresholds, skip_initial_windows=0):
    rows = []
    for threshold in thresholds:
        y_true, y_pred = [], []
        skipped = 0
        for cache in caches:
            sl = _window_slice(cache, skip_initial_windows)
            skipped += min(max(0, int(skip_initial_windows)), len(cache["prob_raw"]))
            probs = cache["prob_raw"][sl]
            enabled = cache["stage1_enabled"][sl].astype(bool)
            preds = ((probs >= float(threshold)) & enabled).astype(int)
            y_true.extend([int(cache["target"])] * len(preds))
            y_pred.extend(preds.tolist())
        if y_true:
            rows.append({
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "n_windows": int(len(y_true)),
                "skipped_initial_windows": int(skipped),
            })
    return pd.DataFrame(rows)


def summarize_guardrail(caches, params, skip_initial_windows=0):
    metrics = evaluate_params(caches, params, skip_initial_windows=skip_initial_windows)
    all_correct = [c for c in caches if c.get("window_status") == "all_correct"]
    regressed = []
    threshold_offset = float(params.get("threshold_offset", 0.0))
    for cache in all_correct:
        pred, _states, _window_preds, _scores = run_postprocess_on_cache(
            cache, params, skip_initial_windows=skip_initial_windows
        )
        if int(pred) != int(cache["target"]):
            regressed.append(cache["sample_name"])
    return {
        "all_samples_metrics": metrics,
        "all_correct_samples": int(len(all_correct)),
        "all_correct_regressions": int(len(regressed)),
        "all_correct_regressed_samples": regressed,
    }


def _diagnostic_tag(target, pred, stage1_enabled, quality, ood_rate):
    if not stage1_enabled:
        return "stage1_blocked"
    if pred == target:
        return "correct"
    if np.isfinite(ood_rate) and ood_rate >= 0.3:
        return "high_ood_fp" if pred == 1 else "high_ood_fn"
    if np.isfinite(quality) and quality < 0.5:
        return "low_quality_fp" if pred == 1 else "low_quality_fn"
    return "model_fp" if pred == 1 else "model_fn"


def build_window_error_report(caches, model_threshold=None, skip_initial_windows=0):
    rows = []
    for cache in caches:
        threshold = cache["model_threshold"] if model_threshold is None else float(model_threshold)
        sl = _window_slice(cache, skip_initial_windows)
        start = sl.start or 0
        probs = cache["prob_raw"][sl]
        enabled = cache["stage1_enabled"][sl].astype(bool)
        quality = cache["quality"][sl]
        ood = cache.get("ood_rate", np.full(len(cache["prob_raw"]), np.nan))[sl]
        preds = ((probs >= threshold) & enabled).astype(int)
        target = int(cache["target"])
        for offset, pred in enumerate(preds):
            idx = start + offset
            q = float(quality[offset]) if offset < len(quality) else np.nan
            o = float(ood[offset]) if offset < len(ood) else np.nan
            rows.append({
                "sample_name": cache["sample_name"],
                "window_index": int(idx),
                "target": target,
                "prob_raw": float(probs[offset]),
                "window_pred": int(pred),
                "stage1_enabled": int(enabled[offset]),
                "quality": q,
                "ood_rate": o,
                "stage1_dc": float(cache["stage1_dc"][idx]) if "stage1_dc" in cache else np.nan,
                "stage1_acdc": float(cache["stage1_acdc"][idx]) if "stage1_acdc" in cache else np.nan,
                "stage1_dc_margin": float(cache["stage1_dc_margin"][idx]) if "stage1_dc_margin" in cache else np.nan,
                "stage1_acdc_margin": float(cache["stage1_acdc_margin"][idx]) if "stage1_acdc_margin" in cache else np.nan,
                "is_error": int(pred != target),
                "diagnostic_tag": _diagnostic_tag(target, int(pred), bool(enabled[offset]), q, o),
            })
    return pd.DataFrame(rows)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--search_splits", type=str, default="train,valid")
    parser.add_argument("--cache_root", type=str, default="window_outputs")
    parser.add_argument("--fp_cost", type=float, default=4.0)
    parser.add_argument("--skip_initial_windows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for postprocess grid search (default 1=serial)")
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--threshold_offsets", type=str, default="-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3")
    parser.add_argument("--hard_samples_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_all_correct_regressions", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=200,
                        help="Print search progress every N completed candidates.")
    parser.add_argument("--max_candidates", type=int, default=None,
                        help="Deprecated alias for --search_budget.")
    parser.add_argument("--search_budget", type=int, default=240,
                        help="Maximum representative postprocess candidates; <=0 searches full grid.")

    if args is None:
        args = parser.parse_args(_normalize_negative_csv_options(sys.argv[1:], {"--threshold_offsets"}))

    search_splits = parse_csv_splits(getattr(args, "search_splits", "") or getattr(args, "split", "valid"))
    if not search_splits:
        search_splits = [str(args.split).lower()]
    if any(split == "test" for split in search_splits) or str(args.split).lower() == "test":
        raise ValueError(
            "test split cannot be used for postprocess optimization; use valid "
            "and reserve test for final reporting."
        )

    caches, cache_dirs = load_cache_splits(args.artifact_dir, args.cache_root, search_splits)
    hard_caches, hard_summary = filter_hard_samples(
        caches, skip_initial_windows=args.skip_initial_windows
    )
    search_caches = hard_caches if args.hard_samples_only else caches
    if not search_caches:
        print("[WARN] No hard samples available; falling back to all non-test cached samples.")
        search_caches = caches

    threshold_offsets = parse_csv_floats(args.threshold_offsets)
    best, results = search_postprocess(
        search_caches, fp_cost=args.fp_cost, skip_initial_windows=args.skip_initial_windows,
        n_workers=args.workers, threshold_offsets=threshold_offsets,
        max_candidates=args.max_candidates, search_budget=args.search_budget,
        progress_interval=args.progress_interval,
    )
    if best is None:
        raise RuntimeError("postprocess search produced no candidates")
    guardrail = summarize_guardrail(caches, best, skip_initial_windows=args.skip_initial_windows)

    out_dir = os.path.join(args.artifact_dir, "postprocess_opt")
    os.makedirs(out_dir, exist_ok=True)
    split_label = "_".join(search_splits)
    csv_path = os.path.join(out_dir, f"postprocess_search_{split_label}.csv")
    json_path = os.path.join(out_dir, f"postprocess_optimized_{split_label}.json")
    threshold_csv_path = os.path.join(out_dir, f"window_threshold_scan_{split_label}.csv")
    error_csv_path = os.path.join(out_dir, f"window_error_report_{split_label}.csv")
    error_summary_path = os.path.join(out_dir, f"window_error_summary_{split_label}.csv")
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    thresholds = parse_csv_floats(args.thresholds)
    threshold_scan = scan_window_thresholds(
        caches, thresholds, skip_initial_windows=args.skip_initial_windows
    )
    threshold_scan.to_csv(threshold_csv_path, index=False, encoding="utf-8-sig")

    error_report = build_window_error_report(
        caches, model_threshold=None, skip_initial_windows=args.skip_initial_windows
    )
    error_report.to_csv(error_csv_path, index=False, encoding="utf-8-sig")
    if len(error_report) > 0:
        error_summary = (
            error_report.groupby("diagnostic_tag")
            .agg(n_windows=("diagnostic_tag", "size"), n_errors=("is_error", "sum"))
            .reset_index()
        )
        error_summary["error_rate"] = error_summary["n_errors"] / error_summary["n_windows"]
    else:
        error_summary = pd.DataFrame(columns=["diagnostic_tag", "n_windows", "n_errors", "error_rate"])
    error_summary.to_csv(error_summary_path, index=False, encoding="utf-8-sig")

    best_params = {
        "alpha": float(best["alpha"]),
        "T_on": float(best["T_on"]),
        "T_off": float(best["T_off"]),
        "K_on": int(best["K_on"]),
        "K_off": int(best["K_off"]),
        "cooldown_sec": float(best["cooldown_sec"]),
        "median_k": int(best["median_k"]),
        "threshold_offset": float(best.get("threshold_offset", 0.0)),
        "threshold_transform": "clip(prob_raw - (model_threshold + threshold_offset) + 0.5, 0, 1)",
    }
    payload = {
        "split": split_label,
        "search_splits": search_splits,
        "cache_dirs": cache_dirs,
        "n_samples": len(caches),
        "n_search_samples": len(search_caches),
        "hard_samples_only": bool(args.hard_samples_only),
        "hard_sample_summary": hard_summary,
        "threshold_offsets": threshold_offsets,
        "fp_cost": float(args.fp_cost),
        "skip_initial_windows": int(args.skip_initial_windows),
        "search_budget": int(args.search_budget),
        "max_candidates_alias": None if args.max_candidates is None else int(args.max_candidates),
        "evaluated_candidates": int(len(results)),
        "full_grid_candidates": int(len(list(iter_param_grid(threshold_offsets=threshold_offsets)))),
        "best_params": best_params,
        "best_metrics": {
            k: float(best[k])
            for k in ("accuracy", "precision", "recall", "f1", "window_accuracy", "sample_fp_rate", "score")
        },
        "guardrail": guardrail,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    cfg_path = os.path.join(args.artifact_dir, "final_model_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Failed to read {cfg_path}, using defaults: {exc}")
            cfg = {}
    regressions = int(guardrail.get("all_correct_regressions", 0))
    if regressions <= int(args.max_all_correct_regressions):
        cfg["postprocess"] = best_params
        cfg["postprocess_cache_optimization"] = payload
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    else:
        payload["write_skipped_reason"] = (
            f"all_correct_regressions={regressions} exceeds "
            f"max_all_correct_regressions={args.max_all_correct_regressions}"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved search table: {csv_path}")
    print(f"saved threshold scan: {threshold_csv_path}")
    print(f"saved window error report: {error_csv_path}")
    print(f"saved window error summary: {error_summary_path}")
    print(f"saved best config:  {json_path}")


if __name__ == "__main__":
    main()
