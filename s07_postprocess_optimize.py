# s07_postprocess_optimize.py
# -*- coding: utf-8 -*-

"""Search postprocess parameters from per-sample window NPZ caches."""

import argparse
import json
import os
import pickle
from itertools import product
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from s06_deploy_eval import apply_postprocess

# Module-level storage for worker processes (ProcessPoolExecutor initializer pattern)
_SCORE_DATA = None


def _init_score_worker(payload_bytes):
    global _SCORE_DATA
    _SCORE_DATA = pickle.loads(payload_bytes)


def _score_grid_point(params):
    caches, skip_initial_windows, fp_cost = _SCORE_DATA
    metrics = evaluate_params(caches, params, skip_initial_windows=skip_initial_windows)
    score = score_metrics(metrics, fp_cost=fp_cost)
    return {**params, **metrics, "score": float(score)}


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


def iter_param_grid():
    alphas = [0.25, 0.4, 0.6]
    t_ons = [0.55, 0.65, 0.75]
    t_offs = [0.20, 0.35, 0.45]
    k_ons = [1, 2, 3, 5]
    k_offs = [1, 2, 3, 5]
    cooldowns = [0, 2, 5]
    median_ks = [1, 3, 5]
    for alpha, t_on, t_off, k_on, k_off, cooldown, median_k in product(
        alphas, t_ons, t_offs, k_ons, k_offs, cooldowns, median_ks
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
        }


def evaluate_params(caches, params, skip_initial_windows=0):
    y_true, y_pred = [], []
    win_true, win_pred = [], []
    for cache in caches:
        pred, states, _window_preds, _scores = run_postprocess_on_cache(
            cache, params, skip_initial_windows=skip_initial_windows
        )
        target = int(cache["target"])
        y_true.append(target)
        y_pred.append(int(pred))
        for state in states:
            win_true.append(target)
            win_pred.append(int(state))

    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "window_accuracy": 0.0,
            "sample_fp_rate": 0.0,
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg = y_true == 0
    fp_rate = float(np.mean(y_pred[neg] == 1)) if np.any(neg) else 0.0
    skipped = sum(min(max(0, int(skip_initial_windows)), len(c["prob_raw"])) for c in caches)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "window_accuracy": float(accuracy_score(win_true, win_pred)) if win_true else 0.0,
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


def search_postprocess(caches, fp_cost=1.5, skip_initial_windows=0, n_workers=None):
    grid = list(iter_param_grid())
    n_workers = max(1, int(n_workers or 1))

    if n_workers <= 1 or len(grid) <= 4:
        rows = []
        best = None
        for params in grid:
            metrics = evaluate_params(caches, params, skip_initial_windows=skip_initial_windows)
            score = score_metrics(metrics, fp_cost=fp_cost)
            row = {**params, **metrics, "score": float(score)}
            rows.append(row)
            if best is None or score > best["score"]:
                best = row
        return best, pd.DataFrame(rows).sort_values("score", ascending=False)

    payload = (caches, skip_initial_windows, float(fp_cost))
    payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    chunksize = max(1, len(grid) // (n_workers * 4))

    rows = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_score_worker,
        initargs=(payload_bytes,),
    ) as executor:
        for row in executor.map(_score_grid_point, grid, chunksize=chunksize):
            rows.append(row)

    best = max(rows, key=lambda r: r["score"])
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
    parser.add_argument("--cache_root", type=str, default="window_outputs")
    parser.add_argument("--fp_cost", type=float, default=4.0)
    parser.add_argument("--skip_initial_windows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for postprocess grid search (default 1=serial)")
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8")

    if args is None:
        args = parser.parse_args()

    cache_dir = os.path.join(args.artifact_dir, args.cache_root, args.split)
    if not os.path.isdir(cache_dir) and args.cache_root == "window_outputs":
        legacy = os.path.join(args.artifact_dir, "window_cache", args.split)
        if os.path.isdir(legacy):
            cache_dir = legacy
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"window cache directory not found: {cache_dir}")

    caches = load_cache_dir(cache_dir)
    best, results = search_postprocess(
        caches, fp_cost=args.fp_cost, skip_initial_windows=args.skip_initial_windows,
        n_workers=args.workers,
    )

    out_dir = os.path.join(args.artifact_dir, "postprocess_opt")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"postprocess_search_{args.split}.csv")
    json_path = os.path.join(out_dir, f"postprocess_optimized_{args.split}.json")
    threshold_csv_path = os.path.join(out_dir, f"window_threshold_scan_{args.split}.csv")
    error_csv_path = os.path.join(out_dir, f"window_error_report_{args.split}.csv")
    error_summary_path = os.path.join(out_dir, f"window_error_summary_{args.split}.csv")
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
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
    }
    payload = {
        "split": args.split,
        "cache_dir": cache_dir,
        "n_samples": len(caches),
        "fp_cost": float(args.fp_cost),
        "skip_initial_windows": int(args.skip_initial_windows),
        "best_params": best_params,
        "best_metrics": {
            k: float(best[k])
            for k in ("accuracy", "precision", "recall", "f1", "window_accuracy", "sample_fp_rate", "score")
        },
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
    cfg["postprocess"] = best_params
    cfg["postprocess_cache_optimization"] = payload
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved search table: {csv_path}")
    print(f"saved threshold scan: {threshold_csv_path}")
    print(f"saved window error report: {error_csv_path}")
    print(f"saved window error summary: {error_summary_path}")
    print(f"saved best config:  {json_path}")


if __name__ == "__main__":
    main()
