# s06_deploy_eval.py
# -*- coding: utf-8 -*-

"""
Step 6: end-to-end deployment evaluation for PPG + EMG + ACC wearing liveness.

Pipeline:
1. Load 6-channel PPG, 2-channel EMG, and 3-channel ACC.
2. Stage1 uses the 6-channel mean PPG at 100Hz with 3s DC/ACDC windows.
3. Stage2 converts 6-channel PPG into 3-channel PPG, then extracts
   3-channel PPG, 2-channel EMG, and 3-channel ACC features for XGBoost.
4. Optional postprocess/state-machine evaluation and NPZ window-cache export.
"""

"""
步骤6：部署/端到端推理评估（适配 6-ch PPG avg + 2-ch EMG + 3-ch ACC）

数据流：
1. 读取 ir (6-ch), emg (2-ch), acc (3-ch)
2. Stage1: 6-ch PPG 取平均 → 100Hz → 3s 窗口 DC/ACDC 粗筛 (1s stride)
3. Stage2: 单通道 PPG (6-ch avg) + EMG + ACC 特征提取
4. 后处理（状态机）、指标计算、部署产物导出
"""

import os
import sys
import json
import argparse
import logging
import re
import joblib
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb

# Linux/macOS 默认 fork 模式多进程可能死锁，强制 spawn
if sys.platform != "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s03_extract_feature_pool import (
    load_ppg,
    load_emg,
    load_acc,
    build_3ch_ppg,
    stage1_sample_pass,
    stage1_ppg_mean_signal,
    stage1_window_pass_100hz,
    extract_feature_pool_from_window,
    align_emg_window,
    align_acc_window,
    iter_sample_windows,
    is_windowed_array,
    extract_acc_features,
    extract_acc_cross_features,
    extract_emg_features,
    extract_emg_ppg_cross_features,
    FEATURE_FS,
    DEFAULT_FS_PPG,
    DEFAULT_FS_EMG,
    DEFAULT_FS_ACC,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_POSTPROCESS_CONFIG = {
    "alpha": 0.4,
    "median_k": 1,
    "T_on": 0.75,
    "T_off": 0.35,
    "K_on": 5,
    "K_off": 5,
    "cooldown_sec": 5,
}

_BUNDLE = None


def load_bundle(path):
    global _BUNDLE
    try:
        _BUNDLE = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to load model bundle from {path}: {exc}\n"
            "The file may be missing, corrupted, or from an incompatible version.\n"
            "Re-run s05 and s06 --export_deploy to regenerate it."
        ) from exc
    assert_bundle_ok(_BUNDLE)
    return _BUNDLE


def assert_bundle_ok(bundle):
    needed = ["feature_names", "fill_values", "scaler", "model", "threshold"]
    for k in needed:
        if k not in bundle:
            raise ValueError(f"model_bundle missing key: {k}")
    if "meta" not in bundle:
        logger.warning("model_bundle missing 'meta' key; using defaults for fs_ppg/fs_emg/fs_acc")
    miss = [c for c in bundle["feature_names"] if c not in bundle["fill_values"]]
    if miss:
        raise ValueError(f"fill_values missing for features: {miss[:5]} ...")


def apply_preprocess(feat_dict_list, bundle=None):
    b = bundle if bundle is not None else _BUNDLE
    if b is None:
        raise RuntimeError("must call load_bundle() first")
    feature_names = b["feature_names"]
    fill_values = b["fill_values"]
    clip_bounds = b.get("clip_bounds", {})
    df = pd.DataFrame(feat_dict_list)
    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feature_names]
    for c in feature_names:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(fill_values[c])
    for c, bound in clip_bounds.items():
        if c not in df.columns or not isinstance(bound, (list, tuple)) or len(bound) != 2:
            continue
        lo, hi = float(bound[0]), float(bound[1])
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df.values.astype(np.float64)


def predict_proba_windows(feat_dict_list, bundle=None):
    b = bundle if bundle is not None else _BUNDLE
    X = apply_preprocess(feat_dict_list, bundle=b)
    proba = b["model"].predict_proba(X)[:, 1]
    return proba


def predict_label_windows(feat_dict_list, bundle=None):
    b = bundle if bundle is not None else _BUNDLE
    proba = predict_proba_windows(feat_dict_list, bundle=b)
    thr = b["threshold"]
    return (proba >= thr).astype(int), proba


# =========================================================
# Stage1 → Stage2 门控（3 帧滞回）
# =========================================================

class Stage1Gate:
    """Stage1 门控状态机：连续 3 帧同结果才切换门控开关。

    门关 → 需连续 3 帧 Stage1 通过才开门
    门开 → 需连续 3 帧 Stage1 不通过才关门
    混合信号重置计数器，门保持当前状态不变。

    门开时：使用 Stage2 XGBoost 输出
    门关时：输出 = 0，Stage2 结果被忽略
    """

    def __init__(self):
        self.gate_on = False       # 门控开关
        self._cons_pos = 0         # 连续通过计数
        self._cons_neg = 0         # 连续不通过计数
        self._K = 3                # 连续帧数阈值

    def update(self, stage1_pass):
        """输入 Stage1 帧结果 (bool)，返回当前门控状态。"""
        if stage1_pass:
            self._cons_pos += 1
            self._cons_neg = 0
        else:
            self._cons_neg += 1
            self._cons_pos = 0

        if not self.gate_on and self._cons_pos >= self._K:
            self.gate_on = True
            self._cons_pos = 0
        elif self.gate_on and self._cons_neg >= self._K:
            self.gate_on = False
            self._cons_neg = 0

        return self.gate_on


# =========================================================
# 后处理：状态机
# =========================================================

class WearStateMachine:
    def __init__(self, alpha=0.4, T_on=0.75, T_off=0.35, K_on=5, K_off=5, cooldown_sec=5):
        self.alpha = alpha
        self.T_on = T_on
        self.T_off = T_off
        self.K_on = K_on
        self.K_off = K_off
        self.cooldown_sec = cooldown_sec
        self.state = 0
        self.score = 0.0
        self.on_count = 0
        self.off_count = 0
        self._steps_since_flip = 999

    def update(self, p, quality=1.0, stride_sec=1.0):
        eff_alpha = self.alpha * quality
        self.score = eff_alpha * p + (1 - eff_alpha) * self.score
        self._steps_since_flip += 1

        cooldown_steps = int(self.cooldown_sec / stride_sec) if stride_sec > 0 else self.cooldown_sec

        if self.state == 0:
            if self.score > self.T_on:
                self.on_count += 1
            else:
                self.on_count = max(0, self.on_count - 1)
            if self.on_count >= self.K_on and self._steps_since_flip >= cooldown_steps:
                self.state = 1
                self.on_count = 0
                self.off_count = 0
                self._steps_since_flip = 0
        else:
            if self.score < self.T_off:
                self.off_count += 1
            else:
                self.off_count = max(0, self.off_count - 1)
            if self.off_count >= self.K_off and self._steps_since_flip >= cooldown_steps:
                self.state = 0
                self.on_count = 0
                self.off_count = 0
                self._steps_since_flip = 0

        return self.state, self.score


def _quality_soft(violation_ratio, floor=0.5):
    v = max(0.0, min(1.0, float(violation_ratio)))
    return float(max(floor, 1.0 - (1.0 - floor) * v))


def compute_quality(feat_or_meta, thresholds=None):
    if not hasattr(feat_or_meta, "get"):
        return 1.0
    if "quality" in feat_or_meta:
        try:
            q = float(feat_or_meta.get("quality", 1.0))
        except (TypeError, ValueError):
            q = 1.0
        return float(np.clip(q, 0.0, 1.0))
    if thresholds:
        q = 1.0
        for key, spec in thresholds.items():
            if key.startswith("_"):
                continue
            v = feat_or_meta.get(key, None)
            if v is None or not np.isfinite(v):
                continue
            thr = spec.get("thr", None)
            kind = spec.get("type", "high")
            if thr is None or thr == 0:
                continue
            if kind == "high":
                violation = (v - thr) / abs(thr)
            else:
                violation = (thr - abs(v)) / abs(thr)
            q *= _quality_soft(violation, floor=0.5)
        return float(q)

    q = 1.0
    irm = feat_or_meta.get("PPG_mean", None)
    if irm is not None and np.abs(irm) < 1e-6:
        q *= 0.5
    return float(q)


def compute_ood_score(feat_dict, feature_quantiles, feature_names):
    if not feature_quantiles or not feat_dict:
        return None
    out = 0
    total = 0
    for f in feature_names:
        spec = feature_quantiles.get(f)
        if not spec:
            continue
        v = feat_dict.get(f, None)
        if v is None or not np.isfinite(v):
            continue
        total += 1
        if v < spec["q_low"] or v > spec["q_high"]:
            out += 1
    if total == 0:
        return None
    return float(out) / float(total)


def _causal_median_filter(values, k):
    k = int(k or 1)
    arr = np.asarray(values, dtype=float)
    if k <= 1 or arr.size == 0:
        return arr
    out = np.zeros_like(arr, dtype=float)
    for i in range(arr.size):
        start = max(0, i - k + 1)
        out[i] = float(np.median(arr[start:i + 1]))
    return out


def _apply_stage1_window_gate(probs, stage1_frames):
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0 or stage1_frames is None:
        return arr
    flags = np.asarray(list(stage1_frames), dtype=bool)
    if flags.size < arr.size:
        flags = np.pad(flags, (0, arr.size - flags.size), constant_values=False)
    elif flags.size > arr.size:
        flags = flags[:arr.size]
    return np.where(flags, arr, 0.0)


def compute_stage1_window_metrics(ppg_mean_window, dc_threshold, ac_dc_threshold):
    x = np.asarray(ppg_mean_window, dtype=np.float64)
    if len(x) < 2:
        dc = float(np.mean(x)) if len(x) else 0.0
        ac = 0.0
    else:
        neighbor_mean = (x[:-1] + x[1:]) / 2.0
        dc = float(np.min(neighbor_mean))
        ac = float(np.median(np.abs(np.diff(x))))
    acdc = float(ac / (np.abs(dc) + 1e-12))
    return {
        "dc": dc,
        "ac": ac,
        "acdc": acdc,
        "dc_margin": float(dc - dc_threshold),
        "acdc_margin": float(ac_dc_threshold - acdc),
    }



# =========================================================
# 单样本推理
# =========================================================

def _infer_one_sample(sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec, bundle):
    fs_ppg = bundle["meta"].get("fs_ppg", DEFAULT_FS_PPG)
    fs_emg = bundle["meta"].get("fs_emg", DEFAULT_FS_EMG)
    model_threshold = bundle["threshold"]

    sample_name = sample.get("sample_name", "unknown")
    target = int(sample.get("target", 0))

    base = {
        "sample_name": sample_name, "target": target,
        "stage1_pass": False,
        "window_probs": [], "window_preds": [], "quality_metas": [],
        "window_ood_scores": [],
        "window_start_100hz": [],
        "stage1_frame_results": [],  # per-1s-frame Stage1 结果 (bool)
        "fallback": False, "fallback_reason": None,
    }

    # 1. 加载信号
    try:
        ppg_6ch = load_ppg(sample)
        emg = load_emg(sample)
        acc = load_acc(sample)
    except Exception as e:
        base["fallback"] = True
        base["fallback_reason"] = f"load_error: {e}"
        return base

    # 2. Stage1
    try:
        if not stage1_sample_pass(ppg_6ch, dc_threshold, ac_dc_threshold):
            return base
    except Exception as e:
        base["fallback"] = True
        base["fallback_reason"] = f"stage1_error: {e}"
        return base

    base["stage1_pass"] = True

    # 3. 特征提取与推理
    try:
        # 6-ch PPG → 3 通道 PPG @ 100Hz（不降采样）
        if is_windowed_array(ppg_6ch):
            feats_list = []
            quality_metas = []
            stage1_frames = []
            stage1_window_metrics = []
            window_start_100hz = []
            win_samples = int(window_sec * FEATURE_FS)
            stride_samples = int(stride_sec * FEATURE_FS)
            for win in iter_sample_windows(
                    ppg_6ch, emg, acc,
                    win_samples=win_samples,
                    stride_samples=stride_samples,
                    fs_ppg=FEATURE_FS,
                    fs_emg=fs_emg,
                    fs_acc=FEATURE_FS,
                    window_indices=sample.get("window_indices")):
                ppg_win_6ch = win["ppg_6ch"]
                try:
                    s1_mean = stage1_ppg_mean_signal(ppg_win_6ch)
                    s1_meta = compute_stage1_window_metrics(s1_mean, dc_threshold, ac_dc_threshold)
                    s1_pass = (s1_meta["dc_margin"] > 0) and (s1_meta["acdc_margin"] > 0)
                except Exception:
                    s1_meta = {"dc": 0.0, "ac": 0.0, "acdc": 0.0,
                               "dc_margin": -np.inf, "acdc_margin": -np.inf}
                    s1_pass = False
                try:
                    feat, preprocessed = extract_feature_pool_from_window(
                        ppg_signal=build_3ch_ppg(ppg_win_6ch),
                        emg_window=win["emg"],
                        acc_window=win["acc"],
                        fs_ppg=FEATURE_FS,
                        fs_emg=fs_emg,
                        fs_acc=FEATURE_FS,
                        return_preprocessed=True,
                    )
                    feats_list.append(feat)
                    quality_metas.append({
                        "PPG_mean": feat.get("PPG_mean"),
                        "PPG_std": feat.get("PPG_std"),
                    })
                    window_start_100hz.append(int(win["start_100hz"]))
                    stage1_frames.append(bool(s1_pass))
                    stage1_window_metrics.append(s1_meta)
                except Exception:
                    continue

            if len(feats_list) == 0:
                return base
            window_preds, probs = predict_label_windows(feats_list, bundle=bundle)
            feature_quantiles = bundle.get("feature_quantiles")
            feature_names = bundle.get("feature_names", [])
            if feature_quantiles and feature_names:
                ood_scores = [compute_ood_score(f, feature_quantiles, feature_names) for f in feats_list]
            else:
                ood_scores = [None] * len(feats_list)
            base["window_probs"] = np.asarray(probs, dtype=float).tolist()
            base["window_preds"] = np.asarray(window_preds, dtype=int).tolist()
            base["quality_metas"] = quality_metas
            base["window_ood_scores"] = ood_scores
            base["window_start_100hz"] = window_start_100hz
            base["stage1_frame_results"] = stage1_frames
            base["stage1_window_metrics"] = stage1_window_metrics
            return base

        ppg_3ch = build_3ch_ppg(ppg_6ch).astype(np.float64)  # (N, 3)
        n_ppg = len(ppg_3ch)

        win_samples = int(window_sec * FEATURE_FS)
        stride_samples = int(stride_sec * FEATURE_FS)

        feats_list = []
        quality_metas = []
        stage1_frames = []
        stage1_window_metrics = []
        window_start_100hz = []
        ppg_stage1_mean = stage1_ppg_mean_signal(ppg_6ch)
        stage1_win_samples = int(3 * FEATURE_FS)
        for start in range(0, n_ppg - win_samples + 1, stride_samples):
            ppg_win = ppg_3ch[start:start + win_samples]  # (W, 3)
            emg_win = align_emg_window(emg, n_ppg, start, win_samples,
                                        fs_ppg=FEATURE_FS, fs_emg=fs_emg)
            acc_win = align_acc_window(acc, n_ppg, start, win_samples,
                                        fs_ppg=FEATURE_FS, fs_acc=FEATURE_FS)

            # 对应 1s Stage1 帧的 DC/ACDC 检查（使用 ch_A，与 stage1_sample_pass 一致）
            s1_pass = False
            try:
                s1_window = ppg_stage1_mean[start:start + stage1_win_samples]
                s1_meta = compute_stage1_window_metrics(s1_window, dc_threshold, ac_dc_threshold)
                s1_pass = (s1_meta["dc_margin"] > 0) and (s1_meta["acdc_margin"] > 0)
            except Exception:
                s1_meta = {"dc": 0.0, "ac": 0.0, "acdc": 0.0,
                           "dc_margin": -np.inf, "acdc_margin": -np.inf}
                s1_pass = False
            try:
                feat, preprocessed = extract_feature_pool_from_window(
                    ppg_signal=ppg_win, emg_window=emg_win, acc_window=acc_win,
                    fs_ppg=FEATURE_FS, fs_emg=fs_emg, fs_acc=FEATURE_FS,
                    return_preprocessed=True,
                )

                feats_list.append(feat)
                quality_metas.append({
                    "PPG_mean": feat.get("PPG_mean"),
                    "PPG_std": feat.get("PPG_std"),
                })
                window_start_100hz.append(int(start))
                stage1_frames.append(s1_pass)
                stage1_window_metrics.append(s1_meta)
            except Exception:
                continue

        if len(feats_list) == 0:
            return base

        window_preds, probs = predict_label_windows(feats_list, bundle=bundle)

        feature_quantiles = bundle.get("feature_quantiles")
        feature_names = bundle.get("feature_names", [])
        if feature_quantiles and feature_names:
            ood_scores = [compute_ood_score(f, feature_quantiles, feature_names) for f in feats_list]
        else:
            ood_scores = [None] * len(feats_list)

        base["window_probs"] = probs.tolist()
        base["window_preds"] = window_preds.tolist()
        base["quality_metas"] = quality_metas
        base["window_ood_scores"] = ood_scores
        base["window_start_100hz"] = window_start_100hz
        base["stage1_frame_results"] = stage1_frames
        base["stage1_window_metrics"] = stage1_window_metrics
        return base

    except Exception as e:
        base["fallback"] = True
        base["fallback_reason"] = f"feature_or_predict_error: {e}"
        return base


# =========================================================
# 多进程推理
# =========================================================

_WORKER_BUNDLE = None


def _init_worker(bundle_path):
    global _WORKER_BUNDLE
    # 防止子进程中 numpy/scipy BLAS 多线程竞争
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _WORKER_BUNDLE = joblib.load(bundle_path)
    assert_bundle_ok(_WORKER_BUNDLE)
    try:
        _WORKER_BUNDLE["model"].set_params(n_jobs=1)
    except Exception:
        pass


def _worker_infer(args_tuple):
    sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec = args_tuple
    return _infer_one_sample(sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec, _WORKER_BUNDLE)


def run_inference_parallel(samples, dc_threshold, ac_dc_threshold,
                           window_sec, stride_sec, bundle_path, n_workers):
    n_workers = max(1, int(n_workers))
    args_list = [(s, dc_threshold, ac_dc_threshold, window_sec, stride_sec) for s in samples]

    if n_workers == 1:
        bundle = _BUNDLE if _BUNDLE is not None else joblib.load(bundle_path)
        return [_infer_one_sample(s, dc_threshold, ac_dc_threshold, window_sec, stride_sec, bundle)
                for s in samples]

    results = [None] * len(samples)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(bundle_path,)) as ex:
        futures = {ex.submit(_worker_infer, a): i for i, a in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                logger.warning(f"sample {samples[i].get('sample_name')} inference crashed: {e}")
                results[i] = {
                    "sample_name": samples[i].get("sample_name", "unknown"),
                    "target": int(samples[i].get("target", 0)),
                    "stage1_pass": False, "window_probs": [], "window_preds": [],
                    "quality_metas": [], "stage1_frame_results": [],
                    "fallback": True, "fallback_reason": f"worker_crash: {e}",
                }
    return results


# =========================================================
# 三套指标
# =========================================================

def _summarize_ood(results, alert_rate=0.3):
    per_sample = []
    overall_total = 0
    overall_out = 0
    n_alert_samples = 0
    available = False
    for r in results:
        oods = r.get("window_ood_scores", []) or []
        valid = [v for v in oods if v is not None and np.isfinite(v)]
        if valid:
            available = True
            mean_o = float(np.mean(valid))
            high = float(np.mean([1.0 if v > alert_rate else 0.0 for v in valid]))
            overall_out += sum(valid)
            overall_total += len(valid)
            if mean_o > alert_rate:
                n_alert_samples += 1
        else:
            mean_o, high = None, None
        per_sample.append({
            "sample_name": r.get("sample_name"), "target": int(r.get("target", 0)),
            "ood_mean": mean_o, "ood_window_alert_rate": high,
        })
    return {
        "available": available, "alert_rate_threshold": float(alert_rate),
        "global_mean_ood": float(overall_out / overall_total) if overall_total else None,
        "n_alert_samples": int(n_alert_samples), "per_sample": per_sample,
    }


def _safe_confusion(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return int(tn), int(fp), int(fn), int(tp)
    except Exception:
        return 0, 0, 0, 0


def _legacy_apply_postprocess_unused(window_probs, quality_metas, method, cfg, model_threshold,
                                     stride_sec=1.0, stage1_frames=None):
    """统一后处理：支持 state_machine / gated / mean_vote / prob_mean 四种模式。

    - "state_machine": Stage1 窗级门控 + causal median 滤波 + EMA 滞回状态机（默认）
    - "gated": Stage1Gate 3-帧滞回门控 + 状态机
    - "mean_vote": 窗口级多数投票
    - "prob_mean": 概率均值

    返回:
        final_pred (int), states (list[int]), window_preds (list[int]), scores (list[float])
    """
    probs = np.asarray(window_probs, dtype=float)
    if probs.size == 0:
        return 0, [], [], []

    # gated 模式：Stage1Gate 3-帧滞回门控，再跑状态机
    if method == "gated":
        gate = Stage1Gate()
        gated_probs = []
        for i, p in enumerate(probs):
            s1 = bool(stage1_frames[i]) if stage1_frames and i < len(stage1_frames) else True
            gate_on = gate.update(s1)
            gated_probs.append(float(p) if gate_on else 0.0)
        probs = np.asarray(gated_probs, dtype=float)
        method = "state_machine"

    # state_machine 模式：窗级门控 + median 滤波 + 状态机
    if method == "state_machine":
        if stage1_frames is not None:
            probs = _apply_stage1_window_gate(probs, stage1_frames)
        probs = _causal_median_filter(probs, cfg.get("median_k", 1))

    window_preds = (probs >= model_threshold).astype(int).tolist()

    if method == "state_machine":
        sm = WearStateMachine(
            alpha=cfg.get("alpha", 0.4), T_on=cfg.get("T_on", 0.75),
            T_off=cfg.get("T_off", 0.35), K_on=cfg.get("K_on", 5),
            K_off=cfg.get("K_off", 5), cooldown_sec=cfg.get("cooldown_sec", 5),
        )
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        states, scores = [], []
        for i, p in enumerate(probs):
            meta_i = quality_metas[i] if i < len(quality_metas) else None
            q = compute_quality(meta_i, thresholds=qt) if meta_i else 1.0
            state, score = sm.update(p, quality=q, stride_sec=stride_sec)
            states.append(int(state))
            scores.append(float(score))
        return int(states[-1]), states, window_preds, scores

    if method == "mean_vote":
        return int(np.mean(window_preds) >= 0.5), [], window_preds, []

    # prob_mean / 默认
    return int(np.mean(probs) >= model_threshold), [], window_preds, []


def apply_postprocess(window_probs, quality_metas, method, cfg, model_threshold,
                      stride_sec=1.0, stage1_frames=None):
    """Unified postprocess path. The gated method is a compatibility alias."""
    probs = np.asarray(window_probs, dtype=float)
    if probs.size == 0:
        return 0, [], [], []

    use_threshold_transform = bool(cfg) and "threshold_offset" in cfg
    threshold_offset = float(cfg.get("threshold_offset", 0.0)) if use_threshold_transform else 0.0
    if use_threshold_transform:
        adjusted_threshold = float(np.clip(float(model_threshold) + threshold_offset, 0.02, 0.98))
        probs = np.clip(probs - adjusted_threshold + 0.5, 0.0, 1.0)
        model_threshold = 0.5

    if method == "gated":
        method = "state_machine"

    if stage1_frames is not None:
        probs = _apply_stage1_window_gate(probs, stage1_frames)

    if method == "state_machine":
        probs = _causal_median_filter(probs, cfg.get("median_k", 1))

    window_preds = (probs >= model_threshold).astype(int).tolist()

    if method == "state_machine":
        sm = WearStateMachine(
            alpha=cfg.get("alpha", 0.4), T_on=cfg.get("T_on", 0.75),
            T_off=cfg.get("T_off", 0.35), K_on=cfg.get("K_on", 5),
            K_off=cfg.get("K_off", 5), cooldown_sec=cfg.get("cooldown_sec", 5),
        )
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        states, scores = [], []
        for i, p in enumerate(probs):
            meta_i = quality_metas[i] if i < len(quality_metas) else None
            q = compute_quality(meta_i, thresholds=qt) if meta_i else 1.0
            state, score = sm.update(p, quality=q, stride_sec=stride_sec)
            states.append(int(state))
            scores.append(float(score))
        return int(states[-1]), states, window_preds, scores

    if method == "mean_vote":
        return int(np.mean(window_preds) >= 0.5), [], window_preds, []

    return int(np.mean(probs) >= model_threshold), [], window_preds, []


def serialize_postprocess_config(postprocess_cfg):
    return OrderedDict([
        ("pipeline_step", 4),
        ("description", "Temporal postprocess for window probabilities"),
        ("state_machine", OrderedDict([
            ("algorithm", "Stage1 gate + causal median + EMA + hysteresis"),
            ("state", "0=not_worn, 1=worn"),
            ("parameters", OrderedDict([
                ("alpha", float(postprocess_cfg.get("alpha", 0.4))),
                ("median_k", int(postprocess_cfg.get("median_k", 1))),
                ("T_on", float(postprocess_cfg.get("T_on", 0.75))),
                ("T_off", float(postprocess_cfg.get("T_off", 0.35))),
                ("K_on", int(postprocess_cfg.get("K_on", 5))),
                ("K_off", int(postprocess_cfg.get("K_off", 5))),
                ("cooldown_sec", float(postprocess_cfg.get("cooldown_sec", 5))),
                ("threshold_offset", float(postprocess_cfg.get("threshold_offset", 0.0))),
                ("threshold_transform", postprocess_cfg.get(
                    "threshold_transform",
                    "disabled" if "threshold_offset" not in postprocess_cfg
                    else "clip(prob_raw - (model_threshold + threshold_offset) + 0.5, 0, 1)",
                )),
            ])),
        ])),
    ])


def describe_postprocess_config_source(config_path, saved_cfg):
    saved_cfg = saved_cfg or {}
    if "postprocess_optimization" in saved_cfg or "postprocess_cache_optimization" in saved_cfg:
        return f"  (已从 {config_path} 加载优化后的后处理参数)"
    return f"  (已从 {config_path} 加载保存的后处理参数)"


def compute_sample_metrics(results, method, cfg, model_threshold, stride_sec=1.0):
    y_true, y_pred = [], []
    fallback_count = 0
    stage1_pass_count = 0
    details = []

    for r in results:
        target = int(r["target"])
        if r.get("fallback", False):
            fallback_count += 1
        if r.get("stage1_pass", False):
            stage1_pass_count += 1

        probs = r.get("window_probs", [])
        if r.get("fallback", False) or not r.get("stage1_pass", False) or len(probs) == 0:
            final_pred = 0
            states = []
            scores = []
            window_preds = list(r.get("window_preds", []))
        else:
            final_pred, states, window_preds, scores = apply_postprocess(
                probs, r.get("quality_metas", []), method, cfg, model_threshold,
                stride_sec=stride_sec, stage1_frames=r.get("stage1_frame_results"))

        y_true.append(target)
        y_pred.append(int(final_pred))

        fallback = r.get("fallback", False)
        details.append({
            "sample_name": r.get("sample_name"), "target": target, "pred": int(final_pred),
            "stage1_pass": bool(r.get("stage1_pass", False)),
            "fallback": bool(fallback),
            "fallback_reason": r.get("fallback_reason"),
            "window_probs": probs, "window_preds": list(window_preds),
            "window_states": list(states),
            "window_scores": list(scores) if not fallback else [],
            "n_windows": len(probs),
        })

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    tn, fp, fn, tp = _safe_confusion(y_true_a, y_pred_a)

    summary = {
        "method": method, "total_samples": int(len(results)),
        "stage1_pass_samples": int(stage1_pass_count),
        "fallback_samples": int(fallback_count),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)) if len(y_true_a) > 0 else 0.0,
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        "postprocess": cfg,
    }
    return summary, details


def compute_window_model_metrics(results):
    y_true, y_pred = [], []
    samples_with_no_windows = 0
    total_input_samples = len(results)
    stage1_pass_samples = 0
    for r in results:
        wp = r.get("window_preds", [])
        if r.get("fallback", False) or not r.get("stage1_pass", False) or len(wp) == 0:
            samples_with_no_windows += 1
            continue
        stage1_pass_samples += 1
        t = int(r["target"])
        for p in wp:
            y_true.append(t)
            y_pred.append(int(p))

    if len(y_true) == 0:
        return {
            "total_input_samples": total_input_samples,
            "stage1_pass_samples": stage1_pass_samples,
            "samples_with_no_windows": samples_with_no_windows,
            "total_windows": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    tn, fp, fn, tp = _safe_confusion(y_true_a, y_pred_a)
    return {
        "total_input_samples": total_input_samples,
        "stage1_pass_samples": stage1_pass_samples,
        "samples_with_no_windows": samples_with_no_windows,
        "total_windows": int(len(y_true_a)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
    }


def compute_window_stream_metrics(results, cfg, warmup_frames=0, stride_sec=1.0, model_threshold=0.5):
    y_true, y_pred = [], []
    samples_with_no_windows = 0
    skipped_windows = 0
    for r in results:
        probs = r.get("window_probs", [])
        qm = r.get("quality_metas", [])
        if r.get("fallback", False) or not r.get("stage1_pass", False) or len(probs) == 0:
            samples_with_no_windows += 1
            continue

        t = int(r["target"])
        _final_pred, sample_states, _window_preds, _scores = apply_postprocess(
            probs, qm, "state_machine", cfg, model_threshold=model_threshold,
            stride_sec=stride_sec, stage1_frames=r.get("stage1_frame_results"))
        start = min(warmup_frames, len(sample_states))
        skipped_windows += start
        for s in sample_states[start:]:
            y_true.append(t)
            y_pred.append(s)

    if len(y_true) == 0:
        return {
            "total_samples": len(results),
            "samples_with_no_windows": samples_with_no_windows,
            "warmup_frames": int(warmup_frames),
            "skipped_warmup_windows": int(skipped_windows),
            "total_windows": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    tn, fp, fn, tp = _safe_confusion(y_true_a, y_pred_a)
    return {
        "total_samples": len(results),
        "samples_with_no_windows": samples_with_no_windows,
        "warmup_frames": int(warmup_frames),
        "skipped_warmup_windows": int(skipped_windows),
        "total_windows": int(len(y_true_a)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
    }


# =========================================================
# 状态机网格搜索
# =========================================================

_GRID_SCORE_DATA = None  # 子进程通过 initializer 注入，避免每个 task 反复反序列化


def _safe_cache_filename(sample_name):
    name = str(sample_name or "unknown")
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._")
    return (name or "unknown") + ".npz"


def _pad_or_trim_array(values, n, fill_value, dtype):
    arr = np.asarray(values, dtype=dtype)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=fill_value)
    elif arr.size > n:
        arr = arr[:n]
    return arr.astype(dtype)


def _quality_values_from_metas(quality_metas, n):
    vals = []
    for meta in list(quality_metas or [])[:n]:
        if isinstance(meta, dict) and "quality" in meta:
            v = meta.get("quality", 1.0)
        else:
            v = 1.0
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 1.0
        vals.append(fv if np.isfinite(fv) else 1.0)
    return _pad_or_trim_array(vals, n, 1.0, np.float64)


def _stage1_metric_values(stage1_metrics, n, key, fill_value=np.nan):
    vals = []
    for meta in list(stage1_metrics or [])[:n]:
        if isinstance(meta, dict):
            vals.append(meta.get(key, fill_value))
        else:
            vals.append(fill_value)
    return _pad_or_trim_array(vals, n, fill_value, np.float64)


def write_window_cache_npz(result, out_dir, window_sec, stride_sec, model_threshold, metadata=None):
    """Write one sample's window-level inference output to an NPZ cache file."""
    metadata = dict(metadata or {})
    os.makedirs(out_dir, exist_ok=True)

    probs = np.asarray(result.get("window_probs", []), dtype=np.float64)
    n = int(probs.size)
    if result.get("window_start_100hz"):
        window_start_sec = _pad_or_trim_array(
            np.asarray(result.get("window_start_100hz"), dtype=np.float64) / float(FEATURE_FS),
            n,
            0.0,
            np.float64,
        )
    else:
        window_start_sec = np.arange(n, dtype=np.float64) * float(stride_sec)
    stage1_enabled = _pad_or_trim_array(
        result.get("stage1_frame_results", []), n, False, np.int8
    )
    if n > 0 and not result.get("stage1_frame_results"):
        stage1_enabled[:] = 1 if result.get("stage1_pass", False) else 0

    path = os.path.join(out_dir, _safe_cache_filename(result.get("sample_name", "unknown")))
    np.savez_compressed(
        path,
        cache_schema_version=np.array(1, dtype=np.int32),
        sample_name=np.array(str(result.get("sample_name", "unknown"))),
        target=np.array(int(result.get("target", 0)), dtype=np.int8),
        stage1_pass=np.array(bool(result.get("stage1_pass", False))),
        fallback=np.array(bool(result.get("fallback", False))),
        fallback_reason=np.array(str(result.get("fallback_reason") or "")),
        window_start_sec=window_start_sec,
        window_end_sec=window_start_sec + float(window_sec),
        prob_raw=probs,
        pred_raw=_pad_or_trim_array(result.get("window_preds", []), n, 0, np.int8),
        stage1_enabled=stage1_enabled,
        quality=_quality_values_from_metas(result.get("quality_metas", []), n),
        ood_rate=_pad_or_trim_array(result.get("window_ood_scores", []), n, 0.0, np.float64),
        stage1_dc=_stage1_metric_values(result.get("stage1_window_metrics", []), n, "dc"),
        stage1_acdc=_stage1_metric_values(result.get("stage1_window_metrics", []), n, "acdc"),
        stage1_dc_margin=_stage1_metric_values(result.get("stage1_window_metrics", []), n, "dc_margin"),
        stage1_acdc_margin=_stage1_metric_values(result.get("stage1_window_metrics", []), n, "acdc_margin"),
        model_threshold=np.array(float(model_threshold), dtype=np.float64),
        window_sec=np.array(float(window_sec), dtype=np.float64),
        stride_sec=np.array(float(stride_sec), dtype=np.float64),
        model_fingerprint_json=np.array(str(metadata.get("model_fingerprint_json", "{}"))),
        feature_names_json=np.array(str(metadata.get("feature_names_json", "[]"))),
    )
    return path


def export_window_cache(results, artifact_dir, split_name, window_sec, stride_sec,
                        model_threshold, metadata=None, cache_root="window_outputs"):
    out_dir = os.path.join(artifact_dir, cache_root, split_name)
    rows = []
    for r in results:
        path = write_window_cache_npz(
            r, out_dir, window_sec, stride_sec, model_threshold, metadata=metadata
        )
        stage1_enabled = r.get("stage1_frame_results", [])
        rows.append({
            "sample_name": r.get("sample_name", "unknown"),
            "target": int(r.get("target", 0)),
            "stage1_pass": bool(r.get("stage1_pass", False)),
            "fallback": bool(r.get("fallback", False)),
            "n_windows": int(len(r.get("window_probs", []))),
            "n_stage1_enabled": int(np.sum(stage1_enabled)) if stage1_enabled else 0,
            "path": path,
        })
    manifest = os.path.join(out_dir, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8-sig")
    return {"out_dir": out_dir, "manifest": manifest, "n_samples": len(rows)}


def _init_grid_worker(cache_pickle):
    global _GRID_SCORE_DATA
    import pickle
    _GRID_SCORE_DATA = pickle.loads(cache_pickle)


def _score_grid_point(args_tuple):
    alpha, T_on, T_off, K_on, K_off, cooldown_sec, stride_sec = args_tuple
    data = _GRID_SCORE_DATA
    cache = data["samples"]
    quality_thresholds = data.get("quality_thresholds")

    y_true, y_pred = [], []
    for s in cache:
        target = s["target"]
        probs = s.get("probs", [])
        qm = s.get("quality_metas", [])
        if not s.get("stage1_pass", True) or len(probs) == 0:
            pred = 0
        else:
            sm = WearStateMachine(alpha=alpha, T_on=T_on, T_off=T_off,
                                  K_on=K_on, K_off=K_off, cooldown_sec=cooldown_sec)
            state = 0
            for i, p in enumerate(probs):
                q = compute_quality(qm[i], thresholds=quality_thresholds) if i < len(qm) and qm[i] else 1.0
                state, _ = sm.update(p, quality=q, stride_sec=stride_sec)
            pred = int(state)
        y_true.append(target)
        y_pred.append(pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    params = {"alpha": alpha, "T_on": T_on, "T_off": T_off,
              "K_on": K_on, "K_off": K_off, "cooldown_sec": cooldown_sec}
    return params, {"recall": float(rec), "precision": float(prec), "f1": float(f1)}


def optimize_state_machine_params(samples, dc_threshold, ac_dc_threshold,
                                   window_sec=3, stride_sec=1, min_recall=0.95,
                                   bundle_path=None, n_workers=None):
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    T_ons = [0.5, 0.6, 0.7, 0.8]
    T_offs = [0.2, 0.3, 0.4]
    K_ons = [3, 5, 7, 10, 15]
    K_offs = [3, 5, 7, 10]
    cooldowns = [2, 5, 10]

    print("状态机参数优化配置:")
    print(f"  搜索空间 alpha={alphas} T_on={T_ons} T_off={T_offs} "
          f"K_on={K_ons} K_off={K_offs} cooldown_sec={cooldowns}")
    print(f"  目标: recall>={min_recall*100}% 前提下最大化 F1")

    if bundle_path is None:
        raise ValueError("optimize_state_machine_params 需要 bundle_path")
    if n_workers is None:
        n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))

    print("预计算样本窗口概率（并行）...")
    results = run_inference_parallel(
        samples, dc_threshold, ac_dc_threshold,
        window_sec, stride_sec, bundle_path, n_workers)

    sample_cache = []
    for r in results:
        sample_cache.append({
            "sample_name": r["sample_name"], "target": int(r["target"]),
            "probs": r.get("window_probs", []),
            "quality_metas": r.get("quality_metas", []),
            "stage1_pass": bool(r.get("stage1_pass", False)) and not r.get("fallback", False),
        })
    print(f"有效样本: {len(sample_cache)}")

    grid = [(a, t_on, t_off, ko, kf, cd)
            for a, t_on, t_off, ko, kf, cd in product(
                alphas, T_ons, T_offs, K_ons, K_offs, cooldowns)
            if t_off < t_on and ko >= kf]
    print(f"待评分网格点: {len(grid)}")

    import pickle
    _bundle = joblib.load(bundle_path) if _BUNDLE is None else _BUNDLE
    score_data = {"samples": sample_cache, "quality_thresholds": _bundle.get("quality_thresholds")}
    cache_pickle = pickle.dumps(score_data, protocol=pickle.HIGHEST_PROTOCOL)
    task_args = [(a, t_on, t_off, ko, kf, cd, stride_sec) for (a, t_on, t_off, ko, kf, cd) in grid]

    scored = []
    if n_workers == 1 or len(grid) <= 4:
        _init_grid_worker(cache_pickle)
        for ta in task_args:
            scored.append(_score_grid_point(ta))
    else:
        chunksize = max(1, len(grid) // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers,
                                  initializer=_init_grid_worker,
                                  initargs=(cache_pickle,)) as ex:
            for res in ex.map(_score_grid_point, task_args, chunksize=chunksize):
                scored.append(res)

    best_params, best_metrics, best_f1 = None, None, -1.0
    for params, metrics in scored:
        if metrics["recall"] >= min_recall and metrics["f1"] > best_f1:
            best_params, best_metrics, best_f1 = params, metrics, metrics["f1"]

    if best_params is None:
        print(f"警告: 未找到 recall>={min_recall} 的组合，放宽约束")
        for params, metrics in scored:
            if best_metrics is None or metrics["recall"] > best_metrics["recall"] or \
               (metrics["recall"] == best_metrics["recall"] and metrics["f1"] > best_f1):
                best_params, best_metrics, best_f1 = params, metrics, metrics["f1"]

    print(f"最优参数: {best_params}")
    print(f"评估指标: {best_metrics}")
    return {"best_params": best_params, "best_metrics": best_metrics}


# =========================================================
# 向后兼容 API
# =========================================================

def get_deploy_stage1_threshold(th):
    if "deploy_stage1_threshold" in th:
        return {
            "dc_threshold": float(th["deploy_stage1_threshold"]["dc_threshold"]),
            "ac_dc_threshold": float(th["deploy_stage1_threshold"]["ac_dc_threshold"]),
        }
    return {
        "dc_threshold": float(th["dc_threshold"]),
        "ac_dc_threshold": float(th["ac_dc_threshold"]),
    }


def predict_sample_with_bundle(sample, dc_threshold, ac_dc_threshold,
                                window_sec=3, stride_sec=1,
                                method="state_machine", postprocess_cfg=None):
    if _BUNDLE is None:
        raise RuntimeError("must call load_bundle() first")
    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)

    r = _infer_one_sample(sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec, _BUNDLE)
    target = int(r["target"])
    if not r["stage1_pass"] or len(r["window_probs"]) == 0 or r["fallback"]:
        return {
            "sample_name": r["sample_name"], "target": target, "pred": 0,
            "stage1_pass": bool(r["stage1_pass"]),
            "window_probs": [], "window_preds": [],
        }

    final_pred, _states, window_preds, _scores = apply_postprocess(
        r["window_probs"], r["quality_metas"], method, postprocess_cfg, _BUNDLE["threshold"],
        stride_sec=stride_sec, stage1_frames=r.get("stage1_frame_results"))
    return {
        "sample_name": r["sample_name"], "target": target, "pred": int(final_pred),
        "stage1_pass": True, "window_probs": list(r["window_probs"]),
        "window_preds": list(window_preds),
    }


# =========================================================
# 部署产物导出
# =========================================================

def build_feature_formula_map(selected_features):
    """
    为每个入选特征生成计算公式描述。
    按特征命名约定自动匹配公式模板（适配 IR+EMG+ACC）。
    """
    from collections import OrderedDict

    # ---- PPG 基础 + DC/AC ----
    PPG_BASIC = OrderedDict([
        ("PPG_mean",      "float(np.mean(ppg_mean_raw))"),
        ("PPG_std",       "float(np.std(ppg_mean_raw))"),
        ("PPG_p95",       "float(np.percentile(ppg_mean_raw, 95))"),
        ("PPG_diff_std",  "float(np.std(np.diff(ppg_mean_raw)))"),
        ("PPG_acdc",      "sqrt(mean(ppg_mean_bp²)) / |ppg_mean_dc|"),
        # PPG DC/AC 系列（来自 extract_ppg_features）
        ("PPG_DC_MEDIAN",   "median(ppg_mean_raw)"),
        ("PPG_DC_IQR",      "robust_iqr(ppg_mean_raw)"),
        ("PPG_AC_RMS",      "sqrt(mean(ppg_mean_bp²))"),
        ("PPG_AC_MAD",      "robust_mad(ppg_mean_bp)"),
        ("PPG_AC_DC_RATIO", "safe_div(sqrt(mean(ppg_mean_bp²)), |ppg_mean_dc|)"),
        ("PPG_DERIV_MAD",   "robust_mad(diff(ppg_mean_bp))"),
        # FFT/Autocorr 峰特征（fs_ppg=100）
        ("PPG_FFT_PEAK_MEDIAN_RATIO", "max(|FFT(ppg_mean_bp, fs=100)|[0.5-5Hz]) / median(|FFT|[0.5-5Hz])"),
        ("PPG_DOM_FREQ",              "argmax_freq(|FFT(ppg_mean_bp, fs=100)|[0.5-5Hz])"),
        ("PPG_AUTO_CORR_PEAK",        "max of normalized_autocorr(ppg_mean_bp) in lag[fs*60/180 : fs*60/40]"),
        ("PPG_AUTO_CORR_LAG_SEC",     "(argmax_lag in the same range) / fs"),
    ])

    # ---- PPG Hjorth/Entropy/Derivative/Temporal/Waveform ----
    PPG_COMPLEXITY = OrderedDict([
        ("PPG_Hjorth_Activity",   "float(np.var(ppg_mean_bp))"),
        ("PPG_Hjorth_Mobility",   "sqrt(var(diff(bp)) / var(bp))"),
        ("PPG_Hjorth_Complexity", "sqrt(var(diff2) / var(diff1))"),
        ("PPG_Entropy_Shannon",   "Shannon entropy on 10-bin histogram of bp"),
        ("PPG_Entropy_SampEn",    "Sample Entropy of bp (m=2, r=0.2*std)"),
        ("PPG_Deriv_d1_mean",     "float(np.mean(np.diff(bp)))"),
        ("PPG_Deriv_d1_std",      "float(np.std(np.diff(bp)))"),
        ("PPG_Deriv_d1_max",      "float(np.max(np.diff(bp)))"),
        ("PPG_Deriv_d1_min",      "float(np.min(np.diff(bp)))"),
        ("PPG_Deriv_d1_zcr",      "zero-crossing rate of diff(bp)"),
        ("PPG_Deriv_d2_mean",     "float(np.mean(np.diff(np.diff(bp))))"),
        ("PPG_Deriv_d2_std",      "float(np.std(np.diff(np.diff(bp))))"),
        ("PPG_Deriv_d2_max",      "float(np.max(np.diff(np.diff(bp))))"),
        ("PPG_Deriv_d2_min",      "float(np.min(np.diff(np.diff(bp))))"),
        ("PPG_Deriv_d2_zcr",      "zero-crossing rate of 2nd-order diff"),
        ("PPG_Temporal_slope_mean","linear regression slope of bp vs time"),
        ("PPG_Temporal_slope_std", "std(residuals after linear detrend)"),
        ("PPG_Temporal_peak_prominence","mean peak prominence from find_peaks(bp)"),
        ("PPG_Temporal_peak_ratio",     "len(peaks) / len(bp)"),
        ("PPG_Temporal_valley_ratio",   "len(valleys) / len(bp)"),
        ("PPG_bp_skewness",       "float(mean((bp-mean(bp))³) / std(bp)³)"),
        ("PPG_bp_kurtosis",       "float(mean((bp-mean(bp))⁴) / std(bp)⁴)"),
        ("PPG_FFT_peak_width_Hz", "FFT peak width at half-maximum (Hz)"),
        ("PPG_FFT_SNR",           "in-band power / out-of-band power"),
    ])

    # ---- EMG features ----
    # 注: emg_bp_clean 指 bp_clean（统一 notch 后，用于常规时频特征 MNF/MDF/PKF 等）
    #     emg_bp_leak_ref 指 bp_leak_ref（20-450Hz 带通后 notch 前，含窄带能量，用于 mains + leak 特征）
    #     emg_demean 指仅去均值未做带通的原始信号，用于 baseline drift
    EMG_TEMPLATES = OrderedDict([
        ("{ch}_MAV",   "mean(|emg_env|)"),
        ("{ch}_RMS",   "sqrt(mean(emg_bp²))"),
        ("{ch}_VAR",   "var(emg_bp)"),
        ("{ch}_WL",    "sum(|diff(emg_bp)|)"),
        ("{ch}_ZC",    "zero-crossing rate of emg_bp"),
        ("{ch}_SSC",   "slope sign change rate"),
        ("{ch}_WAMP",  "fraction of |diff| > 0.05*max"),
        ("{ch}_IEMG",  "sum(|emg_env|)"),
        ("{ch}_P2P",   "percentile(env,95) - percentile(env,5) — robust peak-to-peak"),
        ("{ch}_AMP_CV","std(env) / mean(env) — envelope variability"),
        ("{ch}_MNF",   "mean frequency in 20-450Hz band"),
        ("{ch}_MDF",   "median frequency in 20-450Hz band"),
        ("{ch}_PKF",   "peak frequency in 20-450Hz band"),
        ("{ch}_PSR",   "power ratio (20-100Hz)/(100-450Hz)"),
        ("{ch}_POW_20_60",   "power fraction 20-60Hz / total 20-450Hz"),
        ("{ch}_POW_60_150",  "power fraction 60-150Hz / total 20-450Hz"),
        ("{ch}_POW_150_450", "power fraction 150-450Hz / total 20-450Hz"),
        ("{ch}_POW_LH_RATIO","low/high power ratio (20-60Hz)/(150-450Hz)"),
        ("{ch}_SE95",  "frequency at 95% cumulative power — spectral concentration"),
        ("{ch}_SampEn","Sample Entropy (m=2, r=0.2*std)"),
        ("{ch}_SKEWNESS","skewness of emg_bp amplitude distribution"),
        ("{ch}_KURTOSIS","kurtosis of emg_bp amplitude distribution"),
        ("{ch}_SNR",   "RMS / (MAV + 1e-12) — signal-to-noise proxy"),
        # 50Hz 工频拾取（在 emg_bp_leak_ref 上算，20-450Hz 带通后 notch 前）
        ("{ch}_PWR_50HZ",         "log1p(power(48-52Hz) of emg_bp_leak_ref)"),
        ("{ch}_50HZ_RATIO",       "power(48-52Hz) / power(2-450Hz) on emg_bp_leak_ref"),
        # 谐波 ratio 包含 50/150/250Hz（150/250Hz 可能含 PPG 串扰，由 LEAK_* 特征分离）
        ("{ch}_50HZ_HARM_RATIO",  "(P(48-52)+P(148-152)+P(248-252)) / P(2-450) on emg_bp_leak_ref"),
        # Baseline drift（在 emg_demean 上算，1-10Hz 已被 emg_bp 砍掉）
        ("{ch}_BASELINE_DRIFT_POW","log1p(mean(bandpass(emg_demean, 1-10Hz, 2nd_order, fs=1000)²))"),
        ("{ch}_DRIFT_HF_RATIO",    "mean(lf²) / mean(emg_bp_leak_ref²) — 1-10Hz / 20-450Hz 能量比"),
        # PPG 窄带串扰显式特征（在 emg_bp_leak_ref 上算，notch 前）
        ("{ch}_LEAK_100_RATIO", "power(99.2-100.8Hz) / power(20-450Hz) on emg_bp_leak_ref"),
        ("{ch}_LEAK_150_RATIO", "power(149.2-150.8Hz) / power(20-450Hz) on emg_bp_leak_ref"),
        ("{ch}_LEAK_200_RATIO", "power(199.2-200.8Hz) / power(20-450Hz) on emg_bp_leak_ref"),
        ("{ch}_LEAK_250_RATIO", "power(249.2-250.8Hz) / power(20-450Hz) on emg_bp_leak_ref"),
        ("{ch}_LEAK_300_RATIO", "power(299.2-300.8Hz) / power(20-450Hz) on emg_bp_leak_ref"),
        ("{ch}_LEAK_SUM_RATIO","sum of above 5 LEAK_*_RATIO frequency ratios"),
        ("{ch}_LEAK_MAX_RATIO","max of above 5 LEAK_*_RATIO frequency ratios"),
        ("{ch}_LEAK_MAX_FREQ", "frequency (Hz) of the max LEAK_*_RATIO among 100/150/200/250/300"),
    ])

    # ---- ACC features ----
    ACC_TEMPLATES = OrderedDict([
        ("ACC_GRAV_MAG_MEAN",     "mean(grav_mag) — 低通<0.5Hz分离重力"),
        ("ACC_GRAV_DOM_RATIO",    "max(|mean(acc_grav)|) / sum(|mean|) — 重力轴主导度"),
        ("ACC_MOTION_RMS",        "sqrt(mean(motion_mag²)) — 运动分量 RMS"),
        ("ACC_MOTION_STD",        "std(motion_mag)"),
        ("ACC_MOTION_MAD",        "robust_mad(motion_mag)"),
        ("ACC_AXIS_STD_SUM",     "sum(std(acc, axis=0))"),
        ("ACC_DIFF_MAD",         "robust_mad(diff(motion_mag))"),
        ("ACC_STILL_SCORE",      "1/(1+50*mag_std/|mag_mean|)"),
        ("ACC_MAG_P50",          "float(np.percentile(acc_mag, 50))"),
        ("ACC_MAG_P90",          "float(np.percentile(acc_mag, 90))"),
        ("ACC_SAT_FRAC",         "mean(abs(acc_axis) >= 0.98 * max(abs(acc_axis))) across all ACC samples/axes"),
        ("ACC_CLIP_RATE",        "fraction of near-zero adjacent differences in ACC axes, abs(diff(acc,axis=0)) < 1e-10"),
        # 8-12Hz 生理震颤
        ("ACC_TREMOR_POW_8_12",  "log1p(power_8_12Hz(acc_mag-mean, fs=100))"),
        ("ACC_TREMOR_RATIO",     "power_8_12Hz / power_0.5-15Hz(acc_mag-mean)"),
    ])

    # ---- Cross-modal ----
    CROSS_MODAL = OrderedDict([
        ("ACC_PPG_BP_CORR",  "|safe_corr(bandpass(acc_mag-mean, 0.5-5Hz, fs=100), ppg_bp)|"),
        ("ACC_EMG_CORR",    "|safe_corr(acc_mag, resample(emg0_env, 1000→100Hz))|"),
        ("EMG_PPG_CORR",     "|safe_corr(resample(emg0_env, 1000→100Hz), ppg_bp)|"),
        ("EMG_PPG_ENV_CORR", "|safe_corr(smooth_env(resample(emg0_env,100Hz)), smooth_env(ppg_bp))|"),
        # ACC-PPG 频域 coherence (Welch)
        ("ACC_PPG_COH_MICRO", "mean(|Cxy(f)|² for f∈[0.5,3]Hz) of welch_coherence(acc_mag-mean, ppg_bp, fs=100, nperseg=min(N,2*fs))"),
        ("ACC_PPG_COH_HR",    "mean(|Cxy(f)|² for f∈[0.8,3]Hz) of welch_coherence(acc_mag-mean, ppg_bp, fs=100, nperseg=min(N,2*fs))"),
    ])

    # ---- 新增: PPG Perfusion / Morphology / HRV ----
    PPG_ANTI_SPOOF = OrderedDict([
        # Perfusion Index
        ("PPG_PI",              "sqrt(mean(ppg_bp²)) / |median(ppg_raw)|  — AC/DC 灌注指标"),
        ("PPG_PI_SUBWIN_IQR",   "robust_iqr([PI(sub_i) for sub_i in 1s non-overlapping subwindows of (ppg_raw, ppg_bp)])"),
        # 脉搏波形态学（基于 _detect_ppg_peaks(ppg_bp, fs=100)）
        ("PPG_DICROTIC_RATIO",  "在相邻峰区间 [p_i, p_{i+1}] 的 25%-75% 段内出现次级峰(>mean+0.2*std)的比例"),
        ("PPG_AUG_INDEX_MEAN",  "mean(secondary_peak/main_peak) 对存在 dicrotic 的拍"),
        ("PPG_PULSE_WIDTH_CV",  "std(pulse_widths_sec) / mean(pulse_widths_sec)"),
        # 短窗 HRV (3s 窗 3-5 拍，作为防伪占位)
        ("PPG_RR_RMSSD",        "sqrt(mean(diff(RR)²)) — RR 间隔差的均方根 (秒)"),
        ("PPG_RR_CV",           "std(RR) / mean(RR) — 归一化 RR 离散度"),
        ("PPG_RR_PNN30",        "mean(|diff(RR)| > 0.030) — 相邻 RR 差超 30ms 的比例"),
    ])

    PPG_SPATIAL = OrderedDict([
        ("PPG_ch_imbalance_mean", "mean(std(ppg_3ch, axis=channel) / (abs(mean(ppg_3ch, axis=channel)) + eps))"),
        ("PPG_ch_imbalance_p90", "percentile(channel_imbalance, 90)"),
        ("PPG_ch_imbalance_iqr", "robust_iqr(channel_imbalance)"),
        ("PPG_ch_rangeNorm_mean", "mean((max(ppg_3ch)-min(ppg_3ch)) / (abs(mean(ppg_3ch)) + eps) per sample)"),
        ("PPG_ch_rangeNorm_p90", "percentile(channel_range_norm, 90)"),
        ("PPG_ch_vmag_mean", "mean(sqrt(vx^2+vy^2)/(abs(channel_mean)+eps)), vx=ch0-0.5*(ch1+ch2), vy=sqrt(3)/2*(ch1-ch2)"),
        ("PPG_ch_vmag_p90", "percentile(channel_vector_magnitude, 90)"),
        ("PPG_ch_vmag_iqr", "robust_iqr(channel_vector_magnitude)"),
        ("PPG_ch_vmag_std", "std(channel_vector_magnitude)"),
        ("PPG_ch_dc_cv", "std(median(ppg_ch_i)) / (abs(mean(median(ppg_ch_i))) + eps)"),
        ("PPG_ch_dc_max_min_ratio", "max(abs(median(ppg_ch_i))) / (min(abs(median(ppg_ch_i))) + eps)"),
        ("PPG_ch_bp_corr_mean", "mean(pairwise safe_corr(ppg_bp_channel_i, ppg_bp_channel_j))"),
        ("PPG_ch_bp_corr_min", "min(pairwise safe_corr(ppg_bp_channel_i, ppg_bp_channel_j))"),
        ("PPG_ch_bp_corr_std", "std(pairwise safe_corr(ppg_bp_channel_i, ppg_bp_channel_j))"),
        ("PPG_ch_bp_lag_std", "std(best_lag_samples(pairwise cross-correlation of bandpassed PPG channels))"),
        ("PPG_corr_mean_imbalance", "safe_corr(ppg_mean_raw, channel_imbalance)"),
        ("PPG_corr_mean_vmag", "safe_corr(ppg_mean_raw, channel_vector_magnitude)"),
        ("PPG_corr_IR_imbalance", "safe_corr(ir_raw, channel_imbalance)"),
    ])

    # ---- Meta ----
    META = OrderedDict([
        ("SIG_LEN", "float(len(ppg_mean_raw)) — 窗口采样点数 (3s @ 100Hz = 300)"),
        ("SIG_SEC", "float(len(ppg_mean_raw) / fs_ppg) — 窗口秒数"),
    ])

    # 合并所有模板
    ALL_TEMPLATES = OrderedDict()
    ALL_TEMPLATES.update(PPG_BASIC)
    ALL_TEMPLATES.update(PPG_COMPLEXITY)
    ALL_TEMPLATES.update(PPG_ANTI_SPOOF)
    ALL_TEMPLATES.update(PPG_SPATIAL)
    ALL_TEMPLATES.update(ACC_TEMPLATES)
    ALL_TEMPLATES.update(CROSS_MODAL)
    ALL_TEMPLATES.update(META)

    # 补充 EMG (EMG0_/EMG1_)
    for ch in ["EMG0", "EMG1"]:
        for tmpl, formula in EMG_TEMPLATES.items():
            fname = tmpl.format(ch=ch)
            if fname not in ALL_TEMPLATES:
                ALL_TEMPLATES[fname] = formula

    # 补充 EMG 跨通道
    ALL_TEMPLATES["EMG_CROSS_CORR"] = "safe_corr(emg_ch0_bp, emg_ch1_bp)"
    ALL_TEMPLATES["EMG_RMS_RATIO"] = "EMG0_RMS / EMG1_RMS"
    emg_consensus_source = {
        "RMS": "sqrt(mean(emg_bp^2))",
        "MAV": "mean(abs(emg envelope))",
        "WL": "sum(abs(diff(emg_bp)))",
        "ZC": "zero-crossing rate of emg_bp",
        "MNF": "mean frequency in 20-450Hz band",
        "MDF": "median frequency in 20-450Hz band",
        "PKF": "peak frequency in 20-450Hz band",
        "PSR": "power ratio (20-100Hz)/(100-450Hz)",
    }
    for base, desc in emg_consensus_source.items():
        pair = f"[EMG0_{base}, EMG1_{base}] ({desc})"
        ALL_TEMPLATES[f"EMG_consensus_{base}_min"] = f"min({pair})"
        ALL_TEMPLATES[f"EMG_consensus_{base}_max"] = f"max({pair})"
        ALL_TEMPLATES[f"EMG_consensus_{base}_range"] = f"max({pair}) - min({pair})"
        ALL_TEMPLATES[f"EMG_consensus_{base}_cv"] = f"std({pair}) / (mean(abs({pair})) + eps)"

    # 为每个 selected feature 查找公式
    result = OrderedDict()
    for f in selected_features:
        info = OrderedDict()
        info["feature"] = f
        if f in ALL_TEMPLATES:
            info["formula"] = ALL_TEMPLATES[f]
        else:
            info["formula"] = "[未匹配] — 请查看 s03_extract_feature_pool.py"

        # 确定类别
        if f in PPG_ANTI_SPOOF:
            info["category"] = "ppg_anti_spoof"
        elif f in PPG_SPATIAL:
            info["category"] = "ppg_spatial"
        elif f.startswith("PPG_DC_") or f.startswith("PPG_AC_") or f.startswith("PPG_DERIV_") or \
           f.startswith("PPG_FFT_") or f.startswith("PPG_AUTO_") or f.startswith("PPG_bp_") or \
           f in PPG_BASIC:
            info["category"] = "ppg_basic_dc_ac"
        elif f.startswith("PPG_Hjorth") or f.startswith("PPG_Entropy") or \
             f.startswith("PPG_Deriv") or f.startswith("PPG_Temporal"):
            info["category"] = "ppg_complexity_morphology"
        elif f in ("ACC_PPG_COH_MICRO", "ACC_PPG_COH_HR",
                   "EMG_PPG_CORR", "EMG_PPG_ENV_CORR",
                   "ACC_PPG_BP_CORR", "ACC_EMG_CORR"):
            info["category"] = "cross_modal"
        elif f.startswith("EMG0_"):
            info["category"] = "emg_ch0"
        elif f.startswith("EMG1_"):
            info["category"] = "emg_ch1"
        elif f.startswith("EMG_"):
            info["category"] = "emg_cross"
        elif f.startswith("ACC_"):
            info["category"] = "acc"
        elif f in ("SIG_LEN", "SIG_SEC"):
            info["category"] = "meta"
        else:
            info["category"] = "unknown"
        result[f] = info
    return result


def validate_feature_formula_map(formula_map):
    """Fail deployment export when selected features lack formula documentation."""
    missing = []
    for feature, info in formula_map.items():
        formula = str(info.get("formula", ""))
        category = str(info.get("category", ""))
        if "未匹配" in formula or category == "unknown":
            missing.append(feature)
    if missing:
        raise ValueError(
            "missing deploy feature formula docs for selected features: "
            + ", ".join(missing[:20])
        )


def xgboost_feature_name(feature_token, selected_features):
    """Map XGBoost feature tokens such as 0 or f0 to selected feature names."""
    if feature_token == "Leaf":
        return "Leaf"
    token = str(feature_token)
    match = re.fullmatch(r"f?(\d+)", token)
    if not match:
        return token
    idx = int(match.group(1))
    return selected_features[idx] if 0 <= idx < len(selected_features) else token


def parse_xgboost_dump_nodes(trees_txt, selected_features):
    """Parse text dump nodes into structured rows for deployment review."""
    rows = []
    for tidx, tree in enumerate(trees_txt):
        for line in tree.split("\n"):
            line = line.strip()
            if not line or line.startswith("booster"):
                continue
            node_id_text, sep, node_str = line.partition(":")
            if not sep:
                continue
            node_id = int(node_id_text.strip())
            node_str = node_str.strip()

            if "leaf=" in node_str:
                leaf_match = re.search(r"leaf=([^,\s]+)", node_str)
                cover_match = re.search(r"cover=([^,\s]+)", node_str)
                rows.append({
                    "Tree": tidx,
                    "Node": node_id,
                    "ID": f"{tidx}-{node_id}",
                    "Feature": "Leaf",
                    "FeatureName": "Leaf",
                    "Split": "",
                    "Yes": "",
                    "No": "",
                    "Missing": "",
                    "Gain": "",
                    "Cover": cover_match.group(1) if cover_match else "",
                    "LeafValue": float(leaf_match.group(1)) if leaf_match else "",
                })
                continue

            split_match = re.search(r"\[f(\d+)([<>=!]+)([^\]]+)\]", node_str)
            if not split_match:
                continue
            feat_idx = int(split_match.group(1))

            def _field(name):
                match = re.search(rf"{name}=([^,\s]+)", node_str)
                return match.group(1) if match else ""

            rows.append({
                "Tree": tidx,
                "Node": node_id,
                "ID": f"{tidx}-{node_id}",
                "Feature": f"f{feat_idx}",
                "FeatureName": xgboost_feature_name(f"f{feat_idx}", selected_features),
                "Split": split_match.group(3),
                "Yes": _field("yes"),
                "No": _field("no"),
                "Missing": _field("missing"),
                "Gain": _field("gain"),
                "Cover": _field("cover"),
                "LeafValue": "",
            })
    return rows


def export_deploy_artifacts(artifact_dir):
    """
    导出所有部署所需产物到 artifacts/deploy_package/。

    产生:
      - deploy_config.json       总配置
      - stage1_config.json       Stage1 参数 + 公式
      - feature_formulas.json    每个入选特征的计算公式
      - xgboost_trees.txt        所有树的文本 dump
      - xgboost_nodes.csv        所有节点的结构化 CSV
      - model_params.json        模型超参 / threshold / fill_values / clip_bounds / quality_thresholds / fingerprint
      - postprocess_config.json  状态机参数 + quality 阈值 + OOD 监控
    """
    import os as _os
    import shutil

    out_dir = _os.path.join(artifact_dir, "deploy_package")
    _os.makedirs(out_dir, exist_ok=True)
    print(f"\n导出部署产物到: {out_dir}")

    stage1_path = _os.path.join(artifact_dir, "stage1_threshold.json")
    bundle_path = _os.path.join(artifact_dir, "model_bundle.pkl")
    config_path = _os.path.join(artifact_dir, "final_model_config.json")

    with open(stage1_path, "r", encoding="utf-8") as f:
        stage1 = json.load(f)
    bundle = joblib.load(bundle_path)
    postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
    if _os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            fcfg = json.load(f)
        if "postprocess" in fcfg:
            postprocess_cfg.update(fcfg["postprocess"])

    selected_features = list(bundle["feature_names"])
    model = bundle["model"]
    booster = model.get_booster()

    # =========================================================
    # 1. stage1_config.json
    # =========================================================
    deploy_th = stage1.get("deploy_stage1_threshold", {})
    stage1_config = OrderedDict([
        ("pipeline_step", 1),
        ("description", "Stage1 IR/PPG DC/ACDC gate: six-channel mean @100Hz, no downsampling"),
        ("input", "PPG data, 6 channels @100Hz"),
        ("preprocessing", [
            "ppg_mean = mean(ppg_ch0..ppg_ch5)",
            "keep 100Hz samples",
        ]),
        ("window", {"duration_sec": 3.0, "points_100hz": 300, "stride_sec": 1.0}),
        ("features_per_window", OrderedDict([
            ("DC", {
                "formula": "min(neighbor_mean), neighbor_mean[i]=(x[i]+x[i+1])/2",
                "purpose": "minimum adjacent-sample mean",
            }),
            ("AC", {
                "formula": "median(abs(diff(x)))",
                "purpose": "median adjacent-sample variation",
            }),
            ("AC_DC_RATIO", {
                "formula": "AC / (abs(DC) + eps)",
                "purpose": "normalized AC variation",
            }),
        ])),
        ("window_pass_rule", "DC > dc_threshold AND AC_DC_RATIO < ac_dc_threshold"),
        ("sample_pass_rule", "any 3s Stage1 window passes -> sample enters Stage2"),
        ("thresholds", OrderedDict([
            ("dc_threshold", float(deploy_th.get("dc_threshold", 0))),
            ("ac_dc_threshold", float(deploy_th.get("ac_dc_threshold", 0))),
        ])),
        ("search_source", deploy_th.get("search_source", "unknown")),
    ])
    with open(_os.path.join(out_dir, "stage1_config.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_config, f, indent=2, ensure_ascii=False)
    print("  [OK] stage1_config.json")

    # =========================================================
    # 2. feature_formulas.json
    # =========================================================
    formula_map = build_feature_formula_map(selected_features)
    validate_feature_formula_map(formula_map)
    formulas_out = OrderedDict([
        ("pipeline_step", 2),
        ("description", "Stage2 — 3s 滑动窗口特征提取 (单通道 PPG + EMG + ACC) + XGBoost"),
        ("window_config", {
            "duration_sec": float(bundle["meta"].get("win_sec", 3.0)),
            "stride_sec": float(bundle["meta"].get("step_sec", 1.0)),
            "fs_ppg": float(bundle["meta"].get("fs_ppg", DEFAULT_FS_PPG)),
            "fs_emg": float(bundle["meta"].get("fs_emg", DEFAULT_FS_EMG)),
            "fs_acc": float(bundle["meta"].get("fs_acc", DEFAULT_FS_ACC)),
        }),
        ("preprocessing", {
            "ppg": "remove_burr → remove_step → medfilt(50ms) → movavg(30ms) → BP(0.4-6Hz, order=4)",
            "emg": "demean → robust_clean → BP(20-450Hz, order=4) → unified_notch(50/100/150/200/250/300, ±0.8Hz) → abs(envelope) + leak_ref",
            "acc": "magnitude → demean → BP(0.5-5Hz, order=2)",
        }),
        ("n_selected_features", len(selected_features)),
        ("features", formula_map),
    ])
    with open(_os.path.join(out_dir, "feature_formulas.json"), "w", encoding="utf-8") as f:
        json.dump(formulas_out, f, indent=2, ensure_ascii=False)
    print("  [OK] feature_formulas.json")

    # =========================================================
    # 3. xgboost_trees.txt
    # =========================================================
    trees_txt = booster.get_dump(with_stats=True)
    with open(_os.path.join(out_dir, "xgboost_trees.txt"), "w", encoding="utf-8") as f:
        for i, tree in enumerate(trees_txt):
            f.write(f"booster[{i}]:\n")
            f.write(tree)
            f.write("\n\n")
    print(f"  [OK] xgboost_trees.txt ({len(trees_txt)} trees)")

    # =========================================================
    # 4. xgboost_nodes.csv
    # =========================================================
    try:
        nodes_df = booster.trees_to_data_frame()
        if "Feature" in nodes_df.columns:
            nodes_df["FeatureName"] = nodes_df["Feature"].apply(
                lambda token: xgboost_feature_name(token, selected_features))
        nodes_df.to_csv(_os.path.join(out_dir, "xgboost_nodes.csv"), index=False)
        print(f"  [OK] xgboost_nodes.csv ({len(nodes_df)} nodes)")
    except Exception:
        try:
            rows = parse_xgboost_dump_nodes(trees_txt, selected_features)
            if rows:
                pd.DataFrame(rows).to_csv(_os.path.join(out_dir, "xgboost_nodes.csv"), index=False)
                print(f"  [OK] xgboost_nodes.csv ({len(rows)} nodes, parsed from tree dump)")
            else:
                print("[WARN] xgboost_nodes.csv: parsed 0 nodes")
        except Exception as e2:
            print(f"[WARN] xgboost_nodes.csv 生成失败: {e2}")

    # =========================================================
    # 5. model_params.json
    # =========================================================
    model_params = OrderedDict([
        ("pipeline_step", 2),
        ("model_type", "XGBoost (XGBClassifier)"),
        ("signal_types", {
            "ppg": "6-channel PPG @100Hz -> virtual 3-channel PPG: ch_A=avg(ch0,ch1), ch_B=avg(ch2,ch4), ch_C=avg(ch3,ch5)",
            "emg": "2-channel @ 1000Hz",
            "acc": "3-channel @ 100Hz",
        }),
        ("n_estimators", int(model.n_estimators)),
        ("hyperparameters", {k: v for k, v in model.get_params().items()
                              if k not in ("missing", "n_jobs", "random_state", "verbosity", "n_estimators")}),
        ("window_threshold", float(bundle["threshold"])),
        ("threshold_policy", bundle.get("threshold_policy", {})),
        ("n_selected_features", len(selected_features)),
        ("selected_features", selected_features),
        ("fill_values", bundle["fill_values"]),
        ("clip_bounds", bundle.get("clip_bounds", {})),
        ("preprocess_order", [
            "select selected_features in order",
            "fill NaN/inf with fill_values",
            "clip each selected feature by clip_bounds",
        ]),
        ("quality_thresholds", bundle.get("quality_thresholds", {})),
        ("feature_quantiles", bundle.get("feature_quantiles", {})),
        ("fingerprint", bundle.get("fingerprint", {})),
        ("meta", bundle["meta"]),
    ])
    with open(_os.path.join(out_dir, "model_params.json"), "w", encoding="utf-8") as f:
        json.dump(model_params, f, indent=2, ensure_ascii=False)
    print("  [OK] model_params.json")

    # =========================================================
    # 6. postprocess_config.json
    # =========================================================
    postprocess_out = OrderedDict([
        ("pipeline_step", 4),
        ("description", "时序后处理 — 对窗口概率应用带滞回的状态机平滑"),
        ("state_machine", OrderedDict([
            ("algorithm", "EMA + hysteresis"),
            ("state", "0=not_worn, 1=worn"),
            ("parameters", OrderedDict([
                ("alpha", float(postprocess_cfg.get("alpha", 0.4))),
                ("median_k", int(postprocess_cfg.get("median_k", 1))),
                ("T_on", float(postprocess_cfg.get("T_on", 0.75))),
                ("T_off", float(postprocess_cfg.get("T_off", 0.35))),
                ("K_on", int(postprocess_cfg.get("K_on", 5))),
                ("K_off", int(postprocess_cfg.get("K_off", 5))),
                ("cooldown_sec", float(postprocess_cfg.get("cooldown_sec", 5))),
                ("threshold_offset", float(postprocess_cfg.get("threshold_offset", 0.0))),
                ("threshold_transform", postprocess_cfg.get(
                    "threshold_transform",
                    "disabled" if "threshold_offset" not in postprocess_cfg
                    else "clip(prob_raw - (model_threshold + threshold_offset) + 0.5, 0, 1)",
                )),
            ])),
        ])),
        ("quality_scoring", OrderedDict([
            ("description", "基于特征质量调整 EMA 平滑速度"),
            ("thresholds_source", "train — learned from bundle['quality_thresholds']"),
            ("features_used", ["PPG_mean", "PPG_std"]),
            ("thresholds", bundle.get("quality_thresholds", {})),
        ])),
        ("ood_monitoring", OrderedDict([
            ("description", "OOD 窗比例监控 — 窗特征超出 train 分位 [q_low, q_high] 的比例"),
            ("alert_rate", 0.3),
            ("quantiles", bundle.get("feature_quantiles", {})),
        ])),
    ])
    with open(_os.path.join(out_dir, "postprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump(postprocess_out, f, indent=2, ensure_ascii=False)
    print("  [OK] postprocess_config.json")

    # =========================================================
    # 7. deploy_config.json
    # =========================================================
    deploy_config = OrderedDict([
        ("title", "手表佩戴活体检测 (PPG+EMG+ACC) — 部署配置"),
        ("pipeline_overview", [
            "Stage 1: 6-ch PPG → 通道平均 → 100Hz → 3s 窗 DC/ACDC 阈值粗筛 (1s stride)",
            "Stage 2: 3s 滑窗 → 单通道 PPG + EMG + ACC 特征提取 → XGBoost 窗口级概率",
            "Stage 4: WearStateMachine 时序后处理 (EMA + hysteresis)",
        ]),
        ("stage1", stage1_config),
        ("stage2_features", OrderedDict([("n_features", len(selected_features)), ("names", selected_features)])),
        ("stage2_model", OrderedDict([
            ("type", "XGBoost"), ("n_trees", int(model.n_estimators)),
            ("window_threshold", float(bundle["threshold"])),
            ("fill_strategy", "train median"),
        ])),
        ("stage4_postprocess", OrderedDict([
            ("alpha", float(postprocess_cfg.get("alpha", 0.4))),
            ("median_k", int(postprocess_cfg.get("median_k", 1))),
            ("T_on", float(postprocess_cfg.get("T_on", 0.75))),
            ("T_off", float(postprocess_cfg.get("T_off", 0.35))),
            ("K_on", int(postprocess_cfg.get("K_on", 5))),
            ("K_off", int(postprocess_cfg.get("K_off", 5))),
            ("cooldown_sec", float(postprocess_cfg.get("cooldown_sec", 5))),
            ("threshold_offset", float(postprocess_cfg.get("threshold_offset", 0.0))),
            ("threshold_transform", postprocess_cfg.get(
                "threshold_transform",
                "disabled" if "threshold_offset" not in postprocess_cfg
                else "clip(prob_raw - (model_threshold + threshold_offset) + 0.5, 0, 1)",
            )),
        ])),
        ("bundle_fingerprint", bundle.get("fingerprint", {})),
    ])
    with open(_os.path.join(out_dir, "deploy_config.json"), "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2, ensure_ascii=False)
    print("  [OK] deploy_config.json")

    model_json_path = _os.path.join(artifact_dir, "final_model.json")
    if _os.path.exists(model_json_path):
        shutil.copy2(model_json_path, _os.path.join(out_dir, "xgboost_model.json"))
        print("  [OK] xgboost_model.json")

    print(f"\n部署产物导出完成: {out_dir}/")
    print(f"  共 {len(_os.listdir(out_dir))} 个文件")


# =========================================================
# main
# =========================================================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--method", type=str, default="state_machine",
                        choices=["mean_vote", "prob_mean", "state_machine", "gated"])
    parser.add_argument("--window_sec", type=int, default=3)
    parser.add_argument("--stride_sec", type=int, default=1)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optimize_split", type=str, default="valid")
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--warmup_frames", type=int, default=3)
    parser.add_argument("--ood_alert_rate", type=float, default=0.3)
    parser.add_argument("--export_deploy", action="store_true")
    parser.add_argument("--export_window_cache", action="store_true",
                        help="Save per-sample window outputs as NPZ caches for postprocess search.")
    parser.add_argument("--window_output_root", type=str, default="window_outputs",
                        help="Subdirectory under artifact_dir for NPZ window caches.")
    parser.add_argument("--optimize_thresholds", type=str, default="",
                        help="对一组候选窗口阈值在缓存 probs 上做窗口级指标扫描（如 '0.3,0.4,0.5,0.6'），输出 P/R/F0.5/F1 表。")

    if args is None:
        args = parser.parse_args()

    if args.optimize and str(args.optimize_split).lower() == "test":
        raise ValueError(
            "test split cannot be used for optimization; use valid for "
            "state-machine parameter search and reserve test for final reporting."
        )

    with open(os.path.join(args.artifact_dir, "splits.json"), "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(os.path.join(args.artifact_dir, "stage1_threshold.json"), "r", encoding="utf-8") as f:
        th = json.load(f)

    deploy_th = get_deploy_stage1_threshold(th)
    bundle_path = os.path.join(args.artifact_dir, "model_bundle.pkl")

    print("=" * 80)
    print("加载统一模型包")
    print("=" * 80)
    bundle = load_bundle(bundle_path)
    print(f"feature_names: {len(bundle['feature_names'])} 个特征")
    print(f"threshold: {bundle['threshold']}")
    print(f"meta: {bundle['meta']}")

    print("\n" + "=" * 80)
    print("Deploy Stage1 threshold")
    print("=" * 80)
    print(f"dc_threshold   = {deploy_th['dc_threshold']}")
    print(f"acdc_threshold = {deploy_th['ac_dc_threshold']}")

    if args.optimize:
        print("\n" + "=" * 80)
        print(f"运行状态机参数优化 (split={args.optimize_split})")
        print("=" * 80)
        opt = optimize_state_machine_params(
            samples=split[args.optimize_split],
            dc_threshold=deploy_th["dc_threshold"],
            ac_dc_threshold=deploy_th["ac_dc_threshold"],
            window_sec=args.window_sec, stride_sec=args.stride_sec,
            bundle_path=bundle_path, n_workers=args.n_workers)
        best_params = opt["best_params"]
        best_metrics = opt["best_metrics"]
        print("\n最优参数:")
        print(json.dumps(best_params, indent=2, ensure_ascii=False))

        config_path = os.path.join(args.artifact_dir, "final_model_config.json")
        cfg_old = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_old = json.load(f)
        cfg_old["postprocess"] = best_params
        cfg_old["postprocess_optimization"] = {
            "optimized_on_split": args.optimize_split, "metrics": best_metrics,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_old, f, indent=2, ensure_ascii=False)
        print(f"\n最优参数已保存到: {config_path}")
        postprocess_cfg = best_params
    else:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
        # 尝试读取 optimize 阶段（s06_opt 或 s07_postprocess_optimize）保存的最优参数
        config_path = os.path.join(args.artifact_dir, "final_model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
                if "postprocess" in saved_cfg:
                    postprocess_cfg.update(saved_cfg["postprocess"])
                    print(describe_postprocess_config_source(config_path, saved_cfg))
            except Exception:
                pass

    print("\n" + "=" * 80)
    print(f"并行推理 split={args.split} (n_workers={args.n_workers})")
    print("=" * 80)
    results = run_inference_parallel(
        samples=split[args.split],
        dc_threshold=deploy_th["dc_threshold"],
        ac_dc_threshold=deploy_th["ac_dc_threshold"],
        window_sec=args.window_sec, stride_sec=args.stride_sec,
        bundle_path=bundle_path, n_workers=args.n_workers)

    sample_summary, details = compute_sample_metrics(
        results, args.method, postprocess_cfg, bundle["threshold"],
        stride_sec=args.stride_sec)
    window_model_summary = compute_window_model_metrics(results)
    window_stream_summary = compute_window_stream_metrics(
        results, postprocess_cfg, warmup_frames=args.warmup_frames,
        stride_sec=args.stride_sec, model_threshold=float(bundle["threshold"]))

    ood_summary = _summarize_ood(results, alert_rate=args.ood_alert_rate)

    sample_summary["split"] = args.split
    sample_summary["selected_features"] = bundle["feature_names"]
    sample_summary["model_threshold"] = float(bundle["threshold"])

    print("\n" + "=" * 80)
    print("指标1: 端到端评估 (Stage1→Stage2→Stage3)")
    print("=" * 80)
    summary_for_print = {k: v for k, v in sample_summary.items() if k != "selected_features"}
    print(json.dumps(summary_for_print, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("指标2: Stage2 模型评估 (仅通过Stage1的数据)")
    print("=" * 80)
    wm_print = {k: v for k, v in window_model_summary.items() if k != "confusion_matrix"}
    print(json.dumps(wm_print, indent=2, ensure_ascii=False))
    print(f"  confusion_matrix: {window_model_summary['confusion_matrix']}")

    print("\n" + "=" * 80)
    print("参考: Stage2+3 流式状态")
    print("=" * 80)
    ws_print = {k: v for k, v in window_stream_summary.items() if k != "confusion_matrix"}
    print(json.dumps(ws_print, indent=2, ensure_ascii=False))
    print(f"  confusion_matrix: {window_stream_summary['confusion_matrix']}")

    print("\n" + "=" * 80)
    print("准确率对比")
    print("=" * 80)
    print(f"  端到端 (Stage1→3):       {sample_summary['accuracy']:.4f}")
    print(f"  Stage2 模型 (逐窗):      {window_model_summary['accuracy']:.4f}")
    print(f"  Stage2+3 状态机 (逐窗):  {window_stream_summary['accuracy']:.4f}")

    # 窗口阈值扫描
    threshold_sweep = None
    if args.optimize_thresholds.strip():
        thr_list = []
        for t in args.optimize_thresholds.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                thr_list.append(float(t))
            except ValueError:
                pass
        if thr_list:
            print("\n" + "=" * 80)
            print(f"窗口阈值扫描 (window-level, warmup={args.warmup_frames})")
            print("=" * 80)
            print(f"{'threshold':>10}  {'precision':>10}  {'recall':>10}  "
                  f"{'F0.5':>10}  {'F1':>10}  {'n_win':>8}")
            sweep_rows = []
            for thr in thr_list:
                y_t, y_p = [], []
                for r in results:
                    if (r.get("fallback", False) or not r.get("stage1_pass", False)
                            or len(r.get("window_probs", [])) == 0):
                        continue
                    t_target = int(r["target"])
                    probs = np.asarray(r["window_probs"], dtype=float)
                    start = min(args.warmup_frames, len(probs))
                    for p in probs[start:]:
                        y_t.append(t_target)
                        y_p.append(int(p >= thr))
                if not y_t:
                    continue
                y_t = np.asarray(y_t)
                y_p = np.asarray(y_p)
                prec = float(precision_score(y_t, y_p, zero_division=0))
                rec = float(recall_score(y_t, y_p, zero_division=0))
                f1v = float(f1_score(y_t, y_p, zero_division=0))
                denom = 0.25 * prec + rec
                f05 = 1.25 * prec * rec / denom if denom > 0 else 0.0
                row = {"threshold": float(thr), "precision": prec, "recall": rec,
                        "f0.5": float(f05), "f1": f1v, "n_windows": int(len(y_t))}
                sweep_rows.append(row)
                print(f"{thr:>10.3f}  {prec:>10.4f}  {rec:>10.4f}  "
                      f"{f05:>10.4f}  {f1v:>10.4f}  {len(y_t):>8d}")
            threshold_sweep = {"thresholds": thr_list, "rows": sweep_rows,
                               "note": "用 F0.5（偏 precision）选合适操作点，在 s05 重训时固化。"}

    out_path = os.path.join(args.artifact_dir, f"end_to_end_eval_{args.split}_{args.method}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": sample_summary,
            "window_summary": window_stream_summary,
            "window_model_summary": window_model_summary,
            "window_stream_summary": window_stream_summary,
            "ood_summary": ood_summary,
            "threshold_sweep": threshold_sweep,
            "details": details,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {out_path}")

    if args.export_window_cache:
        metadata = {
            "model_fingerprint_json": json.dumps(bundle.get("fingerprint", {}), ensure_ascii=False),
            "feature_names_json": json.dumps(bundle.get("feature_names", []), ensure_ascii=False),
        }
        cache_info = export_window_cache(
            results,
            artifact_dir=args.artifact_dir,
            split_name=args.split,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            model_threshold=bundle["threshold"],
            metadata=metadata,
            cache_root=args.window_output_root,
        )
        print("\nWindow cache exported")
        print(json.dumps(cache_info, indent=2, ensure_ascii=False))

    if args.export_deploy:
        export_deploy_artifacts(args.artifact_dir)


if __name__ == "__main__":
    main()
