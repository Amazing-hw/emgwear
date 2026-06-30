# s03_extract_feature_pool.py
# -*- coding: utf-8 -*-

"""
步骤3：滑窗特征池提取（适配 6-ch PPG avg → 单通道 PPG + 2-ch EMG + 3-ch ACC）

信号说明：
  - PPG: 6 通道 @ 100Hz
  - EMG: 2 通道 @ 1000Hz
  - ACC: 3 通道 @ 100Hz

Stage1:
  6-ch PPG 取平均 → 保持 100Hz → 3s 窗 DC/ACDC 阈值粗筛

Stage2 特征提取（3s 滑窗）：
  A. PPG 单通道（6-ch 取平均）@ 100Hz
     - 基础统计 + DC/AC
     - 频域/自相关特征
     - Hjorth / Entropy / Derivative / Temporal
  B. EMG 双通道 @ 1000Hz
     - 时域: MAV, RMS, VAR, WL, ZC, SSC, WAMP
     - 频域: MNF, MDF, PKF, PSR
     - 非线性: Sample Entropy
     - 通道间: 相关性
  C. ACC 三通道 @ 100Hz
  D. 跨模态特征: EMG-PPG, ACC-PPG, ACC-EMG
"""

import os
import sys
import json
import time
import argparse
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd

from scipy.signal import resample_poly, butter, filtfilt, medfilt, correlate, find_peaks, iirnotch

# Linux/macOS 默认 fork 模式多进程读 H5 可能死锁，强制 spawn
if sys.platform != "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 已经设置过了

# =========================================================
# 基本配置
# =========================================================

EPS = 1e-12

# EMG 滤波配置：常规特征使用 highpass + narrow bandstop + high-Q notch。
_EMG_NOTCH_BW_HZ = 0.8             # leak 特征的单边统计带宽
_EMG_CLEAN_BANDSTOP_RANGES = ((49.8, 50.2), (149.8, 150.2))
_EMG_CLEAN_NOTCH_FREQS = (50.0, 100.0, 200.0, 300.0, 400.0)
_EMG_CLEAN_NOTCH_Q = 100.0
_EMG_LEAK_FREQS = (100.0, 150.0, 200.0, 250.0, 300.0)  # PPG 串扰特征频点（不含工频 50Hz）
_EMG_MAD_CLIP_K = 10.0          # 超过 K×MAD 视为离群点 (电极 pop)
_ACC_BURR_K = 6.0               # ACC 每轴去毛刺阈值倍数
DEFAULT_FS_PPG = 100.0
DEFAULT_FS_EMG = 1000.0
DEFAULT_FS_ACC = 100.0
FEATURE_FS = 100  # PPG/ACC 保持原始 100Hz

# 已知高冗余/不稳定特征 — s03 阶段跳过以节省计算
# 这些在 s04 清洗阶段也会被 VIF/高相关移除，提前跳过加速提取
_REDUNDANT_FEATURES = {
    # -- Hjorth_Complexity 二阶导出，极不稳定 --
    "PPG_Hjorth_Complexity",
    # -- valley_ratio ≈ 1 - peak_ratio，信息重叠 --
    "PPG_Temporal_valley_ratio",
    # -- 二阶导数 ≈ 一阶导数的变化，相关性极高 --
    "PPG_Deriv_d2_mean", "PPG_Deriv_d2_std",
    "PPG_Deriv_d2_max", "PPG_Deriv_d2_min",
    "PPG_Deriv_d2_zcr",
    # -- ApEn ≈ SampEn（SampEn 更无偏，保留 SampEn）--
    "PPG_Entropy_ApEn",
    # -- Shannon Entropy 对连续信号区分度低 --
    "PPG_Entropy_Shannon",
    # -- AC_RMS ≈ 1.48 × AC_MAD（MAD 更鲁棒）--
    "PPG_AC_RMS",
    # -- AUTO_CORR_LAG_SEC ≈ 1 / DOM_FREQ --
    "PPG_AUTO_CORR_LAG_SEC",
    # -- Hjorth_Activity = var(bp) ≈ (AC_RMS)^2，保留 RMS --
    "PPG_Hjorth_Activity",
}

# =========================================================
# H5 读取
# =========================================================

def normalize_sensor_array(arr, n_channels):
    """Normalize continuous arrays to (N, C), and windowed (W, C, N) to (W, N, C)."""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        if x.shape[1] == n_channels:
            return x
        if x.shape[0] == n_channels:
            return x.T
        return x
    if x.ndim == 3:
        if x.shape[2] == n_channels:
            return x
        if x.shape[1] == n_channels:
            return np.transpose(x, (0, 2, 1))
        raise ValueError(f"cannot infer channel axis for shape={x.shape}, channels={n_channels}")
    raise ValueError(f"unsupported sensor array shape={x.shape}")


def is_windowed_array(arr):
    return np.asarray(arr).ndim == 3


def _load_named_windows(sample, dataset_name, n_channels):
    window_names = sample.get("window_names") or []
    if not window_names:
        return None
    arrs = []
    with h5py.File(sample["h5_file"], "r") as f:
        parent = f[sample["sample_name"]]
        for window_name in window_names:
            if dataset_name not in parent[window_name]:
                if dataset_name in ("emg", "acc"):
                    return None
                raise KeyError(f"{dataset_name} not found in {sample['sample_name']}/{window_name}")
            arrs.append(normalize_sensor_array(parent[window_name][dataset_name][:], n_channels))
    return np.stack(arrs, axis=0)


def load_ppg(sample):
    """读取 6-ch PPG @ 100Hz，返回 (N, 6)。"""
    named = _load_named_windows(sample, "ppg", 6)
    if named is not None:
        return named
    with h5py.File(sample["h5_file"], "r") as f:
        ppg = f[sample["sample_name"]]["ppg"][:]
    return normalize_sensor_array(ppg, 6)


def load_emg(sample):
    """读取 2-ch EMG @ 1000Hz，返回 (N, 2)，不存在返回 None。"""
    named = _load_named_windows(sample, "emg", 2)
    if named is not None:
        return named
    with h5py.File(sample["h5_file"], "r") as f:
        if "emg" not in f[sample["sample_name"]]:
            return None
        emg = f[sample["sample_name"]]["emg"][:]
    return normalize_sensor_array(emg, 2)


def load_acc(sample):
    """读取 3-ch ACC @ 100Hz，返回 (N, 3)，不存在返回 None。"""
    named = _load_named_windows(sample, "acc", 3)
    if named is not None:
        return named
    with h5py.File(sample["h5_file"], "r") as f:
        if "acc" not in f[sample["sample_name"]]:
            return None
        acc = f[sample["sample_name"]]["acc"][:]
    return normalize_sensor_array(acc, 3)


def build_3ch_ppg(ppg_6ch):
    """6 通道 PPG → 3 通道虚拟 PPG（相邻通道配对）。

    ch_A = avg(ch1, ch2)   — 中心上方
    ch_B = avg(ch3, ch5)   — 左下+右下对角
    ch_C = avg(ch4, ch6)   — 左上+右上对角
    返回 (N, 3) @ 100Hz。
    """
    ppg = np.asarray(ppg_6ch, dtype=np.float64)
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)
    # 原始为 (N, 6) 或 (6, N)；统一转为 (N, 6)
    if ppg.shape[1] != 6:
        ppg = ppg.T
    ch_A = (ppg[:, 0] + ppg[:, 1]) / 2.0
    ch_B = (ppg[:, 2] + ppg[:, 4]) / 2.0
    ch_C = (ppg[:, 3] + ppg[:, 5]) / 2.0
    return np.column_stack([ch_A, ch_B, ch_C])


def stage1_ppg_mean_signal(ppg_6ch):
    ppg = np.asarray(ppg_6ch, dtype=np.float64)
    if ppg.ndim == 3:
        return np.mean(ppg, axis=2)
    if ppg.ndim == 1:
        return ppg
    if ppg.shape[1] != 6 and ppg.shape[0] == 6:
        ppg = ppg.T
    return np.mean(ppg, axis=1)


def stage1_window_pass_100hz(ppg_mean_window, dc_threshold, ac_dc_threshold):
    x = np.asarray(ppg_mean_window, dtype=np.float64)
    if len(x) < 2:
        return False
    neighbor_mean = (x[:-1] + x[1:]) / 2.0
    dc = float(np.min(neighbor_mean))
    ac = float(np.median(np.abs(np.diff(x))))
    ac_dc_ratio = ac / (np.abs(dc) + EPS)
    return (dc > dc_threshold) and (ac_dc_ratio < ac_dc_threshold)


def stage1_sample_pass(ppg_6ch, dc_threshold, ac_dc_threshold):
    """Stage1: six-channel mean PPG at 100Hz, 3s window / 1s stride."""
    if is_windowed_array(ppg_6ch):
        for ppg_win in ppg_6ch:
            ppg_mean_win = stage1_ppg_mean_signal(ppg_win)
            if stage1_window_pass_100hz(ppg_mean_win, dc_threshold, ac_dc_threshold):
                return True
        return False
    ppg_mean = stage1_ppg_mean_signal(ppg_6ch)
    win = int(3 * 100)
    stride = int(1 * 100)
    for i in range(0, len(ppg_mean) - win + 1, stride):
        if stage1_window_pass_100hz(ppg_mean[i:i + win], dc_threshold, ac_dc_threshold):
            return True
    return False


# =========================================================
# 鲁棒基础工具
# =========================================================

def safe_div(a, b, eps=EPS):
    return float(a) / (float(b) + eps)


def robust_mad(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def safe_corr(x, y, winsorize=False):
    """Pearson 相关系数，对异常值鲁棒。

    winsorize=True 时先对两信号做 [p5, p95] 裁剪，再算 Pearson。
    适用于 EMG/ACC 等偶有电极 pop 或碰撞瞬变的信号。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    if n < 8:
        return 0.0
    x, y = x[:n], y[:n]
    if winsorize:
        x = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
        y = np.clip(y, np.percentile(y, 5), np.percentile(y, 95))
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx < EPS or sy < EPS:
        return 0.0
    v = np.mean((x / sx) * (y / sy))
    if not np.isfinite(v):
        return 0.0
    return float(v)


def moving_average_filter(x, window_size=5):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window_size or window_size < 2:
        return x.copy()
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(x, kernel, mode="same")


# =========================================================
# 鲁棒预处理
# =========================================================

def remove_burr(x, burr_k=6.0):
    x = np.asarray(x, dtype=np.float64).copy()
    if len(x) < 3:
        return x
    d = np.diff(x)
    mad_d = robust_mad(d)
    thr = max(burr_k * mad_d, EPS)
    left = x[:-2]
    mid = x[1:-1]
    right = x[2:]
    bad = (np.abs(mid - left) > thr) & (np.abs(mid - right) > thr)
    if bad.any():
        replaced = 0.5 * (left + right)
        x[1:-1] = np.where(bad, replaced, mid)
    return x


def remove_step(x, step_k=10.0):
    x = np.asarray(x, dtype=np.float64).copy()
    if len(x) < 2:
        return x
    d = np.diff(x)
    mad_d = robust_mad(d)
    thr = max(step_k * mad_d, EPS)
    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) > thr:
            x[i] = x[i - 1]
    return x


_BUTTER_CACHE = {}
_IIR_NOTCH_CACHE = {}


def _get_butter_coeffs(fs, lowcut, highcut, order):
    key = (float(fs), float(lowcut), float(highcut), int(order))
    if key in _BUTTER_CACHE:
        return _BUTTER_CACHE[key]
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        _BUTTER_CACHE[key] = None
        return None
    try:
        b, a = butter(order, [low, high], btype="band")
        _BUTTER_CACHE[key] = (b, a)
        return _BUTTER_CACHE[key]
    except Exception:
        _BUTTER_CACHE[key] = None
        return None


def _get_butter_highpass_coeffs(fs, cutoff, order):
    key = (float(fs), "highpass", float(cutoff), int(order))
    if key in _BUTTER_CACHE:
        return _BUTTER_CACHE[key]
    nyq = 0.5 * fs
    wn = min(max(cutoff / nyq, 1e-6), 0.999)
    try:
        b, a = butter(order, wn, btype="highpass")
        _BUTTER_CACHE[key] = (b, a)
        return _BUTTER_CACHE[key]
    except Exception:
        _BUTTER_CACHE[key] = None
        return None


def highpass_filter(x, fs, cutoff, order=2):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16:
        return x.copy()
    coeffs = _get_butter_highpass_coeffs(fs, cutoff, order)
    if coeffs is None:
        return x.copy()
    b, a = coeffs
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x - np.median(x)


def bandpass_filter(x, fs, lowcut, highcut, order=2):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16:
        return x.copy()
    coeffs = _get_butter_coeffs(fs, lowcut, highcut, order)
    if coeffs is None:
        return x.copy()
    b, a = coeffs
    try:
        y = filtfilt(b, a, x)
    except Exception:
        y = x - np.median(x)
    return y


def preprocess_signal(x, fs, bp_low=0.4, bp_high=6.0):
    """PPG 预处理：去毛刺 → 去跳变 → 中值滤波 → 滑动平均 → 带通。"""
    x = np.asarray(x, dtype=np.float64).copy()
    x = remove_burr(x, burr_k=6.0)
    x = remove_step(x, step_k=10.0)

    mf_kernel = max(3, int(round(0.05 * fs)))
    if mf_kernel % 2 == 0:
        mf_kernel += 1
    if len(x) >= mf_kernel:
        try:
            x = medfilt(x, kernel_size=mf_kernel)
        except Exception:
            pass

    ma_win = max(2, int(round(0.03 * fs)))
    x = moving_average_filter(x, window_size=ma_win)

    dc = float(np.median(x))
    bp = bandpass_filter(x, fs, lowcut=bp_low, highcut=bp_high, order=4)
    bp = moving_average_filter(bp, window_size=ma_win)
    return x, bp, dc


def preprocess_emg_signal(x, fs=1000.0):
    """
    EMG 预处理：去均值 → 鲁棒清理 → highpass(20Hz) → 指定窄带滤波 → 全波整流。

    返回 (bp_leak_ref, bp_clean, env)：
      - bp_leak_ref: highpass(20Hz) 后、bandstop/notch 前参考，用于 leak + mains 特征
      - bp_clean:    bandstop(49.8-50.2,149.8-150.2) + notch(50/100/200/300/400Hz,Q=100) 后信号
      - env:         基于 bp_clean 的全波整流包络

    注: baseline drift (1-10Hz) 特征需要的原始去均值信号通过 preprocess_emg_signal_with_raw 获取。
    """
    bp_leak_ref, bp_clean, env, _ = preprocess_emg_signal_with_raw(x, fs)
    return bp_leak_ref, bp_clean, env


_NARROW_NOTCH_CACHE = {}


def _narrow_notch(x, fs, f0, bw_hz=0.8, order=2):
    """单频率窄带 bandstop。f0±bw_hz Hz (默认 ±0.8Hz)。
    滤波器系数基于 (fs, f0, bw_hz, order) 缓存，避免重复设计。
    """
    key = (float(fs), float(f0), float(bw_hz), int(order))
    if key not in _NARROW_NOTCH_CACHE:
        nyq = 0.5 * fs
        w0 = f0 / nyq
        wbw = bw_hz / nyq
        lo = max(w0 - wbw, 1e-6)
        hi = min(w0 + wbw, 0.999)
        if not (0 < lo < hi < 1):
            _NARROW_NOTCH_CACHE[key] = None
        else:
            try:
                b, a = butter(order, [lo, hi], btype="bandstop")
                _NARROW_NOTCH_CACHE[key] = (b, a)
            except Exception:
                _NARROW_NOTCH_CACHE[key] = None
    coeffs = _NARROW_NOTCH_CACHE[key]
    if coeffs is None:
        return x
    b, a = coeffs
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x


def _iir_notch_filter(x, fs, f0, q=100.0):
    key = (float(fs), float(f0), float(q))
    if key not in _IIR_NOTCH_CACHE:
        try:
            b, a = iirnotch(float(f0), float(q), fs=float(fs))
            _IIR_NOTCH_CACHE[key] = (b, a)
        except Exception:
            _IIR_NOTCH_CACHE[key] = None
    coeffs = _IIR_NOTCH_CACHE[key]
    if coeffs is None:
        return x
    b, a = coeffs
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x


def _emg_robust_clean(x):
    """EMG 鲁棒清理：3 点中值（消除孤立尖峰）+ MAD 钳位（电极 pop 截断）。

    输入应为已去均值的 EMG。输出同长度。
    """
    if len(x) < 5:
        return x
    try:
        x = medfilt(x, kernel_size=3)
    except Exception:
        pass
    mad = float(np.median(np.abs(x - np.median(x))))
    if mad > EPS:
        clip = _EMG_MAD_CLIP_K * mad
        np.clip(x, -clip, clip, out=x)
    return x


def preprocess_emg_signal_with_raw(x, fs=1000.0):
    """同 preprocess_emg_signal，但额外返回去均值（无带通）的原始信号，用于 baseline drift。

    流水线：demean → [鲁棒清理: medfilt(3) + MAD 钳位] → highpass(20Hz)
            → 保存 bp_leak_ref (bandstop/notch 前参考，含 50Hz 和 PPG 串扰)
            → bandstop(49.8-50.2,149.8-150.2)
            → notch(50/100/200/300/400Hz,Q=100) → 包络。

    返回 (bp_leak_ref, bp_clean, env, x_demean)：
      - bp_leak_ref:  highpass(20Hz) 后、bandstop/notch 前参考，用于 leak + mains 特征
      - bp_clean:     指定窄带滤波后的信号，用于 MNF/MDF/PKF 等
      - env:          abs(bp_clean)
      - x_demean:     仅去均值（未做鲁棒清理），用于 baseline drift (1-10Hz)
    """
    x = np.asarray(x, dtype=np.float64).copy()
    x_demean = x - np.mean(x)

    # 鲁棒清理后再高通，避免尖峰被 filtfilt 抹成长尾
    x_clean = _emg_robust_clean(x_demean.copy())
    bp = highpass_filter(x_clean, fs, cutoff=20.0, order=2)

    # 保存 bandstop/notch 前参考信号：含全部窄带能量，用于 leak 特征和 mains 特征
    bp_leak_ref = bp.copy()

    # 指定窄带滤波：工频/PPG 串扰主要频点。
    for lo, hi in _EMG_CLEAN_BANDSTOP_RANGES:
        center = (lo + hi) / 2.0
        bp = _narrow_notch(bp, fs, center, bw_hz=(hi - lo) / 2.0, order=2)
    for f0 in _EMG_CLEAN_NOTCH_FREQS:
        bp = _iir_notch_filter(bp, fs, f0, q=_EMG_CLEAN_NOTCH_Q)
    bp_clean = bp

    env = np.abs(bp_clean)
    return bp_leak_ref, bp_clean, env, x_demean


# =========================================================
# FFT 与自相关
# =========================================================

def compute_fft_cache(x, fs, fmin=0.5, fmax=5.0):
    x = np.asarray(x, dtype=np.float64)
    result = {'peak_ratio': 0.0, 'dom_freq': 0.0, 'spec': None, 'freqs': None,
              'band_spec': None, 'band_freqs': None}
    if len(x) < 16:
        return result
    x = x - np.mean(x)
    xw = x * np.hamming(len(x))
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    nfft = max(256, nfft)
    spec = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    result['spec'] = spec
    result['freqs'] = freqs
    if not np.any(mask):
        return result
    band_spec = spec[mask]
    band_freqs = freqs[mask]
    result['band_spec'] = band_spec
    result['band_freqs'] = band_freqs
    med = np.median(band_spec)
    if med < EPS:
        peak_ratio = 0.0
    else:
        peak_ratio = float(np.max(band_spec) / (med + EPS))
    dom_freq = float(band_freqs[np.argmax(band_spec)])
    result['peak_ratio'] = peak_ratio
    result['dom_freq'] = dom_freq
    return result


def fft_peak_features(x, fs, fmin=0.5, fmax=5.0):
    cache = compute_fft_cache(x, fs, fmin, fmax)
    return cache['peak_ratio'], cache['dom_freq']


def normalized_autocorr(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return np.zeros(1, dtype=np.float64)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(x) - 1:]
    if corr[0] < EPS:
        return np.zeros_like(corr)
    return corr / (corr[0] + EPS)


def autocorr_periodicity_features(x, fs, bpm_min=40.0, bpm_max=180.0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < int(fs * 1.5):
        return 0.0, 0.0
    ac = normalized_autocorr(x)
    lag_min = int(fs * 60.0 / bpm_max)
    lag_max = int(fs * 60.0 / bpm_min)
    lag_min = max(1, lag_min)
    lag_max = min(len(ac) - 1, lag_max)
    if lag_max <= lag_min:
        return 0.0, 0.0
    seg = ac[lag_min:lag_max + 1]
    if len(seg) == 0:
        return 0.0, 0.0
    idx = int(np.argmax(seg))
    peak = float(seg[idx])
    lag = lag_min + idx
    lag_sec = float(lag / fs)
    return peak, lag_sec


def max_norm_xcorr(x, y, max_lag_samples):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    if n < 8:
        return 0.0
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    sx = np.std(x)
    sy = np.std(y)
    if sx < EPS or sy < EPS:
        return 0.0
    corr = correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= max_lag_samples
    corr = corr[mask]
    corr = corr / (n * sx * sy + EPS)
    if len(corr) == 0:
        return 0.0
    return float(np.max(np.abs(corr)))


def smooth_envelope(x, fs, win_sec=0.25):
    x = np.abs(np.asarray(x, dtype=np.float64))
    win = max(3, int(round(win_sec * fs)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, kernel, mode="same")


# =========================================================
# PPG 单通道特征（6-ch 取平均）
# =========================================================

def extract_ppg_features(raw, bp, dc, fs, prefix="PPG", fft_cache=None):
    feat = OrderedDict()
    raw = np.asarray(raw, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)
    if len(raw) == 0 or len(bp) == 0:
        return feat

    ac_rms = float(np.sqrt(np.mean(bp ** 2)))
    ac_mad = robust_mad(bp)
    dc_iqr = robust_iqr(raw)
    deriv = np.diff(bp) if len(bp) > 1 else np.array([0.0])
    deriv_mad = robust_mad(deriv)

    if fft_cache is not None:
        fft_peak_ratio = fft_cache.get('peak_ratio', 0.0)
        dom_freq = fft_cache.get('dom_freq', 0.0)
    else:
        fft_peak_ratio, dom_freq = fft_peak_features(bp, fs, fmin=0.5, fmax=5.0)

    ac_peak, ac_lag_sec = autocorr_periodicity_features(bp, fs, bpm_min=40.0, bpm_max=180.0)

    feat[f"{prefix}_DC_MEDIAN"] = float(dc)
    feat[f"{prefix}_DC_IQR"] = dc_iqr
    feat[f"{prefix}_AC_RMS"] = ac_rms
    feat[f"{prefix}_AC_MAD"] = ac_mad
    feat[f"{prefix}_AC_DC_RATIO"] = safe_div(ac_rms, abs(dc) + EPS)
    feat[f"{prefix}_DERIV_MAD"] = deriv_mad
    feat[f"{prefix}_FFT_PEAK_MEDIAN_RATIO"] = fft_peak_ratio
    feat[f"{prefix}_DOM_FREQ"] = dom_freq
    feat[f"{prefix}_AUTO_CORR_PEAK"] = ac_peak
    feat[f"{prefix}_AUTO_CORR_LAG_SEC"] = ac_lag_sec
    return feat


# =========================================================
# Hjorth / Entropy / Derivative / Temporal (通用)
# =========================================================

def extract_hjorth_parameters(x, prefix=""):
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    if len(x) < 4:
        for k in ["Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity"]:
            feat[f"{pf}{k}"] = 0.0
        return feat

    activity = float(np.var(x))
    feat[f"{pf}Hjorth_Activity"] = activity

    d1 = np.diff(x)
    if len(d1) < 2:
        feat[f"{pf}Hjorth_Mobility"] = 0.0
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat

    var_d1 = np.var(d1)
    feat[f"{pf}Hjorth_Mobility"] = float(np.sqrt(var_d1 / activity)) if activity > EPS else 0.0

    d2 = np.diff(d1)
    if len(d2) < 2:
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat
    var_d2 = np.var(d2)
    feat[f"{pf}Hjorth_Complexity"] = float(np.sqrt(var_d2 / var_d1)) if var_d1 > EPS else 0.0
    return feat


def extract_entropy_features(x, r_std_ratio=0.2, m=2, prefix=""):
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    if len(x) < 10:
        for k in ["Entropy_Shannon", "Entropy_ApEn", "Entropy_SampEn"]:
            feat[f"{pf}{k}"] = 0.0
        return feat

    try:
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        feat[f"{pf}Entropy_Shannon"] = float(-np.sum(hist * np.log(hist + EPS)))
    except Exception:
        feat[f"{pf}Entropy_Shannon"] = 0.0

    try:
        N = len(x)
        r = r_std_ratio * np.std(x)
        if r < EPS:
            apen = 0.0
        else:
            def phi(m_val):
                patterns = np.array([x[i:i + m_val] for i in range(N - m_val)])
                if len(patterns) == 0:
                    return 0.0
                distances = np.max(np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
                count = np.sum(distances <= r, axis=1) / (N - m_val)
                count = count[count > 0]
                if len(count) == 0:
                    return 0.0
                return np.mean(np.log(count + EPS))
            apen = float(phi(m) - phi(m + 1))
            if not np.isfinite(apen):
                apen = 0.0
    except Exception:
        apen = 0.0
    feat[f"{pf}Entropy_ApEn"] = apen

    try:
        N = len(x)
        r = r_std_ratio * np.std(x)
        if r < EPS:
            sampen = 0.0
        else:
            def sampen_count(m_val):
                patterns = np.array([x[i:i + m_val] for i in range(N - m_val)])
                if len(patterns) < 2:
                    return 0.0
                distances = np.max(np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
                np.fill_diagonal(distances, np.inf)
                return float(np.sum(distances <= r))

            B_m = sampen_count(m)
            B_m1 = sampen_count(m + 1)

            if B_m > 0 and B_m1 > 0:
                sampen = float(-np.log(B_m1 / (B_m + EPS)))
            else:
                sampen = 0.0
            if not np.isfinite(sampen):
                sampen = 0.0
    except Exception:
        sampen = 0.0
    feat[f"{pf}Entropy_SampEn"] = sampen
    return feat


def extract_derivative_features(x, fs=100.0, prefix=""):
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    if len(x) < 4:
        for k in ["Deriv_d1_mean", "Deriv_d1_std", "Deriv_d1_max", "Deriv_d1_min", "Deriv_d1_zcr",
                   "Deriv_d2_mean", "Deriv_d2_std", "Deriv_d2_max", "Deriv_d2_min", "Deriv_d2_zcr"]:
            feat[f"{pf}{k}"] = 0.0
        return feat

    d1 = np.diff(x)
    if len(d1) > 0:
        feat[f"{pf}Deriv_d1_mean"] = float(np.mean(d1))
        feat[f"{pf}Deriv_d1_std"] = float(np.std(d1))
        feat[f"{pf}Deriv_d1_max"] = float(np.max(d1))
        feat[f"{pf}Deriv_d1_min"] = float(np.min(d1))
        feat[f"{pf}Deriv_d1_zcr"] = float(np.sum(np.abs(np.diff(np.sign(d1)))) / (2.0 * len(d1)))
    else:
        for k in ["Deriv_d1_mean", "Deriv_d1_std", "Deriv_d1_max", "Deriv_d1_min", "Deriv_d1_zcr"]:
            feat[f"{pf}{k}"] = 0.0

    d2 = np.diff(d1) if len(d1) > 1 else np.array([])
    if len(d2) > 0:
        feat[f"{pf}Deriv_d2_mean"] = float(np.mean(d2))
        feat[f"{pf}Deriv_d2_std"] = float(np.std(d2))
        feat[f"{pf}Deriv_d2_max"] = float(np.max(d2))
        feat[f"{pf}Deriv_d2_min"] = float(np.min(d2))
        feat[f"{pf}Deriv_d2_zcr"] = float(np.sum(np.abs(np.diff(np.sign(d2)))) / (2.0 * len(d2)))
    else:
        for k in ["Deriv_d2_mean", "Deriv_d2_std", "Deriv_d2_max", "Deriv_d2_min", "Deriv_d2_zcr"]:
            feat[f"{pf}{k}"] = 0.0
    return feat


def extract_temporal_dynamic_features(x, fs=100.0, prefix=""):
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    if len(x) < 4:
        for k in ["Temporal_slope_mean", "Temporal_slope_std", "Temporal_peak_prominence",
                   "Temporal_peak_ratio", "Temporal_valley_ratio"]:
            feat[f"{pf}{k}"] = 0.0
        return feat

    t = np.arange(len(x))
    t_mean = np.mean(t)
    x_mean = np.mean(x)
    slope_num = np.sum((t - t_mean) * (x - x_mean))
    slope_den = np.sum((t - t_mean) ** 2)
    slope = slope_num / slope_den if slope_den > EPS else 0.0
    fitted = x_mean + slope * (t - t_mean)
    residuals = x - fitted
    feat[f"{pf}Temporal_slope_mean"] = float(slope)
    feat[f"{pf}Temporal_slope_std"] = float(np.std(residuals))

    try:
        peaks, peak_props = find_peaks(x, prominence=0)
        if len(peaks) > 0:
            feat[f"{pf}Temporal_peak_prominence"] = float(np.mean(peak_props["prominences"]))
            feat[f"{pf}Temporal_peak_ratio"] = float(len(peaks) / len(x))
        else:
            feat[f"{pf}Temporal_peak_prominence"] = 0.0
            feat[f"{pf}Temporal_peak_ratio"] = 0.0
        valleys, _ = find_peaks(-x, prominence=0)
        feat[f"{pf}Temporal_valley_ratio"] = float(len(valleys) / len(x)) if len(valleys) > 0 else 0.0
    except Exception:
        feat[f"{pf}Temporal_peak_prominence"] = 0.0
        feat[f"{pf}Temporal_peak_ratio"] = 0.0
        feat[f"{pf}Temporal_valley_ratio"] = 0.0
    return feat


# =========================================================
# PPG 3 通道空间特征（基于 ch_A/ch_B/ch_C）
# =========================================================

def _extract_ppg_spatial_features(ppg_ch_raw):
    """PPG 3 通道空间特征：不平衡度、范围、空间向量幅值、DC 一致性。

    ppg_ch_raw: (N, 3) @ 100Hz，3 通道原始 PPG 信号。

    佩戴时三通道由于组织异质性而存在空间差异；
    橡胶/硅胶伪造物光学均匀，三通道高度一致 → 不平衡度≈0。
    """
    feat = OrderedDict()
    ch = np.asarray(ppg_ch_raw, dtype=np.float64)
    if ch.ndim != 2 or ch.shape[1] < 3 or len(ch) < 4:
        for k in ["PPG_ch_imbalance_mean", "PPG_ch_imbalance_p90", "PPG_ch_imbalance_iqr",
                   "PPG_ch_rangeNorm_mean", "PPG_ch_rangeNorm_p90",
                   "PPG_ch_vmag_mean", "PPG_ch_vmag_p90", "PPG_ch_vmag_iqr", "PPG_ch_vmag_std",
                   "PPG_ch_dc_cv", "PPG_ch_dc_max_min_ratio"]:
            feat[k] = 0.0
        return feat

    g = ch.T  # (3, N)
    spatial_std = np.std(g, axis=0)   # 每个时间点三通道标准差
    spatial_mean = np.mean(g, axis=0)  # 每个时间点三通道均值
    imbalance = spatial_std / (np.abs(spatial_mean) + EPS)

    feat["PPG_ch_imbalance_mean"] = float(np.mean(imbalance))
    feat["PPG_ch_imbalance_p90"] = float(np.percentile(imbalance, 90))
    feat["PPG_ch_imbalance_iqr"] = robust_iqr(imbalance)

    g_max = np.max(g, axis=0)
    g_min = np.min(g, axis=0)
    denom = np.abs(g[0]) + np.abs(g[1]) + np.abs(g[2]) + EPS
    range_norm = (g_max - g_min) / denom
    feat["PPG_ch_rangeNorm_mean"] = float(np.mean(range_norm))
    feat["PPG_ch_rangeNorm_p90"] = float(np.percentile(range_norm, 90))

    # 120° 对称空间向量
    vx = g[0] - 0.5 * g[1] - 0.5 * g[2]
    vy = (np.sqrt(3.0) / 2.0) * (g[1] - g[2])
    vmag = np.sqrt(vx * vx + vy * vy) / (denom + EPS)
    feat["PPG_ch_vmag_mean"] = float(np.mean(vmag))
    feat["PPG_ch_vmag_p90"] = float(np.percentile(vmag, 90))
    feat["PPG_ch_vmag_iqr"] = robust_iqr(vmag)
    feat["PPG_ch_vmag_std"] = float(np.std(vmag))

    # 三通道 DC 一致性
    ch_dc = np.array([float(np.median(g[0])), float(np.median(g[1])), float(np.median(g[2]))])
    ch_dc_mean = np.mean(np.abs(ch_dc))
    feat["PPG_ch_dc_cv"] = float(np.std(ch_dc) / (ch_dc_mean + EPS))
    feat["PPG_ch_dc_max_min_ratio"] = float(np.max(np.abs(ch_dc)) / (np.min(np.abs(ch_dc)) + EPS))

    return feat, imbalance, vmag


def _extract_ppg_ch_consistency(ppg_ch_bp):
    """PPG 3 通道带通信号一致性：互相关和相位延迟。

    ppg_ch_bp: (N, 3) @ 100Hz，3 通道带通 PPG。
    真实脉搏波来自同一源头 → 三通道高度相关且相位一致；
    噪声/回放 → 相关低或相位随机。
    """
    feat = OrderedDict()
    keys = ["PPG_ch_bp_corr_mean", "PPG_ch_bp_corr_min", "PPG_ch_bp_corr_std",
            "PPG_ch_bp_lag_std"]
    ch = np.asarray(ppg_ch_bp, dtype=np.float64)
    if ch.ndim != 2 or ch.shape[1] < 3 or len(ch) < 4:
        for k in keys:
            feat[k] = 0.0
        return feat

    c0 = ch[:, 0] - np.mean(ch[:, 0])
    c1 = ch[:, 1] - np.mean(ch[:, 1])
    c2 = ch[:, 2] - np.mean(ch[:, 2])

    c01 = safe_corr(c0, c1)
    c12 = safe_corr(c1, c2)
    c20 = safe_corr(c2, c0)
    feat["PPG_ch_bp_corr_mean"] = float(np.mean([c01, c12, c20]))
    feat["PPG_ch_bp_corr_min"] = float(np.min([c01, c12, c20]))
    feat["PPG_ch_bp_corr_std"] = float(np.std([c01, c12, c20]))

    # 相位延迟：互相关峰值位置
    N = len(c0)
    xc01 = np.correlate(c0, c1, mode="same")
    xc12 = np.correlate(c1, c2, mode="same")
    lag01 = np.argmax(np.abs(xc01)) - N // 2
    lag12 = np.argmax(np.abs(xc12)) - N // 2
    feat["PPG_ch_bp_lag_std"] = float(np.std([lag01, lag12]))

    return feat


def _extract_ppg_spatial_coupling(ppg_mean_raw, imbalance, vmag, ir_raw=None):
    """PPG 空间特征与均值信号/IR 的耦合度。

    真实佩戴时空间不平衡随血流脉动变化，与均值信号存在耦合；
    伪造物空间模式固定，耦合≈0。
    """
    feat = OrderedDict()
    pm = np.asarray(ppg_mean_raw, dtype=np.float64)
    imb = np.asarray(imbalance, dtype=np.float64)
    vm = np.asarray(vmag, dtype=np.float64)
    n = min(len(pm), len(imb), len(vm))

    feat["PPG_corr_mean_imbalance"] = safe_corr(pm[:n], imb[:n])
    feat["PPG_corr_mean_vmag"] = safe_corr(pm[:n], vm[:n])

    if ir_raw is not None:
        ir = np.asarray(ir_raw, dtype=np.float64)
        n2 = min(len(ir), n)
        feat["PPG_corr_IR_imbalance"] = safe_corr(ir[:n2], imb[:n2])
    else:
        feat["PPG_corr_IR_imbalance"] = 0.0

    return feat


# =========================================================
# EMG 特征提取
# =========================================================

def extract_emg_time_domain_features(emg_bp, emg_env, fs=1000.0, prefix="EMG"):
    """
    EMG 时域特征。
    emg_bp: 带通滤波后的 EMG (N,)
    emg_env: 全波整流包络 (N,)
    """
    _KEYS = ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "IEMG",
             "P2P", "AMP_CV"]
    feat = OrderedDict()
    x = np.asarray(emg_env, dtype=np.float64)
    bp = np.asarray(emg_bp, dtype=np.float64)
    if len(x) < 4:
        for k in _KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    feat[f"{prefix}_MAV"] = float(np.mean(x))
    feat[f"{prefix}_RMS"] = float(np.sqrt(np.mean(bp ** 2)))
    feat[f"{prefix}_VAR"] = float(np.var(bp))
    feat[f"{prefix}_WL"] = float(np.sum(np.abs(np.diff(bp))))
    zc = np.sum(np.abs(np.diff(np.sign(bp)))) / (2.0 * len(bp))
    feat[f"{prefix}_ZC"] = float(zc)
    d1 = np.diff(bp)
    if len(d1) >= 3:
        ssc = np.sum((d1[:-1] * d1[1:]) < 0) / len(bp)
    else:
        ssc = 0.0
    feat[f"{prefix}_SSC"] = float(ssc)
    thr = 0.05 * max(np.max(np.abs(bp)), EPS)
    wamp = np.sum(np.abs(np.diff(bp)) > thr) / len(bp)
    feat[f"{prefix}_WAMP"] = float(wamp)
    feat[f"{prefix}_IEMG"] = float(np.sum(x))

    # P2P: 鲁棒峰峰值 (p95-p05) — 佩戴时动态范围大
    feat[f"{prefix}_P2P"] = float(np.percentile(x, 95) - np.percentile(x, 5))
    # AMP_CV: 包络变异系数，描述 3s 窗整体幅值波动程度
    feat[f"{prefix}_AMP_CV"] = float(np.std(x) / (np.mean(x) + EPS))

    return feat


def _emg_welch_spectrum(x, fs=1000.0, nperseg=512):
    """Welch 功率谱估计，返回 (freqs, Pxx) for 20-450Hz band。"""
    from scipy.signal import welch
    x = np.asarray(x, dtype=np.float64)
    if len(x) < nperseg:
        return None, None
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (f >= 20) & (f <= 450)
    if not np.any(mask) or np.sum(Pxx[mask]) < EPS:
        return None, None
    return f[mask], Pxx[mask]


def extract_emg_frequency_features(emg_bp, fs=1000.0, prefix="EMG"):
    """EMG 频域特征: MNF, MDF, PKF, PSR + 子频段能量占比/比值 + SE95。
    Welch 方法 (nperseg=512) 替代单次 FFT，频谱方差更低。
    """
    _KEYS = ["MNF", "MDF", "PKF", "PSR",
             "POW_20_60", "POW_60_150", "POW_150_450", "POW_LH_RATIO",
             "SE95"] + _EMG_FINE_BAND_KEYS + _EMG_BAND_RATIO_KEYS
    feat = OrderedDict()
    x = np.asarray(emg_bp, dtype=np.float64)
    if len(x) < 16:
        for k in _KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    band_freqs, Pxx = _emg_welch_spectrum(x, fs, nperseg=512)
    if band_freqs is None:
        for k in _KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    total_power = np.sum(Pxx) + EPS

    feat[f"{prefix}_MNF"] = float(np.sum(band_freqs * Pxx) / total_power)
    cumsum = np.cumsum(Pxx)
    mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    feat[f"{prefix}_MDF"] = float(band_freqs[min(mdf_idx, len(band_freqs) - 1)])
    feat[f"{prefix}_PKF"] = float(band_freqs[np.argmax(Pxx)])

    low_100 = (band_freqs >= 20) & (band_freqs <= 100)
    high_100 = (band_freqs > 100) & (band_freqs <= 450)
    feat[f"{prefix}_PSR"] = float(np.sum(Pxx[low_100]) / (np.sum(Pxx[high_100]) + EPS))

    pow_20_60 = np.sum(Pxx[(band_freqs >= 20) & (band_freqs <= 60)])
    feat[f"{prefix}_POW_20_60"] = float(pow_20_60 / total_power)
    pow_60_150 = np.sum(Pxx[(band_freqs > 60) & (band_freqs <= 150)])
    feat[f"{prefix}_POW_60_150"] = float(pow_60_150 / total_power)
    pow_150_450 = np.sum(Pxx[(band_freqs > 150) & (band_freqs <= 450)])
    feat[f"{prefix}_POW_150_450"] = float(pow_150_450 / total_power)
    feat[f"{prefix}_POW_LH_RATIO"] = float(pow_20_60 / (pow_150_450 + EPS))

    def _pow(lo, hi, include_low=True):
        lo_mask = band_freqs >= lo if include_low else band_freqs > lo
        return float(np.sum(Pxx[lo_mask & (band_freqs <= hi)]))

    fine_specs = [
        ("POW_20_40", 20.0, 40.0, True),
        ("POW_40_60", 40.0, 60.0, False),
        ("POW_60_90", 60.0, 90.0, False),
        ("POW_90_120", 90.0, 120.0, False),
        ("POW_120_180", 120.0, 180.0, False),
        ("POW_180_250", 180.0, 250.0, False),
        ("POW_250_350", 250.0, 350.0, False),
        ("POW_350_450", 350.0, 450.0, False),
    ]
    band_cache = {}
    for key, lo, hi, include_low in fine_specs:
        band_cache[(lo, hi, include_low)] = _pow(lo, hi, include_low=include_low)
        feat[f"{prefix}_{key}"] = float(band_cache[(lo, hi, include_low)] / total_power)

    p_60_180 = _pow(60.0, 180.0, include_low=False)
    p_250_450 = _pow(250.0, 450.0, include_low=False)
    p_20_90 = _pow(20.0, 90.0, include_low=True)
    p_180_450 = _pow(180.0, 450.0, include_low=False)
    p_40_120 = _pow(40.0, 120.0, include_low=False)
    p_120_350 = _pow(120.0, 350.0, include_low=False)
    feat[f"{prefix}_RATIO_60_150_TO_20_60"] = float(pow_60_150 / (pow_20_60 + EPS))
    feat[f"{prefix}_RATIO_60_180_TO_250_450"] = float(p_60_180 / (p_250_450 + EPS))
    feat[f"{prefix}_RATIO_20_90_TO_180_450"] = float(p_20_90 / (p_180_450 + EPS))
    feat[f"{prefix}_RATIO_40_120_TO_120_350"] = float(p_40_120 / (p_120_350 + EPS))

    se95_idx = np.searchsorted(cumsum, cumsum[-1] * 0.95)
    feat[f"{prefix}_SE95"] = float(band_freqs[min(se95_idx, len(band_freqs) - 1)])

    return feat


def _emg_band_power(spec_sq, freqs, low, high):
    """从已计算好的 |X(f)|² 中取一个 band 的总能量。"""
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec_sq[mask]))


def extract_emg_mains_features(bp_leak_ref, fs=1000.0, prefix="EMG0"):
    """50Hz 工频拾取特征。Welch (nperseg=1024) 替代单次 FFT。

    bp_leak_ref 是 highpass(20Hz) 后、bandstop/notch 前参考信号（含 50Hz 和 PPG 串扰）。
    主工频检测区间为 49.5-50.5Hz，宽带兜底区间为 40-60Hz，谐波检测区间为 150/250Hz ±0.5Hz。
    注：50HZ_HARM_RATIO 在 150/250Hz 处可能包含 PPG 串扰贡献，
        由新增的 LEAK_* 特征显式分离。
    """
    feat = OrderedDict()
    x = np.asarray(bp_leak_ref, dtype=np.float64)
    if len(x) < 16:
        for k in ["PWR_50HZ", "50HZ_RATIO", "50HZ_HARM_RATIO", "40_60HZ_RATIO"]:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    freqs, Pxx = _emg_welch_spectrum(x, fs, nperseg=1024)
    if freqs is None:
        for k in ["PWR_50HZ", "50HZ_RATIO", "50HZ_HARM_RATIO", "40_60HZ_RATIO"]:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    def _band_sum(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.sum(Pxx[m])) if np.any(m) else 0.0

    total_pow = _band_sum(2.0, 450.0) + EPS
    p_50  = _band_sum(49.5, 50.5)
    p_150 = _band_sum(149.5, 150.5)
    p_250 = _band_sum(249.5, 250.5)
    p_40_60 = _band_sum(40.0, 60.0)

    feat[f"{prefix}_PWR_50HZ"] = float(np.log1p(p_50))
    feat[f"{prefix}_50HZ_RATIO"] = float(p_50 / total_pow)
    feat[f"{prefix}_50HZ_HARM_RATIO"] = float((p_50 + p_150 + p_250) / total_pow)
    feat[f"{prefix}_40_60HZ_RATIO"] = float(p_40_60 / total_pow)
    return feat


def extract_emg_leakage_features(bp_leak_ref, fs=1000.0, prefix="EMG0"):
    """PPG 窄带串扰显式特征。

    bp_leak_ref: highpass(20Hz) 后、bandstop/notch 前参考信号。
    特征：对 _EMG_LEAK_FREQS 中每个频点计算 f0±NOTCH_BW 频段能量占比。
    ratio 分母为 20-450Hz 总能量。

    返回:
        {prefix}_LEAK_{FREQ}_RATIO: 各频点串扰能量占比
        {prefix}_LEAK_SUM_RATIO:    总串扰占比
        {prefix}_LEAK_MAX_RATIO:    最大单频串扰占比
        {prefix}_LEAK_MAX_FREQ:     最大串扰频点 (Hz)
    """
    feat = OrderedDict()
    x = np.asarray(bp_leak_ref, dtype=np.float64)
    if len(x) < 16:
        for k in _EMG_LEAK_KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    freqs, Pxx = _emg_welch_spectrum(x, fs, nperseg=512)
    if freqs is None:
        for k in _EMG_LEAK_KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    total_pow = np.sum(Pxx) + EPS
    bw = _EMG_NOTCH_BW_HZ

    ratios = []
    for f0 in _EMG_LEAK_FREQS:
        lo = f0 - bw
        hi = f0 + bw
        mask = (freqs >= lo) & (freqs <= hi)
        p_band = float(np.sum(Pxx[mask])) if np.any(mask) else 0.0
        ratio = p_band / total_pow
        feat[f"{prefix}_LEAK_{int(f0)}_RATIO"] = float(ratio)
        ratios.append(ratio)

    if ratios:
        feat[f"{prefix}_LEAK_SUM_RATIO"] = float(np.sum(ratios))
        max_idx = int(np.argmax(ratios))
        feat[f"{prefix}_LEAK_MAX_RATIO"] = float(ratios[max_idx])
        feat[f"{prefix}_LEAK_MAX_FREQ"] = float(_EMG_LEAK_FREQS[max_idx])
    else:
        feat[f"{prefix}_LEAK_SUM_RATIO"] = 0.0
        feat[f"{prefix}_LEAK_MAX_RATIO"] = 0.0
        feat[f"{prefix}_LEAK_MAX_FREQ"] = 0.0

    return feat


def extract_emg_baseline_drift(x_demean, bp_leak_ref, fs=1000.0, prefix="EMG0"):
    """EMG 接触噪声底：基于原始去均值信号的低频 1-10Hz baseline drift。

    入参:
      x_demean:    去均值但未做高通/陷波的原始 EMG（来自 preprocess_emg_signal_with_raw）
      bp_leak_ref: highpass(20Hz) 后、bandstop/notch 前参考信号，用于 HF 参考能量

    返回:
      {prefix}_BASELINE_DRIFT_POW: log1p(P(1-10Hz)) 绝对量
      {prefix}_DRIFT_HF_RATIO:     P(1-10) / P(20-450) — 真接触时偏高（皮肤位移驱动）
    """
    feat = OrderedDict()
    x = np.asarray(x_demean, dtype=np.float64)
    if len(x) < 16:
        feat[f"{prefix}_BASELINE_DRIFT_POW"] = 0.0
        feat[f"{prefix}_DRIFT_HF_RATIO"] = 0.0
        return feat

    try:
        lf = bandpass_filter(x, fs, lowcut=1.0, highcut=10.0, order=2)
        p_lf = float(np.mean(lf * lf))
    except Exception:
        p_lf = 0.0

    bp = np.asarray(bp_leak_ref, dtype=np.float64)
    p_hf = float(np.mean(bp * bp)) + EPS

    feat[f"{prefix}_BASELINE_DRIFT_POW"] = float(np.log1p(p_lf))
    feat[f"{prefix}_DRIFT_HF_RATIO"] = float(p_lf / p_hf)
    return feat


_EMG_ANTI_SPOOF_KEYS = ["PWR_50HZ", "50HZ_RATIO", "50HZ_HARM_RATIO", "40_60HZ_RATIO",
                        "BASELINE_DRIFT_POW", "DRIFT_HF_RATIO"]

_EMG_LEAK_KEYS = ["LEAK_100_RATIO", "LEAK_150_RATIO", "LEAK_200_RATIO",
                   "LEAK_250_RATIO", "LEAK_300_RATIO",
                   "LEAK_SUM_RATIO", "LEAK_MAX_RATIO", "LEAK_MAX_FREQ"]

_EMG_FINE_BAND_KEYS = [
    "POW_20_40", "POW_40_60", "POW_60_90", "POW_90_120",
    "POW_120_180", "POW_180_250", "POW_250_350", "POW_350_450",
]
_EMG_BAND_RATIO_KEYS = [
    "RATIO_60_150_TO_20_60",
    "RATIO_60_180_TO_250_450",
    "RATIO_20_90_TO_180_450",
    "RATIO_40_120_TO_120_350",
]
_EMG_SUBWIN_KEYS = ["RMS_SUBWIN_CV", "MDF_SUBWIN_IQR", "WL_SUBWIN_CV"]
_EMG_SPEC_SHAPE_KEYS = ["SPEC_ENTROPY", "SPEC_FLATNESS", "SPEC_CENTROID", "SPEC_ROLLOFF_85"]


def _emg_downsample_for_sampen(x, fs=1000.0):
    """降采样 EMG 到 250Hz 供 SampEn 使用。1000→250Hz 减少 4× 采样点，SampEn 加速 ~16×。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 50:
        return None
    if fs > 250 and len(x) > 250:
        gcd = np.gcd(int(fs), 250)
        up = 250 // gcd
        down = int(fs) // gcd
        x = resample_poly(x.astype(np.float32, copy=False), up, down).astype(np.float64)
    if len(x) > 1200:
        start = (len(x) - 1200) // 2
        x = x[start:start + 1200]
    return x


def _emg_sample_entropy(x):
    """EMG 样本熵：分块计算防内存溢出。输入应为已降采样到 ~250Hz 的信号。"""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 50:
        return 0.0

    r = 0.2 * np.std(x)
    if r < EPS:
        return 0.0

    def _count_matches(m_val, chunk=500):
        patterns = np.array([x[i:i + m_val] for i in range(N - m_val)], dtype=np.float64)
        n_pat = len(patterns)
        if n_pat < 2:
            return 0.0
        total = 0.0
        for i in range(0, n_pat, chunk):
            end = min(i + chunk, n_pat)
            chunk_pat = patterns[i:end]
            d = np.max(np.abs(chunk_pat[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
            total += np.sum(d <= r)
            for j in range(len(chunk_pat)):
                idx = i + j
                if idx < n_pat and d[j, idx] <= r:
                    total -= 1.0
        return total

    B = _count_matches(2)
    A = _count_matches(3)
    if B > EPS and A > EPS:
        sampen = float(-np.log(A / B))
        return sampen if np.isfinite(sampen) else 0.0
    return 0.0


def extract_emg_subwindow_features(emg_bp, emg_env, fs=1000.0, prefix="EMG"):
    feat = OrderedDict()
    bp = np.asarray(emg_bp, dtype=np.float64)
    if len(bp) < int(fs):
        for k in _EMG_SUBWIN_KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    win = max(16, int(round(fs)))
    rms_vals = []
    wl_vals = []
    mdf_vals = []
    for start in range(0, len(bp) - win + 1, win):
        seg = bp[start:start + win]
        if len(seg) < win:
            continue
        rms_vals.append(float(np.sqrt(np.mean(seg * seg))))
        wl_vals.append(float(np.sum(np.abs(np.diff(seg)))))
        freq_feat = extract_emg_frequency_features(seg, fs=fs, prefix="_TMP")
        mdf_vals.append(float(freq_feat["_TMP_MDF"]))

    if len(rms_vals) < 2:
        for k in _EMG_SUBWIN_KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    wl_arr = np.asarray(wl_vals, dtype=np.float64)
    mdf_arr = np.asarray(mdf_vals, dtype=np.float64)
    feat[f"{prefix}_RMS_SUBWIN_CV"] = float(np.std(rms_arr) / (np.mean(rms_arr) + EPS))
    feat[f"{prefix}_MDF_SUBWIN_IQR"] = robust_iqr(mdf_arr)
    feat[f"{prefix}_WL_SUBWIN_CV"] = float(np.std(wl_arr) / (np.mean(wl_arr) + EPS))
    return feat


def extract_emg_spectral_shape_features(emg_bp, fs=1000.0, prefix="EMG"):
    feat = OrderedDict()
    freqs, pxx = _emg_welch_spectrum(emg_bp, fs, nperseg=512)
    if freqs is None:
        for k in _EMG_SPEC_SHAPE_KEYS:
            feat[f"{prefix}_{k}"] = 0.0
        return feat

    total = float(np.sum(pxx)) + EPS
    p = np.asarray(pxx, dtype=np.float64) / total
    entropy = -float(np.sum(p * np.log(p + EPS)) / np.log(len(p) + EPS))
    flatness = float(np.exp(np.mean(np.log(pxx + EPS))) / (np.mean(pxx) + EPS))
    centroid = float(np.sum(freqs * pxx) / total)
    cumsum = np.cumsum(pxx)
    roll_idx = np.searchsorted(cumsum, cumsum[-1] * 0.85)
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])

    feat[f"{prefix}_SPEC_ENTROPY"] = entropy
    feat[f"{prefix}_SPEC_FLATNESS"] = flatness
    feat[f"{prefix}_SPEC_CENTROID"] = centroid
    feat[f"{prefix}_SPEC_ROLLOFF_85"] = rolloff
    return feat


def extract_emg_channel_balance_features(ch0_env, ch1_env, ch0_bp, ch1_bp, fs=1000.0):
    feat = OrderedDict()
    if ch0_env is None or ch1_env is None or ch0_bp is None or ch1_bp is None:
        for k in ["EMG_ENV_CORR", "EMG_MAV_RATIO", "EMG_CONTACT_IMBALANCE"]:
            feat[k] = 0.0
        return feat

    n_env = min(len(ch0_env), len(ch1_env))
    n_bp = min(len(ch0_bp), len(ch1_bp))
    if n_env < 4 or n_bp < 4:
        for k in ["EMG_ENV_CORR", "EMG_MAV_RATIO", "EMG_CONTACT_IMBALANCE"]:
            feat[k] = 0.0
        return feat

    env0 = np.asarray(ch0_env[:n_env], dtype=np.float64)
    env1 = np.asarray(ch1_env[:n_env], dtype=np.float64)
    mav0 = float(np.mean(env0))
    mav1 = float(np.mean(env1))
    feat["EMG_ENV_CORR"] = safe_corr(env0, env1, winsorize=True)
    feat["EMG_MAV_RATIO"] = float(np.clip(safe_div(mav0, mav1), 0.0, 1000.0)) if mav1 > EPS else 0.0
    feat["EMG_CONTACT_IMBALANCE"] = float(abs(mav0 - mav1) / (mav0 + mav1 + EPS))
    return feat


def extract_emg_features(emg_window, fs=1000.0, return_signals=False):
    """提取双通道 EMG 的完整特征集（含窄带串扰特征）。emg_window: (N, 2) @ 1000Hz

    return_signals=True 时额外返回 ch0_env，避免调用方重复预处理。
    """
    feat = OrderedDict()
    ch0_env_for_cross = None
    if emg_window is None or len(emg_window) < 4:
        for ch in [0, 1]:
            for k in ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "IEMG",
                       "P2P", "AMP_CV",
                       "MNF", "MDF", "PKF", "PSR",
                       "POW_20_60", "POW_60_150", "POW_150_450", "POW_LH_RATIO",
                       "SE95",
                       "SampEn", "SKEWNESS", "KURTOSIS", "SNR"] + _EMG_FINE_BAND_KEYS + _EMG_BAND_RATIO_KEYS + _EMG_ANTI_SPOOF_KEYS + _EMG_LEAK_KEYS + _EMG_SUBWIN_KEYS + _EMG_SPEC_SHAPE_KEYS:
                feat[f"EMG{ch}_{k}"] = 0.0
        feat["EMG_CROSS_CORR"] = 0.0
        feat["EMG_RMS_RATIO"] = 0.0
        feat.update(extract_emg_channel_balance_features(None, None, None, None, fs))
        return (feat, None) if return_signals else feat

    emg = np.asarray(emg_window, dtype=np.float64)
    if emg.ndim == 1:
        emg = emg.reshape(-1, 1)

    # 5 值返回: (bp_leak_ref, bp_clean, env, x_demean)
    ch0_leak, ch0_bp, ch0_env, ch0_demean = preprocess_emg_signal_with_raw(emg[:, 0], fs)
    ch0_env_for_cross = ch0_env  # 缓存，避免调用方重复预处理
    ch1_leak = ch1_bp = ch1_env = ch1_demean = None
    if emg.shape[1] >= 2:
        ch1_leak, ch1_bp, ch1_env, ch1_demean = preprocess_emg_signal_with_raw(emg[:, 1], fs)

    # SampEn: 提前降采样到 250Hz，避免函数内部重复计算 gcd+resample
    emg0_ds = _emg_downsample_for_sampen(ch0_bp, fs)

    feat.update(extract_emg_time_domain_features(ch0_bp, ch0_env, fs, "EMG0"))
    feat.update(extract_emg_frequency_features(ch0_bp, fs, "EMG0"))
    feat.update(extract_emg_mains_features(ch0_leak, fs, "EMG0"))
    feat.update(extract_emg_leakage_features(ch0_leak, fs, "EMG0"))
    feat.update(extract_emg_baseline_drift(ch0_demean, ch0_leak, fs, "EMG0"))
    feat.update(extract_emg_subwindow_features(ch0_bp, ch0_env, fs, "EMG0"))
    feat.update(extract_emg_spectral_shape_features(ch0_bp, fs, "EMG0"))
    feat["EMG0_SampEn"] = _emg_sample_entropy(emg0_ds) if emg0_ds is not None else 0.0

    if ch1_bp is not None:
        emg1_ds = _emg_downsample_for_sampen(ch1_bp, fs)
        feat.update(extract_emg_time_domain_features(ch1_bp, ch1_env, fs, "EMG1"))
        feat.update(extract_emg_frequency_features(ch1_bp, fs, "EMG1"))
        feat.update(extract_emg_mains_features(ch1_leak, fs, "EMG1"))
        feat.update(extract_emg_leakage_features(ch1_leak, fs, "EMG1"))
        feat.update(extract_emg_baseline_drift(ch1_demean, ch1_leak, fs, "EMG1"))
        feat.update(extract_emg_subwindow_features(ch1_bp, ch1_env, fs, "EMG1"))
        feat.update(extract_emg_spectral_shape_features(ch1_bp, fs, "EMG1"))
        feat["EMG1_SampEn"] = _emg_sample_entropy(emg1_ds) if emg1_ds is not None else 0.0

        n = min(len(ch0_bp), len(ch1_bp))
        feat["EMG_CROSS_CORR"] = safe_corr(ch0_bp[:n], ch1_bp[:n], winsorize=True)
        rms0 = float(np.sqrt(np.mean(ch0_bp ** 2)))
        rms1 = float(np.sqrt(np.mean(ch1_bp ** 2)))
        # rms 比值：rms1 接近 0 时返回 0 而非爆炸到 1e12；正常段做上限 clip
        if rms1 > EPS:
            feat["EMG_RMS_RATIO"] = float(np.clip(rms0 / rms1, 0.0, 1000.0))
        else:
            feat["EMG_RMS_RATIO"] = 0.0

        # 通道分布特征 ch1
        feat["EMG1_SKEWNESS"] = float(np.mean((ch1_bp - np.mean(ch1_bp)) ** 3) / (np.std(ch1_bp) ** 3 + EPS))
        feat["EMG1_KURTOSIS"] = float(np.mean((ch1_bp - np.mean(ch1_bp)) ** 4) / (np.std(ch1_bp) ** 4 + EPS))
        mav1 = feat.get("EMG1_MAV", 0.0)
        feat["EMG1_SNR"] = float(rms1 / mav1) if mav1 > EPS else 0.0
        feat.update(extract_emg_channel_balance_features(ch0_env, ch1_env, ch0_bp, ch1_bp, fs))
    else:
        for k in ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "IEMG",
                   "P2P", "AMP_CV",
                   "MNF", "MDF", "PKF", "PSR",
                   "POW_20_60", "POW_60_150", "POW_150_450", "POW_LH_RATIO",
                   "SE95",
                   "SampEn", "SKEWNESS", "KURTOSIS", "SNR"] + _EMG_FINE_BAND_KEYS + _EMG_BAND_RATIO_KEYS + _EMG_ANTI_SPOOF_KEYS + _EMG_LEAK_KEYS + _EMG_SUBWIN_KEYS + _EMG_SPEC_SHAPE_KEYS:
            feat[f"EMG1_{k}"] = 0.0
        feat["EMG_CROSS_CORR"] = 0.0
        feat["EMG_RMS_RATIO"] = 0.0
        feat.update(extract_emg_channel_balance_features(None, None, None, None, fs))

    # 通道分布特征 ch0（始终计算，与 ch1 是否存在无关）
    rms0 = float(np.sqrt(np.mean(ch0_bp ** 2)))
    feat["EMG0_SKEWNESS"] = float(np.mean((ch0_bp - np.mean(ch0_bp)) ** 3) / (np.std(ch0_bp) ** 3 + EPS))
    feat["EMG0_KURTOSIS"] = float(np.mean((ch0_bp - np.mean(ch0_bp)) ** 4) / (np.std(ch0_bp) ** 4 + EPS))
    mav0 = feat.get("EMG0_MAV", 0.0)
    feat["EMG0_SNR"] = float(rms0 / mav0) if mav0 > EPS else 0.0

    return (feat, ch0_env_for_cross) if return_signals else feat


# =========================================================
# ACC 特征
# =========================================================

def _acc_magnitude(acc_window):
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    return np.sqrt(np.sum(acc * acc, axis=1) + 1e-12)


def _acc_robust_clean(acc_window):
    """每轴独立做 remove_burr，抹掉敲击/碰撞瞬变。

    输入 (N, 3) 或 (N,)，输出同 shape。
    """
    if acc_window is None:
        return acc_window
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1 or len(acc) < 3:
        return acc.copy()
    out = acc.copy()
    for ax in range(out.shape[1]):
        out[:, ax] = remove_burr(out[:, ax], burr_k=_ACC_BURR_K)
    return out


def extract_acc_features(acc_window, fs=100.0, prefix="ACC"):
    """ACC 特征：低通分离重力，带通提取运动，两路独立计算。"""
    feats = OrderedDict()
    if acc_window is None or len(acc_window) < 4:
        for k in ["GRAV_MAG_MEAN", "GRAV_DOM_RATIO",
                   "MOTION_RMS", "MOTION_STD", "MOTION_MAD",
                   "AXIS_STD_SUM", "DIFF_MAD", "STILL_SCORE",
                   "MAG_P50", "MAG_P90"]:
            feats[f"{prefix}_{k}"] = 0.0
        return feats

    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    n_axes = acc.shape[1]

    # 重力分量：每轴低通 <0.5Hz（去均值后低通 + 原始均值 = 重力投影）
    axis_grav = []
    for ax in range(n_axes):
        ax_raw = acc[:, ax]
        ax_mean = np.mean(ax_raw)
        ax_centered = ax_raw - ax_mean
        try:
            ax_lp = bandpass_filter(ax_centered, fs, lowcut=0.1, highcut=0.5, order=2)
        except Exception:
            ax_lp = np.zeros_like(ax_centered)
        axis_grav.append(ax_lp + ax_mean)  # 恢复 DC 分量
    acc_grav = np.column_stack(axis_grav)
    grav_mag = _acc_magnitude(acc_grav)

    # 运动分量：带通 0.5–15Hz（去掉重力和高频电子噪声）
    acc_motion = acc - acc_grav
    motion_mag = _acc_magnitude(acc_motion)

    # 重力特征
    grav_mean_abs = np.abs(np.mean(acc_grav, axis=0))
    feats[f"{prefix}_GRAV_MAG_MEAN"] = float(np.mean(grav_mag))
    feats[f"{prefix}_GRAV_DOM_RATIO"] = float(
        np.max(grav_mean_abs) / (np.sum(grav_mean_abs) + 1e-8))

    # 运动特征
    motion_rms = float(np.sqrt(np.mean(motion_mag ** 2)))
    motion_std = float(np.std(motion_mag))
    motion_mad = robust_mad(motion_mag)
    motion_diff_mad = robust_mad(np.diff(motion_mag)) if len(motion_mag) > 1 else 0.0
    feats[f"{prefix}_MOTION_RMS"] = motion_rms
    feats[f"{prefix}_MOTION_STD"] = motion_std
    feats[f"{prefix}_MOTION_MAD"] = motion_mad
    feats[f"{prefix}_DIFF_MAD"] = motion_diff_mad

    # 跨轴 + 静止分数
    feats[f"{prefix}_AXIS_STD_SUM"] = float(np.sum(np.std(acc, axis=0)))
    rel_std = motion_std / (abs(float(np.mean(motion_mag))) + 1e-6)
    feats[f"{prefix}_STILL_SCORE"] = float(1.0 / (1.0 + 50.0 * rel_std))

    # 原始 mag 的分位数（保持对整体幅值的监控）
    raw_mag = _acc_magnitude(acc)
    feats[f"{prefix}_MAG_P50"] = float(np.percentile(raw_mag, 50))
    feats[f"{prefix}_MAG_P90"] = float(np.percentile(raw_mag, 90))
    return feats


def extract_acc_cross_features(acc_window, ppg_bp, emg_env, fs_ppg=100.0, fs_emg=1000.0):
    """ACC 与 PPG / EMG 的跨模态特征。"""
    feats = OrderedDict()
    if acc_window is None or len(acc_window) < 4:
        feats["ACC_PPG_BP_CORR"] = 0.0
        feats["ACC_EMG_CORR"] = 0.0
        return feats

    mag = _acc_magnitude(acc_window)
    mag_centered = mag - np.mean(mag)

    try:
        mag_bp = bandpass_filter(mag_centered, fs_ppg, lowcut=0.5, highcut=5.0, order=2)
    except Exception:
        mag_bp = mag_centered

    if ppg_bp is not None:
        n = min(len(mag_bp), len(ppg_bp))
        if n >= 8:
            feats["ACC_PPG_BP_CORR"] = abs(safe_corr(mag_bp[:n], ppg_bp[:n], winsorize=True))
        else:
            feats["ACC_PPG_BP_CORR"] = 0.0
    else:
        feats["ACC_PPG_BP_CORR"] = 0.0

    if emg_env is not None:
        try:
            emg_ds = resample_poly(emg_env, fs_ppg, fs_emg)
            n = min(len(mag), len(emg_ds))
            if n >= 8:
                feats["ACC_EMG_CORR"] = abs(safe_corr(mag[:n], emg_ds[:n], winsorize=True))
            else:
                feats["ACC_EMG_CORR"] = 0.0
        except Exception:
            feats["ACC_EMG_CORR"] = 0.0
    else:
        feats["ACC_EMG_CORR"] = 0.0

    return feats


# =========================================================
# EMG-PPG 跨模态特征
# =========================================================

def extract_emg_ppg_cross_features(emg_env, ppg_bp, fs_emg=1000.0, fs_ppg=100.0):
    """EMG 包络与 PPG 带通信号的跨模态相关。"""
    feats = OrderedDict()
    if emg_env is None or ppg_bp is None or len(emg_env) < 4 or len(ppg_bp) < 4:
        for k in ["CORR", "ENV_CORR"]:
            feats[f"EMG_PPG_{k}"] = 0.0
        return feats

    try:
        emg_ds = resample_poly(emg_env, fs_ppg, fs_emg)
        n = min(len(emg_ds), len(ppg_bp))
        if n >= 8:
            feats["EMG_PPG_CORR"] = abs(safe_corr(emg_ds[:n], ppg_bp[:n], winsorize=True))
        else:
            feats["EMG_PPG_CORR"] = 0.0
    except Exception:
        feats["EMG_PPG_CORR"] = 0.0

    try:
        emg_env_smooth = smooth_envelope(emg_env, fs_emg, win_sec=0.25)
        emg_env_ds = resample_poly(emg_env_smooth, fs_ppg, fs_emg)
        ppg_env = smooth_envelope(ppg_bp, fs_ppg, win_sec=0.25)
        n = min(len(emg_env_ds), len(ppg_env))
        if n >= 8:
            feats["EMG_PPG_ENV_CORR"] = abs(safe_corr(emg_env_ds[:n], ppg_env[:n], winsorize=True))
        else:
            feats["EMG_PPG_ENV_CORR"] = 0.0
    except Exception:
        feats["EMG_PPG_ENV_CORR"] = 0.0

    return feats


# =========================================================
# 单窗口特征提取主函数
# =========================================================

# =========================================================
# 新增：ACC tremor / PPG PI / pulse morphology / HRV / coherence
# =========================================================

def extract_acc_tremor_features(acc_window, fs=100.0):
    """ACC 8-12Hz 生理震颤带能量比。

    静止佩戴时存在低能量生理震颤；放桌上是宽带电子噪声，比值很小。
    """
    feat = OrderedDict()
    if acc_window is None or len(acc_window) < 16:
        feat["ACC_TREMOR_POW_8_12"] = 0.0
        feat["ACC_TREMOR_RATIO"] = 0.0
        return feat

    mag = _acc_magnitude(acc_window)
    x = mag - np.mean(mag)
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    nfft = max(256, nfft)
    spec = np.abs(np.fft.rfft(x * np.hamming(len(x)), n=nfft))
    spec_sq = spec * spec
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    p_tremor = float(np.sum(spec_sq[(freqs >= 8.0) & (freqs <= 12.0)]))
    p_total = float(np.sum(spec_sq[(freqs >= 0.5) & (freqs <= 15.0)])) + EPS

    feat["ACC_TREMOR_POW_8_12"] = float(np.log1p(p_tremor))
    feat["ACC_TREMOR_RATIO"] = float(p_tremor / p_total)
    return feat


def extract_ppg_pi_features(ppg_raw, ppg_bp, fs=100.0):
    """Perfusion Index 及其窗内 1s 子窗稳定性。"""
    feat = OrderedDict()
    raw = np.asarray(ppg_raw, dtype=np.float64)
    bp = np.asarray(ppg_bp, dtype=np.float64)
    if len(raw) < int(fs) or len(bp) < int(fs):
        feat["PPG_PI"] = 0.0
        feat["PPG_PI_SUBWIN_IQR"] = 0.0
        return feat

    ac_rms = float(np.sqrt(np.mean(bp ** 2)))
    dc = float(np.median(raw))
    pi = safe_div(ac_rms, abs(dc) + EPS)
    feat["PPG_PI"] = float(pi)

    # 1s 子窗 PI 的 IQR
    sub_n = int(fs)
    pis = []
    for i in range(0, len(raw) - sub_n + 1, sub_n):
        sub_raw = raw[i:i + sub_n]
        sub_bp = bp[i:i + sub_n]
        if len(sub_raw) < sub_n:
            continue
        sub_ac = float(np.sqrt(np.mean(sub_bp ** 2)))
        sub_dc = float(np.median(sub_raw))
        pis.append(sub_ac / (abs(sub_dc) + EPS))
    feat["PPG_PI_SUBWIN_IQR"] = robust_iqr(np.asarray(pis)) if len(pis) >= 2 else 0.0
    return feat


def _detect_ppg_peaks(ppg_bp, fs):
    """单次 PPG 收缩峰检测，供形态学和 HRV 共用。

    返回 peaks 索引。心率 40-180 BPM → 峰间最小距离 ~0.33s。
    """
    from scipy.signal import find_peaks
    x = np.asarray(ppg_bp, dtype=np.float64)
    if len(x) < int(fs):
        return np.array([], dtype=int)
    min_dist = max(1, int(0.33 * fs))
    prom = max(np.std(x) * 0.3, EPS)
    try:
        peaks, _ = find_peaks(x, distance=min_dist, prominence=prom)
    except Exception:
        peaks = np.array([], dtype=int)
    return peaks


def extract_ppg_pulse_morphology(ppg_bp, peaks, fs=100.0):
    """脉搏波形态学：dicrotic notch 出现率、AI、脉宽变异。

    3s 窗约 3-5 拍，统计很噪。这些特征主要用于防橡胶模拟物 — 单纯机械泵
    通常波形单峰、形态高度一致，缺 dicrotic notch。
    """
    feat = OrderedDict()
    keys = ["PPG_DICROTIC_RATIO", "PPG_AUG_INDEX_MEAN", "PPG_PULSE_WIDTH_CV"]
    for k in keys:
        feat[k] = 0.0

    x = np.asarray(ppg_bp, dtype=np.float64)
    if len(x) < int(fs) or len(peaks) < 2:
        return feat

    # 用相邻峰之间的段做形态分析
    dicrotic_hits = 0
    aug_indices = []
    pulse_widths = []
    for i in range(len(peaks) - 1):
        p0 = int(peaks[i])
        p1 = int(peaks[i + 1])
        seg = x[p0:p1]
        if len(seg) < int(0.2 * fs):
            continue
        pulse_widths.append(len(seg) / fs)

        # 找次级峰（dicrotic notch 后的反射波）：在主峰后 30%-70% 段内有次级局部极大
        s_start = int(0.25 * len(seg))
        s_end = int(0.75 * len(seg))
        if s_end - s_start < 3:
            continue
        sub = seg[s_start:s_end]
        if len(sub) < 3:
            continue
        # 次级峰 = sub 内最大值且大于段平均 + 0.2 * 段标准差
        sub_peak = float(np.max(sub))
        sub_mean = float(np.mean(seg))
        sub_std = float(np.std(seg)) + EPS
        main_peak = float(np.max(seg[:s_start])) if s_start > 0 else float(seg[0])
        if sub_peak > sub_mean + 0.2 * sub_std and main_peak > EPS:
            dicrotic_hits += 1
            aug_indices.append(sub_peak / main_peak)

    n_segments = max(1, len(peaks) - 1)
    feat["PPG_DICROTIC_RATIO"] = float(dicrotic_hits / n_segments)
    feat["PPG_AUG_INDEX_MEAN"] = float(np.mean(aug_indices)) if aug_indices else 0.0
    if len(pulse_widths) >= 2:
        m = float(np.mean(pulse_widths))
        feat["PPG_PULSE_WIDTH_CV"] = float(np.std(pulse_widths) / (m + EPS))
    return feat


def extract_ppg_hrv_short(peaks, fs=100.0):
    """短窗 HRV：3-5 拍 RR 统计。信噪比差，作为防伪占位特征。"""
    feat = OrderedDict()
    keys = ["PPG_RR_RMSSD", "PPG_RR_CV", "PPG_RR_PNN30"]
    for k in keys:
        feat[k] = 0.0

    if len(peaks) < 3:
        return feat

    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / fs  # 秒
    if len(rr) < 2:
        return feat

    drr = np.diff(rr)
    feat["PPG_RR_RMSSD"] = float(np.sqrt(np.mean(drr * drr)))
    rr_mean = float(np.mean(rr))
    feat["PPG_RR_CV"] = float(np.std(rr) / (rr_mean + EPS))
    feat["PPG_RR_PNN30"] = float(np.mean(np.abs(drr) > 0.030))
    return feat


def extract_acc_ppg_coherence(acc_window, ppg_bp, fs_acc=100.0, fs_ppg=100.0):
    """ACC-PPG 频域相干性，替代单一时域相关。

    COH_MICRO (0.5-3Hz): 微动→PPG 伪影耦合；真佩戴有，回放为 0。
    COH_HR    (HR 带): 反映运动伪影污染心率信号；干净佩戴应低。
    """
    feat = OrderedDict()
    feat["ACC_PPG_COH_MICRO"] = 0.0
    feat["ACC_PPG_COH_HR"] = 0.0

    if acc_window is None or ppg_bp is None:
        return feat

    mag = _acc_magnitude(acc_window)
    mag = mag - np.mean(mag)
    bp = np.asarray(ppg_bp, dtype=np.float64)

    # 重采样到相同采样率（如果不一致），用 PPG fs 作为目标
    if abs(fs_acc - fs_ppg) > 1e-3:
        try:
            from scipy.signal import resample_poly as _rp
            gcd = np.gcd(int(round(fs_acc)), int(round(fs_ppg)))
            up = int(round(fs_ppg)) // gcd
            down = int(round(fs_acc)) // gcd
            mag = _rp(mag, up, down)
        except Exception:
            return feat

    n = min(len(mag), len(bp))
    if n < int(2 * fs_ppg):
        return feat
    mag = mag[:n]
    bp = bp[:n]

    try:
        from scipy.signal import coherence
        nperseg = min(n, int(2 * fs_ppg))
        f, Cxy = coherence(mag, bp, fs=fs_ppg, nperseg=nperseg)
    except Exception:
        return feat

    mask_micro = (f >= 0.5) & (f <= 3.0)
    mask_hr = (f >= 0.8) & (f <= 3.0)  # HR 带 48-180 BPM
    if np.any(mask_micro):
        feat["ACC_PPG_COH_MICRO"] = float(np.mean(Cxy[mask_micro]))
    if np.any(mask_hr):
        feat["ACC_PPG_COH_HR"] = float(np.mean(Cxy[mask_hr]))
    return feat


def extract_feature_pool_from_window(ppg_signal, emg_window, acc_window,
                                      fs_ppg=100.0, fs_emg=1000.0, fs_acc=100.0,
                                      return_preprocessed=False):
    """
    输入：
        ppg_signal: 3 通道 PPG @ 100Hz, shape (N, 3)
        emg_window: 2-ch EMG @ 1000Hz, shape (M, 2) or None
        acc_window: 3-ch ACC @ 100Hz, shape (K, 3) or None

    输出：
        OrderedDict 特征
    """
    feat = OrderedDict()

    ppg = np.asarray(ppg_signal, dtype=np.float64)
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)

    n = len(ppg)
    if n < int(1.0 * fs_ppg):
        raise ValueError(f"窗口太短: n={n}, 需要 >= {int(fs_ppg)}")

    # ===== PPG 预处理：三通道独立预处理，取均值做单通道特征 =====
    n_ch = ppg.shape[1]
    if n_ch >= 3:
        raw_list, bp_list = [], []
        for ch_idx in range(3):
            ch_raw, ch_bp, _ = preprocess_signal(ppg[:, ch_idx], fs_ppg)
            raw_list.append(ch_raw)
            bp_list.append(ch_bp)
        ppg_ch_raw = np.column_stack(raw_list)  # (N, 3)
        ppg_ch_bp = np.column_stack(bp_list)    # (N, 3)
    else:
        ppg_ch_raw, ppg_ch_bp, _ = preprocess_signal(ppg[:, 0], fs_ppg)
        ppg_ch_raw = ppg_ch_raw.reshape(-1, 1)
        ppg_ch_bp = ppg_ch_bp.reshape(-1, 1)

    # 三通道均值用于单通道特征
    ppg_raw = np.mean(ppg_ch_raw, axis=1)
    ppg_bp = np.mean(ppg_ch_bp, axis=1)
    ppg_dc = float(np.median(ppg_raw))
    ir_raw = ppg_ch_raw[:, 0]  # ch_A 原始信号用于耦合特征

    # ===== FFT 缓存 (使用均值 bp) =====
    fft_cache = compute_fft_cache(ppg_bp, fs_ppg, fmin=0.5, fmax=5.0)

    # ===== 基础长度 =====
    feat["SIG_LEN"] = float(n)
    feat["SIG_SEC"] = float(n / fs_ppg)

    # ===== A. PPG 基础统计 (基于 ch_A 原始) =====
    feat["PPG_mean"] = float(np.mean(ir_raw))
    feat["PPG_std"] = float(np.std(ir_raw))
    feat["PPG_p95"] = float(np.percentile(ir_raw, 95))
    feat["PPG_diff_std"] = float(np.std(np.diff(ir_raw)))
    feat["PPG_acdc"] = safe_div(np.sqrt(np.mean(ppg_bp ** 2)), abs(ppg_dc) + EPS)

    # ===== B. PPG DC/AC 特征 =====
    feat.update(extract_ppg_features(ppg_raw, ppg_bp, ppg_dc, fs_ppg, "PPG", fft_cache))

    # ===== C. PPG BP 波形形态 =====
    _m = float(np.mean(ppg_bp))
    _s = float(np.std(ppg_bp))
    if _s > EPS:
        feat["PPG_bp_skewness"] = float(np.mean((ppg_bp - _m) ** 3) / (_s ** 3))
        feat["PPG_bp_kurtosis"] = float(np.mean((ppg_bp - _m) ** 4) / (_s ** 4))
    else:
        feat["PPG_bp_skewness"] = 0.0
        feat["PPG_bp_kurtosis"] = 0.0

    # ===== D. PPG FFT 峰值宽度 + SNR =====
    if fft_cache.get('band_spec') is not None and len(fft_cache.get('band_spec', [])) > 0:
        _bs = fft_cache['band_spec']
        _bf = fft_cache['band_freqs']
        _peak_val = np.max(_bs)
        _above_half = _bs > _peak_val * 0.5
        feat["PPG_FFT_peak_width_Hz"] = float(_bf[_above_half][-1] - _bf[_above_half][0]) if np.any(_above_half) else 0.0
        _in_band = float(np.sum(_bs ** 2))
        _out_band = float(np.sum(fft_cache['spec'] ** 2)) - _in_band
        feat["PPG_FFT_SNR"] = float(_in_band / (_out_band + EPS))
    else:
        feat["PPG_FFT_peak_width_Hz"] = 0.0
        feat["PPG_FFT_SNR"] = 0.0

    # ===== E. PPG Hjorth / Entropy / Derivative / Temporal =====
    feat.update(extract_hjorth_parameters(ppg_bp, prefix="PPG"))
    feat.update(extract_entropy_features(ppg_bp, prefix="PPG"))
    feat.update(extract_derivative_features(ppg_bp, fs_ppg, prefix="PPG"))
    feat.update(extract_temporal_dynamic_features(ppg_bp, fs_ppg, prefix="PPG"))

    # ===== E2. PPG Perfusion Index + 脉搏形态学 + 短窗 HRV =====
    feat.update(extract_ppg_pi_features(ppg_raw, ppg_bp, fs_ppg))
    ppg_peaks = _detect_ppg_peaks(ppg_bp, fs_ppg)
    feat.update(extract_ppg_pulse_morphology(ppg_bp, ppg_peaks, fs_ppg))
    feat.update(extract_ppg_hrv_short(ppg_peaks, fs_ppg))

    # ===== E3. PPG 3 通道空间特征 =====
    if n_ch >= 3:
        spatial_feats, _imbalance, _vmag = _extract_ppg_spatial_features(ppg_ch_raw)
        feat.update(spatial_feats)
        # 复用 _extract_ppg_spatial_features 已计算的中间变量，避免重复计算
        feat.update(_extract_ppg_ch_consistency(ppg_ch_bp))
        feat.update(_extract_ppg_spatial_coupling(ir_raw, _imbalance, _vmag, ir_raw))
    else:
        for k in ["PPG_ch_imbalance_mean", "PPG_ch_imbalance_p90", "PPG_ch_imbalance_iqr",
                   "PPG_ch_rangeNorm_mean", "PPG_ch_rangeNorm_p90",
                   "PPG_ch_vmag_mean", "PPG_ch_vmag_p90", "PPG_ch_vmag_iqr", "PPG_ch_vmag_std",
                   "PPG_ch_dc_cv", "PPG_ch_dc_max_min_ratio",
                   "PPG_ch_bp_corr_mean", "PPG_ch_bp_corr_min", "PPG_ch_bp_corr_std",
                   "PPG_ch_bp_lag_std",
                   "PPG_corr_mean_imbalance", "PPG_corr_mean_vmag", "PPG_corr_IR_imbalance"]:
            feat[k] = 0.0

    # ===== F. EMG 特征（同时获取 ch0 包络供跨模态使用，避免重复预处理）=====
    emg_feat, emg_env_for_cross = extract_emg_features(
        emg_window, fs=fs_emg, return_signals=True)
    feat.update(emg_feat)

    # ===== G. ACC 特征 (先做鲁棒去毛刺, 抹掉敲击/碰撞瞬变) =====
    acc_clean = _acc_robust_clean(acc_window)
    feat.update(extract_acc_features(acc_clean, fs=fs_acc, prefix="ACC"))
    feat.update(extract_acc_tremor_features(acc_clean, fs=fs_acc))

    # ===== H. 跨模态特征 (统一使用 acc_clean) =====
    feat.update(extract_acc_cross_features(acc_clean, ppg_bp, emg_env_for_cross, fs_ppg, fs_emg))
    feat.update(extract_emg_ppg_cross_features(emg_env_for_cross, ppg_bp, fs_emg, fs_ppg))
    feat.update(extract_acc_ppg_coherence(acc_clean, ppg_bp, fs_acc=fs_acc, fs_ppg=fs_ppg))

    # ===== I. EMG 双通道 consensus 统计 =====
    # 对 EMG0/EMG1 的关键特征计算跨通道 min/max/range/cv，
    # 检测单侧电极接触不良（类似绿光三通道的翘起检测）。
    _emg_consensus_base = ["RMS", "MAV", "WL", "ZC", "MNF", "MDF", "PKF", "PSR"]
    for _base in _emg_consensus_base:
        _v0 = float(feat.get(f"EMG0_{_base}", 0.0))
        _v1 = float(feat.get(f"EMG1_{_base}", 0.0))
        _arr = np.array([_v0, _v1], dtype=np.float64)
        feat[f"EMG_consensus_{_base}_min"] = float(np.min(_arr))
        feat[f"EMG_consensus_{_base}_max"] = float(np.max(_arr))
        feat[f"EMG_consensus_{_base}_range"] = float(np.max(_arr) - np.min(_arr))
        _mean_abs = np.mean(np.abs(_arr))
        feat[f"EMG_consensus_{_base}_cv"] = float(np.std(_arr) / (_mean_abs + EPS))

    # ===== J. ACC 信号质量特征 =====
    # 检测 ACC 传感器饱和、削顶等接触问题
    if acc_clean is not None and len(acc_clean) > 0:
        _acc_arr = np.asarray(acc_clean, dtype=np.float64)
        _acc_max = np.max(np.abs(_acc_arr)) + EPS
        feat["ACC_SAT_FRAC"] = float(np.mean(np.abs(_acc_arr) >= 0.98 * _acc_max))
        _d = np.diff(_acc_arr, axis=0) if _acc_arr.ndim > 1 else np.diff(_acc_arr)
        feat["ACC_CLIP_RATE"] = float(np.mean(np.abs(_d) < 1e-10))
    else:
        feat["ACC_SAT_FRAC"] = 0.0
        feat["ACC_CLIP_RATE"] = 0.0

    # ===== 移除冗余特征 + 清理异常数值 =====
    # 统计 invalid 特征数量（在 0.0 填充之前）
    _invalid_total = 0
    for k, v in feat.items():
        if v is None or not np.isfinite(v):
            _invalid_total += 1
    feat["TOTAL_INVALID_COUNT"] = float(_invalid_total)

    for k in list(feat.keys()):
        if k in _REDUNDANT_FEATURES:
            del feat[k]
            continue
        v = feat[k]
        if v is None or not np.isfinite(v):
            feat[k] = 0.0
        else:
            feat[k] = float(v)

    if return_preprocessed:
        preprocessed = {
            'ppg_bp': ppg_bp,
            'emg_env': emg_env_for_cross,
            'ppg_raw': ppg_raw,
        }
        return feat, preprocessed

    return feat


def align_emg_window(emg, ppg_len, start_ppg, win_ppg, fs_ppg=100.0, fs_emg=1000.0):
    """按时间比例对齐 EMG 窗口。"""
    if emg is None or len(emg) == 0:
        return None
    ratio = fs_emg / fs_ppg
    start_emg = int(start_ppg * ratio)
    end_emg = start_emg + int(win_ppg * ratio)
    if start_emg >= len(emg):
        return None
    return emg[start_emg:min(end_emg, len(emg))]


def align_acc_window(acc, ppg_len, start_ppg, win_ppg, fs_ppg=100.0, fs_acc=100.0):
    """按时间比例对齐 ACC 窗口。"""
    if acc is None or len(acc) == 0:
        return None
    ratio = fs_acc / fs_ppg
    start_acc = int(start_ppg * ratio)
    end_acc = start_acc + int(win_ppg * ratio)
    if start_acc >= len(acc):
        return None
    return acc[start_acc:min(end_acc, len(acc))]


def iter_sample_windows(ppg_6ch, emg, acc, win_samples, stride_samples,
                        fs_ppg=FEATURE_FS, fs_emg=DEFAULT_FS_EMG,
                        fs_acc=FEATURE_FS, window_indices=None):
    """Yield existing H5 windows if present; otherwise yield sliding windows."""
    ppg_6ch = np.asarray(ppg_6ch, dtype=np.float64)
    if is_windowed_array(ppg_6ch):
        n_windows = ppg_6ch.shape[0]
        emg_windowed = emg is not None and is_windowed_array(emg)
        acc_windowed = acc is not None and is_windowed_array(acc)
        for idx in range(n_windows):
            if window_indices is not None and idx < len(window_indices):
                start_100hz = int(window_indices[idx] * stride_samples)
            else:
                start_100hz = int(idx * win_samples)
            yield {
                "start_100hz": start_100hz,
                "ppg_6ch": ppg_6ch[idx],
                "emg": emg[idx] if emg_windowed and idx < len(emg) else None,
                "acc": acc[idx] if acc_windowed and idx < len(acc) else None,
            }
        return

    n_ppg = len(ppg_6ch)
    for start in range(0, n_ppg - win_samples + 1, stride_samples):
        yield {
            "start_100hz": int(start),
            "ppg_6ch": ppg_6ch[start:start + win_samples],
            "emg": align_emg_window(emg, n_ppg, start, win_samples,
                                    fs_ppg=fs_ppg, fs_emg=fs_emg),
            "acc": align_acc_window(acc, n_ppg, start, win_samples,
                                    fs_ppg=fs_ppg, fs_acc=fs_acc),
        }


# =========================================================
# 单样本特征提取
# =========================================================

def _extract_rows_for_sample(sample, dc_threshold, ac_dc_threshold,
                              win_samples, stride_samples, fs_ppg_orig):
    """单样本抽窗特征。win_samples/stride_samples 已由调用方转为采样点数。"""
    try:
        ppg_6ch = load_ppg(sample)
        emg = load_emg(sample)
        acc = load_acc(sample)
    except Exception as e:
        print(f"读取失败 {sample.get('sample_name')}: {e}")
        return []

    if is_windowed_array(ppg_6ch):
        rows = []
        for win in iter_sample_windows(
                ppg_6ch, emg, acc,
                win_samples=win_samples,
                stride_samples=stride_samples,
                fs_ppg=FEATURE_FS,
                fs_emg=DEFAULT_FS_EMG,
                fs_acc=FEATURE_FS,
                window_indices=sample.get("window_indices")):
            start = int(win["start_100hz"])
            ppg_win_6ch = win["ppg_6ch"]
            if not stage1_sample_pass(ppg_win_6ch, dc_threshold, ac_dc_threshold):
                continue
            try:
                feat = extract_feature_pool_from_window(
                    ppg_signal=build_3ch_ppg(ppg_win_6ch),
                    emg_window=win["emg"],
                    acc_window=win["acc"],
                    fs_ppg=FEATURE_FS,
                    fs_emg=DEFAULT_FS_EMG,
                    fs_acc=FEATURE_FS,
                    return_preprocessed=False,
                )
                feat["sample_name"] = sample["sample_name"]
                feat["h5_file"] = sample["h5_file"]
                feat["target"] = int(sample["target"])
                feat["start_100hz"] = start
                rows.append(feat)
            except Exception as e:
                print(f"特征提取失败: sample={sample.get('sample_name')}, "
                      f"start={start}, error={e}")
                continue
        return rows

    if len(ppg_6ch) < win_samples:
        return []

    if not stage1_sample_pass(ppg_6ch, dc_threshold, ac_dc_threshold):
        return []

    sample_target = int(sample.get("target", 0))

    # 6-ch PPG → 3 通道 PPG @ 100Hz（不降采样）
    ppg_3ch = build_3ch_ppg(ppg_6ch)  # (N, 3)
    n_ppg = len(ppg_3ch)

    # ACC 保持 100Hz（不降采样）
    acc_ppg = acc  # (N, 3) @ 100Hz

    stride_actual = stride_samples

    if n_ppg < win_samples:
        return []

    rows = []
    for start in range(0, n_ppg - win_samples + 1, stride_actual):
        ppg_win = ppg_3ch[start:start + win_samples]  # (W, 3)

        emg_win = align_emg_window(emg, n_ppg, start, win_samples,
                                    fs_ppg=FEATURE_FS, fs_emg=DEFAULT_FS_EMG)

        acc_win = align_acc_window(acc_ppg, n_ppg, start, win_samples,
                                    fs_ppg=FEATURE_FS, fs_acc=FEATURE_FS)

        try:
            feat = extract_feature_pool_from_window(
                ppg_signal=ppg_win,
                emg_window=emg_win,
                acc_window=acc_win,
                fs_ppg=FEATURE_FS,
                fs_emg=DEFAULT_FS_EMG,
                fs_acc=FEATURE_FS,
                return_preprocessed=False,
            )

            feat["sample_name"] = sample["sample_name"]
            feat["h5_file"] = sample["h5_file"]
            feat["target"] = int(sample["target"])
            feat["start_100hz"] = int(start * (fs_ppg_orig / FEATURE_FS))
            rows.append(feat)
        except Exception as e:
            print(f"特征提取失败: sample={sample.get('sample_name')}, "
                  f"start={start}, error={e}")
            continue

    return rows


def _worker_extract(args_tuple):
    """单样本特征提取（进程池 worker）。捕获所有异常避免进程静默崩溃。"""
    try:
        (sample, dc_threshold, ac_dc_threshold, window_len, stride_len, fs) = args_tuple[:6]
        return _extract_rows_for_sample(
            sample, dc_threshold, ac_dc_threshold, window_len, stride_len, fs
        )
    except Exception as e:
        sample_name = args_tuple[0].get("sample_name", "?") if args_tuple else "?"
        print(f"\n[worker error] {sample_name}: {e}")
        return []


def _init_feature_worker():
    """Limit BLAS thread fan-out inside feature extraction workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def extract_features_for_split(samples,
                               dc_threshold,
                               ac_dc_threshold,
                               window_sec=3,
                               stride_sec=1,
                               fs=100,
                               n_workers=None):
    """提取特征池（样本级并行）。默认 3s 窗口, 1s 步长。"""
    window_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)

    if n_workers is None:
        n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    n_workers = max(1, int(n_workers))

    args_list = [
        (s, dc_threshold, ac_dc_threshold, window_len, stride_len, fs,
         False, int(1 * fs), int(3 * fs))
        for s in samples
    ]

    n_total = len(args_list)
    use_mp = n_workers > 1 and len(samples) > 2

    # tqdm 进度条 (无依赖时降级为简单打印)
    try:
        from tqdm import tqdm as _tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    print(f"  样本数: {n_total}, 窗口={window_sec}s, 步长={stride_sec}s, "
          f"workers={n_workers if use_mp else 1}{'(mp)' if use_mp else ''}",
          flush=True)
    t_start = time.time()

    all_rows = []

    def _iterate(iterable):
        if _HAS_TQDM:
            return _tqdm(iterable, total=n_total, unit="sample",
                          desc="  特征提取", dynamic_ncols=True,
                          mininterval=0.5, smoothing=0.3)
        return iterable

    if not use_mp:
        it = _iterate(args_list)
        for i, a in enumerate(it):
            rows = _worker_extract(a)
            all_rows.extend(rows)
            if _HAS_TQDM:
                it.set_postfix(wins=len(all_rows))
            elif (i + 1) % max(1, n_total // 20) == 0 or i == n_total - 1:
                elapsed = time.time() - t_start
                eta = (elapsed / (i + 1)) * (n_total - i - 1) if i > 0 else 0
                print(f"\r  [{i+1}/{n_total}] {len(all_rows)} wins, "
                      f"{elapsed:.0f}s, ETA {eta:.0f}s", end="", flush=True)
        if not _HAS_TQDM:
            print()
    else:
        from concurrent.futures import ProcessPoolExecutor
        # chunksize 调到每 worker ~8 个 chunk 平衡 IPC 开销与负载均衡
        chunksize = max(1, n_total // (n_workers * 8))
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_feature_worker) as ex:
            results_iter = ex.map(_worker_extract, args_list, chunksize=chunksize)
            it = _iterate(results_iter)
            for i, rows in enumerate(it):
                all_rows.extend(rows)
                if _HAS_TQDM:
                    it.set_postfix(wins=len(all_rows))
                elif (i + 1) % max(1, n_total // 20) == 0 or i == n_total - 1:
                    elapsed = time.time() - t_start
                    eta = (elapsed / (i + 1)) * (n_total - i - 1) if i > 0 else 0
                    print(f"\r  [{i+1}/{n_total}] {len(all_rows)} wins, "
                          f"{elapsed:.0f}s, ETA {eta:.0f}s", end="", flush=True)
        if not _HAS_TQDM:
            print()

    elapsed = time.time() - t_start
    n_wins = len(all_rows)
    n_with_wins = len(set(r.get("sample_name", "") for r in all_rows if r.get("sample_name")))
    print(f"  完成: {n_wins} windows from {n_with_wins}/{n_total} samples "
          f"passed stage1 ({elapsed:.1f}s, {n_wins / max(elapsed, 0.001):.0f} wins/s)")

    return pd.DataFrame(all_rows)


# =========================================================
# 统一窗口级特征提取（供部署调用）
# =========================================================

def extract_window_features(ppg_win, emg_win, acc_win,
                             fs_ppg=100.0, fs_emg=1000.0, fs_acc=100.0):
    """
    统一的窗口级特征提取函数。
    训练（s03）和部署（s06）都调用此函数。
    """
    return extract_feature_pool_from_window(
        ppg_signal=ppg_win,
        emg_window=emg_win,
        acc_window=acc_win,
        fs_ppg=fs_ppg,
        fs_emg=fs_emg,
        fs_acc=fs_acc,
        return_preprocessed=False,
    )


# =========================================================
# 阈值解析
# =========================================================

def resolve_stage1_thresholds(th):
    if "deploy_stage1_threshold" in th:
        deploy_dc = th["deploy_stage1_threshold"]["dc_threshold"]
        deploy_acdc = th["deploy_stage1_threshold"]["ac_dc_threshold"]
    else:
        deploy_dc = th["dc_threshold"]
        deploy_acdc = th["ac_dc_threshold"]

    if "train_stage1_threshold" in th:
        train_dc = th["train_stage1_threshold"]["dc_threshold"]
        train_acdc = th["train_stage1_threshold"]["ac_dc_threshold"]
    else:
        train_dc = deploy_dc
        train_acdc = deploy_acdc

    return {
        "deploy": {"dc_threshold": float(deploy_dc), "ac_dc_threshold": float(deploy_acdc)},
        "train": {"dc_threshold": float(train_dc), "ac_dc_threshold": float(train_acdc)},
    }


# =========================================================
# main
# =========================================================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--window_sec", type=int, default=3)
    parser.add_argument("--stride_sec", type=int, default=1)
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()

    split_path = os.path.join(args.artifact_dir, "splits.json")
    th_path = os.path.join(args.artifact_dir, "stage1_threshold.json")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(th_path, "r", encoding="utf-8") as f:
        th = json.load(f)

    thresholds = resolve_stage1_thresholds(th)

    print("=" * 80)
    print("Stage1 thresholds")
    print("=" * 80)
    print("deploy threshold:")
    print(f"  dc_threshold    = {thresholds['deploy']['dc_threshold']}")
    print(f"  acdc_threshold  = {thresholds['deploy']['ac_dc_threshold']}")
    print("train/feature threshold:")
    print(f"  dc_threshold    = {thresholds['train']['dc_threshold']}")
    print(f"  acdc_threshold  = {thresholds['train']['ac_dc_threshold']}")

    for part in ["train", "valid", "test"]:
        print("=" * 80)
        print(f"提取 {part} 特征")
        print("=" * 80)

        gate_name = "train" if part in ["train", "valid"] else "deploy"
        dc_threshold = thresholds[gate_name]["dc_threshold"]
        ac_dc_threshold = thresholds[gate_name]["ac_dc_threshold"]

        print(f"{part} 使用 Stage1 gate: {gate_name}")
        print(f"  dc_threshold    = {dc_threshold}")
        print(f"  ac_dc_threshold  = {ac_dc_threshold}")

        df = extract_features_for_split(
            samples=split[part],
            dc_threshold=dc_threshold,
            ac_dc_threshold=ac_dc_threshold,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            fs=100,
            n_workers=args.n_workers,
        )

        out_path = os.path.join(args.artifact_dir, f"feature_pool_{part}.csv")

        if len(df) == 0:
            print(f"[警告] {part} 特征池为空！可能原因:")
            print(f"  1. Stage1 阈值过严：dc>{dc_threshold:.1e}, acdc<{ac_dc_threshold}")
            print(f"  2. 样本时长不足 {args.window_sec}s")
            print(f"  3. H5 字段名不匹配（期望 'ppg', 'emg', 'acc'）")
            # 写一个带列名的空文件，避免下游 s04 读空文件崩溃
            meta_cols = ["sample_name", "h5_file", "target", "start_100hz"]
            dummy_cols = meta_cols + ["PPG_mean"]
            pd.DataFrame(columns=dummy_cols).to_csv(out_path, index=False)
            print(f"  已生成占位 CSV: {out_path}")
            continue
        df.to_csv(out_path, index=False)

        print(f"{part} 特征提取完成: {len(df)} windows")
        print(f"保存到: {out_path}")

        if len(df) > 0 and "target" in df.columns:
            print(f"  target=0: {np.sum(df['target'].values == 0)}")
            print(f"  target=1: {np.sum(df['target'].values == 1)}")
            meta_cols = ["sample_name", "h5_file", "target", "start_100hz"]
            print(f"  特征列数: {len([c for c in df.columns if c not in meta_cols])}")


if __name__ == "__main__":
    main()
