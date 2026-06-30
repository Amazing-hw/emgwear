"""
deploy_feature_extractor.py
Auto-generated standalone feature extraction script (PPG+EMG+ACC).
Extracts 2 features from a 3s@100Hz PPG window + EMG + ACC.
Input: ppg (Nx6 raw PPG @ 100Hz, Nx3 virtual PPG, or 1D PPG), emg (Nx2 @ 1000Hz or None), acc (Nx3 @ 100Hz or None)
Output: feature vector (list of 2 floats)
Dependencies: numpy, scipy
"""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, correlate, find_peaks, iirnotch

EPS = 1e-12
FEATURE_ORDER = ["EMG_consensus_PKF_range", "EMG_consensus_ZC_max"]
FILL_VALUES = {"EMG_consensus_PKF_range": 22.4609375, "EMG_consensus_ZC_max": 0.2803333333333333}
CLIP_BOUNDS = {"EMG_consensus_PKF_range": [-39.0625, 85.9375], "EMG_consensus_ZC_max": [0.26579166666666676, 0.29812499999999986]}


# ========== Utilities ==========

def _safe_div(a, b):
    return float(a) / (float(b) + EPS)

def _robust_mad(x):
    return float(np.median(np.abs(x - np.median(x))))

def _robust_iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def _safe_corr(x, y, winsorize=False):
    n = min(len(x), len(y))
    if n < 8:
        return 0.0
    x = np.asarray(x[:n], dtype=np.float64)
    y = np.asarray(y[:n], dtype=np.float64)
    if winsorize:
        x = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
        y = np.clip(y, np.percentile(y, 5), np.percentile(y, 95))
    x, y = x - np.mean(x), y - np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx < EPS or sy < EPS:
        return 0.0
    v = np.mean((x / sx) * (y / sy))
    return float(v) if np.isfinite(v) else 0.0

def _remove_burr(x, k=6.0):
    if len(x) < 3:
        return x
    d = np.diff(x)
    thr = max(k * _robust_mad(d), EPS)
    left, mid, right = x[:-2], x[1:-1], x[2:]
    bad = (np.abs(mid - left) > thr) & (np.abs(mid - right) > thr)
    x[1:-1] = np.where(bad, 0.5 * (left + right), mid)
    return x

def _remove_step(x, k=10.0):
    if len(x) < 2:
        return x
    d = np.diff(x)
    thr = max(k * _robust_mad(d), EPS)
    for i in range(1, len(x)):
        if abs(x[i] - x[i-1]) > thr:
            x[i] = x[i-1]
    return x

_BUTTER_CACHE = {}
_IIR_NOTCH_CACHE = {}

def _bandpass(x, fs=100, lowcut=0.4, highcut=6.0, order=4):
    if len(x) < 16:
        return x.copy()
    key = (float(fs), float(lowcut), float(highcut), int(order))
    if key not in _BUTTER_CACHE:
        nyq = 0.5 * fs
        b, a = butter(order, [max(lowcut/nyq, 1e-6), min(highcut/nyq, 0.999)], btype="band")
        _BUTTER_CACHE[key] = (b, a)
    b, a = _BUTTER_CACHE[key]
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x.copy()


def _highpass(x, fs=1000, cutoff=20.0, order=2):
    if len(x) < 16:
        return x.copy()
    key = (float(fs), "highpass", float(cutoff), int(order))
    if key not in _BUTTER_CACHE:
        nyq = 0.5 * fs
        b, a = butter(order, min(max(cutoff/nyq, 1e-6), 0.999), btype="highpass")
        _BUTTER_CACHE[key] = (b, a)
    b, a = _BUTTER_CACHE[key]
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x.copy()


# ========== FFT / Autocorr / Entropy ==========

def _fft_features(bp, fs=100, fmin=0.5, fmax=5.0):
    """Returns (peak_ratio, dom_freq, spec, freqs, band_spec, band_freqs)."""
    if len(bp) < 16:
        return 0.0, 0.0, None, None, None, None
    x = bp - np.mean(bp)
    xw = x * np.hamming(len(x))
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    nfft = max(256, nfft)
    spec = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0, spec, freqs, None, None
    bs, bf = spec[mask], freqs[mask]
    med = np.median(bs)
    r = float(np.max(bs) / (med + EPS)) if med > EPS else 0.0
    d = float(bf[np.argmax(bs)])
    return r, d, spec, freqs, bs, bf

def _autocorr_features(bp, fs=100, bpm_min=40, bpm_max=180):
    """Returns (ac_peak, ac_lag_sec)."""
    if len(bp) < int(fs * 1.5):
        return 0.0, 0.0
    x = bp - np.mean(bp)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(x)-1:]
    if ac[0] < EPS:
        return 0.0, 0.0
    ac = ac / ac[0]
    lag_min = max(1, int(fs * 60.0 / bpm_max))
    lag_max = min(len(ac) - 1, int(fs * 60.0 / bpm_min))
    if lag_max <= lag_min:
        return 0.0, 0.0
    seg = ac[lag_min:lag_max+1]
    idx = int(np.argmax(seg))
    return float(seg[idx]), float((lag_min + idx) / fs)

def _sample_entropy(bp, m=2, r_ratio=0.2):
    N = len(bp)
    if N < 50:
        return 0.0
    r = r_ratio * np.std(bp)
    if r < EPS:
        return 0.0
    def _count(mv, chunk=500):
        pat = np.array([bp[i:i+mv] for i in range(N - mv)], dtype=np.float64)
        n_pat = len(pat)
        if n_pat < 2:
            return 0.0
        total = 0.0
        for i in range(0, n_pat, chunk):
            end = min(i + chunk, n_pat)
            cp = pat[i:end]
            d = np.max(np.abs(cp[:, None, :] - pat[None, :, :]), axis=2)
            total += np.sum(d <= r)
            for j in range(len(cp)):
                idx = i + j
                if idx < n_pat and d[j, idx] <= r:
                    total -= 1.0
        return total
    Bm = _count(m)
    Bm1 = _count(m + 1)
    if Bm1 <= EPS or Bm <= EPS:
        return 0.0
    return float(-np.log(Bm1 / Bm))

def _emg_downsample_for_sampen(x, fs=1000.0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 50:
        return None
    if fs > 250 and len(x) > 250:
        gcd = np.gcd(int(fs), 250)
        up = 250 // gcd
        down = int(fs) // gcd
        x = _resample_poly(x, up, down)
    if len(x) > 1200:
        start = (len(x) - 1200) // 2
        x = x[start:start + 1200]
    return x

def _emg_sample_entropy(bp, fs=1000.0):
    ds = _emg_downsample_for_sampen(bp, fs)
    return _sample_entropy(ds) if ds is not None else 0.0

def _smooth_envelope(x, fs=25, win_sec=0.25):
    x = np.abs(x)
    w = max(3, int(round(win_sec * fs)))
    if w % 2 == 0:
        w += 1
    return np.convolve(x, np.ones(w) / w, mode="same")

def _emg_frequency_features(bp, fs=1000):
    """Returns coarse and fine EMG spectral power features.
    使用 Welch 方法 (nperseg=512)，与 s03 训练代码保持一致。
    """
    if bp is None or len(bp) < 16:
        return (0.0,) * 21
    x = np.asarray(bp, dtype=np.float64)
    try:
        from scipy.signal import welch
        nperseg = 512
        if len(x) < nperseg:
            return (0.0,) * 21
        noverlap = nperseg // 2
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        return (0.0,) * 21
    mask = (f >= 20) & (f <= 450)
    if not np.any(mask) or np.sum(Pxx[mask]) < EPS:
        return (0.0,) * 21
    bf, bs = f[mask], Pxx[mask]
    total_p = np.sum(bs) + EPS
    mnf = float(np.sum(bf * bs) / total_p)
    cumsum = np.cumsum(bs)
    mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    mdf = float(bf[min(mdf_idx, len(bf) - 1)])
    pkf = float(bf[np.argmax(bs)])
    low_100 = (bf >= 20) & (bf <= 100)
    high_100 = (bf > 100) & (bf <= 450)
    psr = float(np.sum(bs[low_100]) / (np.sum(bs[high_100]) + EPS))
    pow_20_60 = float(np.sum(bs[(bf >= 20) & (bf <= 60)]) / total_p)
    pow_60_150 = float(np.sum(bs[(bf > 60) & (bf <= 150)]) / total_p)
    pow_150_450 = float(np.sum(bs[(bf > 150) & (bf <= 450)]) / total_p)
    pow_lh = float(np.sum(bs[(bf >= 20) & (bf <= 60)]) / (np.sum(bs[(bf > 150) & (bf <= 450)]) + EPS))
    se95_idx = np.searchsorted(cumsum, cumsum[-1] * 0.95)
    se95 = float(bf[min(se95_idx, len(bf) - 1)])
    def _pow(lo, hi, include_low=True):
        lo_mask = bf >= lo if include_low else bf > lo
        return float(np.sum(bs[lo_mask & (bf <= hi)]))
    p_20_40 = _pow(20.0, 40.0, True)
    p_40_60 = _pow(40.0, 60.0, False)
    p_60_90 = _pow(60.0, 90.0, False)
    p_90_120 = _pow(90.0, 120.0, False)
    p_120_180 = _pow(120.0, 180.0, False)
    p_180_250 = _pow(180.0, 250.0, False)
    p_250_350 = _pow(250.0, 350.0, False)
    p_350_450 = _pow(350.0, 450.0, False)
    p_60_180 = _pow(60.0, 180.0, False)
    p_250_450 = _pow(250.0, 450.0, False)
    p_20_90 = _pow(20.0, 90.0, True)
    p_180_450 = _pow(180.0, 450.0, False)
    p_40_120 = _pow(40.0, 120.0, False)
    p_120_350 = _pow(120.0, 350.0, False)
    return (
        mnf, mdf, pkf, psr, pow_20_60, pow_60_150, pow_150_450, pow_lh, se95,
        float(p_20_40 / total_p), float(p_40_60 / total_p),
        float(p_60_90 / total_p), float(p_90_120 / total_p),
        float(p_120_180 / total_p), float(p_180_250 / total_p),
        float(p_250_350 / total_p), float(p_350_450 / total_p),
        float((pow_60_150 * total_p) / ((pow_20_60 * total_p) + EPS)),
        float(p_60_180 / (p_250_450 + EPS)),
        float(p_20_90 / (p_180_450 + EPS)),
        float(p_40_120 / (p_120_350 + EPS)),
    )

def _emg_subwindow_features(bp, env, fs=1000):
    if bp is None or len(bp) < int(fs):
        return 0.0, 0.0, 0.0
    x = np.asarray(bp, dtype=np.float64)
    win = max(16, int(round(fs)))
    rms_vals = []
    wl_vals = []
    mdf_vals = []
    for start in range(0, len(x) - win + 1, win):
        seg = x[start:start + win]
        if len(seg) < win:
            continue
        rms_vals.append(float(np.sqrt(np.mean(seg * seg))))
        wl_vals.append(float(np.sum(np.abs(np.diff(seg)))))
        mdf_vals.append(float(_emg_frequency_features(seg, fs)[1]))
    if len(rms_vals) < 2:
        return 0.0, 0.0, 0.0
    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    wl_arr = np.asarray(wl_vals, dtype=np.float64)
    mdf_arr = np.asarray(mdf_vals, dtype=np.float64)
    return (
        float(np.std(rms_arr) / (np.mean(rms_arr) + EPS)),
        _robust_iqr(mdf_arr),
        float(np.std(wl_arr) / (np.mean(wl_arr) + EPS)),
    )

def _emg_spectral_shape_features(bp, fs=1000):
    if bp is None or len(bp) < 16:
        return 0.0, 0.0, 0.0, 0.0
    x = np.asarray(bp, dtype=np.float64)
    try:
        from scipy.signal import welch
        nperseg = 512
        if len(x) < nperseg:
            return 0.0, 0.0, 0.0, 0.0
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0
    mask = (f >= 20) & (f <= 450)
    if not np.any(mask) or np.sum(Pxx[mask]) < EPS:
        return 0.0, 0.0, 0.0, 0.0
    bf = f[mask]
    bs = Pxx[mask]
    total = float(np.sum(bs)) + EPS
    p = bs / total
    entropy = -float(np.sum(p * np.log(p + EPS)) / np.log(len(p) + EPS))
    flatness = float(np.exp(np.mean(np.log(bs + EPS))) / (np.mean(bs) + EPS))
    centroid = float(np.sum(bf * bs) / total)
    cumsum = np.cumsum(bs)
    roll_idx = np.searchsorted(cumsum, cumsum[-1] * 0.85)
    rolloff = float(bf[min(roll_idx, len(bf) - 1)])
    return entropy, flatness, centroid, rolloff

def _emg_channel_balance_features(ch0_env, ch1_env, ch0_bp, ch1_bp):
    if ch0_env is None or ch1_env is None or ch0_bp is None or ch1_bp is None:
        return 0.0, 0.0, 0.0
    n_env = min(len(ch0_env), len(ch1_env))
    n_bp = min(len(ch0_bp), len(ch1_bp))
    if n_env < 4 or n_bp < 4:
        return 0.0, 0.0, 0.0
    env0 = np.asarray(ch0_env[:n_env], dtype=np.float64)
    env1 = np.asarray(ch1_env[:n_env], dtype=np.float64)
    mav0 = float(np.mean(env0))
    mav1 = float(np.mean(env1))
    env_corr = _safe_corr(env0, env1, winsorize=True)
    mav_ratio = float(np.clip(_safe_div(mav0, mav1), 0.0, 1000.0)) if mav1 > EPS else 0.0
    imbalance = float(abs(mav0 - mav1) / (mav0 + mav1 + EPS))
    return env_corr, mav_ratio, imbalance

def _resample_poly(data, up, down):
    from scipy.signal import resample_poly as _rp
    return _rp(data.astype(np.float32, copy=False), up, down).astype(np.float64)


# ========== Preprocessing ==========

def _preprocess_ppg(x, fs=100):
    """Returns (raw_clean, bp, dc)."""
    x = np.asarray(x, dtype=np.float64).copy()
    x = _remove_burr(x)
    x = _remove_step(x)
    mf_k = max(3, int(round(0.05 * fs)))
    if mf_k % 2 == 0:
        mf_k += 1
    if len(x) >= mf_k:
        try:
            x = medfilt(x, kernel_size=mf_k)
        except Exception:
            pass
    ma_w = max(2, int(round(0.03 * fs)))
    kernel = np.ones(ma_w) / ma_w
    x = np.convolve(x, kernel, mode="same")
    dc = float(np.median(x))
    bp = _bandpass(x, fs)
    bp = np.convolve(bp, kernel, mode="same")
    return x, bp, dc

def _build_3ch_ppg(ppg):
    """Convert raw six-channel PPG to the same three virtual channels used in training."""
    ppg = np.asarray(ppg, dtype=np.float64)
    if ppg.ndim == 1:
        return ppg.reshape(-1, 1)
    if ppg.ndim != 2:
        raise ValueError(f"ppg must be 1D or 2D, got shape={ppg.shape}")
    if ppg.shape[1] == 6:
        raw = ppg
    elif ppg.shape[0] == 6:
        raw = ppg.T
    else:
        return ppg
    ch_A = (raw[:, 0] + raw[:, 1]) / 2.0
    ch_B = (raw[:, 2] + raw[:, 4]) / 2.0
    ch_C = (raw[:, 3] + raw[:, 5]) / 2.0
    return np.column_stack([ch_A, ch_B, ch_C])

def _narrow_notch(x, fs, f0, bw_hz=0.8, order=2):
    nyq = 0.5 * fs
    w0 = f0 / nyq
    wbw = bw_hz / nyq
    lo = max(w0 - wbw, 1e-6)
    hi = min(w0 + wbw, 0.999)
    if not (0 < lo < hi < 1):
        return x
    try:
        b, a = butter(order, [lo, hi], btype="bandstop")
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

def _compute_leak_ratios(bp_ref, fs, leak_freqs, bw_hz=0.8):
    """Compute narrowband leakage ratios on notch-free reference signal.
    使用 Welch 方法 (nperseg=512)，与 s03 extract_emg_leakage_features 一致。
    """
    if bp_ref is None or len(bp_ref) < 16:
        return (0.0,) * len(leak_freqs)
    x = np.asarray(bp_ref, dtype=np.float64)
    try:
        from scipy.signal import welch
        nperseg = min(512, len(x) // 2)
        if nperseg < 16:
            return (0.0,) * len(leak_freqs)
        noverlap = nperseg // 2
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        return (0.0,) * len(leak_freqs)
    mask_total = (f >= 20) & (f <= 450)
    total_pow = float(np.sum(Pxx[mask_total]))
    if total_pow < EPS:
        return (0.0,) * len(leak_freqs)
    ratios = []
    for f0 in leak_freqs:
        mask = (f >= f0 - bw_hz) & (f <= f0 + bw_hz)
        p_band = float(np.sum(Pxx[mask]))
        ratios.append(p_band / total_pow)
    return tuple(ratios)


def _acc_robust_clean(acc, burr_k=6.0):
    if acc is None:
        return acc
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim == 1 or len(acc) < 3:
        return acc.copy()
    out = acc.copy()
    for ax in range(out.shape[1]):
        out[:, ax] = _remove_burr(out[:, ax], k=burr_k)
    return out

def _preprocess_emg(x, fs=1000):
    """Returns (bp_leak_ref, bp_clean, env, x_raw_ref)."""
    if x is None or len(x) < 4:
        return None, None, None, None
    x = np.asarray(x, dtype=np.float64).copy()
    x_raw_ref = x.copy()
    bp = _highpass(x_raw_ref, fs, cutoff=20.0, order=2)
    # 保存 bandstop/notch 前参考信号
    bp_leak_ref = bp.copy()
    for lo, hi in ((49.8, 50.2), (149.8, 150.2)):
        center = (lo + hi) / 2.0
        bp = _narrow_notch(bp, fs, center, bw_hz=(hi - lo) / 2.0, order=2)
    for f0 in (50.0, 100.0, 200.0, 300.0, 400.0):
        bp = _iir_notch_filter(bp, fs, f0, q=100.0)
    bp_clean = bp
    env = np.abs(bp_clean)
    return bp_leak_ref, bp_clean, env, x_raw_ref


def _band_power(x, low, high, fs):
    """EMG Welch power over the 20-450Hz training spectrum."""
    if x is None:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    nperseg = 1024
    if len(x) < nperseg:
        return 0.0
    try:
        from scipy.signal import welch
        noverlap = nperseg // 2
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        return 0.0
    train_mask = (f >= 20.0) & (f <= 450.0)
    if not np.any(train_mask) or np.sum(Pxx[train_mask]) < EPS:
        return 0.0
    f = f[train_mask]
    Pxx = Pxx[train_mask]
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(Pxx[mask]))

def _acc_tremor_features(acc_mag, fs=100.0):
    if acc_mag is None or len(acc_mag) < 16:
        return 0.0, 0.0
    x = np.asarray(acc_mag, dtype=np.float64)
    x = x - np.mean(x)
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    nfft = max(256, nfft)
    spec = np.abs(np.fft.rfft(x * np.hamming(len(x)), n=nfft))
    spec_sq = spec * spec
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    p_tremor = float(np.sum(spec_sq[(freqs >= 8.0) & (freqs <= 12.0)]))
    p_total = float(np.sum(spec_sq[(freqs >= 0.5) & (freqs <= 15.0)])) + EPS
    return float(np.log1p(p_tremor)), float(p_tremor / p_total)


def _welch_coherence(x, y, fs, nperseg=None):
    try:
        from scipy.signal import coherence as _coh
    except Exception:
        return None, None
    n = min(len(x), len(y))
    if n < int(2 * fs):
        return None, None
    nps = int(2 * fs) if nperseg is None else min(nperseg, n)
    return _coh(x[:n], y[:n], fs=fs, nperseg=nps)


def _detect_ppg_peaks(ppg_bp, fs=100):
    if ppg_bp is None or len(ppg_bp) < int(fs):
        return np.array([], dtype=int)
    min_dist = max(1, int(0.33 * fs))
    prom = max(float(np.std(ppg_bp)) * 0.3, EPS)
    try:
        peaks, _ = find_peaks(ppg_bp, distance=min_dist, prominence=prom)
    except Exception:
        peaks = np.array([], dtype=int)
    return peaks


def _ppg_morphology_stats(ppg_bp, peaks, fs=100):
    """Returns (dicrotic_ratio, aug_index_mean, pulse_width_cv)."""
    if ppg_bp is None or len(peaks) < 2:
        return 0.0, 0.0, 0.0
    x = np.asarray(ppg_bp, dtype=np.float64)
    dicrotic_hits = 0
    aug_indices = []
    pulse_widths = []
    for i in range(len(peaks) - 1):
        p0, p1 = int(peaks[i]), int(peaks[i+1])
        seg = x[p0:p1]
        if len(seg) < int(0.2 * fs):
            continue
        pulse_widths.append(len(seg) / fs)
        s_start = int(0.25 * len(seg))
        s_end = int(0.75 * len(seg))
        if s_end - s_start < 3:
            continue
        sub = seg[s_start:s_end]
        if len(sub) < 3:
            continue
        sub_peak = float(np.max(sub))
        sub_mean = float(np.mean(seg))
        sub_std = float(np.std(seg)) + EPS
        main_peak = float(np.max(seg[:s_start])) if s_start > 0 else float(seg[0])
        if sub_peak > sub_mean + 0.2 * sub_std and main_peak > EPS:
            dicrotic_hits += 1
            aug_indices.append(sub_peak / main_peak)
    n_seg = max(1, len(peaks) - 1)
    dicrotic_ratio = dicrotic_hits / n_seg
    aug_mean = float(np.mean(aug_indices)) if aug_indices else 0.0
    if len(pulse_widths) >= 2:
        pwm = float(np.mean(pulse_widths))
        pw_cv = float(np.std(pulse_widths) / (pwm + EPS))
    else:
        pw_cv = 0.0
    return dicrotic_ratio, aug_mean, pw_cv


def _ppg_hrv_stats(peaks, fs=100):
    """Returns (rmssd, cv, pnn30)."""
    if len(peaks) < 3:
        return 0.0, 0.0, 0.0
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / fs
    if len(rr) < 2:
        return 0.0, 0.0, 0.0
    drr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(drr * drr)))
    rr_mean = float(np.mean(rr))
    cv = float(np.std(rr) / (rr_mean + EPS))
    pnn30 = float(np.mean(np.abs(drr) > 0.030))
    return rmssd, cv, pnn30


def _acc_ppg_coherence(acc_mag, ppg_bp, fs_acc=100, fs_ppg=100):
    """Returns (coh_micro, coh_hr)."""
    if acc_mag is None or ppg_bp is None:
        return 0.0, 0.0
    m = np.asarray(acc_mag, dtype=np.float64)
    m = m - np.mean(m)
    if abs(fs_acc - fs_ppg) > 1e-3:
        try:
            m = _resample_poly(m, int(round(fs_ppg)), int(round(fs_acc)))
        except Exception:
            return 0.0, 0.0
    f, Cxy = _welch_coherence(m, ppg_bp, fs=fs_ppg)
    if f is None:
        return 0.0, 0.0
    mask_micro = (f >= 0.5) & (f <= 3.0)
    mask_hr = (f >= 0.8) & (f <= 3.0)
    micro = float(np.mean(Cxy[mask_micro])) if np.any(mask_micro) else 0.0
    hr = float(np.mean(Cxy[mask_hr])) if np.any(mask_hr) else 0.0
    return micro, hr


# ========== Main ==========

def extract_features(ppg, emg=None, acc=None, fs=100, fs_emg=1000):
    """Extract 2 features from a 3s@100Hz PPG window + EMG + ACC.

    ppg: 2D float array (N,6) raw PPG @100Hz or (N,3) virtual PPG @100Hz.
    """
    # ---- Preprocess PPG (raw 6 channels -> training virtual 3 channels) ----
    ppg = np.asarray(ppg, dtype=np.float64)
    ppg = _build_3ch_ppg(ppg)
    n_ch = ppg.shape[1]
    raw_list, bp_list = [], []
    for ch_idx in range(min(n_ch, 3)):
        ch_raw, ch_bp, _ = _preprocess_ppg(ppg[:, ch_idx], fs)
        raw_list.append(ch_raw)
        bp_list.append(ch_bp)
    ppg_ch_raw = np.column_stack(raw_list)
    ppg_ch_bp = np.column_stack(bp_list)

    # 3-ch mean for single-channel features
    ppg_raw = np.mean(ppg_ch_raw, axis=1)
    ppg_bp = np.mean(ppg_ch_bp, axis=1)
    ppg_dc = float(np.median(ppg_raw))
    ir = ppg_ch_raw[:, 0]  # ch_A for spatial coupling

    # ---- 3ch spatial intermediates ----
    if n_ch >= 3:
        g = ppg_ch_raw.T
        _sp_std = np.std(g, axis=0)
        _sp_mean = np.mean(g, axis=0)
        imb = _sp_std / (np.abs(_sp_mean) + EPS)
        g_max = np.max(g, axis=0)
        g_min = np.min(g, axis=0)
        denom_abs = np.abs(g[0]) + np.abs(g[1]) + np.abs(g[2]) + EPS
        rn = (g_max - g_min) / denom_abs
        vx = g[0] - 0.5 * g[1] - 0.5 * g[2]
        vy = (np.sqrt(3.0) / 2.0) * (g[1] - g[2])
        vmag = np.sqrt(vx * vx + vy * vy) / denom_abs
        dc3 = np.array([float(np.median(g[0])), float(np.median(g[1])), float(np.median(g[2]))])
        # 3ch bandpass consistency
        c0 = ppg_ch_bp[:, 0] - np.mean(ppg_ch_bp[:, 0])
        c1 = ppg_ch_bp[:, 1] - np.mean(ppg_ch_bp[:, 1])
        c2 = ppg_ch_bp[:, 2] - np.mean(ppg_ch_bp[:, 2])
        c01 = _safe_corr(c0, c1)
        c12 = _safe_corr(c1, c2)
        c20 = _safe_corr(c2, c0)
        c3 = [c01, c12, c20]
        xc01 = np.correlate(c0, c1, mode="same")
        xc12 = np.correlate(c1, c2, mode="same")
        l01 = np.argmax(np.abs(xc01)) - len(c0) // 2
        l12 = np.argmax(np.abs(xc12)) - len(c0) // 2
    else:
        imb = rn = vmag = np.zeros(len(ppg_raw))
        dc3 = np.zeros(3)
        c3 = [0.0, 0.0, 0.0]
        l01 = l12 = 0

    # ---- Preprocess EMG ----
    EMG_LEAK_FREQS = (100.0, 150.0, 200.0, 250.0, 300.0)
    if emg is not None and len(emg) >= 4:
        emg = np.asarray(emg, dtype=np.float64)
        if emg.ndim == 1:
            emg = emg.reshape(-1, 1)
        emg0_leak_ref, emg0_bp, emg0_env, emg0_raw_ref = _preprocess_emg(emg[:, 0], fs_emg)
        if emg.shape[1] >= 2:
            emg1_leak_ref, emg1_bp, emg1_env, emg1_raw_ref = _preprocess_emg(emg[:, 1], fs_emg)
        else:
            emg1_leak_ref = emg1_bp = emg1_env = emg1_raw_ref = None
        # leak ratio array per channel
        emg0_leak = _compute_leak_ratios(emg0_leak_ref, fs_emg, EMG_LEAK_FREQS) if emg0_leak_ref is not None else (0,)*5
        emg1_leak = _compute_leak_ratios(emg1_leak_ref, fs_emg, EMG_LEAK_FREQS) if emg1_leak_ref is not None else (0,)*5
        emg0_freq = _emg_frequency_features(emg0_bp, fs_emg)
        emg1_freq = _emg_frequency_features(emg1_bp, fs_emg) if emg1_bp is not None else (0,) * 21
        emg0_subwin = _emg_subwindow_features(emg0_bp, emg0_env, fs_emg)
        emg1_subwin = _emg_subwindow_features(emg1_bp, emg1_env, fs_emg) if emg1_bp is not None else (0.0, 0.0, 0.0)
        emg0_spec_shape = _emg_spectral_shape_features(emg0_bp, fs_emg)
        emg1_spec_shape = _emg_spectral_shape_features(emg1_bp, fs_emg) if emg1_bp is not None else (0.0, 0.0, 0.0, 0.0)
        emg_channel_balance = _emg_channel_balance_features(emg0_env, emg1_env, emg0_bp, emg1_bp)
        if emg0_env is not None:
            emg0_env_ds = _resample_poly(emg0_env, fs, fs_emg)
            emg0_env_smooth = _smooth_envelope(emg0_env, fs_emg)
            emg0_env_smooth_ds = _resample_poly(emg0_env_smooth, fs, fs_emg)
        else:
            emg0_env_ds = emg0_env_smooth_ds = None
    else:
        emg0_leak_ref = emg0_bp = emg0_env = emg0_raw_ref = None
        emg1_leak_ref = emg1_bp = emg1_env = emg1_raw_ref = None
        emg0_leak = emg1_leak = (0,) * 5
        emg0_freq = emg1_freq = (0,) * 21
        emg0_subwin = emg1_subwin = (0.0, 0.0, 0.0)
        emg0_spec_shape = emg1_spec_shape = (0.0, 0.0, 0.0, 0.0)
        emg_channel_balance = (0.0, 0.0, 0.0)
        emg0_env_ds = emg0_env_smooth_ds = None

    # ---- ACC (gravity/motion separation) ----
    if acc is not None and len(acc) >= 4:
        acc = _acc_robust_clean(np.asarray(acc, dtype=np.float64))
        # 低通 <0.5Hz 分离重力
        axis_grav = []
        for ax in range(acc.shape[1]):
            ax_raw = acc[:, ax]
            ax_mean = np.mean(ax_raw)
            try:
                ax_lp = _bandpass(ax_raw - ax_mean, fs, 0.1, 0.5, order=2)
            except Exception:
                ax_lp = np.zeros(len(ax_raw))
            axis_grav.append(ax_lp + ax_mean)
        acc_grav = np.column_stack(axis_grav)
        grav_mag = np.sqrt(np.sum(acc_grav**2, axis=1) + EPS)
        gm = np.mean(acc_grav, axis=0)
        # 运动分量 = 原始 - 重力
        acc_motion = acc - acc_grav
        motion_mag = np.sqrt(np.sum(acc_motion**2, axis=1) + EPS)
        acc_mag = np.sqrt(np.sum(acc**2, axis=1) + EPS)
        acc_mag_bp = _bandpass(acc_mag - np.mean(acc_mag), fs, 0.5, 5.0, order=2)
    else:
        grav_mag = motion_mag = acc_mag = None
        acc_mag_bp = None
        gm = np.zeros(3)
    am = acc_mag is not None
    ambp = acc_mag_bp

    # ---- FFT / autocorr caches ----
    fft_p = _fft_features(ppg_bp, fs)
    ac_p = _autocorr_features(ppg_bp, fs)

    # ---- Smooth envelopes for cross-modal ----
    ppg_env = _smooth_envelope(ppg_bp, fs)

    # ---- Temporal peak detection ----
    Np = len(ppg_bp)
    pk = find_peaks(ppg_bp, prominence=0)
    vk = find_peaks(-ppg_bp, prominence=0)

    # ---- Anti-spoof intermediates ----
    ppg_peaks = _detect_ppg_peaks(ppg_bp, fs)
    ppg_dicr, ppg_aug, ppg_pw_cv = _ppg_morphology_stats(ppg_bp, ppg_peaks, fs)
    ppg_rmssd, ppg_cv, ppg_pnn30 = _ppg_hrv_stats(ppg_peaks, fs)
    acc_tremor = _acc_tremor_features(acc_mag, fs) if am else (0.0, 0.0)
    acc_coh_micro, acc_coh_hr = _acc_ppg_coherence(acc_mag, ppg_bp, fs_acc=fs, fs_ppg=fs)

    # PPG PI sub-window IQR
    _pis = []
    _sub_n = int(fs)
    for _i in range(0, len(ppg_raw) - _sub_n + 1, _sub_n):
        _sr = ppg_raw[_i:_i+_sub_n]
        _sb = ppg_bp[_i:_i+_sub_n]
        if len(_sr) < _sub_n:
            continue
        _sac = float(np.sqrt(np.mean(_sb ** 2)))
        _sdc = float(np.median(_sr))
        _pis.append(_sac / (abs(_sdc) + EPS))
    ppg_pi_sub_iqr = _robust_iqr(np.asarray(_pis)) if len(_pis) >= 2 else 0.0

    # ---- Linear-regression slope ----
    _t = np.arange(Np, dtype=np.float64)
    _t_mean = float(np.mean(_t))
    p_mean = float(np.mean(ppg_bp))
    _num = float(np.sum((_t - _t_mean) * (ppg_bp - p_mean)))
    _den = float(np.sum((_t - _t_mean) ** 2))
    slope = (_num / _den) if _den > EPS else 0.0
    _fitted = p_mean + slope * (_t - _t_mean)
    slope_std = float(np.std(ppg_bp - _fitted))

    # ---- Per-feature computation ----
    f = {}
    f["EMG_consensus_PKF_range"] = float(np.max(np.array([emg0_freq[2], emg1_freq[2]], dtype=np.float64)) - np.min(np.array([emg0_freq[2], emg1_freq[2]], dtype=np.float64)))
    f["EMG_consensus_ZC_max"] = float(np.max(np.array([float(np.sum(np.abs(np.diff(np.sign(emg0_bp))))/(2.0*len(emg0_bp))) if emg0_bp is not None else 0.0, float(np.sum(np.abs(np.diff(np.sign(emg1_bp))))/(2.0*len(emg1_bp))) if emg1_bp is not None else 0.0], dtype=np.float64)))

    # ---- Build output vector with fill values and clip bounds ----
    vec = []
    for name in FEATURE_ORDER:
        v = f.get(name, 0.0)
        if v is None or not np.isfinite(v):
            v = FILL_VALUES.get(name, 0.0)
        # Apply training clip bounds (IQR-based, matches s05 clip_outliers)
        bound = CLIP_BOUNDS.get(name)
        if bound is not None and isinstance(bound, (list, tuple)) and len(bound) == 2:
            lo, hi = float(bound[0]), float(bound[1])
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
        vec.append(float(v))
    return vec


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ppg = rng.normal(0, 1, 300)
    emg = rng.normal(0, 1, (3000, 2))
    acc = rng.normal(0, 1, (300, 3))
    vec = extract_features(ppg, emg, acc)
    print(f"Feature vector: {len(vec)} values")
    for i, (name, val) in enumerate(zip(FEATURE_ORDER, vec)):
        print(f"  {i:2d} {name:35s} = {val:.6f}")
