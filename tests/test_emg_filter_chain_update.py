import numpy as np


def _tone(fs=1000, duration_sec=3, hz=50.0, amp=20.0, seed=123):
    rng = np.random.default_rng(seed)
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    return rng.normal(0, 1, size=n) + amp * np.sin(2 * np.pi * hz * t)


def test_emg_filter_chain_suppresses_400hz_in_clean_features():
    from s03_extract_feature_pool import (
        extract_emg_frequency_features,
        preprocess_emg_signal_with_raw,
    )
    from scipy.signal import welch

    emg = _tone(hz=400.0, amp=80.0)
    bp_leak_ref, bp_clean, env, x_raw_ref = preprocess_emg_signal_with_raw(emg)

    # 验证 400Hz 被 notch 抑制：clean 信号的 395-405Hz 频段能量应显著低于 ref
    f_ref, p_ref = welch(bp_leak_ref, fs=1000.0, nperseg=512, noverlap=256)
    f_clean, p_clean = welch(bp_clean, fs=1000.0, nperseg=512, noverlap=256)
    ref_band = float(np.sum(p_ref[(f_ref >= 395.0) & (f_ref <= 405.0)]))
    clean_band = float(np.sum(p_clean[(f_clean >= 395.0) & (f_clean <= 405.0)]))

    assert clean_band < ref_band * 0.1


def test_mains_features_include_40_60hz_ratio_for_drifted_mains():
    from s03_extract_feature_pool import extract_emg_features, extract_emg_mains_features

    quiet = _tone(hz=80.0, amp=0.0)
    drifted = _tone(hz=49.6, amp=20.0)

    feat_quiet = extract_emg_mains_features(quiet, fs=1000.0, prefix="TEST")
    feat_drifted = extract_emg_mains_features(drifted, fs=1000.0, prefix="TEST")

    assert "TEST_40_60HZ_RATIO" in feat_drifted
    assert feat_drifted["TEST_40_60HZ_RATIO"] > feat_quiet["TEST_40_60HZ_RATIO"] * 5

    all_feat = extract_emg_features(drifted.reshape(-1, 1), fs=1000.0)
    assert "EMG0_40_60HZ_RATIO" in all_feat
    assert "EMG1_40_60HZ_RATIO" in all_feat
    assert all_feat["EMG1_40_60HZ_RATIO"] == 0.0
