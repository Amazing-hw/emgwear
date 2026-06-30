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

    emg = _tone(hz=400.0, amp=80.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)

    feat_ref = extract_emg_frequency_features(bp_leak_ref, fs=1000.0, prefix="EMG")
    feat_clean = extract_emg_frequency_features(bp_clean, fs=1000.0, prefix="EMG")

    assert abs(feat_ref["EMG_PKF"] - 400.0) < 10.0
    assert abs(feat_clean["EMG_PKF"] - 400.0) > 20.0


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
