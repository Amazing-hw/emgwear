import numpy as np


def _synthetic_emg_with_tone(fs=1000, duration_sec=3, tone_hz=100.0, tone_amp=50.0):
    rng = np.random.default_rng(42)
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    emg = rng.normal(0, 1, size=n)
    emg += tone_amp * np.sin(2 * np.pi * tone_hz * t)
    return emg


def test_leak_ratio_detects_100hz_crosstalk():
    from s03_extract_feature_pool import extract_emg_leakage_features, preprocess_emg_signal_with_raw

    emg = _synthetic_emg_with_tone(tone_hz=100.0, tone_amp=50.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)
    feat = extract_emg_leakage_features(bp_leak_ref, fs=1000.0, prefix="EMG0")

    assert feat["EMG0_LEAK_100_RATIO"] > 0.1


def test_leak_ratio_low_for_clean_signal():
    from s03_extract_feature_pool import extract_emg_leakage_features, preprocess_emg_signal_with_raw

    rng = np.random.default_rng(99)
    emg = rng.normal(0, 1, size=3000)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)
    feat = extract_emg_leakage_features(bp_leak_ref, fs=1000.0, prefix="EMG0")

    assert feat["EMG0_LEAK_100_RATIO"] < 0.3
    assert feat["EMG0_LEAK_SUM_RATIO"] < 0.5
    assert feat["EMG0_LEAK_MAX_RATIO"] < 0.2


def test_filter_chain_suppresses_150hz_peak():
    from s03_extract_feature_pool import (
        preprocess_emg_signal_with_raw,
    )
    from scipy.signal import welch

    emg = _synthetic_emg_with_tone(tone_hz=150.0, tone_amp=50.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)

    f_ref, p_ref = welch(bp_leak_ref, fs=1000.0, nperseg=512, noverlap=256)
    f_clean, p_clean = welch(bp_clean, fs=1000.0, nperseg=512, noverlap=256)
    ref_band = float(np.sum(p_ref[(f_ref >= 148.0) & (f_ref <= 152.0)]))
    clean_band = float(np.sum(p_clean[(f_clean >= 148.0) & (f_clean <= 152.0)]))

    assert clean_band < ref_band * 0.1


def test_50hz_filter_chain_covers_drifted_mains():
    from s03_extract_feature_pool import preprocess_emg_signal_with_raw

    emg = _synthetic_emg_with_tone(tone_hz=49.5, tone_amp=30.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)

    assert float(np.std(bp_clean)) < float(np.std(bp_leak_ref)) * 0.9


def test_mains_features_on_leak_ref():
    from s03_extract_feature_pool import extract_emg_mains_features, preprocess_emg_signal_with_raw

    rng = np.random.default_rng(1)
    emg_quiet = rng.normal(0, 1, size=3000)
    t = np.arange(3000) / 1000.0
    emg_loud = emg_quiet + 20.0 * np.sin(2 * np.pi * 50.0 * t)

    bp_leak_quiet, _, _, _ = preprocess_emg_signal_with_raw(emg_quiet)
    bp_leak_loud, _, _, _ = preprocess_emg_signal_with_raw(emg_loud)

    feat_quiet = extract_emg_mains_features(bp_leak_quiet, fs=1000.0, prefix="TEST")
    feat_loud = extract_emg_mains_features(bp_leak_loud, fs=1000.0, prefix="TEST")

    assert feat_loud["TEST_PWR_50HZ"] > feat_quiet["TEST_PWR_50HZ"]
    assert feat_loud["TEST_50HZ_RATIO"] > feat_quiet["TEST_50HZ_RATIO"] * 5
    assert feat_loud["TEST_40_60HZ_RATIO"] > feat_quiet["TEST_40_60HZ_RATIO"] * 5


def test_emg_missing_all_leak_and_mains_features_zero():
    from s03_extract_feature_pool import extract_emg_features

    feat = extract_emg_features(None, fs=1000.0)

    for ch in [0, 1]:
        assert feat[f"EMG{ch}_LEAK_100_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_150_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_SUM_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_MAX_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_MAX_FREQ"] == 0.0
        assert feat[f"EMG{ch}_40_60HZ_RATIO"] == 0.0


def test_emg_single_channel_leak_features():
    from s03_extract_feature_pool import extract_emg_features

    rng = np.random.default_rng(7)
    emg = rng.normal(0, 1, size=(3000, 1))

    feat = extract_emg_features(emg, fs=1000.0)

    assert feat["EMG0_LEAK_100_RATIO"] >= 0.0
    assert feat["EMG0_LEAK_SUM_RATIO"] >= 0.0
    assert feat["EMG1_LEAK_100_RATIO"] == 0.0
    assert feat["EMG1_LEAK_SUM_RATIO"] == 0.0


def test_preprocess_returns_four_values():
    from s03_extract_feature_pool import preprocess_emg_signal_with_raw

    rng = np.random.default_rng(3)
    emg = rng.normal(0, 1, size=3000)
    result = preprocess_emg_signal_with_raw(emg)

    assert len(result) == 4
    bp_leak_ref, bp_clean, env, x_demean = result
    assert bp_leak_ref is not None
    assert bp_clean is not None
    assert env is not None
    assert x_demean is not None
    assert len(bp_leak_ref) == len(bp_clean) == len(emg)


def test_leak_features_names_consistent():
    from s03_extract_feature_pool import extract_emg_features

    rng = np.random.default_rng(5)
    emg = rng.normal(0, 1, size=(3000, 2))
    feat = extract_emg_features(emg, fs=1000.0)

    expected_leak_keys = []
    for ch in [0, 1]:
        for hz in [100, 150, 200, 250, 300]:
            expected_leak_keys.append(f"EMG{ch}_LEAK_{hz}_RATIO")
        expected_leak_keys.append(f"EMG{ch}_LEAK_SUM_RATIO")
        expected_leak_keys.append(f"EMG{ch}_LEAK_MAX_RATIO")
        expected_leak_keys.append(f"EMG{ch}_LEAK_MAX_FREQ")

    for key in expected_leak_keys:
        assert key in feat
        assert isinstance(feat[key], float)


def test_leak_max_freq_is_valid():
    from s03_extract_feature_pool import extract_emg_features, _EMG_LEAK_FREQS

    emg0 = _synthetic_emg_with_tone(tone_hz=200.0, tone_amp=80.0)
    emg1 = _synthetic_emg_with_tone(tone_hz=150.0, tone_amp=30.0)
    emg = np.column_stack([emg0, emg1])

    feat = extract_emg_features(emg, fs=1000.0)

    assert feat["EMG0_LEAK_MAX_FREQ"] in _EMG_LEAK_FREQS
    assert feat["EMG0_LEAK_200_RATIO"] > feat["EMG0_LEAK_100_RATIO"]


def test_notch_config_constant_values():
    from s03_extract_feature_pool import (
        _EMG_CLEAN_BANDSTOP_RANGES,
        _EMG_CLEAN_NOTCH_FREQS,
        _EMG_CLEAN_NOTCH_Q,
        _EMG_LEAK_FREQS,
        _EMG_NOTCH_BW_HZ,
    )

    assert _EMG_NOTCH_BW_HZ == 0.8
    assert (49.8, 50.2) in _EMG_CLEAN_BANDSTOP_RANGES
    assert (149.8, 150.2) in _EMG_CLEAN_BANDSTOP_RANGES
    assert 50.0 in _EMG_CLEAN_NOTCH_FREQS
    assert 100.0 in _EMG_CLEAN_NOTCH_FREQS
    assert 150.0 not in _EMG_CLEAN_NOTCH_FREQS
    assert 250.0 not in _EMG_CLEAN_NOTCH_FREQS
    assert 400.0 in _EMG_CLEAN_NOTCH_FREQS
    assert _EMG_CLEAN_NOTCH_Q == 100.0
    assert 50.0 not in _EMG_LEAK_FREQS
    assert 300.0 in _EMG_LEAK_FREQS
