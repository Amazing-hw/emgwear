import numpy as np


def _steady_emg(fs=1000, duration_sec=3, amp=1.0):
    n = int(fs * duration_sec)
    t = np.arange(n, dtype=np.float64) / fs
    ch0 = amp * np.sin(2.0 * np.pi * 80.0 * t)
    ch1 = amp * np.sin(2.0 * np.pi * 82.0 * t)
    return np.column_stack([ch0, ch1])


def _bursty_emg(fs=1000, duration_sec=3):
    emg = _steady_emg(fs=fs, duration_sec=duration_sec, amp=0.15)
    burst_windows = [(0.45, 0.75), (1.35, 1.65), (2.20, 2.45)]
    n = emg.shape[0]
    t = np.arange(n, dtype=np.float64) / fs
    for start, end in burst_windows:
        mask = (t >= start) & (t < end)
        emg[mask, 0] += 3.0 * np.sin(2.0 * np.pi * 95.0 * t[mask])
        emg[mask, 1] += 2.5 * np.sin(2.0 * np.pi * 95.0 * t[mask])
    return emg


def test_emg_feature_pool_includes_subwindow_spectral_and_channel_features_without_burst():
    from s03_extract_feature_pool import extract_emg_features

    feat = extract_emg_features(_bursty_emg(), fs=1000.0)

    expected = [
        "EMG0_RMS_SUBWIN_CV",
        "EMG0_MDF_SUBWIN_IQR",
        "EMG0_WL_SUBWIN_CV",
        "EMG0_SPEC_ENTROPY",
        "EMG0_SPEC_FLATNESS",
        "EMG0_SPEC_CENTROID",
        "EMG0_SPEC_ROLLOFF_85",
        "EMG1_RMS_SUBWIN_CV",
        "EMG1_SPEC_ENTROPY",
        "EMG_ENV_CORR",
        "EMG_MAV_RATIO",
        "EMG_CONTACT_IMBALANCE",
        "EMG0_SAT_FRAC",
        "EMG1_SAT_FRAC",
        "EMG0_CLIP_RATE",
        "EMG1_CLIP_RATE",
        "EMG0_FLATLINE_FRAC",
        "EMG1_FLATLINE_FRAC",
        "EMG0_MNF_SUBWIN_CV",
        "EMG1_MNF_SUBWIN_CV",
        "EMG0_PKF_SUBWIN_IQR",
        "EMG1_PKF_SUBWIN_IQR",
        "EMG0_SPEC_ENTROPY_SUBWIN_CV",
        "EMG1_SPEC_ENTROPY_SUBWIN_CV",
        "EMG_RMS_RATIO_SUBWIN_CV",
        "EMG_ENV_LAG_SEC",
        "EMG0_CLEAN_105_145_RATIO",
        "EMG1_CLEAN_105_145_RATIO",
        "EMG0_CLEAN_155_195_RATIO",
        "EMG1_CLEAN_155_195_RATIO",
        "EMG0_CLEAN_205_245_RATIO",
        "EMG1_CLEAN_205_245_RATIO",
        "EMG0_CLEAN_255_295_RATIO",
        "EMG1_CLEAN_255_295_RATIO",
        "EMG0_CLEAN_TO_NOISE_RATIO",
        "EMG1_CLEAN_TO_NOISE_RATIO",
    ]
    for name in expected:
        assert name in feat
        assert np.isfinite(feat[name])

    forbidden = [
        "EMG0_ENV_BURST_FRAC",
        "EMG0_ENV_BURST_COUNT",
        "EMG0_ENV_DUTY_CYCLE",
        "EMG1_ENV_BURST_FRAC",
        "EMG1_ENV_BURST_COUNT",
        "EMG1_ENV_DUTY_CYCLE",
        "EMG_BURST_SYNC",
        "EMG0_IEMG",
        "EMG1_IEMG",
        "EMG0_VAR",
        "EMG1_VAR",
        "EMG0_PWR_50HZ",
        "EMG1_PWR_50HZ",
        "EMG0_BASELINE_DRIFT_POW",
        "EMG1_BASELINE_DRIFT_POW",
        "EMG0_LEAK_MAX_FREQ",
        "EMG1_LEAK_MAX_FREQ",
    ]
    for name in forbidden:
        assert name not in feat


def test_new_emg_features_are_assigned_to_existing_selection_groups():
    import s04_feature_selection as s04

    expected_groups = {
        "EMG0_RMS_SUBWIN_CV": "emg_activity",
        "EMG0_MNF_SUBWIN_CV": "emg_frequency",
        "EMG0_PKF_SUBWIN_IQR": "emg_frequency",
        "EMG0_SPEC_ENTROPY_SUBWIN_CV": "emg_frequency",
        "EMG0_SPEC_ENTROPY": "emg_frequency",
        "EMG_ENV_CORR": "emg_cross",
        "EMG_RMS_RATIO_SUBWIN_CV": "emg_cross",
        "EMG_ENV_LAG_SEC": "emg_cross",
        "EMG0_SAT_FRAC": "emg_contact",
        "EMG0_CLIP_RATE": "emg_contact",
        "EMG0_FLATLINE_FRAC": "emg_contact",
        "EMG0_CLEAN_105_145_RATIO": "emg_leakage",
        "EMG0_CLEAN_TO_NOISE_RATIO": "emg_leakage",
    }
    for feature, group in expected_groups.items():
        assert s04.feature_to_group(feature) == group

    assert s04.feature_to_group("EMG0_ENV_BURST_FRAC") == "other"
    assert s04.feature_to_group("EMG_BURST_SYNC") == "other"


def test_deploy_code_map_covers_added_emg_features():
    import s08_run_pipeline as s08

    formula_map = s08._build_feature_code_map()
    expected = [
        "EMG0_SAT_FRAC",
        "EMG1_SAT_FRAC",
        "EMG0_CLIP_RATE",
        "EMG1_CLIP_RATE",
        "EMG0_FLATLINE_FRAC",
        "EMG1_FLATLINE_FRAC",
        "EMG0_MNF_SUBWIN_CV",
        "EMG1_MNF_SUBWIN_CV",
        "EMG0_PKF_SUBWIN_IQR",
        "EMG1_PKF_SUBWIN_IQR",
        "EMG0_SPEC_ENTROPY_SUBWIN_CV",
        "EMG1_SPEC_ENTROPY_SUBWIN_CV",
        "EMG_RMS_RATIO_SUBWIN_CV",
        "EMG_ENV_LAG_SEC",
        "EMG0_50HZ_NARROW_RATIO",
        "EMG1_50HZ_NARROW_RATIO",
        "EMG0_CLEAN_105_145_RATIO",
        "EMG1_CLEAN_105_145_RATIO",
        "EMG0_CLEAN_155_195_RATIO",
        "EMG1_CLEAN_155_195_RATIO",
        "EMG0_CLEAN_205_245_RATIO",
        "EMG1_CLEAN_205_245_RATIO",
        "EMG0_CLEAN_255_295_RATIO",
        "EMG1_CLEAN_255_295_RATIO",
        "EMG0_CLEAN_TO_NOISE_RATIO",
        "EMG1_CLEAN_TO_NOISE_RATIO",
    ]

    for name in expected:
        assert name in formula_map
