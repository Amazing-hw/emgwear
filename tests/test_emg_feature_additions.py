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
    ]
    for name in forbidden:
        assert name not in feat


def test_new_emg_features_are_assigned_to_existing_selection_groups():
    import s04_feature_selection as s04

    expected_groups = {
        "EMG0_RMS_SUBWIN_CV": "emg_activity",
        "EMG0_SPEC_ENTROPY": "emg_frequency",
        "EMG_ENV_CORR": "emg_cross",
    }
    for feature, group in expected_groups.items():
        assert s04.feature_to_group(feature) == group

    assert s04.feature_to_group("EMG0_ENV_BURST_FRAC") == "other"
    assert s04.feature_to_group("EMG_BURST_SYNC") == "other"
