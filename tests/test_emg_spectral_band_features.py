import numpy as np


FINE_BAND_KEYS = [
    "POW_20_40",
    "POW_40_60",
    "POW_60_90",
    "POW_90_120",
    "POW_120_180",
    "POW_180_250",
    "POW_250_350",
    "POW_350_450",
]

RATIO_KEYS = [
    "RATIO_60_150_TO_20_60",
    "RATIO_60_180_TO_250_450",
    "RATIO_20_90_TO_180_450",
    "RATIO_40_120_TO_120_350",
]


def _broadband_emg(fs=1000, duration_sec=3):
    rng = np.random.default_rng(51)
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    x = rng.normal(0, 0.2, size=n)
    for hz, amp in [(35, 1.0), (75, 0.8), (135, 0.6), (220, 0.4), (360, 0.3)]:
        x += amp * np.sin(2 * np.pi * hz * t)
    return x


def test_emg_frequency_features_include_fine_band_ratios():
    from s03_extract_feature_pool import extract_emg_frequency_features

    feat = extract_emg_frequency_features(_broadband_emg(), fs=1000.0, prefix="EMG")

    for key in FINE_BAND_KEYS + RATIO_KEYS:
        full_key = f"EMG_{key}"
        assert full_key in feat
        assert np.isfinite(feat[full_key])
        assert feat[full_key] >= 0.0

    fine_total = sum(feat[f"EMG_{key}"] for key in FINE_BAND_KEYS)
    assert 0.95 <= fine_total <= 1.05


def test_emg_feature_pool_zero_fills_fine_band_ratios_when_emg_missing():
    from s03_extract_feature_pool import extract_emg_features

    feat = extract_emg_features(None, fs=1000.0)

    for ch in [0, 1]:
        for key in FINE_BAND_KEYS + RATIO_KEYS:
            assert feat[f"EMG{ch}_{key}"] == 0.0


def test_deploy_and_formula_docs_cover_fine_band_features():
    import s08_run_pipeline as s08
    import s06_deploy_eval as s06

    features = [f"EMG0_{key}" for key in FINE_BAND_KEYS + RATIO_KEYS]
    features += [f"EMG1_{key}" for key in FINE_BAND_KEYS + RATIO_KEYS]

    deploy_map = s08._build_feature_code_map()
    missing_deploy = sorted(set(features) - set(deploy_map))
    assert missing_deploy == []

    docs = s06.build_feature_formula_map(features)
    s06.validate_feature_formula_map(docs)
    for info in docs.values():
        assert "未匹配" not in info["formula"]
        assert info["category"] != "unknown"
