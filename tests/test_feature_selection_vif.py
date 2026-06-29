import numpy as np
import pandas as pd


def test_vif_values_flag_collinear_features():
    import s04_feature_selection as s04

    rng = np.random.default_rng(123)
    x0 = rng.normal(size=200)
    x1 = x0 * 1.01 + rng.normal(scale=0.01, size=200)
    x2 = rng.normal(size=200)
    X = np.column_stack([x0, x1, x2])

    vif = s04.compute_vif_values(X)

    assert vif.shape == (3,)
    assert vif[0] > 10.0
    assert vif[1] > 10.0
    assert vif[2] < 2.0


def test_clean_features_removes_high_vif_features():
    import s04_feature_selection as s04

    rng = np.random.default_rng(456)
    n = 120
    x0 = rng.normal(size=n)
    df = pd.DataFrame({
        "sample_name": [f"s{i}" for i in range(n)],
        "h5_file": ["dummy.h5"] * n,
        "target": (x0 > 0).astype(int),
        "start_100hz": np.arange(n),
        "F0": x0,
        "F1": x0 * 1.02 + rng.normal(scale=0.01, size=n),
        "F2": rng.normal(size=n),
    })

    _, _, kept, removed, _ = s04.clean_features_by_train(
        df,
        df.copy(),
        ["F0", "F1", "F2"],
        corr_thresh=1.1,
    )

    assert len(kept) < 3
    assert removed["high_vif"]


def test_get_feature_cols_excludes_non_deploy_diagnostic_columns():
    import s04_feature_selection as s04

    df = pd.DataFrame({
        "sample_name": ["s1", "s2"],
        "h5_file": ["a.h5", "b.h5"],
        "target": [0, 1],
        "start_100hz": [0, 100],
        "TOTAL_INVALID_COUNT": [1.0, 0.0],
        "EMG_consensus_WL_max": [2.0, 3.0],
    })

    cols = s04.get_feature_cols(df)

    assert "TOTAL_INVALID_COUNT" not in cols
    assert "EMG_consensus_WL_max" in cols


def test_get_feature_cols_excludes_features_without_deploy_formula():
    import s04_feature_selection as s04

    df = pd.DataFrame({
        "sample_name": ["s1", "s2"],
        "h5_file": ["a.h5", "b.h5"],
        "target": [0, 1],
        "start_100hz": [0, 100],
        "PPG_mean": [1.0, 2.0],
        "UNSUPPORTED_MODEL_FEATURE": [3.0, 4.0],
    })

    cols = s04.get_feature_cols(df)

    assert "PPG_mean" in cols
    assert "UNSUPPORTED_MODEL_FEATURE" not in cols


def test_emg_consensus_features_have_dedicated_group_and_redundant_features_are_other():
    import s04_feature_selection as s04

    consensus_features = [
        "EMG_consensus_RMS_min",
        "EMG_consensus_RMS_max",
        "EMG_consensus_RMS_range",
        "EMG_consensus_RMS_cv",
        "EMG_consensus_MDF_min",
        "EMG_consensus_PSR_range",
    ]
    for feature in consensus_features:
        assert s04.feature_to_group(feature) == "emg_consensus"

    redundant_features = [
        "EMG0_IEMG",
        "EMG1_IEMG",
        "EMG0_VAR",
        "EMG1_VAR",
        "EMG0_LEAK_MAX_FREQ",
        "EMG1_LEAK_MAX_FREQ",
        "EMG0_PWR_50HZ",
        "EMG1_PWR_50HZ",
        "EMG0_BASELINE_DRIFT_POW",
        "EMG1_BASELINE_DRIFT_POW",
    ]
    for feature in redundant_features:
        assert s04.feature_to_group(feature) == "other"


def test_default_emg_group_limits_balance_frequency_cross_leakage_and_consensus():
    import s04_feature_selection as s04

    limits = s04.GROUP_LIMITS_DEFAULT

    assert limits["emg_frequency"] == 4
    assert limits["emg_cross"] == 2
    assert limits["emg_leakage"] == 4
    assert limits["emg_consensus"] == 2


def test_group_limit_zero_excludes_features_from_selection_and_ranked_candidates():
    import s04_feature_selection as s04

    summary = [
        {"feature": "ACC_MOTION_RMS", "group": "acc_features", "combined_score": 0.99},
        {"feature": "PPG_DICROTIC_RATIO", "group": "anti_spoof", "combined_score": 0.98},
        {"feature": "PPG_ch_vmag_mean", "group": "ppg_spatial", "combined_score": 0.97},
        {"feature": "PPG_mean", "group": "ppg_quality", "combined_score": 0.50},
        {"feature": "EMG0_MAV", "group": "emg_contact", "combined_score": 0.40},
    ]
    limits = dict(s04.GROUP_LIMITS_DEFAULT)
    limits["acc_features"] = 0
    limits["anti_spoof"] = 0
    limits["ppg_spatial"] = 0

    selected, group_count = s04.select_by_group_from_combined(
        summary,
        max_features=5,
        group_limits=limits,
        min_acc_features=1,
        min_anti_spoof_features=3,
    )
    ranked = s04.filter_summary_by_group_limits(summary, limits)

    assert selected == ["PPG_mean", "EMG0_MAV"]
    assert group_count == {"ppg_quality": 1, "emg_contact": 1}
    assert [item["feature"] for item in ranked] == ["PPG_mean", "EMG0_MAV"]


def test_fast_group_preselection_skips_zero_limit_tiny_groups():
    import s04_feature_selection as s04

    df = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "PPG_mean": [1.0, 2.0, 1.1, 2.1],
        "PPG_std": [0.1, 0.2, 0.1, 0.2],
        "EMG0_MAV": [0.0, 1.0, 0.0, 1.0],
    })
    limits = dict(s04.GROUP_LIMITS_DEFAULT)
    limits["ppg_quality"] = 0

    selected = s04.fast_group_preselection(
        df,
        ["PPG_mean", "PPG_std", "EMG0_MAV"],
        group_limits=limits,
    )

    assert "PPG_mean" not in selected
    assert "PPG_std" not in selected
    assert "EMG0_MAV" in selected


def test_stability_selection_handles_mixed_type_sample_names():
    import s04_feature_selection as s04

    rng = np.random.default_rng(789)
    n = 24
    target = np.tile([0, 1], n // 2)
    sample_names = [
        i if i % 2 == 0 else f"s{i}"
        for i in range(n)
    ]
    df = pd.DataFrame({
        "sample_name": sample_names,
        "h5_file": ["mixed.h5"] * n,
        "target": target,
        "start_100hz": np.arange(n),
        "F0": target + rng.normal(scale=0.01, size=n),
        "F1": rng.normal(size=n),
    })

    summary = s04.stability_selection(
        df,
        ["F0", "F1"],
        max_splits=3,
        seeds=[42],
        n_workers=1,
        min_fold_auc=0.0,
    )

    assert {item["feature"] for item in summary} == {"F0", "F1"}
