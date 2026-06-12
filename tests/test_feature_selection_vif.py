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
