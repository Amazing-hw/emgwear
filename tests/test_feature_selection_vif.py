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
