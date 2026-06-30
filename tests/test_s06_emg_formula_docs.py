import s06_deploy_eval as s06


def test_s06_formula_map_covers_s03_generated_deployable_features():
    import numpy as np
    import s03_extract_feature_pool as s03
    import s04_feature_selection as s04

    rng = np.random.default_rng(29)
    ppg_6ch = rng.normal(50000, 4000, size=(300, 6))
    emg = rng.normal(0, 1, size=(3000, 2))
    acc = rng.normal(0, 1, size=(300, 3))
    generated = s03.extract_feature_pool_from_window(
        s03.build_3ch_ppg(ppg_6ch),
        emg,
        acc,
    )
    non_deploy = set(getattr(s04, "NON_DEPLOY_FEATURES", set()))
    deployable = sorted(k for k in generated if k not in non_deploy)

    formulas = s06.build_feature_formula_map(deployable)

    s06.validate_feature_formula_map(formulas)


def test_deploy_feature_formula_map_documents_new_emg_features():
    formulas = s06.build_feature_formula_map([
        "EMG0_RMS_SUBWIN_CV",
        "EMG1_MDF_SUBWIN_IQR",
        "EMG0_SPEC_ENTROPY",
        "EMG1_SPEC_ROLLOFF_85",
        "EMG_ENV_CORR",
        "EMG_MAV_RATIO",
        "EMG_CONTACT_IMBALANCE",
        "EMG0_40_60HZ_RATIO",
        "EMG1_40_60HZ_RATIO",
    ])

    s06.validate_feature_formula_map(formulas)
    for info in formulas.values():
        assert "未匹配" not in info["formula"]
        assert info["category"] != "unknown"
