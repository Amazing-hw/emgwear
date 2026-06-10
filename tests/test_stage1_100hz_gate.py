import numpy as np


def _ppg_6ch_with_mean(mean_signal):
    mean_signal = np.asarray(mean_signal, dtype=np.float64)
    offsets = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], dtype=np.float64)
    return mean_signal[:, None] + offsets[None, :]


def test_stage1_sample_pass_uses_6ch_mean_100hz_three_second_windows():
    """Stage1 使用 100Hz 3s 窗 (300 samples)，不降采样。"""
    import s03_extract_feature_pool as s03

    ppg = _ppg_6ch_with_mean(np.full(300, 10.0))

    # 300 samples @ 100Hz = 3s 窗口
    assert s03.stage1_sample_pass(ppg, dc_threshold=1.0, ac_dc_threshold=0.1)

    # 窗口不足 3s 应正确拒绝 (STAGE1_WINDOW_SEC=3, STAGE1_STRIDE_SEC=1)
    ppg_short = _ppg_6ch_with_mean(np.full(200, 10.0))
    assert not s03.stage1_sample_pass(ppg_short, dc_threshold=1.0, ac_dc_threshold=0.1)


def test_stage1_threshold_features_use_100hz_windows():
    """Stage1 阈值学习的 DC/ACDC 特征基于 100Hz 窗口计算。"""
    import s02_ir_dc_threshold as s02

    # 3s @ 100Hz = 300 samples
    ppg = np.full(300, 10.0, dtype=np.float64)

    feats = s02.extract_dc_acdc_features(ppg)

    assert len(feats) == 1
    # 常值信号: DC ≈ 10.0, AC ≈ 0
    assert feats[0]["dc"] == 10.0
    assert feats[0]["ac_dc_ratio"] == 0.0

    # 窗口不足 3s 应返回空
    ppg_short = np.full(200, 10.0, dtype=np.float64)
    assert len(s02.extract_dc_acdc_features(ppg_short)) == 0


def test_stage1_threshold_artifact_uses_fixed_engineering_threshold():
    import pandas as pd
    import s02_ir_dc_threshold as s02

    df = pd.DataFrame({
        "sample_name": ["p1", "n1"],
        "target": [1, 0],
        "dc": [3.0e6, 1.0e6],
        "ac_dc_ratio": [0.01, 0.02],
    })

    result = s02.build_fixed_stage1_threshold_artifact(df, df)

    assert result["threshold_search_enabled"] is False
    assert result["dc_threshold"] == 2.2e6
    assert result["ac_dc_threshold"] == 0.35
    assert result["deploy_stage1_threshold"]["dc_threshold"] == 2.2e6
    assert result["deploy_stage1_threshold"]["ac_dc_threshold"] == 0.35
    assert result["deploy_stage1_threshold"]["search_source"] == "fixed_engineering_threshold"
    assert result["deploy_stage1_threshold"]["fs"] == 100


def test_stage1_main_ignores_legacy_search_method(monkeypatch):
    import json
    import shutil
    import uuid
    from pathlib import Path

    import pandas as pd
    import s02_ir_dc_threshold as s02

    artifact_dir = Path.cwd() / ".test_stage1_fixed" / uuid.uuid4().hex
    artifact_dir.mkdir(parents=True)
    try:
        (artifact_dir / "splits.json").write_text(
            json.dumps({"train": [], "valid": [], "test": []}),
            encoding="utf-8",
        )

        df = pd.DataFrame({
            "sample_name": ["p1", "n1"],
            "target": [1, 0],
            "dc": [3.0e6, 1.0e6],
            "ac_dc_ratio": [0.01, 0.02],
        })
        monkeypatch.setattr(s02, "extract_stage1_windows", lambda *args, **kwargs: df.copy())
        monkeypatch.setattr(s02, "plot_stage1_scatter", lambda *args, **kwargs: None)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Stage1 threshold search should be disabled")

        monkeypatch.setattr(s02, "search_deploy_threshold", fail_if_called)
        monkeypatch.setattr(s02, "search_deploy_threshold_multi_stage", fail_if_called)

        s02.main([
            "--artifact_dir", str(artifact_dir),
            "--search_method", "grid",
            "--fixed_dc_threshold", "2200000",
        ])

        out = json.loads((artifact_dir / "stage1_threshold.json").read_text(encoding="utf-8"))
        assert out["threshold_search_enabled"] is False
        assert out["deploy_stage1_threshold"]["search_source"] == "fixed_engineering_threshold"
        assert out["deploy_stage1_threshold"]["dc_threshold"] == 2.2e6
    finally:
        shutil.rmtree(artifact_dir.parent, ignore_errors=True)



def test_stage1_uses_mean_not_ch_a():
    import s03_extract_feature_pool as s03

    ppg = np.zeros((300, 6), dtype=np.float64)
    ppg[:, 0] = 1.0
    ppg[:, 1] = 1.0
    ppg[:, 2:] = 10.0

    assert not s03.stage1_sample_pass(ppg, dc_threshold=8.0, ac_dc_threshold=0.1)
    assert s03.stage1_sample_pass(ppg, dc_threshold=5.0, ac_dc_threshold=0.1)


def test_stage2_window_features_include_emg_and_acc():
    from s03_extract_feature_pool import extract_feature_pool_from_window

    rng = np.random.default_rng(42)
    ppg = np.column_stack([
        2.0e6 + 1000.0 * np.sin(np.linspace(0, 6 * np.pi, 300)),
        2.1e6 + 900.0 * np.sin(np.linspace(0, 6 * np.pi, 300) + 0.1),
        2.2e6 + 800.0 * np.sin(np.linspace(0, 6 * np.pi, 300) + 0.2),
    ])
    emg = rng.normal(0.0, 1.0, size=(3000, 2))
    acc = rng.normal(0.0, 0.01, size=(300, 3))

    feat = extract_feature_pool_from_window(ppg, emg, acc)

    assert "EMG0_RMS" in feat
    assert "EMG1_RMS" in feat
    assert "EMG_CROSS_CORR" in feat
    assert "ACC_MOTION_RMS" in feat or "ACC_MAG_MEAN" in feat
