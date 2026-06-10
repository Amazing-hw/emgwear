import json
from pathlib import Path

import numpy as np

import s06_deploy_eval as s06
import s07_postprocess_optimize as s07


def _cache(sample_name, target, probs, stage1=None, quality=None, ood=None):
    n = len(probs)
    return {
        "sample_name": sample_name,
        "target": int(target),
        "prob_raw": np.asarray(probs, dtype=float),
        "stage1_enabled": np.asarray(stage1 if stage1 is not None else [1] * n, dtype=np.int8),
        "quality": np.asarray(quality if quality is not None else [1.0] * n, dtype=float),
        "ood_rate": np.asarray(ood if ood is not None else [0.0] * n, dtype=float),
        "stage1_dc": np.asarray([10.0] * n, dtype=float),
        "stage1_acdc": np.asarray([0.1] * n, dtype=float),
        "stage1_dc_margin": np.asarray([2.0] * n, dtype=float),
        "stage1_acdc_margin": np.asarray([0.2] * n, dtype=float),
        "model_threshold": 0.5,
        "window_sec": 3.0,
        "stride_sec": 1.0,
    }


def test_window_cache_contains_stage1_margin_fields():
    result = {
        "sample_name": "margin-case",
        "target": 0,
        "stage1_pass": True,
        "fallback": False,
        "window_probs": [0.1, 0.2],
        "window_preds": [0, 0],
        "quality_metas": [{}, {}],
        "stage1_frame_results": [True, False],
        "stage1_window_metrics": [
            {"dc": 5.0, "acdc": 0.10, "dc_margin": 2.0, "acdc_margin": 0.25},
            {"dc": 2.0, "acdc": 0.50, "dc_margin": -1.0, "acdc_margin": -0.15},
        ],
    }
    out_dir = Path.cwd() / "test_outputs" / "stage1_margin_cache"
    try:
        path = s06.write_window_cache_npz(
            result,
            out_dir,
            window_sec=3.0,
            stride_sec=1.0,
            model_threshold=0.5,
            metadata={
                "model_fingerprint_json": json.dumps({}),
                "feature_names_json": json.dumps([]),
            },
        )

        loaded = s07.load_window_cache_npz(path)

        assert loaded["stage1_dc"].tolist() == [5.0, 2.0]
        assert loaded["stage1_acdc"].tolist() == [0.10, 0.50]
        assert loaded["stage1_dc_margin"].tolist() == [2.0, -1.0]
        assert loaded["stage1_acdc_margin"].tolist() == [0.25, -0.15]
    finally:
        for p in out_dir.glob("*.npz"):
            p.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass


def test_scan_window_thresholds_can_skip_initial_windows():
    caches = [
        _cache("neg", 0, [0.9, 0.1, 0.1]),
        _cache("pos", 1, [0.1, 0.9, 0.9]),
    ]

    no_skip = s07.scan_window_thresholds(caches, [0.5], skip_initial_windows=0)
    skip_one = s07.scan_window_thresholds(caches, [0.5], skip_initial_windows=1)

    assert no_skip.iloc[0]["accuracy"] < 1.0
    assert skip_one.iloc[0]["accuracy"] == 1.0
    assert skip_one.iloc[0]["skipped_initial_windows"] == 2


def test_build_window_error_report_stratifies_window_errors():
    caches = [
        _cache("neg", 0, [0.8, 0.7, 0.2], quality=[1.0, 1.0, 1.0], ood=[0.4, 0.0, 0.0]),
        _cache("pos", 1, [0.2, 0.9], stage1=[0, 1], quality=[0.3, 1.0]),
    ]

    report = s07.build_window_error_report(caches, model_threshold=0.5, skip_initial_windows=0)
    tags = set(report["diagnostic_tag"])

    assert "model_fp" in tags
    assert "stage1_blocked" in tags
    assert "correct" in tags
    assert {"sample_name", "window_index", "target", "prob_raw", "window_pred", "diagnostic_tag"}.issubset(report.columns)
