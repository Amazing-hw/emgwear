import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import s06_deploy_eval as s06


def test_s06_rejects_test_split_for_state_machine_optimization():
    args = type("Args", (), {
        "artifact_dir": "artifacts",
        "split": "test",
        "method": "state_machine",
        "window_sec": 3,
        "stride_sec": 1,
        "optimize": True,
        "optimize_split": "test",
        "n_workers": 1,
        "warmup_frames": 3,
        "ood_alert_rate": 0.3,
        "export_deploy": False,
        "export_window_cache": False,
        "window_output_root": "window_outputs",
        "optimize_thresholds": "",
    })()

    with pytest.raises(ValueError, match="test split.*optimization"):
        s06.main(args)


def test_s07_rejects_test_split_for_postprocess_optimization():
    import s07_postprocess_optimize as s07

    args = type("Args", (), {
        "artifact_dir": "artifacts",
        "split": "test",
        "cache_root": "window_outputs",
        "fp_cost": 4.0,
        "skip_initial_windows": 0,
        "workers": 1,
        "thresholds": "0.3,0.4",
        "search_splits": "train,valid",
        "hard_samples_only": True,
        "threshold_offsets": "-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3",
        "max_all_correct_regressions": 0,
    })()

    with pytest.raises(ValueError, match="test split.*postprocess"):
        s07.main(args)


def test_state_machine_zeroes_windows_when_stage1_disabled():
    cfg = {
        "alpha": 1.0,
        "T_on": 0.5,
        "T_off": 0.2,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0,
    }

    final_pred, states, window_preds, scores = s06.apply_postprocess(
        [0.99, 0.99, 0.99],
        quality_metas=[{}, {}, {}],
        method="state_machine",
        cfg=cfg,
        model_threshold=0.5,
        stride_sec=1.0,
        stage1_frames=[False, False, False],
    )

    assert final_pred == 0
    assert states == [0, 0, 0]
    assert window_preds == [0, 0, 0]
    assert scores == [0.0, 0.0, 0.0]


def test_gated_is_state_machine_compatibility_alias():
    cfg = {
        "alpha": 1.0,
        "T_on": 0.5,
        "T_off": 0.2,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0,
    }
    kwargs = {
        "window_probs": [0.9, 0.9, 0.9],
        "quality_metas": [{}, {}, {}],
        "cfg": cfg,
        "model_threshold": 0.5,
        "stride_sec": 1.0,
        "stage1_frames": [True, False, True],
    }

    state_machine = s06.apply_postprocess(method="state_machine", **kwargs)
    gated_alias = s06.apply_postprocess(method="gated", **kwargs)

    assert gated_alias == state_machine


def test_stage1_gate_applies_to_non_state_machine_methods():
    cfg = {"median_k": 1}

    mean_pred, _states, mean_window_preds, _scores = s06.apply_postprocess(
        [0.9, 0.9, 0.9],
        quality_metas=[{}, {}, {}],
        method="mean_vote",
        cfg=cfg,
        model_threshold=0.5,
        stride_sec=1.0,
        stage1_frames=[False, False, False],
    )
    prob_pred, _states, prob_window_preds, _scores = s06.apply_postprocess(
        [0.9, 0.9, 0.9],
        quality_metas=[{}, {}, {}],
        method="prob_mean",
        cfg=cfg,
        model_threshold=0.5,
        stride_sec=1.0,
        stage1_frames=[False, False, False],
    )

    assert mean_pred == 0
    assert prob_pred == 0
    assert mean_window_preds == [0, 0, 0]
    assert prob_window_preds == [0, 0, 0]


def test_window_stream_metrics_uses_same_median_states_as_details():
    cfg = {
        "alpha": 1.0,
        "T_on": 0.5,
        "T_off": 0.2,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0,
        "median_k": 3,
    }
    results = [{
        "sample_name": "sample_a",
        "target": 0,
        "stage1_pass": True,
        "fallback": False,
        "window_probs": [0.0, 1.0, 0.0],
        "window_preds": [0, 1, 0],
        "quality_metas": [{}, {}, {}],
        "stage1_frame_results": [True, True, True],
    }]

    _summary, details = s06.compute_sample_metrics(
        results, "state_machine", cfg, model_threshold=0.5, stride_sec=1.0
    )
    stream = s06.compute_window_stream_metrics(
        results, cfg, warmup_frames=0, stride_sec=1.0
    )

    assert details[0]["window_states"] == [0, 0, 0]
    assert stream["confusion_matrix"] == {"TN": 3, "FP": 0, "FN": 0, "TP": 0}


def test_window_cache_roundtrip_and_postprocess():
    result = {
        "sample_name": "case/001",
        "target": 1,
        "stage1_pass": True,
        "fallback": False,
        "window_probs": [0.9, 0.8, 0.7],
        "window_preds": [1, 1, 1],
        "quality_metas": [{}, {}, {}],
        "stage1_frame_results": [True, False, True],
        "window_ood_scores": [0.0, 0.1, 0.2],
    }
    metadata = {
        "model_fingerprint_json": json.dumps({"features": 3}),
        "feature_names_json": json.dumps(["a", "b", "c"]),
    }

    out_dir = Path.cwd() / "test_outputs" / "window_cache_roundtrip"
    try:
        path = s06.write_window_cache_npz(
            result,
            out_dir,
            window_sec=3.0,
            stride_sec=1.0,
            model_threshold=0.5,
            metadata=metadata,
        )

        import s07_postprocess_optimize as s07

        cache = s07.load_window_cache_npz(path)
        assert cache["sample_name"] == "case/001"
        assert cache["target"] == 1
        assert cache["model_fingerprint_json"] == metadata["model_fingerprint_json"]
        assert cache["stage1_enabled"].tolist() == [1, 0, 1]

        final_pred, states, window_preds, scores = s07.run_postprocess_on_cache(
            cache,
            {
                "alpha": 1.0,
                "T_on": 0.5,
                "T_off": 0.2,
                "K_on": 1,
                "K_off": 1,
                "cooldown_sec": 0,
                "median_k": 1,
            },
            model_threshold=0.5,
        )

        assert window_preds == [1, 0, 1]
        assert states == [1, 0, 1]
        assert final_pred == 1
        assert scores == [0.9, 0.0, 0.7]
    finally:
        for p in out_dir.glob("*.npz"):
            p.unlink()
        manifest = out_dir / "manifest.csv"
        if manifest.exists():
            manifest.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass


def test_cache_quality_value_affects_state_machine_score():
    import s07_postprocess_optimize as s07

    cache = {
        "sample_name": "quality_case",
        "target": 1,
        "prob_raw": np.array([1.0], dtype=float),
        "stage1_enabled": np.array([1], dtype=np.int8),
        "quality": np.array([0.25], dtype=float),
        "model_threshold": 0.5,
        "stride_sec": 1.0,
    }

    _pred, _states, _window_preds, scores = s07.run_postprocess_on_cache(
        cache,
        {
            "alpha": 1.0,
            "T_on": 0.9,
            "T_off": 0.2,
            "K_on": 1,
            "K_off": 1,
            "cooldown_sec": 0,
            "median_k": 1,
        },
        model_threshold=0.5,
    )

    assert scores == [0.25]


def test_threshold_override_recenters_probabilities_for_state_machine():
    import s07_postprocess_optimize as s07

    cache = {
        "sample_name": "threshold_case",
        "target": 1,
        "prob_raw": np.array([0.65], dtype=float),
        "stage1_enabled": np.array([1], dtype=np.int8),
        "quality": np.array([1.0], dtype=float),
        "model_threshold": 0.5,
        "stride_sec": 1.0,
    }
    cfg = {
        "alpha": 1.0,
        "T_on": 0.5,
        "T_off": 0.2,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0,
        "median_k": 1,
        "threshold_offset": 0.0,
    }

    low_threshold = s07.run_postprocess_on_cache(cache, cfg, model_threshold=0.5)
    high_threshold = s07.run_postprocess_on_cache(cache, cfg, model_threshold=0.8)

    assert low_threshold[0] == 1
    assert high_threshold[0] == 0
    assert low_threshold[3] == [0.65]
    assert high_threshold[3] == [0.35]


def test_hard_sample_filter_keeps_only_window_error_samples():
    import s07_postprocess_optimize as s07

    caches = [
        {
            "sample_name": "all_correct_pos",
            "target": 1,
            "prob_raw": np.array([0.8, 0.7], dtype=float),
            "stage1_enabled": np.array([1, 1], dtype=np.int8),
            "model_threshold": 0.5,
        },
        {
            "sample_name": "hard_pos",
            "target": 1,
            "prob_raw": np.array([0.8, 0.2], dtype=float),
            "stage1_enabled": np.array([1, 1], dtype=np.int8),
            "model_threshold": 0.5,
        },
        {
            "sample_name": "empty",
            "target": 0,
            "prob_raw": np.array([], dtype=float),
            "stage1_enabled": np.array([], dtype=np.int8),
            "model_threshold": 0.5,
        },
    ]

    selected, summary = s07.filter_hard_samples(caches)

    assert [c["sample_name"] for c in selected] == ["hard_pos"]
    assert summary["total_samples"] == 3
    assert summary["hard_samples"] == 1
    assert summary["all_correct_samples"] == 1
    assert summary["no_window_samples"] == 1


def test_postprocess_export_includes_median_k():
    cfg = {"alpha": 0.2, "median_k": 5, "T_on": 0.6, "T_off": 0.3, "K_on": 2, "K_off": 3, "cooldown_sec": 1}

    exported = s06.serialize_postprocess_config(cfg)

    assert exported["state_machine"]["parameters"]["median_k"] == 5


def test_apply_preprocess_applies_training_clip_bounds():
    bundle = {
        "feature_names": ["a", "b", "c"],
        "fill_values": {"a": 0.0, "b": 2.0, "c": -1.0},
        "clip_bounds": {"a": [0.0, 1.0], "b": [1.5, 2.5]},
    }

    X = s06.apply_preprocess(
        [{"a": 10.0, "b": np.inf}],
        bundle=bundle,
    )

    assert X.tolist() == [[1.0, 2.0, -1.0]]


def test_xgboost_dump_node_parser_maps_feature_names():
    trees = [
        "0:[f0<39.0625] yes=1,no=2,missing=2,gain=6.7,cover=20\n"
        "\t1:leaf=-0.1,cover=8.5\n"
        "\t2:[f1<0.28] yes=3,no=4,missing=4,gain=3.2,cover=5.25"
    ]

    rows = s06.parse_xgboost_dump_nodes(trees, ["feat_a", "feat_b"])

    assert rows[0]["Feature"] == "f0"
    assert rows[0]["FeatureName"] == "feat_a"
    assert rows[0]["Split"] == "39.0625"
    assert rows[2]["Feature"] == "f1"
    assert rows[2]["FeatureName"] == "feat_b"


def test_deploy_feature_formula_map_documents_consensus_and_acc_features():
    formulas = s06.build_feature_formula_map([
        "EMG_consensus_WL_max",
        "EMG_consensus_MDF_min",
        "ACC_SAT_FRAC",
        "ACC_CLIP_RATE",
    ])

    for info in formulas.values():
        assert "未匹配" not in info["formula"]
        assert info["category"] != "unknown"


def test_deploy_feature_formula_map_documents_ppg_spatial_features():
    formulas = s06.build_feature_formula_map([
        "PPG_ch_vmag_mean",
        "PPG_ch_bp_corr_mean",
        "PPG_corr_mean_vmag",
    ])

    for info in formulas.values():
        assert "未匹配" not in info["formula"]
        assert info["category"] == "ppg_spatial"


def test_validate_feature_formula_map_rejects_unmatched_formula():
    formulas = s06.build_feature_formula_map(["FEATURE_WITHOUT_FORMULA"])

    with pytest.raises(ValueError, match="missing deploy feature formula docs"):
        s06.validate_feature_formula_map(formulas)


def test_postprocess_config_source_message_distinguishes_optimized_from_saved_default():
    default_msg = s06.describe_postprocess_config_source(
        "artifacts/final_model_config.json",
        {"postprocess": {"alpha": 0.4}},
    )
    optimized_msg = s06.describe_postprocess_config_source(
        "artifacts/final_model_config.json",
        {
            "postprocess": {"alpha": 0.4},
            "postprocess_optimization": {"optimized_on_split": "valid"},
        },
    )
    cache_optimized_msg = s06.describe_postprocess_config_source(
        "artifacts/final_model_config.json",
        {
            "postprocess": {"alpha": 0.25},
            "postprocess_cache_optimization": {"split": "valid"},
        },
    )

    assert "保存的后处理参数" in default_msg
    assert "优化后的后处理参数" not in default_msg
    assert "优化后的后处理参数" in optimized_msg
    assert "优化后的后处理参数" in cache_optimized_msg
