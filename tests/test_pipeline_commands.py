import sys
import ast
import importlib.util
import json
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s08_run_pipeline as s08
import s07_postprocess_optimize as s07


class FakeBooster:
    def save_config(self):
        return "{}"

    def get_dump(self, with_stats=False):
        return ["0:leaf=0"]

    def trees_to_data_frame(self):
        import pandas as pd
        return pd.DataFrame([{"Tree": 0, "Node": 0, "Feature": "Leaf"}])


class FakeModel:
    n_estimators = 3

    def get_booster(self):
        return FakeBooster()

    def get_params(self):
        return {"n_estimators": self.n_estimators}


def test_pipeline_commands_include_npz_cache_postprocess_path():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=15,
        window_sec=3,
        stride_sec=1,
    )

    commands = s08.build_pipeline_commands(args)

    assert "--export_window_cache" in commands["s06_cache_train"]
    assert "--split train" in commands["s06_cache_train"]
    assert "--export_window_cache" in commands["s06_cache_valid"]
    assert "--split valid" in commands["s06_cache_valid"]
    assert "s07_postprocess_optimize" in commands["s07_post"]
    assert "--cache_root window_outputs" in commands["s07_post"]
    assert "--search_splits train,valid" in commands["s07_post"]
    assert "--hard_samples_only" in commands["s07_post"]
    assert "--workers 2" in commands["s07_post"]
    assert "--search_budget 240" in commands["s07_post"]
    assert "--threshold_offsets=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3" in commands["s07_post"]


def test_negative_csv_cli_values_are_normalized_before_argparse():
    assert s08._normalize_negative_csv_options(
        ["--postprocess_threshold_offsets", "-0.4,-0.3,0,0.3"],
        {"--postprocess_threshold_offsets"},
    ) == ["--postprocess_threshold_offsets=-0.4,-0.3,0,0.3"]
    assert s07._normalize_negative_csv_options(
        ["--threshold_offsets", "-0.4,-0.3,0,0.3"],
        {"--threshold_offsets"},
    ) == ["--threshold_offsets=-0.4,-0.3,0,0.3"]


def test_pipeline_postprocess_search_stays_on_valid_when_final_eval_uses_test():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=15,
        window_sec=3,
        stride_sec=1,
        split="test",
        postprocess_fp_cost=2.0,
    )

    commands = s08.build_pipeline_commands(args)

    assert "--split train" in commands["s06_cache_train"]
    assert "--split valid" in commands["s06_cache_valid"]
    assert "--search_splits train,valid" in commands["s07_post"]
    assert "--split test" in commands["s06_eval"]
    assert "--split test" in commands["s06_xpt"]


def test_pipeline_commands_enable_model_search_by_default():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=15,
        window_sec=3,
        stride_sec=1,
    )

    cmd = s08.build_pipeline_commands(args)["s05"]

    assert "--model_search" in cmd
    assert "--max_features 15" in cmd
    assert "--model_search_strategy staged_group_cv" in cmd
    assert "--model_search_max_candidates 300" in cmd
    assert "--model_search_stage2_top_k 40" in cmd
    assert "--model_search_cv_folds 3" in cmd
    assert "--model_search_cv_repeats 3" in cmd
    assert "--model_search_random_state 42" in cmd
    assert "--max_model_nodes 0" in cmd
    assert '--model_search_n_estimators "40,45,50,55,60,65,70,75,80"' in cmd


def test_with_postprocess_enables_accuracy_first_model_search_preset():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=15,
        window_sec=3,
        stride_sec=1,
        with_postprocess=True,
        export_window_cache=False,
        optimize_postprocess=False,
        accuracy_first_optimize=False,
        model_search_accuracy_tolerance=0.01,
        model_search_fp_cost=2.0,
        model_search_size_cost=0.1,
    )

    s08.apply_pipeline_presets(args)
    cmd = s08.build_pipeline_commands(args)["s05"]

    assert args.export_window_cache is True
    assert args.optimize_postprocess is True
    assert args.accuracy_first_optimize is True
    assert "--model_search_accuracy_tolerance 0.0" in cmd
    assert "--model_search_fp_cost 0.0" in cmd
    assert "--model_search_size_cost 0.0" in cmd


def test_accuracy_search_budget_expands_candidates_without_extra_cv_repeats():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=15,
        window_sec=3,
        stride_sec=1,
        search_budget="accuracy",
    )

    s08.apply_pipeline_presets(args)
    cmd = s08.build_pipeline_commands(args)["s05"]

    assert "--model_search_max_candidates 600" in cmd
    assert "--model_search_stage2_top_k 80" in cmd
    assert "--model_search_cv_repeats 5" in cmd


def test_fast_search_budget_reduces_candidates_for_short_runs():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=15,
        window_sec=3,
        stride_sec=1,
        search_budget="fast",
    )

    s08.apply_pipeline_presets(args)
    cmd = s08.build_pipeline_commands(args)["s05"]

    assert "--model_search_max_candidates 150" in cmd
    assert "--model_search_stage2_top_k 20" in cmd
    assert "--model_search_cv_repeats 1" in cmd


def test_dry_run_feature_count_search_prints_quick_and_full_search_commands():
    script = Path(__file__).resolve().parents[1] / "s08_run_pipeline.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry_run",
            "--stop_after",
            "s05",
            "--n_workers",
            "1",
            "--model_search_feature_counts",
            "8,12,15",
            "--model_search_full_top_k",
            "2",
        ],
        cwd=script.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    assert result.returncode == 0
    assert "quick feature-count search k=8" in result.stdout
    assert "quick feature-count search k=12" in result.stdout
    assert "quick feature-count search k=15" in result.stdout
    assert "full model search #1/2" in result.stdout
    assert "full model search #2/2" in result.stdout
    assert "--no-model_search" in result.stdout
    quick_lines = [
        line for line in result.stdout.splitlines()
        if "quick feature-count search" in line
    ]
    assert quick_lines
    assert all("--model_search_feature_counts" not in line for line in quick_lines)
    assert '--model_search_feature_counts "15"' in result.stdout


def test_readme_documents_current_search_budget_cli():
    readme = (Path(__file__).resolve().parents[1] / "readme.md").read_text(encoding="utf-8")

    assert "--search_budget" in readme
    assert "--runtime_profile" not in readme
    assert "thorough" not in readme


def test_readme_documents_current_postprocess_search_flow():
    readme = (Path(__file__).resolve().parents[1] / "readme.md").read_text(encoding="utf-8")

    assert "s06_cache_train" in readme
    assert "--search_splits train,valid" in readme
    assert "--hard_samples_only" in readme
    assert "--threshold_offsets=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3" in readme
    assert "postprocess_search_train_valid.csv" in readme


def test_default_pipeline_steps_skip_postprocess_search_before_final_eval():
    steps = s08.default_pipeline_steps()
    step_keys = [key for key, _, _ in steps]
    # default-enabled steps only
    default_keys = [key for key, _, enabled in steps if enabled]

    assert "s06_opt" not in default_keys
    assert "s06_cache_train" not in default_keys
    assert "s06_cache_valid" not in default_keys
    assert "s07_post" not in default_keys
    assert step_keys.index("s05") < step_keys.index("s06_eval")


def test_dry_run_stop_after_stops_printing_later_steps():
    script = Path(__file__).resolve().parents[1] / "s08_run_pipeline.py"

    result = subprocess.run(
        [sys.executable, str(script), "--dry_run", "--stop_after", "s04", "--n_workers", "1"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    assert result.returncode == 0
    assert "稳定性特征筛选" in result.stdout
    assert "XGBoost模型训练" not in result.stdout
    assert "[STOP] 已运行到 s04" in result.stdout


def test_deploy_feature_extractor_is_standalone_and_matches_training_ppg_features():
    feature_order = ["PPG_mean", "PPG_std", "PPG_p95", "PPG_diff_std"]
    formula_map = s08._build_feature_code_map()
    feat_block = "\n".join(f'    f["{name}"] = {formula_map[name]}' for name in feature_order)
    script = s08._build_extractor_script_template(
        len(feature_order),
        json.dumps(feature_order),
        json.dumps({name: 0.0 for name in feature_order}),
        json.dumps({}),  # CLIP_BOUNDS (empty for this test)
        feat_block,
    )
    out_dir = Path.cwd() / "test_outputs" / "deploy_feature_extractor"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / f"deploy_feature_extractor_{uuid.uuid4().hex}.py"
    try:
        script_path.write_text(script, encoding="utf-8")

        parsed = ast.parse(script)
        imported_modules = []
        for node in parsed.body:
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
        assert "s03_extract_feature_pool" not in imported_modules

        spec = importlib.util.spec_from_file_location("deploy_feature_extractor_tmp", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rng = np.random.default_rng(7)
        ppg_6ch = rng.normal(50000, 4000, size=(300, 6))
        emg = rng.normal(0, 1, size=(3000, 2))
        acc = rng.normal(0, 1, size=(300, 3))

        deployed = module.extract_features(ppg_6ch, emg, acc)
        trained = s03.extract_feature_pool_from_window(
            s03.build_3ch_ppg(ppg_6ch),
            emg,
            acc,
        )

        expected = [float(trained[name]) for name in feature_order]
        np.testing.assert_allclose(deployed, expected, rtol=1e-9, atol=1e-9)
    finally:
        if script_path.exists():
            script_path.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass


def test_deploy_feature_map_covers_s03_generated_deployable_features():
    rng = np.random.default_rng(17)
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

    missing = sorted(set(deployable) - set(s08._build_feature_code_map()))

    assert missing == []


def test_deploy_feature_extractor_matches_training_consensus_features():
    feature_order = [
        "EMG_consensus_WL_max",
        "EMG_consensus_MDF_min",
        "EMG_consensus_MDF_max",
        "ACC_SAT_FRAC",
        "ACC_CLIP_RATE",
    ]
    formula_map = s08._build_feature_code_map()
    feat_block = "\n".join(f'    f["{name}"] = {formula_map[name]}' for name in feature_order)
    script = s08._build_extractor_script_template(
        len(feature_order),
        json.dumps(feature_order),
        json.dumps({name: 0.0 for name in feature_order}),
        json.dumps({}),
        feat_block,
    )
    out_dir = Path.cwd() / "test_outputs" / "deploy_feature_extractor_consensus"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / f"deploy_feature_extractor_{uuid.uuid4().hex}.py"
    try:
        script_path.write_text(script, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("deploy_feature_extractor_consensus_tmp", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rng = np.random.default_rng(23)
        ppg_6ch = rng.normal(50000, 4000, size=(300, 6))
        emg = rng.normal(0, 1, size=(3000, 2))
        acc = rng.normal(0, 1, size=(300, 3))

        deployed = module.extract_features(ppg_6ch, emg, acc)
        trained = s03.extract_feature_pool_from_window(
            s03.build_3ch_ppg(ppg_6ch),
            emg,
            acc,
        )

        expected = [float(trained[name]) for name in feature_order]
        np.testing.assert_allclose(deployed, expected, rtol=1e-9, atol=1e-9)
    finally:
        if script_path.exists():
            script_path.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass


def test_deploy_feature_extractor_matches_training_for_all_deployable_features():
    rng = np.random.default_rng(101)
    ppg_6ch = rng.normal(50000, 4000, size=(300, 6))
    emg = rng.normal(0, 1, size=(3000, 2))
    acc = rng.normal(0, 1, size=(300, 3))
    trained = s03.extract_feature_pool_from_window(
        s03.build_3ch_ppg(ppg_6ch),
        emg,
        acc,
    )
    non_deploy = set(getattr(s04, "NON_DEPLOY_FEATURES", set()))
    feature_order = sorted(k for k in trained if k not in non_deploy)
    formula_map = s08._build_feature_code_map()
    feat_block = "\n".join(f'    f["{name}"] = {formula_map[name]}' for name in feature_order)
    script = s08._build_extractor_script_template(
        len(feature_order),
        json.dumps(feature_order),
        json.dumps({name: 0.0 for name in feature_order}),
        json.dumps({}),
        feat_block,
    )
    out_dir = Path.cwd() / "test_outputs" / "deploy_feature_extractor_all_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / f"deploy_feature_extractor_{uuid.uuid4().hex}.py"
    try:
        script_path.write_text(script, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("deploy_feature_extractor_all_tmp", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        deployed = np.array(module.extract_features(ppg_6ch, emg, acc), dtype=float)
        expected = np.array([float(trained[name]) for name in feature_order], dtype=float)

        np.testing.assert_allclose(deployed, expected, rtol=1e-7, atol=1e-7)
    finally:
        if script_path.exists():
            script_path.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass


def test_deploy_feature_extractor_fails_on_missing_feature_formula():
    out_dir = Path.cwd() / "test_outputs" / f"missing_deploy_formula_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_names": ["FEATURE_NOT_IMPLEMENTED_FOR_DEPLOY"],
        "fill_values": {"FEATURE_NOT_IMPLEMENTED_FOR_DEPLOY": 0.0},
        "clip_bounds": {},
        "threshold": 0.5,
        "model": FakeModel(),
    }
    try:
        joblib.dump(bundle, out_dir / "model_bundle.pkl")

        with pytest.raises(ValueError, match="missing executable formulas"):
            s08.export_feature_extractor_script(str(out_dir))
    finally:
        for p in out_dir.glob("*"):
            p.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass


def test_deploy_cookbook_uses_current_postprocess_and_clip_bounds():
    out_dir = Path.cwd() / "test_outputs" / f"deploy_cookbook_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        (out_dir / "stage1_threshold.json").write_text(
            json.dumps({
                "deploy_stage1_threshold": {
                    "dc_threshold": 1.0,
                    "ac_dc_threshold": 0.2,
                }
            }),
            encoding="utf-8",
        )
        postprocess = {
            "alpha": 0.25,
            "T_on": 0.55,
            "T_off": 0.2,
            "K_on": 1,
            "K_off": 1,
            "cooldown_sec": 0.0,
            "median_k": 1,
        }
        (out_dir / "final_model_config.json").write_text(
            json.dumps({"postprocess": postprocess}),
            encoding="utf-8",
        )
        bundle = {
            "feature_names": ["PPG_mean"],
            "fill_values": {"PPG_mean": 10.0},
            "clip_bounds": {"PPG_mean": [1.0, 20.0]},
            "threshold": 0.35,
            "model": FakeModel(),
        }
        joblib.dump(bundle, out_dir / "model_bundle.pkl")

        s08.export_deploy_cookbook(str(out_dir))

        cookbook = json.loads((out_dir / "deploy_cookbook.json").read_text(encoding="utf-8"))
        deploy_xgb = json.loads((out_dir / "deploy_xgboost.json").read_text(encoding="utf-8"))

        params = cookbook["D_stage3_postprocess"]["params"]
        for key, value in postprocess.items():
            assert params[key] == value
        assert params["threshold_offset"] == 0.0
        assert params["threshold_transform"] == "disabled"
        assert deploy_xgb["fill_values"] == bundle["fill_values"]
        assert deploy_xgb["clip_bounds"] == bundle["clip_bounds"]
    finally:
        for p in out_dir.glob("*"):
            p.unlink()
        try:
            out_dir.rmdir()
            out_dir.parent.rmdir()
        except OSError:
            pass
