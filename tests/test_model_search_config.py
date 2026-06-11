import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import s05_train_final_model as s05
import s08_run_pipeline as s08


def test_parse_model_search_grid_exposes_adjustable_values():
    args = SimpleNamespace(
        model_search_n_estimators="20,30,40",
        model_search_max_depth="2,3",
        model_search_learning_rate="0.03,0.05",
        model_search_min_child_weight="20,50",
        model_search_reg_lambda="10,20",
        model_search_reg_alpha="1,2",
        model_search_subsample="0.7,0.8",
        model_search_colsample_bytree="0.7,0.9",
    )

    grid = s05.build_model_search_grid(args, scale_pos_weight=1.25)

    assert len(grid) == 3 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    assert {
        "n_estimators": 20,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "reg_lambda": 10.0,
        "reg_alpha": 1.0,
        "scale_pos_weight": 1.25,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    } in grid


def test_staged_model_search_plan_caps_default_candidate_count():
    args = SimpleNamespace(
        model_search_n_estimators="20,30,40",
        model_search_max_depth="2,3",
        model_search_learning_rate="0.03,0.05,0.08",
        model_search_min_child_weight="20,50",
        model_search_reg_lambda="10,20",
        model_search_reg_alpha="1,2",
        model_search_subsample="0.7,0.8,0.9",
        model_search_colsample_bytree="0.7,0.8,0.9",
        model_search_stage1_top_k=4,
    )

    plan = s05.build_staged_model_search_plan(args, scale_pos_weight=1.0)
    stage1 = [item for item in plan if item["stage"] == "stage1_structure"]
    stage2 = [item for item in plan if item["stage"] == "stage2_refine"]

    assert len(stage1) == 48
    assert len(stage2) == 4 * (27 - 1)
    assert len(plan) == 152
    assert len(plan) < len(s05.build_model_search_grid(args, scale_pos_weight=1.0))


def test_model_search_score_uses_valid_accuracy_as_primary_objective():
    smaller_lower_accuracy = s05.score_model_search_candidate(
        {"accuracy": 0.98, "confusion_matrix": {"TN": 95, "FP": 5, "FN": 2, "TP": 98}},
        total_nodes=120,
        max_model_nodes=400,
    )
    larger_higher_accuracy = s05.score_model_search_candidate(
        {"accuracy": 0.99, "confusion_matrix": {"TN": 80, "FP": 20, "FN": 1, "TP": 99}},
        total_nodes=260,
        max_model_nodes=400,
    )
    oversized = s05.score_model_search_candidate(
        {"accuracy": 0.99, "confusion_matrix": {"TN": 95, "FP": 5, "FN": 1, "TP": 99}},
        total_nodes=401,
        max_model_nodes=400,
    )

    assert smaller_lower_accuracy["eligible"] is True
    assert larger_higher_accuracy["eligible"] is True
    assert larger_higher_accuracy["score"] > smaller_lower_accuracy["score"]
    assert oversized["eligible"] is False
    assert oversized["score"] == float("-inf")


def test_model_search_prefers_smaller_model_within_accuracy_tolerance():
    large = {
        "eligible": True,
        "metrics": {"accuracy": 0.981},
        "total_nodes": 320,
        "fp_rate": 0.02,
        "rank_input_order": 1,
    }
    small = {
        "eligible": True,
        "metrics": {"accuracy": 0.979},
        "total_nodes": 120,
        "fp_rate": 0.03,
        "rank_input_order": 2,
    }

    assert s05.is_better_model_search_record(
        small, large, accuracy_tolerance=0.005
    ) is True
    assert s05.is_better_model_search_record(
        small, large, accuracy_tolerance=0.001
    ) is False


def test_pipeline_commands_pass_model_search_params_to_s05():
    args = SimpleNamespace(
        dataset_dir="dataset",
        artifact_dir="artifacts",
        n_workers=2,
        max_features=12,
        window_sec=3,
        stride_sec=1,
        model_search=True,
        max_model_nodes=260,
        model_search_accuracy_tolerance=0.004,
        model_search_stage1_top_k=3,
        model_search_n_estimators="20,30",
        model_search_max_depth="2",
        model_search_learning_rate="0.05",
        model_search_min_child_weight="30,50",
        model_search_reg_lambda="20",
        model_search_reg_alpha="2",
        model_search_subsample="0.8",
        model_search_colsample_bytree="0.7,0.8",
        model_search_feature_counts="8,12",
    )

    cmd = s08.build_pipeline_commands(args)["s05"]

    assert "--max_features 12" in cmd
    assert "--model_search" in cmd
    assert "--max_model_nodes 260" in cmd
    assert "--model_search_accuracy_tolerance 0.004" in cmd
    assert "--model_search_stage1_top_k 3" in cmd
    assert '--model_search_n_estimators "20,30"' in cmd
    assert '--model_search_max_depth "2"' in cmd
    assert '--model_search_colsample_bytree "0.7,0.8"' in cmd
    assert '--model_search_feature_counts "8,12"' in cmd


def test_default_model_search_axes_include_fine_n_estimators():
    args = SimpleNamespace(
        model_search_n_estimators="20,25,30,35,40,45,50,55,60,70,80",
        model_search_max_depth="2,3,4",
        model_search_learning_rate="0.025,0.03,0.04,0.05,0.06,0.08,0.10",
        model_search_min_child_weight="10,15,20,25,30,40,50",
        model_search_reg_lambda="5,8,10,12,16,20,30",
        model_search_reg_alpha="0,0.5,1,1.5,2,3",
        model_search_subsample="0.70,0.75,0.80,0.85,0.90",
        model_search_colsample_bytree="0.70,0.75,0.80,0.85,0.90",
    )

    axes = s05.build_model_search_axes(args)

    assert 25 in axes["n_estimators"]
    assert 35 in axes["n_estimators"]
    assert 45 in axes["n_estimators"]
    assert 55 in axes["n_estimators"]


def test_sampled_candidates_force_default_params_even_when_user_grid_excludes_them():
    args = SimpleNamespace(
        model_search_n_estimators="20",
        model_search_max_depth="2",
        model_search_learning_rate="0.1",
        model_search_min_child_weight="10",
        model_search_reg_lambda="5",
        model_search_reg_alpha="0",
        model_search_subsample="0.7",
        model_search_colsample_bytree="0.7",
        model_search_max_candidates=1,
        model_search_random_state=42,
    )

    candidates = s05.build_sampled_model_search_candidates(args, scale_pos_weight=1.0)
    default_params = s05.build_default_xgb_params(scale_pos_weight=1.0)

    assert any(c["is_default_params"] for c in candidates)
    assert any(c["params"] == default_params for c in candidates)


def test_cv_selection_keeps_default_when_no_candidate_beats_it():
    default = {
        "eligible": True,
        "is_default_params": True,
        "mean_cv_accuracy": 0.90,
        "std_cv_accuracy": 0.03,
        "mean_cv_fp_rate": 0.02,
        "final_total_nodes": 120,
    }
    candidate = {
        "eligible": True,
        "is_default_params": False,
        "mean_cv_accuracy": 0.90,
        "std_cv_accuracy": 0.01,
        "mean_cv_fp_rate": 0.01,
        "final_total_nodes": 80,
    }

    chosen = s05.select_best_group_cv_record([candidate, default])

    assert chosen is default
    assert chosen["chosen_reason"] == "default_params_protection"


def test_cv_selection_prefers_higher_accuracy_candidate_within_node_budget():
    default = {
        "eligible": True,
        "is_default_params": True,
        "mean_cv_accuracy": 0.90,
        "std_cv_accuracy": 0.01,
        "mean_cv_fp_rate": 0.01,
        "final_total_nodes": 120,
    }
    candidate = {
        "eligible": True,
        "is_default_params": False,
        "mean_cv_accuracy": 0.91,
        "std_cv_accuracy": 0.04,
        "mean_cv_fp_rate": 0.05,
        "final_total_nodes": 200,
    }

    chosen = s05.select_best_group_cv_record([default, candidate])

    assert chosen is candidate
    assert chosen["beats_default_params"] is True


def test_cv_selection_excludes_candidates_over_node_budget():
    default = {
        "eligible": True,
        "is_default_params": True,
        "mean_cv_accuracy": 0.90,
        "std_cv_accuracy": 0.01,
        "mean_cv_fp_rate": 0.01,
        "final_total_nodes": 120,
    }
    oversized = {
        "eligible": False,
        "is_default_params": False,
        "mean_cv_accuracy": 0.99,
        "std_cv_accuracy": 0.0,
        "mean_cv_fp_rate": 0.0,
        "final_total_nodes": 999,
    }

    chosen = s05.select_best_group_cv_record([oversized, default])

    assert chosen is default


def test_model_search_csv_rows_include_group_cv_and_baseline_fields():
    record = {
        "rank_input_order": 1,
        "feature_count": 12,
        "eligible": True,
        "mean_cv_accuracy": 0.91,
        "std_cv_accuracy": 0.02,
        "mean_cv_fp_rate": 0.03,
        "mean_cv_precision": 0.94,
        "mean_cv_recall": 0.88,
        "cv_folds_completed": 6,
        "final_total_nodes": 180,
        "is_default_params": False,
        "beats_default_params": True,
        "chosen_reason": "highest_mean_cv_accuracy",
        "params": s05.build_default_xgb_params(),
    }

    row = s05.model_search_record_to_csv_row(record)

    for key in [
        "feature_count",
        "mean_cv_accuracy",
        "std_cv_accuracy",
        "mean_cv_fp_rate",
        "mean_cv_precision",
        "mean_cv_recall",
        "cv_folds_completed",
        "final_total_nodes",
        "is_default_params",
        "beats_default_params",
        "chosen_reason",
    ]:
        assert key in row


def test_feature_count_candidates_are_sorted_unique_and_bounded():
    counts = s05.parse_feature_count_candidates(
        "15,8,8,99",
        max_features=12,
        ranked_count=20,
    )

    assert counts == [8, 15]


def test_feature_count_selection_uses_cv_stability_fp_and_nodes():
    lower_std = {
        "feature_count": 8,
        "selection_record": {
            "eligible": True,
            "mean_cv_accuracy": 0.91,
            "std_cv_accuracy": 0.01,
            "mean_cv_fp_rate": 0.03,
            "final_total_nodes": 200,
        },
    }
    higher_std = {
        "feature_count": 12,
        "selection_record": {
            "eligible": True,
            "mean_cv_accuracy": 0.91,
            "std_cv_accuracy": 0.04,
            "mean_cv_fp_rate": 0.01,
            "final_total_nodes": 120,
        },
    }

    assert s05.select_best_feature_count_result([higher_std, lower_std]) is lower_std


def test_feature_count_selection_excludes_over_budget_models():
    oversized = {
        "feature_count": 8,
        "selection_record": {
            "eligible": False,
            "mean_cv_accuracy": 0.99,
            "std_cv_accuracy": 0.0,
            "mean_cv_fp_rate": 0.0,
            "final_total_nodes": 999,
        },
    }
    eligible = {
        "feature_count": 12,
        "selection_record": {
            "eligible": True,
            "mean_cv_accuracy": 0.90,
            "std_cv_accuracy": 0.02,
            "mean_cv_fp_rate": 0.03,
            "final_total_nodes": 180,
        },
    }

    assert s05.select_best_feature_count_result([oversized, eligible]) is eligible


def test_group_cv_search_summary_uses_train_group_cv_and_keeps_valid_out(monkeypatch):
    args = SimpleNamespace(
        model_search_strategy="staged_group_cv",
        model_search_n_estimators="5",
        model_search_max_depth="2",
        model_search_learning_rate="0.1",
        model_search_min_child_weight="1",
        model_search_reg_lambda="1",
        model_search_reg_alpha="0",
        model_search_subsample="1",
        model_search_colsample_bytree="1",
        model_search_max_candidates=2,
        model_search_stage2_top_k=2,
        model_search_cv_folds=2,
        model_search_cv_repeats=1,
        model_search_random_state=42,
        model_search_accuracy_tolerance=0.0,
        max_model_nodes=500,
    )
    X_train = np.array([[0.0], [1.0], [0.1], [1.1], [0.2], [1.2], [0.3], [1.3]])
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    X_valid = np.array([[99.0]])
    y_valid = np.array([1])

    class DummyModel:
        n_estimators = 5

        def predict_proba(self, X):
            p = (X[:, 0] > 0.5).astype(float)
            return np.column_stack([1.0 - p, p])

        def get_booster(self):
            class Booster:
                def get_dump(self):
                    return ["0:leaf=0"] * 5
            return Booster()

    monkeypatch.setattr(s05, "train_xgb_with_params", lambda params, X, y: DummyModel())

    _model, summary, records = s05.search_xgb_hyperparameters(
        args, X_train, y_train, X_valid, y_valid, scale_pos_weight=1.0, groups=groups
    )

    assert summary["strategy"] == "staged_group_cv"
    assert summary["selection_data"] == "train_group_cv_only"
    assert summary["valid_used_for_model_selection"] is False
    assert summary["group_source"] == "sample_name"
    assert summary["default_params_baseline"] is not None
    assert any(r["is_default_params"] for r in records)


def test_group_cv_search_falls_back_when_group_folds_make_single_class_train():
    args = SimpleNamespace(
        model_search_strategy="staged_group_cv",
        model_search_n_estimators="5",
        model_search_max_depth="2",
        model_search_learning_rate="0.1",
        model_search_min_child_weight="1",
        model_search_reg_lambda="1",
        model_search_reg_alpha="0",
        model_search_subsample="1",
        model_search_colsample_bytree="1",
        model_search_max_candidates=2,
        model_search_stage2_top_k=2,
        model_search_cv_folds=3,
        model_search_cv_repeats=1,
        model_search_random_state=42,
        model_search_accuracy_tolerance=0.0,
        max_model_nodes=500,
    )
    X_train = np.array([[0.0], [0.1], [1.0], [1.1], [1.2], [1.3]])
    y_train = np.array([0, 0, 1, 1, 1, 1])
    groups = np.array(["neg", "neg", "pos1", "pos1", "pos2", "pos2"])

    _model, summary, records = s05.search_xgb_hyperparameters(
        args,
        X_train,
        y_train,
        np.array([[0.0], [1.0]]),
        np.array([0, 1]),
        scale_pos_weight=1.0,
        groups=groups,
    )

    assert summary["strategy"] == "staged_group_cv"
    assert summary["group_source"] == "stratified_fallback"
    assert all(r["cv_folds_completed"] > 0 for r in records)
