# s05_train_final_model.py
# -*- coding: utf-8 -*-

"""
步骤5：最终模型训练（适配 IR + EMG + ACC 特征集）

原则：
1. train 训练模型
2. valid 只选择窗口概率阈值和确认指标
3. test 完全不参与
4. 缺失填充值只从 train 计算
"""

import os
import json
import argparse
import logging
import joblib
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

from s03_extract_feature_pool import (
    DEFAULT_FS_PPG, DEFAULT_FS_EMG, DEFAULT_FS_ACC, FEATURE_FS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_XGB_PARAMS = {
    "n_estimators": 40,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 10,
    "reg_alpha": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}


# =========================================================
# 异常值Clipping
# =========================================================

def clip_outliers(df, columns, k=1.5, bounds=None, return_bounds=False):
    """
    基于 IQR 的异常值裁剪（向量化版）。

    参数:
        df: 输入 DataFrame
        columns: 需要裁剪的列名列表
        k: IQR 倍数，默认 1.5
        bounds: dict[col -> (lower, upper)] 预先计算好的裁剪边界。给定时跳过 IQR 估计直接应用。
        return_bounds: 是否同时返回 bounds 字典（仅 bounds=None 时有意义）。

    返回:
        若 return_bounds=False：DataFrame
        若 return_bounds=True：  (DataFrame, bounds_dict)
    """
    df = df.copy()

    cols = [c for c in columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return (df, {}) if return_bounds else df

    if bounds is not None:
        valid_cols = [c for c in cols if c in bounds]
        if not valid_cols:
            return (df, {}) if return_bounds else df
        lower = pd.Series({c: bounds[c][0] for c in valid_cols})
        upper = pd.Series({c: bounds[c][1] for c in valid_cols})
        df[valid_cols] = df[valid_cols].clip(lower=lower, upper=upper, axis=1)
        return (df, bounds) if return_bounds else df

    sub = df[cols]
    q1 = sub.quantile(0.25)
    q3 = sub.quantile(0.75)
    iqr = q3 - q1

    valid_mask = iqr.values > 1e-10
    if not valid_mask.any():
        return (df, {}) if return_bounds else df

    valid_cols = [c for c, ok in zip(cols, valid_mask) if ok]
    q1v = q1[valid_cols]
    q3v = q3[valid_cols]
    lower = q1v - k * (q3v - q1v)
    upper = q3v + k * (q3v - q1v)

    before_min = df[valid_cols].min()
    before_max = df[valid_cols].max()

    df[valid_cols] = df[valid_cols].clip(lower=lower, upper=upper, axis=1)

    clipped_cols = []
    for c in valid_cols:
        bmin = before_min[c]
        bmax = before_max[c]
        lo = lower[c]
        hi = upper[c]
        if bmin < lo or bmax > hi:
            clipped_cols.append({
                "column": c, "lower": float(lo), "upper": float(hi),
                "clipped_min": bool(bmin < lo), "clipped_max": bool(bmax > hi),
            })

    if clipped_cols:
        logger.info(f"异常值裁剪统计 (k={k}):")
        for item in clipped_cols:
            logger.info(f"  {item['column']}: lower={item['lower']:.4f}, "
                        f"upper={item['upper']:.4f}, "
                        f"clipped_min={item['clipped_min']}, clipped_max={item['clipped_max']}")

    out_bounds = {c: (float(lower[c]), float(upper[c])) for c in valid_cols}
    return (df, out_bounds) if return_bounds else df


def prepare_fill_values(df_train, selected_features):
    fill_values = {}
    for c in selected_features:
        x = df_train[c].replace([np.inf, -np.inf], np.nan)
        med = x.median()
        if not np.isfinite(med):
            med = 0.0
        fill_values[c] = float(med)
    return fill_values


def apply_fill(df, selected_features, fill_values):
    df = df.copy()
    for c in selected_features:
        if c not in df.columns:
            df[c] = fill_values.get(c, 0.0)
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(fill_values.get(c, 0.0))
    return df


def prepare_xy(df, selected_features, fill_values):
    df = apply_fill(df, selected_features, fill_values)
    X = df[selected_features].values.astype(float)
    y = df["target"].values.astype(int)
    return X, y, None


def eval_model(model, X, y, threshold=0.5):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }

    try:
        if len(np.unique(y)) < 2:
            raise ValueError("auc requires both classes")
        metrics["auc"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["auc"] = None

    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        metrics["confusion_matrix"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    except Exception:
        metrics["confusion_matrix"] = None

    return metrics


def _fbeta(precision, recall, beta):
    """F-beta；beta<1 偏 precision，beta>1 偏 recall。"""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def search_threshold_by_valid(model, X_valid, y_valid, objective="f1",
                               beta=0.5, min_precision=None):
    """
    在 valid 上搜窗口阈值。

    objective:
      - "f1"                 : 默认 F1
      - "precision"          : 仅 precision
      - "recall"             : 仅 recall
      - "fbeta"              : F-beta (默认 beta=0.5 偏 precision)
      - "precision_constrained" : 在 precision >= min_precision 约束下最大化 recall
    """
    probs = model.predict_proba(X_valid)[:, 1]
    best = None
    best_score = -np.inf

    for th in np.linspace(0.05, 0.95, 100):
        pred = (probs >= th).astype(int)
        precision = float(precision_score(y_valid, pred, zero_division=0))
        recall = float(recall_score(y_valid, pred, zero_division=0))
        f1 = float(f1_score(y_valid, pred, zero_division=0))

        if objective == "f1":
            score = f1
        elif objective == "recall":
            score = recall
        elif objective == "precision":
            score = precision
        elif objective == "fbeta":
            score = _fbeta(precision, recall, beta)
        elif objective == "precision_constrained":
            if min_precision is None:
                min_precision = 0.95
            score = recall if precision >= min_precision else -1.0 + precision
        else:
            score = f1

        item = {
            "threshold": float(th),
            "score": float(score),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": float(_fbeta(precision, recall, beta)),
        }

        if score > best_score:
            best = item
            best_score = score

    if best is not None:
        best["objective"] = objective
        if objective == "fbeta":
            best["beta"] = float(beta)
        if objective == "precision_constrained":
            best["min_precision"] = float(min_precision if min_precision is not None else 0.95)

    return best


def parse_model_search_values(raw, cast, name):
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = cast(part)
        except ValueError as exc:
            raise ValueError(f"{name} contains invalid value {part!r}") from exc
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def parse_feature_count_candidates(raw, max_features=None, ranked_count=None):
    """Parse feature-count search candidates.

    Empty input means "use max_features once". Explicit counts are sorted,
    de-duplicated, and capped to the available ranked feature count.
    """
    if str(raw or "").strip():
        counts = parse_model_search_values(raw, int, "model_search_feature_counts")
    else:
        if max_features is None:
            return []
        counts = [int(max_features)]

    limit = int(ranked_count) if ranked_count is not None else None
    out = []
    for count in counts:
        count = int(count)
        if count <= 0:
            raise ValueError("model_search_feature_counts must be positive integers")
        if limit is not None and count > limit:
            continue
        if count not in out:
            out.append(count)
    if not out:
        raise ValueError("model_search_feature_counts has no usable value within available features")
    return sorted(out)


def build_default_xgb_params(scale_pos_weight=1.0):
    params = dict(DEFAULT_XGB_PARAMS)
    params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def build_model_search_axes(args):
    return {
        "n_estimators": parse_model_search_values(
            args.model_search_n_estimators, int, "model_search_n_estimators"),
        "max_depth": parse_model_search_values(
            args.model_search_max_depth, int, "model_search_max_depth"),
        "learning_rate": parse_model_search_values(
            args.model_search_learning_rate, float, "model_search_learning_rate"),
        "subsample": parse_model_search_values(
            args.model_search_subsample, float, "model_search_subsample"),
        "colsample_bytree": parse_model_search_values(
            args.model_search_colsample_bytree, float, "model_search_colsample_bytree"),
        "min_child_weight": parse_model_search_values(
            args.model_search_min_child_weight, int, "model_search_min_child_weight"),
        "reg_lambda": parse_model_search_values(
            args.model_search_reg_lambda, float, "model_search_reg_lambda"),
        "reg_alpha": parse_model_search_values(
            args.model_search_reg_alpha, float, "model_search_reg_alpha"),
    }


def _middle_value(values):
    return values[len(values) // 2]


def _freeze_params(params):
    return tuple(sorted(params.items()))


def build_model_search_grid(args, scale_pos_weight=1.0):
    axes = build_model_search_axes(args)
    keys = list(axes.keys())
    grid = []
    for values in product(*(axes[k] for k in keys)):
        params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
        params.update(dict(zip(keys, values)))
        grid.append(params)
    return grid


def _same_xgb_params(left, right):
    for key in DEFAULT_XGB_PARAMS:
        lv = left.get(key)
        rv = right.get(key)
        if isinstance(lv, float) or isinstance(rv, float):
            if not np.isclose(float(lv), float(rv)):
                return False
        elif lv != rv:
            return False
    return np.isclose(float(left.get("scale_pos_weight", 1.0)),
                      float(right.get("scale_pos_weight", 1.0)))


def build_sampled_model_search_candidates(args, scale_pos_weight=1.0):
    """Build a deterministic sampled candidate set and always include defaults."""
    axes = build_model_search_axes(args)
    keys = list(axes.keys())
    sizes = [len(axes[k]) for k in keys]
    total_grid_size = int(np.prod(sizes))
    max_candidates = max(1, int(getattr(args, "model_search_max_candidates", 600)))
    random_state = int(getattr(args, "model_search_random_state", 42))
    default_params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)

    def params_from_index(index):
        # Decode a Cartesian-product index without materializing the full grid.
        # The default space is intentionally dense, so building every dict would
        # waste memory before we sample.
        index = int(index)
        values = []
        for size in reversed(sizes):
            values.append(index % size)
            index //= size
        values = list(reversed(values))
        params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
        params.update({key: axes[key][idx] for key, idx in zip(keys, values)})
        return params

    if total_grid_size > max_candidates:
        rng = np.random.default_rng(random_state)
        sampled_idx = rng.choice(total_grid_size, size=max_candidates, replace=False)
        selected = [params_from_index(i) for i in sorted(sampled_idx)]
    else:
        selected = [params_from_index(i) for i in range(total_grid_size)]

    # Baseline protection: fixed default params must be evaluated even when the
    # user-provided comma lists do not include them.
    selected.append(default_params)
    seen = set()
    candidates = []
    for params in selected:
        frozen = _freeze_params(params)
        if frozen in seen:
            continue
        seen.add(frozen)
        candidates.append({
            "rank_input_order": len(candidates) + 1,
            "params": params,
            "is_default_params": bool(_same_xgb_params(params, default_params)),
        })
    return candidates


def build_staged_model_search_plan(args, scale_pos_weight=1.0):
    axes = build_model_search_axes(args)
    stage1_keys = ["n_estimators", "max_depth", "min_child_weight", "reg_lambda", "reg_alpha"]
    stage2_keys = ["learning_rate", "subsample", "colsample_bytree"]

    fixed_stage2 = {key: _middle_value(axes[key]) for key in stage2_keys}
    top_k = max(1, int(getattr(args, "model_search_stage1_top_k", 4)))
    seen = set()
    plan = []

    for values in product(*(axes[k] for k in stage1_keys)):
        params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
        params.update(fixed_stage2)
        params.update(dict(zip(stage1_keys, values)))
        frozen = _freeze_params(params)
        if frozen in seen:
            continue
        seen.add(frozen)
        plan.append({"stage": "stage1_structure", "params": params})

    stage1_structures = [item["params"] for item in plan[:top_k]]
    for base in stage1_structures:
        structure = {key: base[key] for key in stage1_keys}
        for values in product(*(axes[k] for k in stage2_keys)):
            params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
            params.update(structure)
            params.update(dict(zip(stage2_keys, values)))
            frozen = _freeze_params(params)
            if frozen in seen:
                continue
            seen.add(frozen)
            plan.append({"stage": "stage2_refine", "params": params})

    return plan


def count_xgb_nodes(model):
    booster = model.get_booster()
    tree_dumps = booster.get_dump()
    return int(sum(
        len([line for line in tree.split("\n")
             if line.strip() and not line.strip().startswith("booster")])
        for tree in tree_dumps
    ))


def _window_fp_rate_from_metrics(metrics):
    cm = metrics.get("confusion_matrix") or {}
    tn = int(cm.get("TN", 0))
    fp = int(cm.get("FP", 0))
    n_neg = max(tn + fp, 1)
    return float(fp) / float(n_neg)


def score_model_search_candidate(metrics, total_nodes, max_model_nodes=500,
                                 fp_cost=2.0, size_cost=0.1):
    total_nodes = int(total_nodes)
    max_model_nodes = int(max_model_nodes)
    if max_model_nodes > 0 and total_nodes > max_model_nodes:
        return {
            "eligible": False,
            "score": float("-inf"),
            "fp_rate": _window_fp_rate_from_metrics(metrics),
            "size_ratio": float(total_nodes) / float(max(max_model_nodes, 1)),
        }
    fp_rate = _window_fp_rate_from_metrics(metrics)
    size_ratio = float(total_nodes) / float(max(max_model_nodes, 1)) if max_model_nodes > 0 else 0.0
    score = float(metrics.get("accuracy", 0.0))
    return {
        "eligible": True,
        "score": float(score),
        "fp_rate": float(fp_rate),
        "size_ratio": float(size_ratio),
    }


def is_better_model_search_record(candidate, incumbent, accuracy_tolerance=0.005):
    if incumbent is None:
        return bool(candidate.get("eligible", False))
    if not candidate.get("eligible", False):
        return False
    if not incumbent.get("eligible", False):
        return True

    cand_acc = float(candidate.get("metrics", {}).get("accuracy", candidate.get("score", 0.0)))
    inc_acc = float(incumbent.get("metrics", {}).get("accuracy", incumbent.get("score", 0.0)))
    tol = max(0.0, float(accuracy_tolerance))

    if cand_acc > inc_acc + tol:
        return True
    if cand_acc < inc_acc - tol:
        return False

    cand_nodes = int(candidate.get("total_nodes", 0))
    inc_nodes = int(incumbent.get("total_nodes", 0))
    if cand_nodes != inc_nodes:
        return cand_nodes < inc_nodes

    cand_fp = float(candidate.get("fp_rate", 0.0))
    inc_fp = float(incumbent.get("fp_rate", 0.0))
    if cand_fp != inc_fp:
        return cand_fp < inc_fp

    return int(candidate.get("rank_input_order", 0)) < int(incumbent.get("rank_input_order", 0))


def _json_safe_float(value):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _json_safe_model_search_record(record):
    if record is None:
        return None
    out = dict(record)
    for key in (
        "score", "fp_rate", "size_ratio", "avg_nodes_per_tree",
        "mean_cv_accuracy", "std_cv_accuracy", "mean_cv_fp_rate",
        "mean_cv_precision", "mean_cv_recall",
    ):
        if key in out and out[key] is not None:
            out[key] = _json_safe_float(out[key])
    return out


def train_xgb_with_params(params, X_train, y_train):
    fit_params = dict(params)
    fit_params["n_jobs"] = -1
    model = xgb.XGBClassifier(**fit_params)
    model.fit(X_train, y_train, verbose=False)
    return model


def _group_cv_splits(y, groups=None, n_folds=3, n_repeats=2, random_state=42):
    y = np.asarray(y)
    n = len(y)
    n_folds = max(2, int(n_folds))
    n_repeats = max(1, int(n_repeats))

    def valid_binary_split(train_idx, valid_idx):
        # XGBoost binary classification fails when a train fold has one class.
        # Requiring both classes in valid also keeps accuracy/fp-rate comparable.
        return (
            len(train_idx) > 0
            and len(valid_idx) > 0
            and len(np.unique(y[train_idx])) >= 2
            and len(np.unique(y[valid_idx])) >= 2
        )

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) == n:
            unique_groups = np.unique(groups)
            if len(unique_groups) >= n_folds:
                splits = []
                for rep in range(n_repeats):
                    rng = np.random.default_rng(int(random_state) + rep)
                    shuffled = unique_groups.copy()
                    rng.shuffle(shuffled)
                    fold_groups = [set() for _ in range(n_folds)]
                    for idx, group in enumerate(shuffled):
                        fold_groups[idx % n_folds].add(group)
                    for fold_id in range(n_folds):
                        valid_mask = np.array([g in fold_groups[fold_id] for g in groups])
                        train_idx = np.where(~valid_mask)[0]
                        valid_idx = np.where(valid_mask)[0]
                        if valid_binary_split(train_idx, valid_idx):
                            splits.append((train_idx, valid_idx))
                if splits:
                    return splits, "sample_name"

    # Some datasets have too few or too imbalanced groups to make legal group
    # folds. Fall back to stratified window-level CV and record that source in
    # the model_search summary instead of crashing mid-run.
    class_counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    folds = min(n_folds, max(2, min_class)) if min_class >= 2 else 2
    if min_class >= 2 and n >= folds:
        splits = []
        for rep in range(n_repeats):
            skf = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=int(random_state) + rep,
            )
            for train_idx, valid_idx in skf.split(np.zeros(n), y):
                if valid_binary_split(train_idx, valid_idx):
                    splits.append((train_idx, valid_idx))
        return splits, "stratified_fallback"

    if n < 2:
        return [], "none"
    idx = np.arange(n)
    mid = max(1, n // 2)
    return [(idx[mid:], idx[:mid])], "simple_fallback"


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _std_or_zero(values):
    return float(np.std(values)) if values else 0.0


def evaluate_group_cv_candidate(candidate, args, X_train, y_train, groups=None,
                                scale_pos_weight=1.0, splits=None, stage="cv"):
    params = dict(candidate["params"])
    if splits is None:
        splits, group_source = _group_cv_splits(
            y_train,
            groups=groups,
            n_folds=getattr(args, "model_search_cv_folds", 3),
            n_repeats=getattr(args, "model_search_cv_repeats", 2),
            random_state=getattr(args, "model_search_random_state", 42),
        )
    else:
        group_source = "provided"

    accuracies, fp_rates, precisions, recalls = [], [], [], []
    for train_idx, valid_idx in splits:
        model = train_xgb_with_params(params, X_train[train_idx], y_train[train_idx])
        metrics = eval_model(model, X_train[valid_idx], y_train[valid_idx], threshold=0.5)
        accuracies.append(float(metrics.get("accuracy", 0.0)))
        fp_rates.append(_window_fp_rate_from_metrics(metrics))
        precisions.append(float(metrics.get("precision", 0.0)))
        recalls.append(float(metrics.get("recall", 0.0)))

    final_model = train_xgb_with_params(params, X_train, y_train)
    final_total_nodes = count_xgb_nodes(final_model)
    eligible = int(final_total_nodes) <= int(getattr(args, "max_model_nodes", 500))

    return {
        "rank_input_order": int(candidate.get("rank_input_order", 0)),
        "stage": str(stage),
        "eligible": bool(eligible),
        "mean_cv_accuracy": _mean_or_zero(accuracies),
        "std_cv_accuracy": _std_or_zero(accuracies),
        "mean_cv_fp_rate": _mean_or_zero(fp_rates),
        "mean_cv_precision": _mean_or_zero(precisions),
        "mean_cv_recall": _mean_or_zero(recalls),
        "cv_folds_completed": int(len(accuracies)),
        "final_total_nodes": int(final_total_nodes),
        "is_default_params": bool(candidate.get("is_default_params", False)),
        "beats_default_params": False,
        "chosen_reason": "",
        "params": params,
        "group_source": group_source,
    }, final_model


def select_best_group_cv_record(records, accuracy_tolerance=0.0):
    eligible = [r for r in records if r.get("eligible", False)]
    if not eligible:
        return None

    default_records = [r for r in eligible if r.get("is_default_params", False)]
    default_record = max(default_records, key=lambda r: float(r.get("mean_cv_accuracy", 0.0))) if default_records else None
    tol = max(0.0, float(accuracy_tolerance))

    def sort_key(r):
        return (
            -float(r.get("mean_cv_accuracy", 0.0)),
            float(r.get("std_cv_accuracy", 0.0)),
            float(r.get("mean_cv_fp_rate", 0.0)),
            int(r.get("final_total_nodes", 0)),
            int(r.get("rank_input_order", 0)),
        )

    best = sorted(eligible, key=sort_key)[0]
    if default_record is not None:
        default_acc = float(default_record.get("mean_cv_accuracy", 0.0))
        best_acc = float(best.get("mean_cv_accuracy", 0.0))
        if not best.get("is_default_params", False) and best_acc <= default_acc + tol:
            default_record["beats_default_params"] = False
            default_record["chosen_reason"] = "default_params_protection"
            return default_record
        best["beats_default_params"] = (not best.get("is_default_params", False)) and best_acc > default_acc + tol

    if best.get("is_default_params", False):
        best["chosen_reason"] = "default_params_best_cv"
    elif not best.get("chosen_reason"):
        best["chosen_reason"] = "highest_mean_cv_accuracy"
    return best


def select_best_feature_count_result(results, accuracy_tolerance=0.0):
    """Select the best feature-count run using the same deployment-oriented keys."""
    eligible = [
        r for r in results
        if r.get("selection_record", {}).get("eligible", False)
    ]
    if not eligible:
        return None

    def sort_key(result):
        record = result.get("selection_record", {})
        return (
            -float(record.get("mean_cv_accuracy", 0.0)),
            float(record.get("std_cv_accuracy", 0.0)),
            float(record.get("mean_cv_fp_rate", 0.0)),
            int(record.get("final_total_nodes", 0)),
            int(result.get("feature_count", record.get("feature_count", 0))),
        )

    return sorted(eligible, key=sort_key)[0]


def model_search_record_to_csv_row(record):
    row = {
        "rank_input_order": record.get("rank_input_order"),
        "feature_count": record.get("feature_count"),
        "chosen_feature_count": record.get("chosen_feature_count"),
        "stage": record.get("stage"),
        "eligible": record.get("eligible"),
        "mean_cv_accuracy": record.get("mean_cv_accuracy"),
        "std_cv_accuracy": record.get("std_cv_accuracy"),
        "mean_cv_fp_rate": record.get("mean_cv_fp_rate"),
        "mean_cv_precision": record.get("mean_cv_precision"),
        "mean_cv_recall": record.get("mean_cv_recall"),
        "cv_folds_completed": record.get("cv_folds_completed"),
        "final_total_nodes": record.get("final_total_nodes"),
        "is_default_params": record.get("is_default_params"),
        "beats_default_params": record.get("beats_default_params"),
        "chosen_reason": record.get("chosen_reason"),
        "group_source": record.get("group_source"),
    }
    row.update({f"param_{k}": v for k, v in record.get("params", {}).items()})
    return row


def evaluate_model_search_candidate(params, idx, stage, args, X_train, y_train, X_valid, y_valid):
    candidate = train_xgb_with_params(params, X_train, y_train)
    best_threshold = search_threshold_by_valid(
        candidate, X_valid, y_valid,
        objective=args.threshold_objective,
        beta=args.threshold_beta,
        min_precision=args.threshold_min_precision,
    )
    metrics = eval_model(candidate, X_valid, y_valid, threshold=best_threshold["threshold"])
    total_nodes = count_xgb_nodes(candidate)
    score = score_model_search_candidate(
        metrics,
        total_nodes=total_nodes,
        max_model_nodes=args.max_model_nodes,
        fp_cost=args.model_search_fp_cost,
        size_cost=args.model_search_size_cost,
    )
    record = {
        "rank_input_order": int(idx),
        "stage": str(stage),
        "eligible": bool(score["eligible"]),
        "score": float(score["score"]),
        "fp_rate": float(score["fp_rate"]),
        "size_ratio": float(score["size_ratio"]),
        "total_nodes": int(total_nodes),
        "avg_nodes_per_tree": float(total_nodes) / float(max(int(params["n_estimators"]), 1)),
        "metrics": metrics,
        "threshold_search": best_threshold,
        "params": params,
    }
    return candidate, record


def search_xgb_hyperparameters_group_cv(args, X_train, y_train, groups=None,
                                        scale_pos_weight=1.0):
    candidates = build_sampled_model_search_candidates(args, scale_pos_weight=scale_pos_weight)
    all_splits, group_source = _group_cv_splits(
        y_train,
        groups=groups,
        n_folds=getattr(args, "model_search_cv_folds", 3),
        n_repeats=getattr(args, "model_search_cv_repeats", 2),
        random_state=getattr(args, "model_search_random_state", 42),
    )
    if not all_splits:
        raise RuntimeError("model_search staged_group_cv could not create any CV split.")

    stage_a_splits = all_splits[:1]
    stage_a_records = []
    stage_a_models = {}
    logger.info("model_search staged_group_cv: stage A evaluating %d sampled candidates",
                len(candidates))
    for cand in candidates:
        record, model = evaluate_group_cv_candidate(
            cand, args, X_train, y_train, groups=groups, scale_pos_weight=scale_pos_weight,
            splits=stage_a_splits, stage="stage_a_sample")
        record["group_source"] = group_source
        stage_a_records.append(record)
        stage_a_models[_freeze_params(record["params"])] = model

    stage2_top_k = max(1, int(getattr(args, "model_search_stage2_top_k", 80)))
    default_records = [r for r in stage_a_records if r.get("is_default_params", False)]
    top_records = sorted(
        [r for r in stage_a_records if r.get("eligible", False)],
        key=lambda r: (
            -float(r.get("mean_cv_accuracy", 0.0)),
            float(r.get("std_cv_accuracy", 0.0)),
            float(r.get("mean_cv_fp_rate", 0.0)),
            int(r.get("final_total_nodes", 0)),
        ),
    )[:stage2_top_k]

    stage_b_candidates = []
    seen = set()
    for record in top_records + default_records:
        frozen = _freeze_params(record["params"])
        if frozen in seen:
            continue
        seen.add(frozen)
        stage_b_candidates.append({
            "rank_input_order": int(record.get("rank_input_order", len(stage_b_candidates) + 1)),
            "params": record["params"],
            "is_default_params": bool(record.get("is_default_params", False)),
        })

    records = []
    models = {}
    logger.info("model_search staged_group_cv: stage B CV evaluating %d candidates (%d folds)",
                len(stage_b_candidates), len(all_splits))
    for cand in stage_b_candidates:
        record, model = evaluate_group_cv_candidate(
            cand, args, X_train, y_train, groups=groups, scale_pos_weight=scale_pos_weight,
            splits=all_splits, stage="stage_b_group_cv")
        record["group_source"] = group_source
        records.append(record)
        models[_freeze_params(record["params"])] = model

    best = select_best_group_cv_record(
        records,
        accuracy_tolerance=getattr(args, "model_search_accuracy_tolerance", 0.0),
    )
    if best is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or adjust the search space."
        )
    best_model = models[_freeze_params(best["params"])]

    default_record = next((r for r in records if r.get("is_default_params", False)), None)
    for record in records:
        if default_record is not None and not record.get("is_default_params", False):
            record["beats_default_params"] = (
                bool(record.get("eligible", False))
                and float(record.get("mean_cv_accuracy", 0.0))
                > float(default_record.get("mean_cv_accuracy", 0.0))
                + max(0.0, float(getattr(args, "model_search_accuracy_tolerance", 0.0)))
            )
        if _freeze_params(record["params"]) == _freeze_params(best["params"]):
            record["chosen_reason"] = best.get("chosen_reason", "")

    records.sort(key=lambda r: (
        not r.get("eligible", False),
        -float(r.get("mean_cv_accuracy", 0.0)),
        float(r.get("std_cv_accuracy", 0.0)),
        float(r.get("mean_cv_fp_rate", 0.0)),
        int(r.get("final_total_nodes", 0)),
    ))
    axes = build_model_search_axes(args)
    summary = {
        "enabled": True,
        "strategy": "staged_group_cv",
        "selection_data": "train_group_cv_only",
        "valid_used_for_model_selection": False,
        "selection_policy": "mean_cv_accuracy_std_fp_nodes",
        "max_model_nodes": int(args.max_model_nodes),
        "accuracy_tolerance": float(getattr(args, "model_search_accuracy_tolerance", 0.0)),
        "max_candidates": int(getattr(args, "model_search_max_candidates", 600)),
        "stage2_top_k": int(stage2_top_k),
        "cv_folds": int(getattr(args, "model_search_cv_folds", 3)),
        "cv_repeats": int(getattr(args, "model_search_cv_repeats", 2)),
        "cv_folds_completed": int(best.get("cv_folds_completed", 0)),
        "random_state": int(getattr(args, "model_search_random_state", 42)),
        "group_source": group_source,
        "sampled_candidate_count": int(len(candidates)),
        "stage_b_candidate_count": int(len(stage_b_candidates)),
        "parameter_space": {k: list(v) for k, v in axes.items()},
        "default_params_baseline": _json_safe_model_search_record(default_record) if default_record else None,
        "best": _json_safe_model_search_record(best),
        "top_candidates": [_json_safe_model_search_record(r) for r in records[:20]],
    }
    return best_model, summary, records


def search_xgb_hyperparameters(args, X_train, y_train, X_valid, y_valid,
                               scale_pos_weight=1.0, groups=None):
    if getattr(args, "model_search_strategy", "staged_group_cv") == "staged_group_cv":
        return search_xgb_hyperparameters_group_cv(
            args, X_train, y_train, groups=groups, scale_pos_weight=scale_pos_weight)

    axes = build_model_search_axes(args)
    stage1_keys = ["n_estimators", "max_depth", "min_child_weight", "reg_lambda", "reg_alpha"]
    stage2_keys = ["learning_rate", "subsample", "colsample_bytree"]
    fixed_stage2 = {key: _middle_value(axes[key]) for key in stage2_keys}
    top_k = max(1, int(args.model_search_stage1_top_k))

    records = []
    best = None
    best_model = None

    stage1_plan = []
    for values in product(*(axes[k] for k in stage1_keys)):
        params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
        params.update(fixed_stage2)
        params.update(dict(zip(stage1_keys, values)))
        stage1_plan.append(params)

    logger.info("model_search enabled: stage1 evaluating %d structure candidates", len(stage1_plan))
    candidate_count = 0
    stage1_records = []
    for params in stage1_plan:
        candidate_count += 1
        candidate, record = evaluate_model_search_candidate(
            params, candidate_count, "stage1_structure", args, X_train, y_train, X_valid, y_valid)
        records.append(record)
        stage1_records.append(record)
        if is_better_model_search_record(
                record, best, accuracy_tolerance=args.model_search_accuracy_tolerance):
            best = record
            best_model = candidate

    stage1_records.sort(key=lambda r: (
        not r["eligible"],
        -float(r["metrics"].get("accuracy", 0.0)),
        int(r["total_nodes"]),
        float(r["fp_rate"]),
        int(r["rank_input_order"]),
    ))
    refine_structures = [r["params"] for r in stage1_records[:top_k]]
    seen = {_freeze_params(r["params"]) for r in records}
    stage2_plan = []
    for base in refine_structures:
        structure = {key: base[key] for key in stage1_keys}
        for values in product(*(axes[k] for k in stage2_keys)):
            params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
            params.update(structure)
            params.update(dict(zip(stage2_keys, values)))
            frozen = _freeze_params(params)
            if frozen in seen:
                continue
            seen.add(frozen)
            stage2_plan.append(params)

    logger.info("model_search enabled: stage2 refining %d candidates from top %d structures",
                len(stage2_plan), len(refine_structures))
    for params in stage2_plan:
        candidate_count += 1
        candidate, record = evaluate_model_search_candidate(
            params, candidate_count, "stage2_refine", args, X_train, y_train, X_valid, y_valid)
        records.append(record)
        if is_better_model_search_record(
                record, best, accuracy_tolerance=args.model_search_accuracy_tolerance):
            best = record
            best_model = candidate

    if best_model is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or shrink the search grid."
        )

    records.sort(key=lambda r: (
        not r["eligible"],
        -float(r["metrics"].get("accuracy", 0.0)),
        int(r["total_nodes"]),
        float(r["fp_rate"]),
        int(r["rank_input_order"]),
    ))
    return best_model, {
        "enabled": True,
        "selection_data": "valid_only",
        "selection_policy": "valid_accuracy_primary_size_tiebreak",
        "accuracy_tolerance": float(args.model_search_accuracy_tolerance),
        "stage1_top_k": int(top_k),
        "max_model_nodes": int(args.max_model_nodes),
        "fp_cost": float(args.model_search_fp_cost),
        "size_cost": float(args.model_search_size_cost),
        "grid_size": int(candidate_count),
        "stage1_grid_size": int(len(stage1_plan)),
        "stage2_grid_size": int(len(stage2_plan)),
        "best": _json_safe_model_search_record(best),
        "top_candidates": [_json_safe_model_search_record(r) for r in records[:20]],
    }, records


def select_features_for_count(fs, ranked, feature_count):
    if ranked:
        k = min(int(feature_count), len(ranked))
        return [r["feature"] for r in ranked[:k]]
    selected = list(fs["selected_features"])
    k = min(int(feature_count), len(selected))
    return selected[:k]


def _feature_count_selection_record(result):
    summary = result.get("model_search_summary", {})
    best = summary.get("best") or {}
    if "mean_cv_accuracy" in best:
        record = dict(best)
    elif "metrics" in best:
        metrics = best.get("metrics", {})
        record = {
            "eligible": bool(best.get("eligible", False)),
            "mean_cv_accuracy": float(metrics.get("accuracy", 0.0)),
            "std_cv_accuracy": 0.0,
            "mean_cv_fp_rate": float(best.get("fp_rate", 0.0)),
            "mean_cv_precision": float(metrics.get("precision", 0.0)),
            "mean_cv_recall": float(metrics.get("recall", 0.0)),
            "final_total_nodes": int(best.get("total_nodes", result.get("total_nodes", 0))),
            "params": best.get("params", {}),
        }
    else:
        metrics = result.get("valid_best", {})
        record = {
            "eligible": True,
            "mean_cv_accuracy": float(metrics.get("accuracy", 0.0)),
            "std_cv_accuracy": 0.0,
            "mean_cv_fp_rate": _window_fp_rate_from_metrics(metrics),
            "mean_cv_precision": float(metrics.get("precision", 0.0)),
            "mean_cv_recall": float(metrics.get("recall", 0.0)),
            "final_total_nodes": int(result.get("total_nodes", 0)),
            "params": result.get("model_params", {}),
        }
    record["feature_count"] = int(result["feature_count"])
    return record


def summarize_feature_count_search(results, chosen_result, accuracy_tolerance=0.0):
    enabled = len(results) > 1
    candidates = []
    for result in results:
        record = result["selection_record"]
        candidates.append({
            "feature_count": int(result["feature_count"]),
            "eligible": bool(record.get("eligible", False)),
            "mean_cv_accuracy": _json_safe_float(record.get("mean_cv_accuracy", 0.0)),
            "std_cv_accuracy": _json_safe_float(record.get("std_cv_accuracy", 0.0)),
            "mean_cv_fp_rate": _json_safe_float(record.get("mean_cv_fp_rate", 0.0)),
            "final_total_nodes": int(record.get("final_total_nodes", 0)),
        })
    return {
        "enabled": enabled,
        "selection_data": "train_group_cv_only" if enabled else "fixed",
        "selection_policy": "mean_cv_accuracy_std_fp_nodes_feature_count" if enabled else "fixed_feature_count",
        "accuracy_tolerance": float(accuracy_tolerance),
        "candidate_feature_counts": [int(r["feature_count"]) for r in results],
        "chosen_feature_count": int(chosen_result["feature_count"]),
        "chosen_reason": "highest_mean_cv_accuracy_lowest_std_fp_nodes",
        "chosen_record": _json_safe_model_search_record(chosen_result["selection_record"]),
        "candidates": candidates,
    }


def train_final_model_for_features(args, fs, df_train_raw, df_valid_raw,
                                   feature_pool_train_path, splits_path,
                                   selected_features, feature_count):
    logger.info("training final model with %d selected features", len(selected_features))

    logger.info("应用异常值裁剪 (train 学边界 -> train/valid 同步应用)...")
    df_train, clip_bounds = clip_outliers(df_train_raw, selected_features, k=1.5,
                                          return_bounds=True)
    df_valid = clip_outliers(df_valid_raw, selected_features, k=1.5, bounds=clip_bounds)

    quality_thresholds = learn_quality_thresholds(df_train, QUALITY_FEATURES_DEFAULT)
    feature_quantiles = compute_feature_quantiles(
        df_train, selected_features,
        q_low=args.ood_q_low, q_high=args.ood_q_high
    )
    fill_values = prepare_fill_values(df_train, selected_features)

    X_train, y_train, _ = prepare_xy(df_train, selected_features, fill_values=fill_values)
    X_valid, y_valid, _ = prepare_xy(df_valid, selected_features, fill_values=fill_values)
    train_groups = df_train["sample_name"].astype(str).values if "sample_name" in df_train.columns else None

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    n_total = max(neg_count + pos_count, 1)
    p_train_pos = pos_count / float(n_total)

    scale_pos_weight_strategy = "balanced_1.0"
    if args.legacy_scale_pos_weight:
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        scale_pos_weight_strategy = "legacy_neg_over_pos"
    elif args.target_deploy_ratio is not None:
        r = float(args.target_deploy_ratio)
        r = min(max(r, 1e-6), 1 - 1e-6)
        if 0.0 < p_train_pos < 1.0:
            scale_pos_weight = (r * (1 - p_train_pos)) / ((1 - r) * p_train_pos)
        else:
            scale_pos_weight = 1.0
        scale_pos_weight_strategy = f"target_deploy_ratio={r}"
    else:
        scale_pos_weight = 1.0

    logger.info("样本分布统计:")
    logger.info("  负样本(target=0): %d", neg_count)
    logger.info("  正样本(target=1): %d", pos_count)
    logger.info("  train 正类占比 p_train_pos: %.4f", p_train_pos)
    logger.info("  scale_pos_weight 策略: %s", scale_pos_weight_strategy)
    logger.info("  scale_pos_weight: %.4f", scale_pos_weight)

    model_search_summary = {"enabled": False}
    model_search_records = []
    if args.model_search:
        model, model_search_summary, model_search_records = search_xgb_hyperparameters(
            args, X_train, y_train, X_valid, y_valid,
            scale_pos_weight=scale_pos_weight,
            groups=train_groups,
        )
        for record in model_search_records:
            record["feature_count"] = int(feature_count)
    else:
        model = train_xgb_with_params(
            build_default_xgb_params(scale_pos_weight=scale_pos_weight),
            X_train, y_train,
        )

    total_nodes = count_xgb_nodes(model)
    avg_nodes = total_nodes / max(model.n_estimators, 1)
    logger.info("trained %d trees, total_nodes=%d, avg_nodes/tree=%.1f",
                model.n_estimators, total_nodes, avg_nodes)
    if total_nodes > int(args.max_model_nodes):
        logger.warning("总节点数 %d 超过 %d 上限", total_nodes, int(args.max_model_nodes))

    valid_default = eval_model(model, X_valid, y_valid, threshold=0.5)
    best_threshold = search_threshold_by_valid(
        model, X_valid, y_valid,
        objective=args.threshold_objective,
        beta=args.threshold_beta,
        min_precision=args.threshold_min_precision,
    )
    valid_best = eval_model(model, X_valid, y_valid, threshold=best_threshold["threshold"])
    fingerprint = build_fingerprint(args.artifact_dir, feature_pool_train_path, splits_path)

    result = {
        "feature_count": int(feature_count),
        "selected_features": selected_features,
        "clip_bounds": clip_bounds,
        "quality_thresholds": quality_thresholds,
        "feature_quantiles": feature_quantiles,
        "fill_values": fill_values,
        "model": model,
        "model_search_summary": model_search_summary,
        "model_search_records": model_search_records,
        "total_nodes": int(total_nodes),
        "avg_nodes": float(avg_nodes),
        "valid_default": valid_default,
        "best_threshold": best_threshold,
        "valid_best": valid_best,
        "fingerprint": fingerprint,
        "neg_count": neg_count,
        "pos_count": pos_count,
        "p_train_pos": float(p_train_pos),
        "scale_pos_weight": float(scale_pos_weight),
        "scale_pos_weight_strategy": scale_pos_weight_strategy,
        "model_params": model.get_params(),
    }
    result["selection_record"] = _feature_count_selection_record(result)
    return result


# =========================================================
# 质量阈值 / OOD 分位数 / fingerprint
# =========================================================

QUALITY_FEATURES_DEFAULT = ["PPG_mean", "PPG_std"]


def learn_quality_thresholds(df_train, features=None, q_high=0.99, q_low=0.01):
    """
    从 train 集学习 compute_quality 用的阈值。
    PPG_std 用 q_high 分位（防过大）；PPG_mean 用 q_low 分位（防过小）。
    """
    if features is None:
        features = QUALITY_FEATURES_DEFAULT
    out = {}
    for f in features:
        if f not in df_train.columns:
            continue
        x = df_train[f].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        if f.endswith("_std"):
            out[f] = {"type": "high", "thr": float(x.quantile(q_high))}
        else:
            out[f] = {"type": "low", "thr": float(np.abs(x).quantile(q_low))}
    out["_meta"] = {
        "learned_from": "train_only",
        "q_high": float(q_high),
        "q_low": float(q_low),
    }
    return out


def compute_feature_quantiles(df_train, features, q_low=0.05, q_high=0.95):
    """从 train 算每个特征的 [q_low, q_high]，给 s06 做 OOD 监控。"""
    out = {}
    for f in features:
        if f not in df_train.columns:
            continue
        x = df_train[f].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        out[f] = {
            "q_low": float(x.quantile(q_low)),
            "q_high": float(x.quantile(q_high)),
        }
    out["_meta"] = {
        "learned_from": "train_only",
        "q_low": float(q_low),
        "q_high": float(q_high),
    }
    return out


def build_fingerprint(artifact_dir, feature_pool_path, splits_path):
    """收集 provenance：版本号、数据 hash、git sha、训练时间。"""
    import hashlib
    import time
    import platform

    info = {
        "train_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        info["numpy"] = np.__version__
    except Exception:
        pass
    try:
        info["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import xgboost as _xgb
        info["xgboost"] = _xgb.__version__
    except Exception:
        pass

    def sha256_head(path, head_bytes=4 * 1024 * 1024):
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(head_bytes))
        return h.hexdigest()

    info["splits_sha256_head"] = sha256_head(splits_path)
    info["feature_pool_train_sha256_head"] = sha256_head(feature_pool_path)

    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=2)
        if r.returncode == 0:
            info["git_sha"] = r.stdout.strip()
    except Exception:
        pass

    return info


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument(
        "--threshold_objective", type=str, default="fbeta",
        choices=["f1", "precision", "recall", "fbeta", "precision_constrained"],
        help="阈值搜索目标。默认 fbeta（偏 precision，因为 FP 代价更高）。"
    )
    parser.add_argument("--threshold_beta", type=float, default=0.5,
                        help="F-beta 的 beta。<1 偏 precision，>1 偏 recall。")
    parser.add_argument("--threshold_min_precision", type=float, default=0.95,
                        help="precision_constrained 模式下的精度下限。")
    parser.add_argument("--max_features", type=int, default=None,
                        help="Compatibility passthrough from s08; selected_features.json remains authoritative.")
    parser.add_argument("--model_search", action="store_true",
                        help="Search XGBoost hyperparameters on valid.")
    parser.add_argument("--max_model_nodes", type=int, default=500)
    parser.add_argument("--model_search_fp_cost", type=float, default=2.0)
    parser.add_argument("--model_search_size_cost", type=float, default=0.1)
    parser.add_argument("--model_search_strategy", type=str, default="staged_group_cv",
                        choices=["staged_group_cv", "staged_valid"])
    parser.add_argument("--model_search_max_candidates", type=int, default=600)
    parser.add_argument("--model_search_stage2_top_k", type=int, default=80)
    parser.add_argument("--model_search_cv_folds", type=int, default=3)
    parser.add_argument("--model_search_cv_repeats", type=int, default=2)
    parser.add_argument("--model_search_random_state", type=int, default=42)
    parser.add_argument("--model_search_accuracy_tolerance", type=float, default=0.0)
    parser.add_argument("--model_search_stage1_top_k", type=int, default=4)
    parser.add_argument("--model_search_n_estimators", type=str, default="20,25,30,35,40,45,50,55,60")
    parser.add_argument("--model_search_max_depth", type=str, default="2,3,4")
    parser.add_argument("--model_search_learning_rate", type=str, default="0.025,0.03,0.04,0.05,0.06,0.08,0.10")
    parser.add_argument("--model_search_min_child_weight", type=str, default="10,15,20,25,30,40,50")
    parser.add_argument("--model_search_reg_lambda", type=str, default="5,8,10,12,16,20,30")
    parser.add_argument("--model_search_reg_alpha", type=str, default="0,0.5,1,1.5,2,3")
    parser.add_argument("--model_search_subsample", type=str, default="0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--model_search_colsample_bytree", type=str, default="0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--model_search_feature_counts", type=str, default="",
                        help="搜参时测试的特征数量，逗号分隔 (如 8,10,12,15,18,20)。留空则使用 --max_features 固定值")
    parser.add_argument("--ood_q_low", type=float, default=0.05)
    parser.add_argument("--ood_q_high", type=float, default=0.95)
    parser.add_argument(
        "--target_deploy_ratio", type=float, default=None,
        help=("部署时 Stage2 输入的期望 P(target=1 | Stage1 pass)。"
              "给出后按公式 r*(1-p_train)/((1-r)*p_train) 计算 scale_pos_weight。"
              "默认 None：scale_pos_weight=1.0。")
    )
    parser.add_argument(
        "--legacy_scale_pos_weight", action="store_true", default=False,
        help=("回退到旧行为 scale_pos_weight = neg/pos。仅用于对照实验。")
    )

    if args is None:
        args = parser.parse_args()

    selected_features_path = os.path.join(args.artifact_dir, "selected_features.json")
    ranked_features_path = os.path.join(args.artifact_dir, "ranked_features.json")

    # 始终加载 selected_features.json（含 selection_policy 元数据）
    with open(selected_features_path, "r", encoding="utf-8") as f:
        fs = json.load(f)

    feature_pool_train_path = os.path.join(args.artifact_dir, "feature_pool_train.csv")
    feature_pool_valid_path = os.path.join(args.artifact_dir, "feature_pool_valid.csv")
    splits_path = os.path.join(args.artifact_dir, "splits.json")

    df_train_raw = pd.read_csv(feature_pool_train_path)
    df_valid_raw = pd.read_csv(feature_pool_valid_path)

    ranked = None
    if os.path.exists(ranked_features_path):
        with open(ranked_features_path, "r", encoding="utf-8") as f:
            ranked = json.load(f)

    ranked_count = len(ranked) if ranked is not None else len(fs["selected_features"])
    default_feature_count = args.max_features if args.max_features is not None else ranked_count
    feature_count_search_enabled = bool(str(args.model_search_feature_counts or "").strip())
    if feature_count_search_enabled and not args.model_search:
        raise ValueError("--model_search_feature_counts requires --model_search")
    if feature_count_search_enabled and args.model_search_strategy != "staged_group_cv":
        raise ValueError("--model_search_feature_counts requires --model_search_strategy staged_group_cv")
    feature_counts = parse_feature_count_candidates(
        args.model_search_feature_counts,
        max_features=default_feature_count,
        ranked_count=ranked_count,
    )
    if not feature_counts:
        feature_counts = [int(default_feature_count)]

    fit_results = []
    for feature_count in feature_counts:
        selected_for_count = select_features_for_count(fs, ranked, feature_count)
        logger.info("feature_count candidate k=%d -> %d selected features",
                    int(feature_count), len(selected_for_count))
        fit_results.append(train_final_model_for_features(
            args, fs, df_train_raw, df_valid_raw,
            feature_pool_train_path, splits_path,
            selected_for_count, feature_count,
        ))

    chosen_result = select_best_feature_count_result(
        fit_results,
        accuracy_tolerance=args.model_search_accuracy_tolerance,
    )
    if chosen_result is None:
        raise RuntimeError("feature-count search found no eligible model under max_model_nodes")

    feature_count_search_summary = summarize_feature_count_search(
        fit_results,
        chosen_result,
        accuracy_tolerance=args.model_search_accuracy_tolerance,
    )
    selected_features = chosen_result["selected_features"]
    fill_values = chosen_result["fill_values"]
    clip_bounds = chosen_result["clip_bounds"]
    quality_thresholds = chosen_result["quality_thresholds"]
    feature_quantiles = chosen_result["feature_quantiles"]
    model = chosen_result["model"]
    best_threshold = chosen_result["best_threshold"]
    model_search_summary = dict(chosen_result["model_search_summary"])
    model_search_summary["feature_count_search"] = feature_count_search_summary
    total_nodes = chosen_result["total_nodes"]
    avg_nodes = chosen_result["avg_nodes"]
    valid_default = chosen_result["valid_default"]
    valid_best = chosen_result["valid_best"]
    fingerprint = chosen_result["fingerprint"]
    neg_count = chosen_result["neg_count"]
    pos_count = chosen_result["pos_count"]
    p_train_pos = chosen_result["p_train_pos"]
    scale_pos_weight = chosen_result["scale_pos_weight"]
    scale_pos_weight_strategy = chosen_result["scale_pos_weight_strategy"]

    model_search_records = []
    for result in fit_results:
        for record in result["model_search_records"]:
            record["chosen_feature_count"] = int(result["feature_count"]) == int(chosen_result["feature_count"])
            model_search_records.append(record)

    print("\nValid 默认阈值 0.5:")
    print(json.dumps(valid_default, indent=2, ensure_ascii=False))
    print("\nValid 选择出的窗口阈值:")
    print(json.dumps(best_threshold, indent=2, ensure_ascii=False))
    print("\nValid 最优阈值指标:")
    print(json.dumps(valid_best, indent=2, ensure_ascii=False))

    model_path = os.path.join(args.artifact_dir, "final_model.json")
    config_path = os.path.join(args.artifact_dir, "final_model_config.json")
    bundle_path = os.path.join(args.artifact_dir, "model_bundle.pkl")

    model.save_model(model_path)

    model_bundle = {
        "version": "v2",
        "feature_names": selected_features,
        "fill_values": fill_values,
        "scaler": None,
        "model": model,
        "threshold": best_threshold["threshold"],
        "threshold_policy": {
            "objective": args.threshold_objective,
            "beta": float(args.threshold_beta),
            "min_precision": float(args.threshold_min_precision),
        },
        "clip_bounds": clip_bounds,
        "quality_thresholds": quality_thresholds,
        "feature_quantiles": feature_quantiles,
        "fingerprint": fingerprint,
        "model_search": model_search_summary,
        "feature_count_search": feature_count_search_summary,
        "xgboost_complexity": {
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": float(avg_nodes),
            "max_model_nodes": int(args.max_model_nodes),
        },
        "meta": {
            "fs_ppg": float(DEFAULT_FS_PPG),
            "fs_emg": float(DEFAULT_FS_EMG),
            "fs_acc": float(DEFAULT_FS_ACC),
            "fs_ppg_feat": float(FEATURE_FS),
            "fs_acc_feat": float(FEATURE_FS),
            "win_sec": 3.0,
            "step_sec": 1.0,
            "n_ppg_channels": 6,
            "n_emg_channels": 2,
            "n_acc_channels": 3,
            "ppg_mode": "raw6_to_virtual3_chA_basic_features",
            "emg_notch_config": {
                "notch_freqs_hz": [50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
                "notch_bw_hz": 0.8,
                "leak_freqs_hz": [100.0, 150.0, 200.0, 250.0, 300.0],
            },
        },
    }
    joblib.dump(model_bundle, bundle_path)
    print(f"统一模型包已保存: {bundle_path}")
    print(f"bundle fingerprint: {fingerprint}")

    config = {
        "selected_features": selected_features,
        "selected_feature_count": int(len(selected_features)),
        "fill_values": fill_values,
        "clip_bounds": clip_bounds,
        "quality_thresholds": quality_thresholds,
        "feature_quantiles": feature_quantiles,
        "preprocess": {
            "feature_order": selected_features,
            "fill_rule": "NaN/inf -> train median fill_values",
            "clip_rule": "after fill, clip selected features by train-learned clip_bounds",
            "scaler": None,
        },
        "model_path": model_path,
        "window_model_threshold": best_threshold["threshold"],

        "anti_overfit_policy": {
            "model_train_data": "train_only",
            "threshold_selection_data": "valid_only",
            "test_used": False,
            "feature_selection_data": fs.get("selection_policy", {}).get("selection_data", "unknown"),
            "feature_count_selection_data": feature_count_search_summary.get("selection_data", "fixed"),
        },

        "class_balance": {
            "neg_count": neg_count,
            "pos_count": pos_count,
            "p_train_pos": float(p_train_pos),
            "scale_pos_weight": float(scale_pos_weight),
            "scale_pos_weight_strategy": scale_pos_weight_strategy,
            "target_deploy_ratio": args.target_deploy_ratio,
        },

        "xgboost_params": model.get_params(),
        "xgboost_complexity": {
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": float(avg_nodes),
            "max_model_nodes": int(args.max_model_nodes),
        },
        "model_search": model_search_summary,
        "feature_count_search": feature_count_search_summary,
        "valid_default_threshold_metrics": valid_default,
        "valid_best_threshold_metrics": valid_best,
        "threshold_search": best_threshold,
        "threshold_policy": {
            "objective": args.threshold_objective,
            "beta": float(args.threshold_beta),
            "min_precision": float(args.threshold_min_precision),
        },
        "fingerprint": fingerprint,

        "postprocess": {
            "alpha": 0.4,
            "median_k": 1,
            "T_on": 0.75,
            "T_off": 0.35,
            "K_on": 5,
            "K_off": 5,
            "cooldown_sec": 5
        }
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")

    if args.model_search:
        records_path = os.path.join(args.artifact_dir, "model_search_records.json")
        records_csv_path = os.path.join(args.artifact_dir, "model_search_records.csv")
        results_csv_path = os.path.join(args.artifact_dir, "model_search_results.csv")
        safe_records = [_json_safe_model_search_record(r) for r in model_search_records]
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(safe_records, f, indent=2, ensure_ascii=False)
        if getattr(args, "model_search_strategy", "staged_group_cv") == "staged_group_cv":
            rows = [model_search_record_to_csv_row(r) for r in safe_records]
        else:
            rows = []
            for r in safe_records:
                row = {
                    "rank_input_order": r["rank_input_order"],
                    "feature_count": r.get("feature_count"),
                    "chosen_feature_count": r.get("chosen_feature_count"),
                    "eligible": r["eligible"],
                    "score": r["score"],
                    "fp_rate": r["fp_rate"],
                    "size_ratio": r["size_ratio"],
                    "total_nodes": r["total_nodes"],
                    "avg_nodes_per_tree": r["avg_nodes_per_tree"],
                    "threshold": r["threshold_search"]["threshold"],
                }
                row.update({f"param_{k}": v for k, v in r["params"].items()})
                row.update({f"metric_{k}": v for k, v in r["metrics"].items()
                            if k != "confusion_matrix"})
                rows.append(row)
        pd.DataFrame(rows).to_csv(records_csv_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(rows).to_csv(results_csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
