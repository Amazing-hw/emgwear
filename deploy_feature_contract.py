# deploy_feature_contract.py
# -*- coding: utf-8 -*-
"""Shared contract for features that may enter the deployable window model."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


# s03 may emit diagnostic columns for audit. They are useful in CSVs but are
# not standalone deploy-time model inputs.
NON_DEPLOY_FEATURES = {"TOTAL_INVALID_COUNT"}

_DEPLOYABLE_FEATURE_NAMES = None


def get_deployable_feature_names() -> set:
    """Return feature names that have executable deploy formulas."""
    global _DEPLOYABLE_FEATURE_NAMES
    if _DEPLOYABLE_FEATURE_NAMES is None:
        # Lazy import avoids making deploy code depend on feature-selection code.
        from s08_run_pipeline import _build_feature_code_map

        _DEPLOYABLE_FEATURE_NAMES = (
            set(_build_feature_code_map().keys()) - set(NON_DEPLOY_FEATURES)
        )
    return set(_DEPLOYABLE_FEATURE_NAMES)


def split_deployable_features(features: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split feature names into deployable and blocked lists, preserving order."""
    allowed = get_deployable_feature_names()
    keep = []
    blocked = []
    for feature in features:
        if feature in allowed:
            keep.append(feature)
        else:
            blocked.append(feature)
    return keep, blocked


def filter_deployable_features(features: Iterable[str]) -> List[str]:
    """Return only features with deploy formulas, preserving order."""
    keep, _blocked = split_deployable_features(features)
    return keep


def filter_ranked_deployable_features(ranked: Sequence[dict] | None):
    """Filter ranked feature dictionaries by deployability."""
    if ranked is None:
        return None, []

    allowed = get_deployable_feature_names()
    keep = []
    blocked = []
    for item in ranked:
        feature = item.get("feature")
        if feature in allowed:
            keep.append(item)
        else:
            blocked.append(feature)
    return keep, blocked
