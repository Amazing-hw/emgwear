# test_feature_report.py
# -*- coding: utf-8 -*-

"""
Test-set feature embedding visualization report for new_new (EMG liveness).

Loads feature_pool_test.csv from an artifact_dir, computes PCA/t-SNE/UMAP
embeddings, and generates publication-quality figures with label-colored points.

This is a self-contained module; it does NOT import from new_codex.
All embedding/plotting functions are inlined and adapted for the new_new
feature pool format (META_COLS = ["sample_name", "h5_file", "target", "start_100hz"]).

Usage:
    python test_feature_report.py --artifact_dir artifacts
    python test_feature_report.py --artifact_dir artifacts --methods pca,tsne --max_points 500 --dpi 200
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Constants (adapted for new_new project)
# ══════════════════════════════════════════════════════════════════════════

META_COLS = ["sample_name", "h5_file", "target", "start_100hz"]

LABEL_COLORS = {
    0: "#4C78A8",
    1: "#D65F5F",
}

METHOD_TITLES = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
}

# ══════════════════════════════════════════════════════════════════════════
# Style & save utilities
# ══════════════════════════════════════════════════════════════════════════

def _set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> List[str]:
    paths = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        save_kwargs = {"dpi": dpi}
        if fmt.lower() in {"png", "tif", "tiff"}:
            save_kwargs["facecolor"] = "white"
        fig.savefig(out, **save_kwargs)
        paths.append(str(out))
    return paths


def _safe_feature_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("._")
    return safe or "feature"


# ══════════════════════════════════════════════════════════════════════════
# Data loading (adapted for new_new META_COLS)
# ══════════════════════════════════════════════════════════════════════════

def load_test_feature_pool(artifact_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load feature_pool_test.csv and return (dataframe, numeric_feature_columns)."""
    path = Path(artifact_dir) / "feature_pool_test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Test feature pool not found: {path}\n"
            "Run s03_extract_feature_pool.py first to generate this file."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read {path}: {exc}. "
            "The feature_pool_test.csv is malformed or partially written."
        ) from exc

    if len(df) == 0:
        raise ValueError(f"Test feature pool is empty: {path}")

    df = df.copy()
    df["split"] = "test"

    if "target" not in df.columns:
        raise ValueError("Feature pool must contain a 'target' column")

    # Identify numeric feature columns by excluding META_COLS
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    meta_set = set(META_COLS) | {"split"}
    feature_cols = [c for c in numeric_cols if c not in meta_set]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after excluding metadata columns")

    return df, feature_cols


def load_selected_features(artifact_dir: Path, available_features: Sequence[str]) -> Tuple[List[str], Dict]:
    """Load selected_features.json if present; return (feature_list, status_dict)."""
    path = Path(artifact_dir) / "selected_features.json"
    if not path.exists():
        return [], {"status": "skipped", "reason": "selected_features.json not found"}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [], {"status": "skipped", "reason": f"Failed to read selected_features.json: {exc}"}

    if isinstance(payload, list):
        selected = [str(x) for x in payload]
    elif isinstance(payload, dict):
        selected = [str(x) for x in payload.get("selected_features", [])]
    else:
        return [], {"status": "skipped", "reason": "Unexpected format in selected_features.json"}

    available = set(available_features)
    present = [name for name in selected if name in available]
    missing = [name for name in selected if name not in available]
    status = {
        "status": "ok" if present else "skipped",
        "n_features": len(present),
        "selected_features": present,
        "missing_features": missing,
    }
    if not present:
        status["reason"] = "No selected features matched available columns"
    return present, status


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    max_points: int = 0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sample, median-impute, and z-score scale the feature matrix."""
    # Sampling
    if max_points and max_points > 0 and len(df) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=max_points, replace=False)
        sampled = df.iloc[np.sort(idx)].reset_index(drop=True)
    else:
        sampled = df.copy()

    raw = sampled.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    scaled = StandardScaler().fit_transform(imputed)
    return sampled.reset_index(drop=True), scaled


# ══════════════════════════════════════════════════════════════════════════
# Embedding computation
# ══════════════════════════════════════════════════════════════════════════

def _pad_components(values: np.ndarray, dim: int) -> np.ndarray:
    if values.shape[1] >= dim:
        return values[:, :dim]
    padded = np.zeros((values.shape[0], dim), dtype=float)
    padded[:, : values.shape[1]] = values
    return padded


def compute_embeddings(
    x: np.ndarray,
    methods: Sequence[str],
    dims: Sequence[int],
    random_state: int = 42,
    perplexity: float = 30.0,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, dict]]:
    """Compute PCA/t-SNE/UMAP embeddings from standardized feature matrix."""
    requested_methods = tuple(str(m).strip().lower() for m in methods if str(m).strip())
    requested_dims = tuple(sorted({int(d) for d in dims if int(d) in {2, 3}}))
    embeddings: Dict[str, Dict[int, np.ndarray]] = {}
    status: Dict[str, dict] = {}

    if x.shape[0] < 3:
        raise ValueError("At least 3 windows are required for 2D/3D embedding figures")

    if "pca" in requested_methods:
        n_components = min(3, x.shape[0], x.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        coords = pca.fit_transform(x)
        embeddings["pca"] = {dim: _pad_components(coords, dim) for dim in requested_dims}
        explained = [float(v) for v in pca.explained_variance_ratio_]
        status["pca"] = {
            "status": "ok",
            "explained_variance_ratio": explained,
        }

    if "tsne" in requested_methods:
        if x.shape[0] < 5:
            status["tsne"] = {"status": "skipped", "reason": "need at least 5 windows for stable t-SNE"}
        else:
            method_embeddings = {}
            effective_perplexity = min(float(perplexity), max(2.0, (x.shape[0] - 1) / 3.0))
            for dim in requested_dims:
                init = "pca" if x.shape[1] >= dim else "random"
                model = TSNE(
                    n_components=dim,
                    init=init,
                    learning_rate="auto",
                    perplexity=effective_perplexity,
                    random_state=random_state,
                    metric="euclidean",
                )
                method_embeddings[dim] = model.fit_transform(x)
            embeddings["tsne"] = method_embeddings
            status["tsne"] = {"status": "ok", "perplexity": float(effective_perplexity)}

    if "umap" in requested_methods:
        try:
            from umap import UMAP  # type: ignore
        except Exception as exc:
            status["umap"] = {"status": "skipped", "reason": f"umap-learn not installed ({exc.__class__.__name__})"}
        else:
            n_neighbors = min(30, max(2, x.shape[0] - 1))
            method_embeddings = {}
            for dim in requested_dims:
                model = UMAP(
                    n_components=dim,
                    n_neighbors=n_neighbors,
                    min_dist=0.12,
                    metric="euclidean",
                    random_state=random_state,
                )
                method_embeddings[dim] = model.fit_transform(x)
            embeddings["umap"] = method_embeddings
            status["umap"] = {"status": "ok", "n_neighbors": int(n_neighbors), "min_dist": 0.12}

    return embeddings, status


# ══════════════════════════════════════════════════════════════════════════
# Scatter / embedding plots
# ══════════════════════════════════════════════════════════════════════════

def _axis_label(method: str, axis_idx: int) -> str:
    if method == "pca":
        return f"PC{axis_idx + 1}"
    if method == "tsne":
        return f"t-SNE {axis_idx + 1}"
    return f"UMAP {axis_idx + 1}"


def _format_axes_2d(ax, method: str) -> None:
    ax.set_xlabel(_axis_label(method, 0))
    ax.set_ylabel(_axis_label(method, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.set_axisbelow(True)


def _format_axes_3d(ax, method: str) -> None:
    ax.set_xlabel(_axis_label(method, 0), labelpad=-2)
    ax.set_ylabel(_axis_label(method, 1), labelpad=-2)
    ax.set_zlabel(_axis_label(method, 2), labelpad=-2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.25, color="#D9D9D9", alpha=0.45)
    ax.view_init(elev=24, azim=38)


def _plot_points(ax, coords: np.ndarray, labels: np.ndarray, dim: int) -> None:
    unique_labels = sorted(pd.Series(labels).dropna().unique().tolist())
    for label in unique_labels:
        mask = labels == label
        color = LABEL_COLORS.get(int(label), "#6F6F6F") if str(label).lstrip("-").isdigit() else "#6F6F6F"
        label_text = f"label={label}"
        if dim == 2:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=6, marker="o", c=color,
                edgecolors="white", linewidths=0.18, alpha=0.76,
                label=label_text,
            )
        else:
            ax.scatter(
                coords[mask, 0], coords[mask, 1], coords[mask, 2],
                s=5, marker="o", c=color,
                edgecolors="white", linewidths=0.12, alpha=0.72,
                depthshade=False, label=label_text,
            )


def _plot_single(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    dim: int,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> List[str]:
    n_total = int(coords.shape[0])
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    title = (
        f"{METHOD_TITLES.get(method, method.upper())} {dim}D{title_suffix}\n"
        f"(n={n_total}, pos={n_pos}, neg={n_neg})"
    )
    if dim == 2:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        _plot_points(ax, coords, labels, dim=2)
        _format_axes_2d(ax, method)
        ax.set_title(title, fontsize=8)
    else:
        fig = plt.figure(figsize=(3.5, 3.1))
        ax = fig.add_subplot(111, projection="3d")
        _plot_points(ax, coords, labels, dim=3)
        _format_axes_3d(ax, method)
        ax.set_title(title, pad=8, fontsize=8)

    ax.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    paths = _save_figure(fig, out_dir / f"{method}_{dim}d{filename_suffix}", formats, dpi)
    plt.close(fig)
    return paths


def _plot_panel(
    embeddings: Mapping[str, Mapping[int, np.ndarray]],
    labels: np.ndarray,
    dim: int,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> List[str]:
    methods = [m for m in ("pca", "tsne", "umap") if dim in embeddings.get(m, {})]
    if not methods:
        return []

    n_total = int(labels.shape[0])
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    count_str = f"n={n_total} pos={n_pos} neg={n_neg}"
    if dim == 2:
        fig, axes = plt.subplots(1, len(methods), figsize=(3.15 * len(methods), 2.8), squeeze=False)
        for ax, method in zip(axes[0], methods):
            _plot_points(ax, embeddings[method][dim], labels, dim=2)
            _format_axes_2d(ax, method)
            ax.set_title(f"{METHOD_TITLES.get(method, method.upper())} 2D{title_suffix}\n({count_str})", fontsize=8)
        axes[0, -1].legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    else:
        fig = plt.figure(figsize=(3.25 * len(methods), 3.0))
        for idx, method in enumerate(methods, start=1):
            ax = fig.add_subplot(1, len(methods), idx, projection="3d")
            _plot_points(ax, embeddings[method][dim], labels, dim=3)
            _format_axes_3d(ax, method)
            ax.set_title(f"{METHOD_TITLES.get(method, method.upper())} 3D{title_suffix}\n({count_str})", pad=6, fontsize=8)
            if idx == len(methods):
                ax.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)

    fig.tight_layout(w_pad=1.0)
    paths = _save_figure(fig, out_dir / f"embedding_panel_{dim}d{filename_suffix}", formats, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════
# Feature distribution plots
# ══════════════════════════════════════════════════════════════════════════

def _plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    feature_index: int,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    random_state: int,
) -> List[str]:
    plot_df = df.loc[:, ["target", feature]].copy()
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["target", feature])

    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = sorted(plot_df["target"].dropna().unique().tolist())
        rng = np.random.default_rng(int(random_state) + int(feature_index))
        positions = np.arange(len(labels), dtype=float)
        groups = [plot_df.loc[plot_df["target"] == label, feature].to_numpy(dtype=float) for label in labels]

        violin = ax.violinplot(
            groups, positions=positions, widths=0.68,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#E8E8E8")
            body.set_edgecolor("#A0A0A0")
            body.set_linewidth(0.5)
            body.set_alpha(0.55)

        box = ax.boxplot(
            groups, positions=positions, widths=0.24,
            patch_artist=True, showfliers=False,
            medianprops={"color": "#202020", "linewidth": 0.9},
            boxprops={"facecolor": "white", "edgecolor": "#404040", "linewidth": 0.65},
            whiskerprops={"color": "#404040", "linewidth": 0.6},
            capprops={"color": "#404040", "linewidth": 0.6},
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.82)

        for pos, label, values in zip(positions, labels, groups):
            jitter = rng.normal(0.0, 0.045, size=len(values))
            color = LABEL_COLORS.get(int(label), "#6F6F6F") if str(label).lstrip("-").isdigit() else "#6F6F6F"
            ax.scatter(
                np.full(len(values), pos) + jitter, values,
                s=4, marker="o", c=color,
                edgecolors="white", linewidths=0.12, alpha=0.55,
                label=f"label={label}", zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"label={label}" for label in labels])
        ax.legend(frameon=False, loc="best", handletextpad=0.25, borderpad=0.2)

    ax.set_title(feature)
    ax.set_xlabel("Target label")
    ax.set_ylabel("Feature value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.set_axisbelow(True)

    stem = out_dir / f"feature_distribution_{feature_index:02d}_{_safe_feature_name(feature)}"
    paths = _save_figure(fig, stem, formats, dpi)
    plt.close(fig)
    return paths


def plot_selected_feature_distributions(
    out_dir: Path,
    df: pd.DataFrame,
    selected_features: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    random_state: int,
) -> Dict[str, List[str]]:
    figure_paths: Dict[str, List[str]] = {}
    for idx, feature in enumerate(selected_features, start=1):
        key = f"feature_distribution_{idx:02d}_{_safe_feature_name(feature)}"
        figure_paths[key] = _plot_feature_distribution(
            df=df, feature=feature, feature_index=idx,
            out_dir=out_dir, formats=formats, dpi=dpi, random_state=random_state,
        )
    return figure_paths


# ══════════════════════════════════════════════════════════════════════════
# Feature correlation heatmap
# ══════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    out_dir: Path, df: pd.DataFrame, selected_features: Sequence[str],
    formats: Sequence[str], dpi: int,
) -> List[str]:
    if len(selected_features) < 2:
        return []
    matrix = df.loc[:, list(selected_features)].apply(pd.to_numeric, errors="coerce")
    corr = matrix.replace([np.inf, -np.inf], np.nan).corr(method="pearson")
    if corr.empty:
        return []

    fig_size = max(3.2, 0.34 * len(selected_features) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title("Selected feature correlation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    paths = _save_figure(fig, out_dir / "selected_feature_correlation_heatmap", formats, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════
# PCA loading top features
# ══════════════════════════════════════════════════════════════════════════

def plot_pca_loading_top_features(
    out_dir: Path, x: np.ndarray, feature_cols: Sequence[str],
    formats: Sequence[str], dpi: int, top_n: int = 15,
) -> List[str]:
    if x.shape[0] < 2 or x.shape[1] < 1:
        return []
    n_components = min(2, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(x)
    loading_strength = np.sum(np.abs(pca.components_), axis=0)
    order = np.argsort(loading_strength)[::-1][: min(int(top_n), len(feature_cols))]
    names = [str(feature_cols[i]) for i in order][::-1]
    values = [float(loading_strength[i]) for i in order][::-1]

    fig, ax = plt.subplots(figsize=(4.2, max(2.6, 0.22 * len(names) + 1.0)))
    ax.barh(np.arange(len(names)), values, color="#4C78A8", height=0.68)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("|PC loading| sum")
    ax.set_title("Top PCA loading features")
    ax.grid(True, axis="x", linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    paths = _save_figure(fig, out_dir / "pca_loading_top_features", formats, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════
# Test-set-specific visualizations
# ══════════════════════════════════════════════════════════════════════════

def plot_test_feature_auc_ranking(
    out_dir: Path, df: pd.DataFrame, feature_cols: Sequence[str],
    top_n: int = 20, formats: Sequence[str] = ("png",), dpi: int = 600,
) -> List[str]:
    """Per-feature ROC AUC ranking on test set (horizontal bar chart)."""
    if "target" not in df.columns:
        return []

    y = pd.to_numeric(df["target"], errors="coerce")
    rankings = []
    for feature in feature_cols:
        values = pd.to_numeric(df.get(feature), errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = pd.DataFrame({"target": y, "value": values}).dropna()
        if valid["target"].nunique() < 2 or len(valid) < 3:
            continue
        try:
            auc = float(roc_auc_score(valid["target"].astype(int), valid["value"].to_numpy(dtype=float)))
            rankings.append({"feature": feature, "auc_sep": float(max(auc, 1.0 - auc))})
        except Exception:
            continue

    if not rankings:
        return []

    table = pd.DataFrame(rankings).sort_values("auc_sep", ascending=True)
    if top_n and len(table) > int(top_n):
        table = table.tail(int(top_n))

    features = table["feature"].tolist()
    values = table["auc_sep"].tolist()

    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.28 * len(features) + 1.5)))
    colors = [
        "#2F7A3C" if v >= 0.80 else "#D49A2A" if v >= 0.65 else "#B8403F"
        for v in values
    ]
    ax.barh(np.arange(len(features)), values, color=colors, height=0.7, zorder=3)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=6)
    ax.set_xlabel("AUC separation on test set", fontsize=7)
    ax.set_title(
        f"Feature AUC ranking (test set)\nn={len(df)}, features={len(feature_cols)}, "
        f"pos={int((df['target'] == 1).sum())}, neg={int((df['target'] == 0).sum())}",
        fontsize=8,
    )
    ax.set_xlim(0.45, 1.02)
    ax.axvline(x=0.5, color="#A0A0A0", linewidth=0.6, linestyle="--", zorder=1)
    ax.grid(True, axis="x", linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=5, color="#333333")

    fig.tight_layout()
    paths = _save_figure(fig, out_dir / "test_feature_auc_ranking", formats, dpi)
    plt.close(fig)
    return paths


def plot_per_sample_feature_heatmap(
    out_dir: Path, df: pd.DataFrame, feature_cols: Sequence[str],
    selected_features: Sequence[str], top_n: int = 15,
    formats: Sequence[str] = ("png",), dpi: int = 600,
) -> Tuple[List[str], Optional[Path]]:
    """Per-sample aggregated feature heatmap (rows=samples, cols=features)."""
    if "sample_name" not in df.columns or "target" not in df.columns:
        return [], None

    use_features = list(selected_features) if selected_features else list(feature_cols)
    if not use_features:
        return [], None

    # Aggregate: mean per sample
    agg_cols = ["sample_name", "target"] + [c for c in df.columns if c in use_features]
    per_sample = df.loc[:, agg_cols].groupby("sample_name", sort=False).agg(
        {"target": "first", **{c: "mean" for c in use_features}}
    ).reset_index()

    if len(per_sample) < 2:
        return [], None

    # Rank features by per-sample AUC
    y = pd.to_numeric(per_sample["target"], errors="coerce")
    auc_scores = {}
    for feat in use_features:
        vals = pd.to_numeric(per_sample[feat], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = pd.DataFrame({"target": y, "value": vals}).dropna()
        if valid["target"].nunique() >= 2 and len(valid) >= 3:
            try:
                auc = float(roc_auc_score(valid["target"].astype(int), valid["value"].to_numpy(dtype=float)))
                auc_scores[feat] = max(auc, 1.0 - auc)
            except Exception:
                pass

    if not auc_scores:
        return [], None

    top_features = sorted(auc_scores, key=auc_scores.get, reverse=True)[:top_n] if top_n else list(auc_scores.keys())
    n_samples = len(per_sample)

    # Scale features
    matrix = per_sample.loc[:, top_features].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix_imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(matrix), columns=matrix.columns
    )
    matrix_scaled = pd.DataFrame(
        StandardScaler().fit_transform(matrix_imputed), columns=matrix.columns
    ).clip(-3, 3)

    # Hierarchical clustering
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        if n_samples >= 3 and len(top_features) >= 2:
            sample_order = leaves_list(linkage(pdist(matrix_scaled.values), method="ward"))
            feat_order = leaves_list(linkage(pdist(matrix_scaled.values.T), method="ward"))
        else:
            sample_order = np.arange(n_samples)
            feat_order = np.arange(len(top_features))
    except Exception:
        sample_order = np.arange(n_samples)
        feat_order = np.arange(len(top_features))

    ordered_features = [top_features[i] for i in feat_order]
    heatmap_data = matrix_scaled.iloc[sample_order][ordered_features].values
    targets = per_sample["target"].values[sample_order]

    fig_w = max(4.5, 0.35 * len(ordered_features) + 2.0)
    fig_h = max(3.5, 0.18 * n_samples + 1.2)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.05, 0.95], wspace=0.02)
    ax_cbar = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    # Target color bar
    target_colors = [LABEL_COLORS.get(int(t), "#6F6F6F") for t in targets]
    ax_cbar.barh(np.arange(n_samples), [1] * n_samples, color=target_colors, height=1.0)
    ax_cbar.set_xlim(0, 1)
    ax_cbar.set_ylim(-0.5, n_samples - 0.5)
    ax_cbar.invert_yaxis()
    ax_cbar.set_xticks([])
    ax_cbar.set_yticks([])
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)

    ax_hm.imshow(heatmap_data, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax_hm.set_xticks(np.arange(len(ordered_features)))
    ax_hm.set_xticklabels(ordered_features, rotation=45, ha="right", fontsize=5)
    ax_hm.set_yticks(np.arange(n_samples))
    ax_hm.set_yticklabels([f"s{i}" for i in sample_order], fontsize=4)
    ax_hm.set_title(
        f"Per-sample feature heatmap (test set)\n{n_samples} samples x {len(top_features)} features",
        fontsize=8,
    )

    fig.subplots_adjust(wspace=0.04, left=0.08, right=0.98)
    paths = _save_figure(fig, out_dir / "test_per_sample_feature_heatmap", formats, dpi)
    plt.close(fig)

    source_path = out_dir / "test_per_sample_feature_source_data.csv"
    per_sample.loc[:, ["sample_name", "target"] + top_features].to_csv(source_path, index=False)
    return paths, source_path


# ══════════════════════════════════════════════════════════════════════════
# Balancing (downsample positives to match negatives)
# ══════════════════════════════════════════════════════════════════════════

def _balance_by_target(
    sampled: pd.DataFrame, x: np.ndarray, feature_cols: Sequence[str], random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Downsample positives to the number of negatives for balanced visualization."""
    if "target" not in sampled.columns:
        return sampled, x, {"status": "skipped", "reason": "no target column"}

    neg_mask = sampled["target"] == 0
    pos_mask = sampled["target"] == 1
    n_neg = int(neg_mask.sum())
    n_pos = int(pos_mask.sum())

    if n_neg == 0 or n_pos == 0:
        return sampled, x, {"status": "skipped", "reason": f"single class: neg={n_neg}, pos={n_pos}"}
    if n_pos <= n_neg:
        return sampled, x, {"status": "skipped", "reason": f"pos({n_pos}) <= neg({n_neg}), already balanced"}

    rng = np.random.default_rng(int(random_state))
    pos_indices = np.where(pos_mask)[0]
    keep_pos = rng.choice(pos_indices, size=n_neg, replace=False)
    keep_idx = np.sort(np.concatenate([np.where(neg_mask)[0], keep_pos]))
    balanced = sampled.iloc[keep_idx].reset_index(drop=True)

    raw = balanced.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    scaled = StandardScaler().fit_transform(imputed)
    return balanced, scaled, {
        "status": "ok",
        "n_neg": n_neg, "n_pos_original": n_pos,
        "n_pos_downsampled": n_neg, "n_total_balanced": n_neg * 2,
    }


# ══════════════════════════════════════════════════════════════════════════
# Source data & report writing
# ══════════════════════════════════════════════════════════════════════════

def _write_source_data(out_dir: Path, sampled: pd.DataFrame,
                       embeddings: Mapping[str, Mapping[int, np.ndarray]]) -> Path:
    columns = [c for c in ["split", "sample_name", "h5_file", "target", "start_100hz"]
               if c in sampled.columns]
    source = sampled.loc[:, columns].copy()
    for method, by_dim in embeddings.items():
        for dim, coords in by_dim.items():
            for idx in range(dim):
                source[f"{method}_{dim}d_{idx + 1}"] = coords[:, idx]
    path = out_dir / "embedding_source_data.csv"
    source.to_csv(path, index=False)
    return path


def _write_report(out_dir: Path, summary: dict, figure_paths: Dict[str, List[str]]) -> Path:
    lines = [
        "# Test Set Feature Embedding Report",
        "",
        "This report visualizes **test set only** window features from the EMG liveness pipeline.",
        "Each point is one 3s window. Color encodes the binary target label.",
        f"Output directory: `{out_dir.name}`.",
        "",
        "## Data",
        f"- Windows plotted: {summary['n_rows']}",
        f"- Numeric features: {summary['n_features']}",
        f"- Embedding feature source: {summary.get('embedding_feature_source', 'all_numeric')}",
        f"- Labels: {summary['label_counts']}",
        "",
        "## Embedding Figures",
        "- PCA 2D/3D: linear separability and dominant variance directions on test data.",
        "- t-SNE 2D/3D: local neighborhood structure on test data.",
        "- UMAP 2D/3D: global/local manifold structure (`umap-learn` required).",
        "- `*_balanced`: positives randomly downsampled to match negative count.",
        "",
    ]

    for key, paths in sorted(figure_paths.items()):
        if not paths:
            continue
        rel = [Path(p).name for p in paths]
        lines.append(f"- {key}: " + ", ".join(rel))

    # Feature Generalization section
    if any("auc_ranking" in k for k in figure_paths):
        lines.extend([
            "",
            "## Feature Generalization (Test Set AUC Ranking)",
            "",
            "![Test feature AUC ranking](test_feature_auc_ranking.png)",
            "",
        ])

    # Per-sample section
    if any("per_sample" in k for k in figure_paths):
        lines.extend([
            "## Per-Sample Feature Analysis",
            "",
            "![Per-sample feature heatmap](test_per_sample_feature_heatmap.png)",
            "",
        ])

    # Selected feature distributions
    dist_figures = summary.get("selected_feature_distributions", {}).get("figures", {})
    if dist_figures:
        lines.extend([
            "## Selected Feature Distributions",
            "",
        ])
        for key, paths in dist_figures.items():
            rel = [Path(p).name for p in paths]
            lines.append(f"- {key}: " + ", ".join(rel))

    lines.extend(["", "## Method Status", ""])
    for method, info in summary["methods"].items():
        lines.append(f"- {METHOD_TITLES.get(method, method.upper())}: {info}")
    lines.append("")

    path = out_dir / "test_embedding_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════════════════════════

def run_test_embedding_report(
    artifact_dir,
    output_dir=None,
    methods: Sequence[str] = ("pca", "tsne"),
    dims: Sequence[int] = (2, 3),
    formats: Sequence[str] = ("png",),
    max_points: int = 0,
    random_state: int = 42,
    perplexity: float = 30.0,
    dpi: int = 600,
) -> dict:
    """Generate test-set-only feature embedding visualization report.

    Parameters
    ----------
    artifact_dir : Path or str
        Directory containing feature_pool_test.csv (and optionally selected_features.json).
    output_dir : Path or str, optional
        Output directory. Defaults to artifact_dir/test_feature_embedding_report.
    methods : sequence of str
        Embedding methods: "pca", "tsne", "umap".
    dims : sequence of int
        Embedding dimensions: 2, 3.
    formats : sequence of str
        Output formats: "png", "pdf", "svg".
    max_points : int
        Max windows to plot (0 = all).
    random_state : int
        Random seed.
    perplexity : float
        t-SNE perplexity.
    dpi : int
        Output figure DPI.

    Returns
    -------
    dict with keys: output_dir, report_path, summary_path, source_path, summary
    """
    artifact_dir = Path(artifact_dir)
    out_dir = Path(output_dir) if output_dir is not None else artifact_dir / "test_feature_embedding_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_nature_style()
    formats = tuple(str(f).lower().strip() for f in formats if str(f).strip())
    if not formats:
        formats = ("png",)

    # 1. Load test feature pool
    df, all_feature_cols = load_test_feature_pool(artifact_dir)
    selected_features, selected_status = load_selected_features(artifact_dir, all_feature_cols)
    if selected_features:
        feature_cols = list(selected_features)
        embedding_feature_source = "selected_features"
    else:
        feature_cols = list(all_feature_cols)
        embedding_feature_source = "all_numeric_features"

    # 2. Prepare matrix
    sampled, x = prepare_matrix(df, feature_cols, int(max_points), int(random_state))

    # 3. Compute embeddings
    embeddings, method_status = compute_embeddings(
        x=x, methods=methods, dims=dims,
        random_state=int(random_state), perplexity=float(perplexity),
    )

    # 4. Generate scatter plots
    labels = sampled["target"].to_numpy()
    figure_paths: Dict[str, List[str]] = {}
    for method in ("pca", "tsne", "umap"):
        for dim in sorted(embeddings.get(method, {})):
            key = f"{method}_{dim}d"
            figure_paths[key] = _plot_single(
                embeddings[method][dim], labels, method, dim, out_dir, formats, int(dpi),
            )

    for dim in sorted({int(d) for d in dims if int(d) in {2, 3}}):
        figure_paths[f"embedding_panel_{dim}d"] = _plot_panel(
            embeddings, labels, dim, out_dir, formats, int(dpi),
        )

    # 5. Balanced version
    balanced_sampled, balanced_x, balance_info = _balance_by_target(
        sampled, x, feature_cols, int(random_state),
    )
    balanced_method_status = balance_info
    if balance_info.get("status") == "ok":
        balanced_embeddings, balanced_method_status = compute_embeddings(
            x=balanced_x, methods=methods, dims=dims,
            random_state=int(random_state), perplexity=float(perplexity),
        )
        balanced_labels = balanced_sampled["target"].to_numpy()
        bal_sfx = " (balanced)"
        for method in ("pca", "tsne", "umap"):
            for dim in sorted(balanced_embeddings.get(method, {})):
                key = f"{method}_{dim}d_balanced"
                figure_paths[key] = _plot_single(
                    balanced_embeddings[method][dim], balanced_labels,
                    method, dim, out_dir, formats, int(dpi),
                    title_suffix=bal_sfx, filename_suffix="_balanced",
                )
        for dim in sorted({int(d) for d in dims if int(d) in {2, 3}}):
            figure_paths[f"embedding_panel_{dim}d_balanced"] = _plot_panel(
                balanced_embeddings, balanced_labels, dim, out_dir, formats, int(dpi),
                title_suffix=bal_sfx, filename_suffix="_balanced",
            )

    # 6. Source data
    source_path = _write_source_data(out_dir, sampled, embeddings)

    # 7. Feature distributions (only if selected_features available)
    distribution_paths = {}
    if selected_features:
        distribution_paths = plot_selected_feature_distributions(
            out_dir=out_dir, df=sampled, selected_features=selected_features,
            formats=formats, dpi=int(dpi), random_state=int(random_state),
        )

    # 8. Test-set-specific visualizations
    auc_paths = plot_test_feature_auc_ranking(
        out_dir=out_dir, df=sampled, feature_cols=feature_cols,
        top_n=min(20, len(feature_cols)), formats=formats, dpi=int(dpi),
    )
    if auc_paths:
        figure_paths["test_feature_auc_ranking"] = auc_paths

    per_sample_paths, _ = plot_per_sample_feature_heatmap(
        out_dir=out_dir, df=sampled, feature_cols=feature_cols,
        selected_features=selected_features, top_n=15,
        formats=formats, dpi=int(dpi),
    )
    if per_sample_paths:
        figure_paths["test_per_sample_feature_heatmap"] = per_sample_paths

    # 9. Explainer figures
    explainer_figures = {}
    if selected_features:
        explainer_figures["selected_feature_correlation_heatmap"] = plot_correlation_heatmap(
            out_dir=out_dir, df=sampled, selected_features=selected_features,
            formats=formats, dpi=int(dpi),
        )
    explainer_figures["pca_loading_top_features"] = plot_pca_loading_top_features(
        out_dir=out_dir, x=x, feature_cols=feature_cols, formats=formats, dpi=int(dpi),
    )

    # 10. Summary & report
    label_counts = sampled["target"].value_counts().sort_index()
    summary = {
        "n_rows": int(len(sampled)),
        "n_rows_available": int(len(df)),
        "max_points": int(max_points),
        "n_features": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "all_numeric_feature_count": int(len(all_feature_cols)),
        "embedding_feature_source": embedding_feature_source,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "split": "test",
        "methods": method_status,
        "balanced": {
            "status": balance_info.get("status", "skipped"),
            "n_neg": balance_info.get("n_neg"),
            "n_pos_original": balance_info.get("n_pos_original"),
            "n_pos_downsampled": balance_info.get("n_pos_downsampled"),
            "n_total_balanced": balance_info.get("n_total_balanced"),
            "methods": balanced_method_status,
        },
        "source_data": str(source_path),
        "figures": figure_paths,
        "explainer_figures": explainer_figures,
        "selected_feature_distributions": {
            **selected_status,
            "figures": distribution_paths,
        },
    }

    summary_path = out_dir / "embedding_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(out_dir, summary, figure_paths)

    return {
        "output_dir": out_dir,
        "report_path": report_path,
        "summary_path": summary_path,
        "source_path": source_path,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════

def _parse_list(value: str, cast=str):
    return tuple(cast(p.strip()) for p in str(value).split(",") if p.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Test-set feature embedding visualization report (EMG liveness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_feature_report.py --artifact_dir artifacts
  python test_feature_report.py --artifact_dir artifacts --methods pca,tsne,umap
  python test_feature_report.py --artifact_dir artifacts --max_points 500 --dpi 200
        """,
    )
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--methods", type=str, default="pca,tsne",
                        help="Embedding methods: pca,tsne,umap (default: pca,tsne)")
    parser.add_argument("--dims", type=str, default="2,3",
                        help="Embedding dimensions: 2,3 (default: 2,3)")
    parser.add_argument("--formats", type=str, default="png",
                        help="Output formats: png,pdf,svg (default: png)")
    parser.add_argument("--max_points", type=int, default=0,
                        help="Max windows to plot (0=all)")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()
    result = run_test_embedding_report(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        methods=_parse_list(args.methods, str),
        dims=_parse_list(args.dims, int),
        formats=_parse_list(args.formats, str),
        max_points=args.max_points,
        random_state=args.random_state,
        perplexity=args.perplexity,
        dpi=args.dpi,
    )
    print(f"\n[OK] Test feature report generated at: {result['output_dir']}")
    print(f"     Report: {result['report_path']}")
    print(f"     Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
