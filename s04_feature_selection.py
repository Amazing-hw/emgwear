# s04_feature_selection.py
# -*- coding: utf-8 -*-

"""
步骤4：稳定性特征筛选（适配 IR pairs + EMG + ACC 特征集）

与旧版主要差异：特征组更新为新的信号类型。
筛选策略（group_kfold + permutation importance + SHAP）保持不变。
"""

import os
import sys
import json
import argparse
import pickle
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Linux/macOS 默认 fork 模式多进程可能死锁，强制 spawn
if sys.platform != "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
import xgboost as xgb

from deploy_feature_contract import NON_DEPLOY_FEATURES, split_deployable_features

from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _elapsed(t0):
    return f"{time.time() - t0:.1f}s"


def _log_timing(name, t0):
    print(f"[timing] {name}: {_elapsed(t0)}", flush=True)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed, skipping SHAP importance")


META_COLS = ["sample_name", "h5_file", "target", "start_100hz"]

# =========================================================
# 特征组定义（适配新信号类型）
# =========================================================

FEATURE_GROUPS = {
    # === EMG 佩戴检测特征（皮肤接触 → 肌电信号）===
    # EMG 接触质量: 幅值 + 分布特征 (8) — limit 3
    # 佩戴时肌电幅值显著高于非佩戴噪声，偏度/峰度反映信号结构
    "emg_contact": [
        "EMG0_MAV", "EMG0_RMS", "EMG0_SNR",
        "EMG0_SKEWNESS", "EMG0_KURTOSIS",
        "EMG1_MAV", "EMG1_RMS", "EMG1_SNR",
        "EMG1_SKEWNESS", "EMG1_KURTOSIS",
    ],
    # EMG 肌肉活动: 波形复杂度 + 3s 窗内稳定性 — limit 2
    # P2P/AMP_CV 描述 3s 窗整体包络动态范围；SUBWIN 描述 1s 子窗间稳定性。
    "emg_activity": [
        "EMG0_WL", "EMG0_ZC", "EMG0_SSC", "EMG0_WAMP",
        "EMG0_P2P", "EMG0_AMP_CV",
        "EMG0_RMS_SUBWIN_CV", "EMG0_MDF_SUBWIN_IQR", "EMG0_WL_SUBWIN_CV",
        "EMG1_WL", "EMG1_ZC", "EMG1_SSC", "EMG1_WAMP",
        "EMG1_P2P", "EMG1_AMP_CV",
        "EMG1_RMS_SUBWIN_CV", "EMG1_MDF_SUBWIN_IQR", "EMG1_WL_SUBWIN_CV",
    ],
    # EMG 频域：粗分带 + 细分带 + 频谱形态
    # 佩戴时功率集中在 20-200Hz 且各子频段呈特征性分布，非佩戴时频谱平坦各段均匀
    # SE95: 95%累积功率频率 — 真实EMG频谱集中(SE95低)，噪声平缓(SE95高)
    "emg_frequency": [
        "EMG0_MNF", "EMG0_MDF", "EMG0_PKF", "EMG0_PSR",
        "EMG0_POW_20_60", "EMG0_POW_60_150", "EMG0_POW_150_450", "EMG0_POW_LH_RATIO",
        "EMG0_POW_20_40", "EMG0_POW_40_60", "EMG0_POW_60_90", "EMG0_POW_90_120",
        "EMG0_POW_120_180", "EMG0_POW_180_250", "EMG0_POW_250_350", "EMG0_POW_350_450",
        "EMG0_RATIO_60_150_TO_20_60", "EMG0_RATIO_60_180_TO_250_450",
        "EMG0_RATIO_20_90_TO_180_450", "EMG0_RATIO_40_120_TO_120_350",
        "EMG0_SE95",
        "EMG0_SPEC_ENTROPY", "EMG0_SPEC_FLATNESS",
        "EMG0_SPEC_CENTROID", "EMG0_SPEC_ROLLOFF_85",
        "EMG1_MNF", "EMG1_MDF", "EMG1_PKF", "EMG1_PSR",
        "EMG1_POW_20_60", "EMG1_POW_60_150", "EMG1_POW_150_450", "EMG1_POW_LH_RATIO",
        "EMG1_POW_20_40", "EMG1_POW_40_60", "EMG1_POW_60_90", "EMG1_POW_90_120",
        "EMG1_POW_120_180", "EMG1_POW_180_250", "EMG1_POW_250_350", "EMG1_POW_350_450",
        "EMG1_RATIO_60_150_TO_20_60", "EMG1_RATIO_60_180_TO_250_450",
        "EMG1_RATIO_20_90_TO_180_450", "EMG1_RATIO_40_120_TO_120_350",
        "EMG1_SE95",
        "EMG1_SPEC_ENTROPY", "EMG1_SPEC_FLATNESS",
        "EMG1_SPEC_CENTROID", "EMG1_SPEC_ROLLOFF_85",
    ],
    # EMG 非线性 (2) — limit 1
    # 活体 EMG 有中等 SampEn，噪声要么极低(0)要么极高(随机)
    "emg_complexity": ["EMG0_SampEn", "EMG1_SampEn"],
    # EMG 通道一致性 (2) — limit 1
    # 佩戴时两通道高度相关，非佩戴时独立噪声
    "emg_cross": [
        "EMG_CROSS_CORR", "EMG_RMS_RATIO",
        "EMG_ENV_CORR", "EMG_MAV_RATIO",
        "EMG_CONTACT_IMBALANCE",
    ],
    # EMG PPG 窄带串扰 (16) — limit 6
    # PPG LED 100Hz 切换在 EMG 上的串扰被显式建模为耦合特征
    # 佩戴时串扰路径稳定（皮肤耦合），非佩戴时串扰可能异常
    "emg_leakage": [
        "EMG0_LEAK_100_RATIO", "EMG0_LEAK_150_RATIO", "EMG0_LEAK_200_RATIO",
        "EMG0_LEAK_250_RATIO", "EMG0_LEAK_300_RATIO",
        "EMG0_LEAK_SUM_RATIO", "EMG0_LEAK_MAX_RATIO",
        "EMG1_LEAK_100_RATIO", "EMG1_LEAK_150_RATIO", "EMG1_LEAK_200_RATIO",
        "EMG1_LEAK_250_RATIO", "EMG1_LEAK_300_RATIO",
        "EMG1_LEAK_SUM_RATIO", "EMG1_LEAK_MAX_RATIO",
    ],

    # EMG 双通道汇总。仅开放少量代表性 min/max/range/cv，避免 32 个 consensus 全落到 other。
    "emg_consensus": [
        "EMG_consensus_RMS_min", "EMG_consensus_RMS_max",
        "EMG_consensus_RMS_range", "EMG_consensus_RMS_cv",
        "EMG_consensus_MDF_min", "EMG_consensus_MDF_max", "EMG_consensus_MDF_range",
        "EMG_consensus_PSR_min", "EMG_consensus_PSR_max", "EMG_consensus_PSR_range",
    ],

    # === PPG 活体检测特征（血流光学信号）===
    # PPG 信号质量 (7) — limit 2
    # DC 水平反映皮肤接触质量，AC/DC 比反映搏动分量，PI 反映灌注稳定性
    "ppg_quality": [
        "PPG_mean", "PPG_std", "PPG_p95",
        "PPG_DC_MEDIAN", "PPG_DC_IQR",
        "PPG_PI", "PPG_PI_SUBWIN_IQR",
    ],
    # PPG 心跳检测 (8) — limit 2
    # FFT 峰值比和自相关峰是心跳存在的核心证据（活体关键）
    "ppg_heartbeat": [
        "PPG_acdc", "PPG_diff_std",
        "PPG_AC_RMS", "PPG_AC_MAD",
        "PPG_AC_DC_RATIO", "PPG_DERIV_MAD",
        "PPG_FFT_PEAK_MEDIAN_RATIO", "PPG_DOM_FREQ",
        "PPG_AUTO_CORR_PEAK", "PPG_AUTO_CORR_LAG_SEC",
    ],
    # PPG 波形形态 (4) — limit 1
    "ppg_waveform": [
        "PPG_bp_skewness", "PPG_bp_kurtosis",
        "PPG_FFT_peak_width_Hz", "PPG_FFT_SNR",
    ],
    # PPG 信号复杂度 (5) — limit 2
    "ppg_complexity": [
        "PPG_Hjorth_Activity", "PPG_Hjorth_Mobility",
        "PPG_Entropy_Shannon",
        "PPG_Entropy_SampEn",
    ],
    # PPG 波形细节 (8) — limit 1
    "ppg_morphology": [
        "PPG_Deriv_d1_mean", "PPG_Deriv_d1_std",
        "PPG_Deriv_d1_max", "PPG_Deriv_d1_min",
        "PPG_Deriv_d1_zcr",
        "PPG_Temporal_slope_mean", "PPG_Temporal_slope_std",
        "PPG_Temporal_peak_prominence",
        "PPG_Temporal_peak_ratio",
    ],

    # === ACC 运动上下文 === (12) — limit 2
    # 加入 8-12Hz 生理震颤带（佩戴时存在，桌面静止时无）
    "acc_features": [
        "ACC_GRAV_MAG_MEAN", "ACC_GRAV_DOM_RATIO",
        "ACC_MOTION_RMS", "ACC_MOTION_STD", "ACC_MOTION_MAD",
        "ACC_AXIS_STD_SUM", "ACC_DIFF_MAD", "ACC_STILL_SCORE",
        "ACC_MAG_P50", "ACC_MAG_P90",
        "ACC_TREMOR_POW_8_12", "ACC_TREMOR_RATIO",
    ],

    # === PPG 3 通道空间一致性 === (18) — limit 3
    # 三通道 PPG 空间差异：真实手腕组织异质 → 不平衡度高；伪造物均匀 → 不平衡≈0
    # 含空间不平衡/范围/向量/DC一致性/BP相关性/相位延迟/耦合
    "ppg_spatial": [
        "PPG_ch_imbalance_mean", "PPG_ch_imbalance_p90", "PPG_ch_imbalance_iqr",
        "PPG_ch_rangeNorm_mean", "PPG_ch_rangeNorm_p90",
        "PPG_ch_vmag_mean", "PPG_ch_vmag_p90", "PPG_ch_vmag_iqr", "PPG_ch_vmag_std",
        "PPG_ch_dc_cv", "PPG_ch_dc_max_min_ratio",
        "PPG_ch_bp_corr_mean", "PPG_ch_bp_corr_min", "PPG_ch_bp_corr_std",
        "PPG_ch_bp_lag_std",
        "PPG_corr_mean_imbalance", "PPG_corr_mean_vmag", "PPG_corr_IR_imbalance",
    ],

    # === 跨模态一致性 === (4) — limit 2
    # 佩戴时 EMG-PPG 存在生理耦合（肌肉活动→血流变化），ACC-PPG 反映运动伪影
    "cross_modal": [
        "ACC_PPG_BP_CORR", "ACC_EMG_CORR",
        "EMG_PPG_CORR", "EMG_PPG_ENV_CORR",
    ],

    # === 防伪靶向特征 === (18) — limit 4，强制保留 ≥3
    # 针对 PPG 回放/橡胶模拟物的威胁:
    #   - EMG 50Hz / 谐波 / baseline drift: 橡胶介电体 + 离体时拾取异常
    #   - PPG dicrotic / AI / pulse width CV: 机械泵单峰、形态高度一致
    #   - PPG RR HRV: 真人微妙不规则，回放/机械源 HRV 极规整
    #   - ACC-PPG coherence: 回放 PPG 不响应当前 ACC 微动 → 相干 ≈ 0
    # 注: 无伪造标签训练数据时这些特征 importance 不会高，靠 min_anti_spoof_features 强制保留
    "anti_spoof": [
        "EMG0_50HZ_RATIO", "EMG0_50HZ_HARM_RATIO",
        "EMG0_40_60HZ_RATIO",
        "EMG1_50HZ_RATIO", "EMG1_50HZ_HARM_RATIO",
        "EMG1_40_60HZ_RATIO",
        "EMG0_DRIFT_HF_RATIO",
        "EMG1_DRIFT_HF_RATIO",
        "PPG_DICROTIC_RATIO", "PPG_AUG_INDEX_MEAN", "PPG_PULSE_WIDTH_CV",
        "PPG_RR_RMSSD", "PPG_RR_CV", "PPG_RR_PNN30",
        "ACC_PPG_COH_MICRO", "ACC_PPG_COH_HR",
    ],

    # === Meta === (2) — limit 0
    "meta": ["SIG_LEN", "SIG_SEC"],
}

GROUP_LIMITS_DEFAULT = {
    # EMG — 佩戴检测核心，给最多名额
    "emg_contact": 3,
    "emg_activity": 2,
    "emg_frequency": 6,
    "emg_complexity": 1,
    "emg_cross": 2,
    "emg_leakage": 4,
    "emg_consensus": 2,
    # PPG — 活体检测核心
    "ppg_quality": 2,
    "ppg_heartbeat": 2,
    "ppg_waveform": 1,
    "ppg_complexity": 2,
    "ppg_morphology": 1,
    # 上下文
    "acc_features": 2,
    "ppg_spatial": 3,
    "cross_modal": 2,
    # 防伪靶向（无标签训练时 importance 低，靠 min_anti_spoof_features 兜底）
    "anti_spoof": 4,
    # Meta
    "meta": 0,
    "other": 2,
}

# 防伪特征最少保留数量（即使训练数据无伪造标签也强制选）
MIN_ANTI_SPOOF_FEATURES_DEFAULT = 3


def get_feature_cols(df):
    exclude = set(META_COLS) | set(NON_DEPLOY_FEATURES)
    cols = [c for c in df.columns if c not in exclude]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    deployable, blocked = split_deployable_features(cols)
    if blocked:
        logger.warning(
            "Dropped %d feature columns without deploy formulas: %s",
            len(blocked),
            blocked[:20],
        )
    return deployable


def feature_to_group(feature):
    for g, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return g
    return "other"


# =========================================================
# 阶段1：组内快速预筛
# =========================================================

def fast_group_preselection(df, feature_cols, group_limits=None, preselect_top=4):
    if group_limits is None:
        group_limits = GROUP_LIMITS_DEFAULT

    y = df["target"].values.astype(int)
    group_features = defaultdict(list)
    for f in feature_cols:
        group_features[feature_to_group(f)].append(f)

    selected = {}
    for group_name, features_in_group in group_features.items():
        n_feat = len(features_in_group)
        limit = group_limits.get(group_name, 1)
        if limit <= 0:
            continue
        # 特征数 ≤3 的小组跳过预筛，全部保留到稳定性选择阶段
        if n_feat <= 3:
            for f in features_in_group:
                selected[f] = {"group": group_name, "importance": 1.0, "method": "tiny_group_pass"}
            continue

        actual_select = min(preselect_top, n_feat, max(limit * 2, 4))
        print(f"  [{group_name}] {n_feat} features, top {actual_select}...",
              end="", flush=True)
        try:
            X_group = df[features_in_group].values.astype(float)
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=20, reg_lambda=10, reg_alpha=1,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=1, verbosity=0,
            )
            model.fit(X_group, y)

            importance_dict = model.get_booster().get_score(importance_type='gain')
            importance = {}
            for idx_str, imp in importance_dict.items():
                try:
                    idx = int(idx_str[1:])
                except Exception:
                    continue
                if 0 <= idx < len(features_in_group):
                    importance[features_in_group[idx]] = imp

            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for i, (f, imp) in enumerate(sorted_features[:actual_select]):
                selected[f] = {
                    "group": group_name, "importance": float(imp),
                    "method": "gain", "rank": i + 1,
                }
            print("done")
        except Exception as e:
            logger.warning(f"fast_group_preselection: group={group_name} 训练失败({e})；"
                           f"回退为顺序保留前 {actual_select} 个")
            for i, f in enumerate(features_in_group[:actual_select]):
                selected[f] = {
                    "group": group_name, "importance": 0.0,
                    "method": "fallback_after_fit_error", "rank": i + 1,
                }
            print("fallback")

    return selected


def compute_vif_values(X, eps=1e-12):
    """Compute VIF for all columns using the pseudo-inverse of the correlation matrix."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_rows, n_cols = X.shape
    if n_cols == 0:
        return np.array([], dtype=np.float64)
    if n_cols == 1 or n_rows < 3:
        return np.ones(n_cols, dtype=np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, ddof=1)
    valid = std > eps
    vif = np.ones(n_cols, dtype=np.float64)
    if int(valid.sum()) <= 1:
        return vif

    Xs = X[:, valid] / std[valid]
    corr = np.corrcoef(Xs, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0, posinf=0.0, neginf=0.0)
    inv_corr = np.linalg.pinv(corr, hermitian=True)
    vif[valid] = np.clip(np.diag(inv_corr), 1.0, np.inf)
    return vif


# =========================================================
# 数据清洗（按 train）
# =========================================================

def clean_features_by_train(df_train, df_valid, feature_cols, missing_thresh=0.3,
                             var_thresh=1e-8, corr_thresh=0.95):
    t_clean = time.time()
    df_train = df_train.copy()
    df_valid = df_valid.copy()
    removed = {"missing": [], "low_variance": [], "high_corr": []}

    df_train[feature_cols] = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
    in_valid = [c for c in feature_cols if c in df_valid.columns]
    if in_valid:
        df_valid[in_valid] = df_valid[in_valid].replace([np.inf, -np.inf], np.nan)

    # 1. high missing
    miss_rate = df_train[feature_cols].isna().mean()
    kept = miss_rate[miss_rate <= missing_thresh].index.tolist()
    removed["missing"] = miss_rate[miss_rate > missing_thresh].index.tolist()
    _log_timing(f"clean/missing filter -> {len(kept)} features", t_clean)

    # 2. median fill
    t_fill = time.time()
    fill_values = {}
    med_series = df_train[kept].median()
    for c in kept:
        med = med_series[c]
        if not np.isfinite(med):
            med = 0.0
        fill_values[c] = float(med)
    df_train[kept] = df_train[kept].fillna(med_series.fillna(0.0))
    valid_kept = [c for c in kept if c in df_valid.columns]
    if valid_kept:
        df_valid[valid_kept] = df_valid[valid_kept].fillna(med_series[valid_kept].fillna(0.0))
    _log_timing("clean/median fill", t_fill)

    # 3. low variance
    t_var = time.time()
    var_series = df_train[kept].var()
    kept2 = var_series[np.isfinite(var_series) & (var_series > var_thresh)].index.tolist()
    removed["low_variance"] = [c for c in kept if c not in kept2]
    _log_timing(f"clean/variance filter -> {len(kept2)} features", t_var)

    # 4. high corr
    t_corr = time.time()
    if len(kept2) > 1:
        print(f"  high-corr matrix: {len(kept2)} features...", end="", flush=True)
        y = df_train["target"].values
        corr = df_train[kept2].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_remove = set()
        for col_i in upper.columns:
            high_corr_with = [col_j for col_j in upper.index if upper.at[col_j, col_i] > corr_thresh]
            if not high_corr_with:
                continue
            for col_j in high_corr_with:
                if col_i in to_remove or col_j in to_remove:
                    continue
                vi = df_train[col_i].fillna(df_train[col_i].median()).values
                vj = df_train[col_j].fillna(df_train[col_j].median()).values
                corr_i = abs(np.corrcoef(vi, y)[0, 1]) if np.isfinite(np.var(vi)) else 0.0
                corr_j = abs(np.corrcoef(vj, y)[0, 1]) if np.isfinite(np.var(vj)) else 0.0
                if corr_i >= corr_j:
                    to_remove.add(col_j)
                else:
                    to_remove.add(col_i)
        kept3 = [c for c in kept2 if c not in to_remove]
        removed["high_corr"] = sorted(list(to_remove))
    else:
        kept3 = kept2

    # 5. VIF (Variance Inflation Factor) — 批量剔除 VIF > 10 的特征
    _log_timing("clean/high corr", t_corr)
    t_vif_all = time.time()
    removed["high_vif"] = []
    if len(kept3) > 2:
        _kept = list(kept3)
        _X = df_train[_kept].values.astype(float)
        _max_iter = max(len(_kept) // 2, 1)
        _orig_n = len(_kept)
        for _iter in range(_max_iter):
            t_vif_round = time.time()
            nf = _X.shape[1]
            if nf <= 2:
                break
            print(f"  VIF round {_iter+1}: {nf} features...", end="", flush=True)
            vif_vals = compute_vif_values(_X)
            # Legacy Ridge path is disabled; compute_vif_values is the active VIF implementation.
            if False:
                if j and (j % progress_step == 0):
                    print(f"{j}/{nf}...", end="", flush=True)
                col_mask[j] = False
                X_rest = _X[:, col_mask]  # 列索引切片，无复制
                y_j = _X[:, j]
                try:
                    ridge = Ridge(alpha=1.0, fit_intercept=True)
                    ridge.fit(X_rest, y_j)
                    y_pred = ridge.predict(X_rest)
                    ss_res = np.sum((y_j - y_pred) ** 2)
                    ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    r2 = max(0.0, min(r2, 0.999))
                    vif_vals[j] = 1.0 / (1.0 - r2)
                except Exception:
                    vif_vals[j] = 100.0
                col_mask[j] = True

            # 一次性移除所有 VIF > 10 的特征
            high_vif_mask = vif_vals > 10.0
            n_remove = int(np.sum(high_vif_mask))
            if n_remove == 0:
                print(f"  done (all VIF <= 10, {_elapsed(t_vif_round)})")
                break
            print(f"  remove {n_remove} ({_elapsed(t_vif_round)})")
            for j in range(nf - 1, -1, -1):
                if high_vif_mask[j]:
                    removed["high_vif"].append(_kept[j])
                    _kept.pop(j)
            _X = df_train[_kept].values.astype(float)  # 重建（仅当有删除时）
        kept3 = _kept
        print(f"  VIF: {_orig_n} -> {len(kept3)} features")
    _log_timing("clean/VIF", t_vif_all)
    _log_timing("clean/total", t_clean)

    return df_train, df_valid, kept3, removed, fill_values


# =========================================================
# 稳定性选择（并行 Permutation Importance）
# =========================================================

_WORKER_DATA = None


def _init_stab_worker(data_pickle):
    global _WORKER_DATA
    _WORKER_DATA = pickle.loads(data_pickle)


def _run_one_fold(args_tuple):
    """
    单 (seed, fold) 训练 + permutation importance。
    返回 (fold_info, top_k 列表 或 [])。
    """
    seed, fold_id, n_folds, tr_idx, va_idx, top_k = args_tuple
    data = _WORKER_DATA
    X = data["X"]
    y = data["y"]
    feature_cols = data["feature_cols"]

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    n_tr = len(y_tr)
    n_va = len(y_va)
    info = {"seed": seed, "fold": fold_id + 1, "n_folds": n_folds,
            "n_train": n_tr, "n_valid": n_va, "auc": None, "kept": False}

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return info, []

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=10, reg_alpha=1,
        objective="binary:logistic", eval_metric="logloss",
        random_state=seed, n_jobs=1, verbosity=0,
    )

    try:
        model.fit(X_tr, y_tr)
    except Exception as e:
        logger.warning(f"fold seed={seed} 训练失败: {e}")
        return info, []

    try:
        proba = model.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, proba)
    except Exception:
        val_auc = 0.5

    info["auc"] = float(val_auc)
    min_fold_auc = data.get("min_fold_auc", 0.55)
    if not np.isfinite(val_auc) or val_auc < min_fold_auc:
        return info, []

    try:
        result = permutation_importance(
            model, X_va, y_va,
            scoring="roc_auc",
            n_repeats=5,
            random_state=seed,
            n_jobs=1,
        )
    except Exception as e:
        logger.warning(f"fold seed={seed} permutation 失败: {e}")
        return info, []

    imps = result.importances_mean
    order = np.argsort(imps)[::-1]
    k = min(top_k, len(feature_cols))
    out = []
    for rank, idx in enumerate(order[:k]):
        out.append((int(idx), float(imps[idx]), int(rank + 1)))
    info["kept"] = True
    return info, out


def _run_one_fold_serial(args_tuple, data):
    global _WORKER_DATA
    _WORKER_DATA = data
    try:
        return _run_one_fold(args_tuple)
    finally:
        _WORKER_DATA = None


def stability_selection(df, feature_cols, max_splits=5, seeds=None, n_workers=None, min_fold_auc=0.55):
    if seeds is None:
        seeds = [42, 7, 123]  # 3 seeds × max_splits folds

    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(int)
    groups = df["sample_name"].astype(str).values

    unique_groups = np.unique(groups)
    n_splits = min(max_splits, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("sample group 数量不足，无法 GroupKFold。")

    # 小数据自动放宽 AUC 门槛
    if X.shape[0] < 200:
        min_fold_auc = max(0.50, min_fold_auc - 0.10)

    # 构造任务: (seed, fold_id, n_folds, tr, va, top_k)
    n_folds_total = len(seeds) * n_splits
    tasks = []
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_splits)
        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            tasks.append((seed, fold_id, n_splits,
                          np.asarray(tr_idx, dtype=np.int64),
                          np.asarray(va_idx, dtype=np.int64),
                          min(15, len(feature_cols))))

    if n_workers is None:
        n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    n_workers = max(1, int(n_workers))

    data = {"X": X, "y": y, "feature_cols": feature_cols, "min_fold_auc": min_fold_auc}

    # 小数据集走单进程，跳过 pickle 序列化开销
    use_mp = n_workers > 1 and len(tasks) > 4
    print(f"\n  稳定性选择: {n_folds_total} folds (seeds={len(seeds)} x splits={n_splits}), "
          f"workers={'mp' if use_mp else 1}")

    fold_results = []
    if use_mp:
        data_pickle = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_stab_worker,
                                 initargs=(data_pickle,)) as ex:
            futures = {ex.submit(_run_one_fold, t): i for i, t in enumerate(tasks)}
            done_count = 0
            kept_count = 0
            total = len(futures)
            bar_width = 40
            for fut in as_completed(futures):
                info, fold_out = fut.result()
                done_count += 1
                if fold_out:
                    kept_count += 1
                pct = done_count / total
                filled = int(bar_width * pct)
                bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
                auc_str = f"auc={info['auc']:.3f}" if info['auc'] else ""
                print(f"\r  {bar} {done_count}/{total} folds  "
                      f"kept={kept_count}  {auc_str}  ", end="", flush=True)
                fold_results.append((info, fold_out))
            print()  # newline after progress bar
    else:
        for i, t in enumerate(tasks):
            info, fold_out = _run_one_fold_serial(t, data)
            pct = (i + 1) / len(tasks)
            bar_width = 40
            filled = int(bar_width * pct)
            bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
            auc_str = f"auc={info['auc']:.3f}" if info['auc'] else ""
            fold_out_list = fold_out or []
            print(f"\r  {bar} {i+1}/{len(tasks)} folds  "
                  f"kept={sum(1 for _, o in fold_results if o)}  {auc_str}  ",
                  end="", flush=True)
            fold_results.append((info, fold_out))
        print()

    # 聚合结果
    selected_count = defaultdict(int)
    importance_sum = defaultdict(float)
    rank_sum = defaultdict(float)
    total_runs = 0
    for _info, fold_out in fold_results:
        if not fold_out:
            continue
        total_runs += 1
        for idx, imp, rank in fold_out:
            f = feature_cols[idx]
            selected_count[f] += 1
            importance_sum[f] += imp
            rank_sum[f] += rank

    print(f"  有效 folds: {total_runs}/{n_folds_total}")

    summary = []
    for f in feature_cols:
        count = selected_count[f]
        freq = count / max(total_runs, 1)
        avg_imp = importance_sum[f] / max(count, 1)
        avg_rank = rank_sum[f] / max(count, 1)
        summary.append({
            "feature": f, "group": feature_to_group(f),
            "freq": float(freq), "avg_importance": float(avg_imp),
            "avg_rank": float(avg_rank), "count": int(count),
        })

    summary = sorted(summary, key=lambda x: (x["freq"], x["avg_importance"], -x["avg_rank"]), reverse=True)
    return summary


# =========================================================
# SHAP
# =========================================================

def shap_importance(df, feature_cols, selected_features=None):
    if not SHAP_AVAILABLE:
        return {}
    if selected_features is None:
        selected_features = feature_cols

    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(int)
    if len(np.unique(y)) < 2:
        return {}

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X, y)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return {f: float(mean_abs_shap[i]) for i, f in enumerate(feature_cols)}
    except Exception as e:
        print(f"SHAP计算失败: {e}")
        return {}


def shap_consistency_check(df_train, df_valid, feature_cols):
    if not SHAP_AVAILABLE:
        return {"available": False}

    X_tr = df_train[feature_cols].values.astype(float)
    y_tr = df_train[["target"]].values.astype(int).ravel()
    X_va = df_valid[feature_cols].values.astype(float)
    if len(np.unique(y_tr)) < 2 or len(X_va) == 0:
        return {"available": False, "reason": "insufficient_data"}

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    try:
        sv_tr = explainer.shap_values(X_tr)
        if isinstance(sv_tr, list):
            sv_tr = sv_tr[1] if len(sv_tr) > 1 else sv_tr[0]
        s_tr = np.abs(sv_tr).mean(axis=0)

        sv_va = explainer.shap_values(X_va)
        if isinstance(sv_va, list):
            sv_va = sv_va[1] if len(sv_va) > 1 else sv_va[0]
        s_va = np.abs(sv_va).mean(axis=0)
    except Exception as e:
        return {"available": False, "reason": f"shap_fail: {e}"}

    from scipy.stats import spearmanr
    rho, _ = spearmanr(s_tr, s_va)

    k = min(10, len(feature_cols))
    top_tr = set(np.argsort(s_tr)[-k:])
    top_va = set(np.argsort(s_va)[-k:])
    overlap = len(top_tr & top_va) / float(k)

    shap_train = {f: float(s_tr[i]) for i, f in enumerate(feature_cols)}
    shap_valid = {f: float(s_va[i]) for i, f in enumerate(feature_cols)}

    suspicious = []
    rank_tr = {f: int(r) for r, f in enumerate(sorted(feature_cols, key=lambda f: shap_train[f], reverse=True))}
    rank_va = {f: int(r) for r, f in enumerate(sorted(feature_cols, key=lambda f: shap_valid[f], reverse=True))}
    for f in feature_cols:
        if rank_tr[f] < k and rank_va[f] > 2 * k:
            suspicious.append({
                "feature": f,
                "rank_train": rank_tr[f], "rank_valid": rank_va[f],
                "shap_train": shap_train[f], "shap_valid": shap_valid[f],
            })

    return {
        "available": True,
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "topk_overlap": float(overlap), "k": int(k),
        "shap_train": shap_train, "shap_valid": shap_valid,
        "suspicious_train_only": suspicious,
    }


# 标量绝对量纲特征
SCALE_DEPENDENT_FEATURES = {
    "PPG_mean", "PPG_std", "PPG_p95",
    "PPG_DC_MEDIAN", "PPG_DC_IQR", "PPG_AC_RMS", "PPG_AC_MAD",
    "EMG0_MAV", "EMG0_RMS", "EMG0_VAR", "EMG0_WL", "EMG0_IEMG", "EMG0_SNR", "EMG0_P2P",
    "EMG1_MAV", "EMG1_RMS", "EMG1_VAR", "EMG1_WL", "EMG1_IEMG", "EMG1_SNR", "EMG1_P2P",
    "ACC_GRAV_MAG_MEAN", "ACC_AXIS_STD_SUM",
    # 3ch PPG spatial — imbalance/vmag 等归一化为 ratio，scale-invariant
    # dc_cv 和 dc_max_min_ratio 也是 ratio，scale-invariant
    # 留空
}


def annotate_scale_dependency(selected_features):
    scale_dep = [f for f in selected_features if f in SCALE_DEPENDENT_FEATURES]
    scale_inv = [f for f in selected_features if f not in SCALE_DEPENDENT_FEATURES]
    return scale_dep, scale_inv


# =========================================================
# 综合评分
# =========================================================

def cross_validate_importance(df, feature_cols, selected_features=None, n_workers=None, max_splits=4):
    perm_summary = stability_selection(df, feature_cols, max_splits=max_splits, n_workers=n_workers, min_fold_auc=0.55)
    shap_imp = shap_importance(df, feature_cols, selected_features)

    combined = []
    for item in perm_summary:
        f = item["feature"]
        shap_val = shap_imp.get(f, 0.0)
        combined.append({
            "feature": f, "group": item["group"],
            "perm_freq": item["freq"], "perm_imp": item["avg_importance"],
            "perm_rank": item["avg_rank"], "shap_imp": shap_val,
            "count": item["count"],
        })

    perm_scores_raw = np.array([it["perm_freq"] * it["perm_imp"] for it in combined])
    shap_scores_raw = np.array([shap_imp.get(it["feature"], 0.0) for it in combined])

    def _norm(arr):
        mx = arr.max() if len(arr) > 0 else 1.0
        return arr / mx if mx > 1e-12 else arr

    perm_scores_n = _norm(perm_scores_raw)
    shap_scores_n = _norm(shap_scores_raw)

    # 自适应权重：permutation 与 SHAP 的 Spearman 相关性越高，越信任 SHAP
    has_shap_mask = np.array([f in shap_imp and shap_imp[f] > 0 for f in [it["feature"] for it in combined]])
    if has_shap_mask.sum() >= 5:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(perm_scores_raw[has_shap_mask], shap_scores_raw[has_shap_mask])
        rho = abs(rho) if np.isfinite(rho) else 0.0
        # rho∈[0,1] → shap_weight∈[0.3, 0.7], perm_weight = 1 - shap_weight
        w_shap = 0.3 + 0.4 * rho
        w_perm = 1.0 - w_shap
    else:
        w_shap, w_perm = 0.5, 0.5

    for i, item in enumerate(combined):
        f = item["feature"]
        has_shap = has_shap_mask[i]
        if has_shap:
            item["combined_score"] = float(w_perm * perm_scores_n[i] + w_shap * shap_scores_n[i])
        else:
            item["combined_score"] = float(perm_scores_n[i])
        item["_w_perm"] = float(w_perm)
        item["_w_shap"] = float(w_shap if has_shap else 0.0)

    combined = sorted(combined, key=lambda x: x["combined_score"], reverse=True)
    return combined, perm_summary, shap_imp


# =========================================================
# 按组选择
# =========================================================

def _supplement_group(selected, group_count, summary, group_name, min_count, max_features):
    """补充某组特征到至少 min_count 个。

    先看 selected 是否还有未满 max_features 的空位：有就直接 append（不挤）。
    位置已满时，按 selected 末尾（最低重要度）挤掉非 anti_spoof / 非 acc_features 的特征。
    """
    if min_count <= 0:
        return selected, group_count

    have = sum(1 for f in selected if f in FEATURE_GROUPS.get(group_name, []))
    if have >= min_count:
        return selected, group_count

    candidates = [item["feature"] for item in summary
                  if item["group"] == group_name and item["feature"] not in selected]

    for f in candidates:
        if have >= min_count:
            break
        if len(selected) < max_features:
            selected.append(f)
            group_count[group_name] = group_count.get(group_name, 0) + 1
            have += 1
            continue

        evict_idx = -1
        for i in range(len(selected) - 1, -1, -1):
            cur_group = feature_to_group(selected[i])
            if cur_group in (group_name, "acc_features"):
                continue
            evict_idx = i
            break
        if evict_idx < 0:
            break

        evicted_group = feature_to_group(selected[evict_idx])
        selected.pop(evict_idx)
        group_count[evicted_group] = max(0, group_count.get(evicted_group, 0) - 1)
        selected.append(f)
        group_count[group_name] = group_count.get(group_name, 0) + 1
        have += 1

    return selected, group_count


def filter_summary_by_group_limits(summary, group_limits=None):
    """Return ranked feature summary entries whose groups are enabled."""
    if group_limits is None:
        group_limits = GROUP_LIMITS_DEFAULT
    return [
        item for item in summary
        if int(group_limits.get(item.get("group", "other"), 1)) > 0
    ]


def _select_by_group_impl(summary, max_features=15, group_limits=None,
                           min_acc_features=1,
                           min_anti_spoof_features=MIN_ANTI_SPOOF_FEATURES_DEFAULT):
    if group_limits is None:
        group_limits = GROUP_LIMITS_DEFAULT

    selected = []
    group_count = defaultdict(int)

    for item in summary:
        f = item["feature"]
        g = item["group"]
        limit = group_limits.get(g, 1)
        if limit <= 0:
            continue
        if group_count[g] < limit:
            selected.append(f)
            group_count[g] += 1
        if len(selected) >= max_features:
            break

    # ACC 兜底（保持原有逻辑）
    acc_selected = [f for f in selected if f in FEATURE_GROUPS.get("acc_features", [])]
    if (group_limits.get("acc_features", 1) > 0
            and len(acc_selected) < min_acc_features and min_acc_features > 0):
        acc_candidates = [item["feature"] for item in summary
                          if item["group"] == "acc_features"]
        for f in acc_candidates:
            if f not in selected:
                selected.append(f)
                group_count["acc_features"] = group_count.get("acc_features", 0) + 1
                break

    # 防伪兜底：训练数据无伪造标签时这些特征 importance 通常不高，必须强制保留 ≥N 个
    if group_limits.get("anti_spoof", 1) > 0:
        selected, group_count = _supplement_group(
            selected, dict(group_count), summary, "anti_spoof",
            min_anti_spoof_features, max_features)

    return selected, dict(group_count)


def select_by_group(summary, max_features=15, group_limits=None,
                     min_acc_features=1,
                     min_anti_spoof_features=MIN_ANTI_SPOOF_FEATURES_DEFAULT):
    return _select_by_group_impl(summary, max_features, group_limits,
                                  min_acc_features, min_anti_spoof_features)


def select_by_group_from_combined(summary, max_features=15, group_limits=None,
                                    min_acc_features=1,
                                    min_anti_spoof_features=MIN_ANTI_SPOOF_FEATURES_DEFAULT):
    return _select_by_group_impl(summary, max_features, group_limits,
                                  min_acc_features, min_anti_spoof_features)


def summarize_valid_selected(df_valid, selected_features):
    out = {}
    for f in selected_features:
        if f not in df_valid.columns:
            out[f] = {"exists": False}
            continue
        x = df_valid[f].replace([np.inf, -np.inf], np.nan)
        out[f] = {
            "exists": True,
            "missing_rate": float(x.isna().mean()),
            "mean": float(x.mean()) if x.notna().any() else None,
            "std": float(x.std()) if x.notna().any() else None,
        }
    return out


# =========================================================
# main
# =========================================================

def main(args=None):
    t_all = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--max_features", type=int, default=15)
    parser.add_argument("--missing_thresh", type=float, default=0.3)
    parser.add_argument("--var_thresh", type=float, default=1e-8)
    parser.add_argument("--corr_thresh", type=float, default=0.90)
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()

    t_load = time.time()
    df_train = pd.read_csv(os.path.join(args.artifact_dir, "feature_pool_train.csv"))
    df_valid = pd.read_csv(os.path.join(args.artifact_dir, "feature_pool_valid.csv"))
    _log_timing(
        f"load feature pools train={df_train.shape}, valid={df_valid.shape}",
        t_load,
    )

    if len(df_train) == 0:
        raise RuntimeError(
            "feature_pool_train.csv 为空。请检查 s03 输出日志：\n"
            "  1. Stage1 阈值是否过严（查看打印的 dc/acdc 阈值）\n"
            "  2. H5 文件字段名是否为 'ppg', 'emg', 'acc'\n"
            "  3. 样本时长是否 >= 3s"
        )

    feature_cols = get_feature_cols(df_train)
    print(f"原始特征列数: {len(feature_cols)}")

    t_stage = time.time()
    df_train_clean, df_valid_clean, kept_features, removed, fill_values = clean_features_by_train(
        df_train, df_valid, feature_cols,
        missing_thresh=args.missing_thresh,
        var_thresh=args.var_thresh,
        corr_thresh=args.corr_thresh,
    )
    _log_timing("stage clean_features_by_train", t_stage)

    print(f"原始特征数: {len(feature_cols)}")
    print(f"train 清洗后特征数: {len(kept_features)}")
    if removed["missing"]:
        print(f"  缺失过多: {removed['missing']}")
    if removed["low_variance"]:
        print(f"  低方差: {removed['low_variance'][:10]}...")
    if removed["high_corr"]:
        print(f"  高相关剔除: {removed['high_corr'][:10]}...")
    if removed["high_vif"]:
        print(f"  高VIF剔除: {removed['high_vif'][:10]}...")

    print("\n" + "=" * 50)
    print("阶段1: 按组快速预筛 (每组Top4, XGBoost Gain重要性)")
    print("=" * 50)
    t_stage = time.time()
    preselected = fast_group_preselection(df_train_clean, kept_features, preselect_top=4)
    preselected_features = list(preselected.keys())
    _log_timing("stage fast_group_preselection", t_stage)
    print(f"预选后特征数: {len(preselected_features)}")
    for f, info in sorted(preselected.items(), key=lambda x: x[1]['importance'], reverse=True)[:20]:
        print(f"  {f}: {info['method']}, importance={info['importance']:.2f}, group={info['group']}")

    print("\n" + "=" * 50)
    print(f"阶段2: 稳定性特征选择 (5折, Permutation + SHAP, n_workers={args.n_workers})")
    print("=" * 50)
    t_stage = time.time()
    combined_summary, perm_summary, shap_imp = cross_validate_importance(
        df_train_clean, preselected_features, n_workers=args.n_workers
    )
    _log_timing("stage cross_validate_importance", t_stage)

    print("\nPermutation 稳定性排序 Top20:")
    for i, item in enumerate(perm_summary[:20]):
        print(f"{i+1:02d}. {item['feature']} | group={item['group']} | "
              f"freq={item['freq']:.3f} | imp={item['avg_importance']:.6f} | "
              f"rank={item['avg_rank']:.2f}")

    if SHAP_AVAILABLE and shap_imp:
        print("\nSHAP Top10:")
        for i, (f, v) in enumerate(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1}. {f}: {v:.6f}")

    print("\n综合排序 Top20 (Permutation + SHAP):")
    for i, item in enumerate(combined_summary[:20]):
        shap_str = f", shap={item['shap_imp']:.6f}" if item['shap_imp'] > 0 else ""
        print(f"{i+1:02d}. {item['feature']} | group={item['group']} | "
              f"freq={item['perm_freq']:.3f}, combined={item['combined_score']:.6f}{shap_str}")

    t_stage = time.time()
    selected, group_count = select_by_group_from_combined(
        combined_summary, max_features=args.max_features,
        group_limits=GROUP_LIMITS_DEFAULT,
    )
    _log_timing("stage select_by_group", t_stage)

    print(f"\n最终选择特征 ({len(selected)}):")
    for i, f in enumerate(selected):
        shap_v = shap_imp.get(f, 0.0) if shap_imp else 0.0
        shap_str = f", shap={shap_v:.4f}" if shap_v > 0 else ""
        print(f"{i+1}. {f} | group={feature_to_group(f)}{shap_str}")

    t_stage = time.time()
    valid_summary = summarize_valid_selected(df_valid_clean, selected)
    _log_timing("stage summarize_valid_selected", t_stage)

    print("\n【SHAP train vs valid 一致性检查】")
    t_stage = time.time()
    shap_check = shap_consistency_check(df_train_clean, df_valid_clean, preselected_features)
    _log_timing("stage shap_consistency_check", t_stage)
    if shap_check.get("available"):
        print(f"  Spearman 相关={shap_check['spearman_rho']}, "
              f"Top{shap_check['k']} 重合度={shap_check['topk_overlap']:.2f}")
        if shap_check["suspicious_train_only"]:
            print(f"  [WARN] train 排名高但 valid 不高的可疑特征 ({len(shap_check['suspicious_train_only'])}):")
            for s in shap_check["suspicious_train_only"][:5]:
                print(f"    {s['feature']}: train rank={s['rank_train']}, valid rank={s['rank_valid']}")
    else:
        print(f"  跳过：{shap_check.get('reason', 'shap unavailable')}")

    scale_dep_in_sel, scale_inv_in_sel = annotate_scale_dependency(selected)
    if scale_dep_in_sel:
        print(f"\n[WARN] 选中特征中含 {len(scale_dep_in_sel)} 个绝对量纲特征:")
        for f in scale_dep_in_sel:
            print(f"    {f}")

    result = {
        "selected_features": selected,
        "max_features": args.max_features,
        "selection_policy": {
            "selection_data": "train_only",
            "valid_used_for_selection": False,
            "test_used_for_selection": False,
            "group_kfold_group": "sample_name",
            "use_shap": SHAP_AVAILABLE,
            "use_permutation": True,
        },
        "permutation_summary": perm_summary,
        "shap_importance": shap_imp if SHAP_AVAILABLE else {},
        "shap_consistency": shap_check,
        "combined_summary": combined_summary,
        "removed_features": removed,
        "group_count": group_count,
        "group_limits": GROUP_LIMITS_DEFAULT,
        "train_fill_values": fill_values,
        "valid_selected_feature_summary": valid_summary,
        "scale_dependency": {
            "scale_dependent_selected": scale_dep_in_sel,
            "scale_invariant_selected": scale_inv_in_sel,
            "ratio_invariant": (len(scale_inv_in_sel) / max(len(selected), 1)),
            "note": "scale_dependent 特征依赖原始 ADC 数值，建议线上做基线自适应或换成 ratio。",
        },
    }

    out_path = os.path.join(args.artifact_dir, "selected_features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 输出完整排序列表（供 s05 搜参时测试不同 max_features）
    ranked = sorted(
        filter_summary_by_group_limits(combined_summary, GROUP_LIMITS_DEFAULT),
        key=lambda x: x["combined_score"], reverse=True,
    )
    ranked_path = os.path.join(args.artifact_dir, "ranked_features.json")
    with open(ranked_path, "w", encoding="utf-8") as f:
        json.dump([{
            "feature": r["feature"],
            "group": r["group"],
            "combined_score": r["combined_score"],
            "perm_freq": r["perm_freq"],
            "perm_imp": r["perm_imp"],
            "perm_rank": r["perm_rank"],
            "shap_imp": r.get("shap_imp", 0.0),
        } for r in ranked], f, indent=2, ensure_ascii=False)

    _log_timing("s04 total", t_all)
    print(f"\n特征选择结果已保存: {out_path}")
    print(f"特征排序列表已保存: {ranked_path}")


if __name__ == "__main__":
    main()
