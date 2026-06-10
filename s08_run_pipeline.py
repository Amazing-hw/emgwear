# s08_run_pipeline.py
# -*- coding: utf-8 -*-
"""
主控脚本：一键运行全流程 s01→s06（适配 PPG+EMG+ACC 版本）

用法:
    # 全量运行
    python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts

流程:
    s01: 数据扫描 & train/valid/test 切分
    s02: Stage1 PPG DC/ACDC 阈值搜索 (6-ch PPG 取平均, 100Hz, 3s 窗)
    s03: 滑窗特征池提取 (单通道 PPG + EMG + ACC, 3s 窗)
    s04: 稳定性特征筛选
    s05: XGBoost 最终模型训练
    s06_opt:  状态机参数网格搜索
    s06_eval: 端到端评估
    s06_xpt: 导出部署产物
    s06_feat: 导出独立特征提取脚本
    s06_plot: 画错误样本图
    s06_cb:   导出部署配方

注: s07_postprocess_optimize.py 为独立脚本，基于 s06 生成的 NPZ 缓存做后处理参数搜索。
"""

import argparse
import os
import json
import sys
import time
import subprocess
import joblib
from datetime import timedelta

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def _arg(args, name, default):
    return getattr(args, name, default)


def _script_path(name):
    return os.path.join(SCRIPTS_DIR, f"{name}.py")


def _run(name, cmd):
    """执行一个子步骤。返回 True/False。"""
    print(f"\n{'─' * 70}")
    print(f"[RUN] {name}")
    print(f"  {cmd}")
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True)
    dt = time.time() - t0
    if rc == 0:
        print(f"[OK] {name}  [{timedelta(seconds=int(dt))}]")
        return True
    else:
        print(f"[FAIL] {name}  FAILED (exit={rc})")
        return False


def build_pipeline_commands(args):
    s05_extra = f' --max_features {args.max_features}'
    if getattr(args, "model_search", True):
        s05_extra += (
            f' --model_search'
            f' --max_model_nodes {_arg(args, "max_model_nodes", 500)}'
            f' --model_search_strategy {_arg(args, "model_search_strategy", "staged_group_cv")}'
            f' --model_search_max_candidates {_arg(args, "model_search_max_candidates", 600)}'
            f' --model_search_stage2_top_k {_arg(args, "model_search_stage2_top_k", 80)}'
            f' --model_search_cv_folds {_arg(args, "model_search_cv_folds", 3)}'
            f' --model_search_cv_repeats {_arg(args, "model_search_cv_repeats", 2)}'
            f' --model_search_random_state {_arg(args, "model_search_random_state", 42)}'
            f' --model_search_accuracy_tolerance {_arg(args, "model_search_accuracy_tolerance", 0.0)}'
            f' --model_search_stage1_top_k {_arg(args, "model_search_stage1_top_k", 4)}'
            f' --model_search_n_estimators "{_arg(args, "model_search_n_estimators", "20,25,30,35,40,45,50,55,60")}"'
            f' --model_search_max_depth "{_arg(args, "model_search_max_depth", "2,3,4")}"'
            f' --model_search_learning_rate "{_arg(args, "model_search_learning_rate", "0.025,0.03,0.04,0.05,0.06,0.08,0.10")}"'
            f' --model_search_min_child_weight "{_arg(args, "model_search_min_child_weight", "10,15,20,25,30,40,50")}"'
            f' --model_search_reg_lambda "{_arg(args, "model_search_reg_lambda", "5,8,10,12,16,20,30")}"'
            f' --model_search_reg_alpha "{_arg(args, "model_search_reg_alpha", "0,0.5,1,1.5,2,3")}"'
            f' --model_search_subsample "{_arg(args, "model_search_subsample", "0.70,0.75,0.80,0.85,0.90")}"'
            f' --model_search_colsample_bytree "{_arg(args, "model_search_colsample_bytree", "0.70,0.75,0.80,0.85,0.90")}"'
        )
    return {
        's01': f'"{PYTHON}" "{_script_path("s01_data_split")}" --dataset_dir "{args.dataset_dir}" --artifact_dir "{args.artifact_dir}" --n_workers {args.n_workers}',
        's02': f'"{PYTHON}" "{_script_path("s02_ir_dc_threshold")}" --artifact_dir "{args.artifact_dir}" --n_workers {args.n_workers}',
        's03': f'"{PYTHON}" "{_script_path("s03_extract_feature_pool")}" --artifact_dir "{args.artifact_dir}" --window_sec {args.window_sec} --stride_sec {args.stride_sec} --n_workers {args.n_workers}',
        's04': f'"{PYTHON}" "{_script_path("s04_feature_selection")}" --artifact_dir "{args.artifact_dir}" --max_features {args.max_features} --n_workers {args.n_workers}',
        's05': f'"{PYTHON}" "{_script_path("s05_train_final_model")}" --artifact_dir "{args.artifact_dir}"{s05_extra}',
        's06_opt': f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" --artifact_dir "{args.artifact_dir}" --split valid --n_workers {args.n_workers} --optimize --window_sec {args.window_sec} --stride_sec {args.stride_sec}',
        's06_cache_valid': f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" --artifact_dir "{args.artifact_dir}" --split valid --n_workers {args.n_workers} --window_sec {args.window_sec} --stride_sec {args.stride_sec} --export_window_cache --window_output_root window_outputs',
        's07_post': f'"{PYTHON}" "{_script_path("s07_postprocess_optimize")}" --artifact_dir "{args.artifact_dir}" --split valid --cache_root window_outputs --fp_cost {_arg(args, "postprocess_fp_cost", 4.0)}',
        's06_eval': f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" --artifact_dir "{args.artifact_dir}" --split {_arg(args, "split", "test")} --n_workers {args.n_workers} --window_sec {args.window_sec} --stride_sec {args.stride_sec}',
        's06_xpt': f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" --artifact_dir "{args.artifact_dir}" --split {_arg(args, "split", "test")} --n_workers {args.n_workers} --window_sec {args.window_sec} --stride_sec {args.stride_sec} --export_deploy',
        's06_feat': '__extractor__',
        's06_plot': '__plot__',
        's06_cb': '__cookbook__',
    }


def _step_list():
    """Return all known pipeline steps (key, display_name, default_enabled)."""
    return [
        ("s01",   "数据扫描 & 切分",             True),
        ("s02",   "Stage1 阈值筛选",              True),
        ("s03",   "特征池提取",                   True),
        ("s04",   "稳定性特征筛选",               True),
        ("s05",   "XGBoost模型训练",              True),
        ("s06_opt","状态机参数优化",              False),
        ("s06_cache_valid", "导出valid NPZ缓存",  False),
        ("s07_post", "FP敏感后处理搜参",          False),
        ("s06_eval","端到端评估(test)",           True),
        ("s06_xpt","导出部署产物",                True),
        ("s06_feat","导出特征提取脚本",           True),
        ("s06_plot","画错误样本图",               True),
        ("s06_cb", "导出部署配方",                True),
    ]

# backward compat: old test code references default_pipeline_steps
default_pipeline_steps = _step_list


def _load_eval_details(artifact_dir, split="test", method="state_machine"):
    eval_path = os.path.join(artifact_dir, f"end_to_end_eval_{split}_{method}.json")
    if not os.path.exists(eval_path):
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f).get("details", [])


# =========================================================
# 独立特征提取脚本生成
# =========================================================

def export_feature_extractor_script(artifact_dir):
    """从 bundle 读取入选特征，生成独立可运行的特征提取 Python 脚本。"""
    import numpy as np

    bp = os.path.join(artifact_dir, "model_bundle.pkl")
    if not os.path.exists(bp):
        print("[WARN] model_bundle.pkl not found, skip feature extractor script")
        return

    b = joblib.load(bp)
    sel = b["feature_names"]
    fv = b["fill_values"]
    cb = b.get("clip_bounds", {})
    FILL_JSON = json.dumps({k: float(v) if v is not None else 0.0 for k, v in fv.items() if k in sel})
    CLIP_JSON = json.dumps({k: [float(lo), float(hi)] for k, (lo, hi) in cb.items() if k in sel})
    ORDER_JSON = json.dumps(sel)

    # 每个入选特征的 Python 表达式（PPG + EMG + ACC 版本）
    FC = _build_feature_code_map()

    feat_lines = []
    for f in sel:
        code = FC.get(f, '0.0')
        feat_lines.append(f'    f["{f}"] = {code}')
    feat_block = '\n'.join(feat_lines)

    script = _build_extractor_script_template(len(sel), ORDER_JSON, FILL_JSON, CLIP_JSON, feat_block)

    out_path = os.path.join(artifact_dir, "deploy_feature_extractor.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"[OK] deploy_feature_extractor.py -> {out_path}")


def _build_feature_code_map():
    """构建特征名 → Python 表达式的映射。"""
    FC = {
        # PPG basic: match s03.extract_feature_pool_from_window, which uses
        # virtual channel A (avg raw ch0/ch1 after preprocessing), not the
        # three-channel mean.
        "PPG_mean": "float(np.mean(ir))",
        "PPG_std": "float(np.std(ir))",
        "PPG_p95": "float(np.percentile(ir, 95))",
        "PPG_diff_std": "float(np.std(np.diff(ir)))",
        "PPG_acdc": "_safe_div(np.sqrt(np.mean(ppg_bp**2)), abs(ppg_dc))",
        # PPG DC/AC
        "PPG_DC_MEDIAN": "float(ppg_dc)",
        "PPG_DC_IQR": "_robust_iqr(ppg_raw)",
        "PPG_AC_RMS": "float(np.sqrt(np.mean(ppg_bp**2)))",
        "PPG_AC_MAD": "_robust_mad(ppg_bp)",
        "PPG_AC_DC_RATIO": "_safe_div(np.sqrt(np.mean(ppg_bp**2)), abs(ppg_dc))",
        "PPG_DERIV_MAD": "_robust_mad(np.diff(ppg_bp))",
        # PPG FFT / autocorr
        "PPG_FFT_PEAK_MEDIAN_RATIO": "fft_p[0]",
        "PPG_DOM_FREQ": "fft_p[1]",
        "PPG_AUTO_CORR_PEAK": "ac_p[0]",
        "PPG_AUTO_CORR_LAG_SEC": "ac_p[1]",
        "PPG_FFT_peak_width_Hz": "float(fft_p[5][np.where(fft_p[4]>np.max(fft_p[4])*0.5)[0]][-1]-fft_p[5][np.where(fft_p[4]>np.max(fft_p[4])*0.5)[0]][0]) if fft_p[4] is not None and len(fft_p[4])>0 and np.any(fft_p[4]>np.max(fft_p[4])*0.5) else 0.0",
        "PPG_FFT_SNR": "float(np.sum(fft_p[4]**2)/(np.sum(fft_p[2]**2)-np.sum(fft_p[4]**2)+EPS)) if fft_p[4] is not None and fft_p[2] is not None else 0.0",
        # PPG waveform
        "PPG_bp_skewness": "float(np.mean((ppg_bp-np.mean(ppg_bp))**3)/(np.std(ppg_bp)**3+EPS))",
        "PPG_bp_kurtosis": "float(np.mean((ppg_bp-np.mean(ppg_bp))**4)/(np.std(ppg_bp)**4+EPS))",
        # PPG Hjorth
        "PPG_Hjorth_Activity": "float(np.var(ppg_bp))",
        "PPG_Hjorth_Mobility": "float(np.sqrt(np.var(np.diff(ppg_bp))/(np.var(ppg_bp)+EPS)))",
        # PPG Entropy
        "PPG_Entropy_Shannon": "float(-np.sum(h*np.log(h+EPS))) if len(h:=np.histogram(ppg_bp,bins=10,density=True)[0])>0 else 0.0",
        "PPG_Entropy_SampEn": "_sample_entropy(ppg_bp)",
        # PPG Derivative
        "PPG_Deriv_d1_mean": "float(np.mean(np.diff(ppg_bp)))",
        "PPG_Deriv_d1_std": "float(np.std(np.diff(ppg_bp)))",
        "PPG_Deriv_d1_max": "float(np.max(np.diff(ppg_bp)))",
        "PPG_Deriv_d1_min": "float(np.min(np.diff(ppg_bp)))",
        "PPG_Deriv_d1_zcr": "float(np.sum(np.abs(np.diff(np.sign(np.diff(ppg_bp)))))/(2.0*len(np.diff(ppg_bp))))",
        # PPG Temporal
        "PPG_Temporal_slope_mean": "float(slope)",
        "PPG_Temporal_slope_std": "slope_std",
        "PPG_Temporal_peak_prominence": "float(np.mean(pk[1]['prominences'])) if len(pk[0])>0 else 0.0",
        "PPG_Temporal_peak_ratio": "float(len(pk[0])/Np)",
        # EMG ch0
        "EMG0_MAV": "float(np.mean(emg0_env)) if emg0_env is not None else 0.0",
        "EMG0_RMS": "float(np.sqrt(np.mean(emg0_bp**2))) if emg0_bp is not None else 0.0",
        "EMG0_VAR": "float(np.var(emg0_bp)) if emg0_bp is not None else 0.0",
        "EMG0_WL": "float(np.sum(np.abs(np.diff(emg0_bp)))) if emg0_bp is not None else 0.0",
        "EMG0_ZC": "float(np.sum(np.abs(np.diff(np.sign(emg0_bp))))/(2.0*len(emg0_bp))) if emg0_bp is not None else 0.0",
        "EMG0_SSC": "float(np.sum((np.diff(emg0_bp)[:-1]*np.diff(emg0_bp)[1:])<0)/len(emg0_bp)) if emg0_bp is not None and len(emg0_bp)>=3 else 0.0",
        "EMG0_WAMP": "float(np.sum(np.abs(np.diff(emg0_bp))>0.05*max(np.max(np.abs(emg0_bp)),EPS))/len(emg0_bp)) if emg0_bp is not None else 0.0",
        "EMG0_IEMG": "float(np.sum(emg0_env)) if emg0_env is not None else 0.0",
        "EMG0_P2P": "float(np.percentile(emg0_env,95)-np.percentile(emg0_env,5)) if emg0_env is not None else 0.0",
        "EMG0_AMP_CV": "float(np.std(emg0_env)/(np.mean(emg0_env)+EPS)) if emg0_env is not None else 0.0",
        "EMG0_MNF": "emg0_freq[0]", "EMG0_MDF": "emg0_freq[1]",
        "EMG0_PKF": "emg0_freq[2]", "EMG0_PSR": "emg0_freq[3]",
        "EMG0_POW_20_60": "emg0_freq[4]", "EMG0_POW_60_150": "emg0_freq[5]",
        "EMG0_POW_150_450": "emg0_freq[6]", "EMG0_POW_LH_RATIO": "emg0_freq[7]",
        "EMG0_SE95": "emg0_freq[8]",
        "EMG0_SampEn": "_sample_entropy(emg0_bp) if emg0_bp is not None else 0.0",
        "EMG0_SKEWNESS": "float(np.mean((emg0_bp-np.mean(emg0_bp))**3)/(np.std(emg0_bp)**3+EPS)) if emg0_bp is not None else 0.0",
        "EMG0_KURTOSIS": "float(np.mean((emg0_bp-np.mean(emg0_bp))**4)/(np.std(emg0_bp)**4+EPS)) if emg0_bp is not None else 0.0",
        "EMG0_SNR": "_safe_div(float(np.sqrt(np.mean(emg0_bp**2))), float(np.mean(np.abs(emg0_env))+EPS)) if emg0_bp is not None else 0.0",
        # EMG ch1
        "EMG1_MAV": "float(np.mean(emg1_env)) if emg1_env is not None else 0.0",
        "EMG1_RMS": "float(np.sqrt(np.mean(emg1_bp**2))) if emg1_bp is not None else 0.0",
        "EMG1_VAR": "float(np.var(emg1_bp)) if emg1_bp is not None else 0.0",
        "EMG1_WL": "float(np.sum(np.abs(np.diff(emg1_bp)))) if emg1_bp is not None else 0.0",
        "EMG1_ZC": "float(np.sum(np.abs(np.diff(np.sign(emg1_bp))))/(2.0*len(emg1_bp))) if emg1_bp is not None else 0.0",
        "EMG1_SSC": "float(np.sum((np.diff(emg1_bp)[:-1]*np.diff(emg1_bp)[1:])<0)/len(emg1_bp)) if emg1_bp is not None and len(emg1_bp)>=3 else 0.0",
        "EMG1_WAMP": "float(np.sum(np.abs(np.diff(emg1_bp))>0.05*max(np.max(np.abs(emg1_bp)),EPS))/len(emg1_bp)) if emg1_bp is not None else 0.0",
        "EMG1_IEMG": "float(np.sum(emg1_env)) if emg1_env is not None else 0.0",
        "EMG1_P2P": "float(np.percentile(emg1_env,95)-np.percentile(emg1_env,5)) if emg1_env is not None else 0.0",
        "EMG1_AMP_CV": "float(np.std(emg1_env)/(np.mean(emg1_env)+EPS)) if emg1_env is not None else 0.0",
        "EMG1_MNF": "emg1_freq[0]", "EMG1_MDF": "emg1_freq[1]",
        "EMG1_PKF": "emg1_freq[2]", "EMG1_PSR": "emg1_freq[3]",
        "EMG1_POW_20_60": "emg1_freq[4]", "EMG1_POW_60_150": "emg1_freq[5]",
        "EMG1_POW_150_450": "emg1_freq[6]", "EMG1_POW_LH_RATIO": "emg1_freq[7]",
        "EMG1_SE95": "emg1_freq[8]",
        "EMG1_SampEn": "_sample_entropy(emg1_bp) if emg1_bp is not None else 0.0",
        "EMG1_SKEWNESS": "float(np.mean((emg1_bp-np.mean(emg1_bp))**3)/(np.std(emg1_bp)**3+EPS)) if emg1_bp is not None else 0.0",
        "EMG1_KURTOSIS": "float(np.mean((emg1_bp-np.mean(emg1_bp))**4)/(np.std(emg1_bp)**4+EPS)) if emg1_bp is not None else 0.0",
        "EMG1_SNR": "_safe_div(float(np.sqrt(np.mean(emg1_bp**2))), float(np.mean(np.abs(emg1_env))+EPS)) if emg1_bp is not None else 0.0",
        # EMG cross
        "EMG_CROSS_CORR": "_safe_corr(emg0_bp, emg1_bp) if emg0_bp is not None and emg1_bp is not None else 0.0",
        "EMG_RMS_RATIO": "_safe_div(float(np.sqrt(np.mean(emg0_bp**2))), float(np.sqrt(np.mean(emg1_bp**2)))) if emg0_bp is not None and emg1_bp is not None else 0.0",
        # ACC (gravity/motion separated)
        "ACC_GRAV_MAG_MEAN": "float(np.mean(grav_mag))",
        "ACC_GRAV_DOM_RATIO": "float(np.max(np.abs(gm))/(np.sum(np.abs(gm))+1e-8))",
        "ACC_MOTION_RMS": "float(np.sqrt(np.mean(motion_mag**2)))",
        "ACC_MOTION_STD": "float(np.std(motion_mag))",
        "ACC_MOTION_MAD": "_robust_mad(motion_mag)",
        "ACC_AXIS_STD_SUM": "float(np.sum(np.std(acc, axis=0))) if acc is not None and len(acc) >= 4 else 0.0",
        "ACC_DIFF_MAD": "_robust_mad(np.diff(motion_mag))",
        "ACC_STILL_SCORE": "float(1.0/(1.0+50.0*np.std(motion_mag)/(abs(np.mean(motion_mag))+1e-6)))",
        "ACC_MAG_P50": "float(np.percentile(acc_mag, 50)) if am else 0.0",
        "ACC_MAG_P90": "float(np.percentile(acc_mag, 90)) if am else 0.0",
        # Cross-modal
        "ACC_PPG_BP_CORR": "abs(_safe_corr(ambp, ppg_bp)) if ambp is not None else 0.0",
        "ACC_EMG_CORR": "abs(_safe_corr(acc_mag, emg0_env_ds)) if am and emg0_env_ds is not None else 0.0",
        "EMG_PPG_CORR": "abs(_safe_corr(emg0_env_ds, ppg_bp)) if emg0_env_ds is not None else 0.0",
        "EMG_PPG_ENV_CORR": "abs(_safe_corr(emg0_env_smooth_ds, ppg_env)) if emg0_env_smooth_ds is not None else 0.0",
        # Meta
        "SIG_LEN": "float(len(ppg))",
        "SIG_SEC": "float(len(ppg)/fs)",
        # EMG 50Hz mains (on bp_leak_ref, before any notch)
        "EMG0_PWR_50HZ": "float(np.log1p(_band_power(emg0_leak_ref, 48, 52, fs_emg))) if emg0_leak_ref is not None else 0.0",
        "EMG0_50HZ_RATIO": "(_band_power(emg0_leak_ref, 48, 52, fs_emg) / (_band_power(emg0_leak_ref, 2, 450, fs_emg) + EPS)) if emg0_leak_ref is not None else 0.0",
        "EMG0_50HZ_HARM_RATIO": "((_band_power(emg0_leak_ref, 48, 52, fs_emg)+_band_power(emg0_leak_ref, 148, 152, fs_emg)+_band_power(emg0_leak_ref, 248, 252, fs_emg)) / (_band_power(emg0_leak_ref, 2, 450, fs_emg) + EPS)) if emg0_leak_ref is not None else 0.0",
        "EMG1_PWR_50HZ": "float(np.log1p(_band_power(emg1_leak_ref, 48, 52, fs_emg))) if emg1_leak_ref is not None else 0.0",
        "EMG1_50HZ_RATIO": "(_band_power(emg1_leak_ref, 48, 52, fs_emg) / (_band_power(emg1_leak_ref, 2, 450, fs_emg) + EPS)) if emg1_leak_ref is not None else 0.0",
        "EMG1_50HZ_HARM_RATIO": "((_band_power(emg1_leak_ref, 48, 52, fs_emg)+_band_power(emg1_leak_ref, 148, 152, fs_emg)+_band_power(emg1_leak_ref, 248, 252, fs_emg)) / (_band_power(emg1_leak_ref, 2, 450, fs_emg) + EPS)) if emg1_leak_ref is not None else 0.0",
        # EMG baseline drift
        "EMG0_BASELINE_DRIFT_POW": "float(np.log1p(float(np.mean(_bandpass(emg0_demean, fs_emg, 1.0, 10.0, order=2)**2)))) if emg0_demean is not None else 0.0",
        "EMG0_DRIFT_HF_RATIO": "(float(np.mean(_bandpass(emg0_demean, fs_emg, 1.0, 10.0, order=2)**2)) / (float(np.mean(emg0_leak_ref**2)) + EPS)) if emg0_demean is not None and emg0_leak_ref is not None else 0.0",
        "EMG1_BASELINE_DRIFT_POW": "float(np.log1p(float(np.mean(_bandpass(emg1_demean, fs_emg, 1.0, 10.0, order=2)**2)))) if emg1_demean is not None else 0.0",
        "EMG1_DRIFT_HF_RATIO": "(float(np.mean(_bandpass(emg1_demean, fs_emg, 1.0, 10.0, order=2)**2)) / (float(np.mean(emg1_leak_ref**2)) + EPS)) if emg1_demean is not None and emg1_leak_ref is not None else 0.0",
        # EMG narrowband leakage features (on bp_leak_ref, before any notch)
        "EMG0_LEAK_100_RATIO": "emg0_leak[0] if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_150_RATIO": "emg0_leak[1] if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_200_RATIO": "emg0_leak[2] if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_250_RATIO": "emg0_leak[3] if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_300_RATIO": "emg0_leak[4] if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_SUM_RATIO": "float(np.sum(emg0_leak)) if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_MAX_RATIO": "float(np.max(emg0_leak)) if emg0_leak_ref is not None else 0.0",
        "EMG0_LEAK_MAX_FREQ": "float([100,150,200,250,300][int(np.argmax(emg0_leak))]) if emg0_leak_ref is not None else 0.0",
        "EMG1_LEAK_100_RATIO": "emg1_leak[0] if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_150_RATIO": "emg1_leak[1] if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_200_RATIO": "emg1_leak[2] if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_250_RATIO": "emg1_leak[3] if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_300_RATIO": "emg1_leak[4] if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_SUM_RATIO": "float(np.sum(emg1_leak)) if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_MAX_RATIO": "float(np.max(emg1_leak)) if emg1_leak_ref is not None else 0.0",
        "EMG1_LEAK_MAX_FREQ": "float([100,150,200,250,300][int(np.argmax(emg1_leak))]) if emg1_leak_ref is not None else 0.0",
        # ACC tremor
        "ACC_TREMOR_POW_8_12": "float(np.log1p(_band_power(acc_mag - np.mean(acc_mag), 8, 12, fs))) if am else 0.0",
        "ACC_TREMOR_RATIO": "(_band_power(acc_mag - np.mean(acc_mag), 8, 12, fs) / (_band_power(acc_mag - np.mean(acc_mag), 0.5, 15, fs) + EPS)) if am else 0.0",
        # PPG PI
        "PPG_PI": "_safe_div(float(np.sqrt(np.mean(ppg_bp**2))), abs(float(np.median(ppg_raw))))",
        "PPG_PI_SUBWIN_IQR": "ppg_pi_sub_iqr",
        # PPG morphology
        "PPG_DICROTIC_RATIO": "ppg_dicr",
        "PPG_AUG_INDEX_MEAN": "ppg_aug",
        "PPG_PULSE_WIDTH_CV": "ppg_pw_cv",
        # PPG HRV
        "PPG_RR_RMSSD": "ppg_rmssd",
        "PPG_RR_CV": "ppg_cv",
        "PPG_RR_PNN30": "ppg_pnn30",
        # ACC-PPG coherence
        "ACC_PPG_COH_MICRO": "acc_coh_micro",
        "ACC_PPG_COH_HR": "acc_coh_hr",
        # PPG 3ch spatial
        "PPG_ch_imbalance_mean": "float(np.mean(imb))",
        "PPG_ch_imbalance_p90": "float(np.percentile(imb, 90))",
        "PPG_ch_imbalance_iqr": "_robust_iqr(imb)",
        "PPG_ch_rangeNorm_mean": "float(np.mean(rn))",
        "PPG_ch_rangeNorm_p90": "float(np.percentile(rn, 90))",
        "PPG_ch_vmag_mean": "float(np.mean(vmag))",
        "PPG_ch_vmag_p90": "float(np.percentile(vmag, 90))",
        "PPG_ch_vmag_iqr": "_robust_iqr(vmag)",
        "PPG_ch_vmag_std": "float(np.std(vmag))",
        "PPG_ch_dc_cv": "float(np.std(dc3)/(abs(np.mean(dc3))+EPS))",
        "PPG_ch_dc_max_min_ratio": "float(np.max(np.abs(dc3))/(np.min(np.abs(dc3))+EPS))",
        "PPG_ch_bp_corr_mean": "float(np.mean(c3))",
        "PPG_ch_bp_corr_min": "float(np.min(c3))",
        "PPG_ch_bp_corr_std": "float(np.std(c3))",
        "PPG_ch_bp_lag_std": "float(np.std([l01,l12]))",
        "PPG_corr_mean_imbalance": "_safe_corr(ir, imb)",
        "PPG_corr_mean_vmag": "_safe_corr(ir, vmag)",
        "PPG_corr_IR_imbalance": "_safe_corr(ir, imb)",
    }
    return FC


def _build_extractor_script_template(n_features, ORDER_JSON, FILL_JSON, CLIP_JSON, feat_block):
    """构建 deploy_feature_extractor.py 的完整脚本模板。"""
    script = f'''"""
deploy_feature_extractor.py
Auto-generated standalone feature extraction script (PPG+EMG+ACC).
Extracts {n_features} features from a 3s@100Hz PPG window + EMG + ACC.
Input: ppg (Nx6 raw PPG @ 100Hz, Nx3 virtual PPG, or 1D PPG), emg (Nx2 @ 1000Hz or None), acc (Nx3 @ 100Hz or None)
Output: feature vector (list of {n_features} floats)
Dependencies: numpy, scipy
"""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, correlate, find_peaks

EPS = 1e-12
FEATURE_ORDER = {ORDER_JSON}
FILL_VALUES = {FILL_JSON}
CLIP_BOUNDS = {CLIP_JSON}


# ========== Utilities ==========

def _safe_div(a, b):
    return float(a) / (float(b) + EPS)

def _robust_mad(x):
    return float(np.median(np.abs(x - np.median(x))))

def _robust_iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def _safe_corr(x, y):
    n = min(len(x), len(y))
    if n < 8:
        return 0.0
    x, y = x[:n] - np.mean(x[:n]), y[:n] - np.mean(y[:n])
    sx, sy = np.std(x), np.std(y)
    if sx < EPS or sy < EPS:
        return 0.0
    v = np.mean((x / sx) * (y / sy))
    return float(v) if np.isfinite(v) else 0.0

def _remove_burr(x, k=6.0):
    if len(x) < 3:
        return x
    d = np.diff(x)
    thr = max(k * _robust_mad(d), EPS)
    left, mid, right = x[:-2], x[1:-1], x[2:]
    bad = (np.abs(mid - left) > thr) & (np.abs(mid - right) > thr)
    x[1:-1] = np.where(bad, 0.5 * (left + right), mid)
    return x

def _remove_step(x, k=10.0):
    if len(x) < 2:
        return x
    d = np.diff(x)
    thr = max(k * _robust_mad(d), EPS)
    for i in range(1, len(x)):
        if abs(x[i] - x[i-1]) > thr:
            x[i] = x[i-1]
    return x

_BUTTER_CACHE = {{}}

def _bandpass(x, fs=100, lowcut=0.4, highcut=6.0, order=4):
    if len(x) < 16:
        return x.copy()
    key = (float(fs), float(lowcut), float(highcut), int(order))
    if key not in _BUTTER_CACHE:
        nyq = 0.5 * fs
        b, a = butter(order, [max(lowcut/nyq, 1e-6), min(highcut/nyq, 0.999)], btype="band")
        _BUTTER_CACHE[key] = (b, a)
    b, a = _BUTTER_CACHE[key]
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x - np.median(x)


# ========== FFT / Autocorr / Entropy ==========

def _fft_features(bp, fs=100, fmin=0.5, fmax=5.0):
    """Returns (peak_ratio, dom_freq, spec, freqs, band_spec, band_freqs)."""
    if len(bp) < 16:
        return 0.0, 0.0, None, None, None, None
    x = bp - np.mean(bp)
    xw = x * np.hamming(len(x))
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    nfft = max(256, nfft)
    spec = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0, spec, freqs, None, None
    bs, bf = spec[mask], freqs[mask]
    med = np.median(bs)
    r = float(np.max(bs) / (med + EPS)) if med > EPS else 0.0
    d = float(bf[np.argmax(bs)])
    return r, d, spec, freqs, bs, bf

def _autocorr_features(bp, fs=100, bpm_min=40, bpm_max=180):
    """Returns (ac_peak, ac_lag_sec)."""
    if len(bp) < int(fs * 1.5):
        return 0.0, 0.0
    x = bp - np.mean(bp)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(x)-1:]
    if ac[0] < EPS:
        return 0.0, 0.0
    ac = ac / ac[0]
    lag_min = max(1, int(fs * 60.0 / bpm_max))
    lag_max = min(len(ac) - 1, int(fs * 60.0 / bpm_min))
    if lag_max <= lag_min:
        return 0.0, 0.0
    seg = ac[lag_min:lag_max+1]
    idx = int(np.argmax(seg))
    return float(seg[idx]), float((lag_min + idx) / fs)

def _sample_entropy(bp, m=2, r_ratio=0.2):
    N = len(bp)
    if N < 50:
        return 0.0
    r = r_ratio * np.std(bp)
    if r < EPS:
        return 0.0
    def _count(mv, chunk=500):
        pat = np.array([bp[i:i+mv] for i in range(N - mv)], dtype=np.float64)
        n_pat = len(pat)
        if n_pat < 2:
            return 0.0
        total = 0.0
        for i in range(0, n_pat, chunk):
            end = min(i + chunk, n_pat)
            cp = pat[i:end]
            d = np.max(np.abs(cp[:, None, :] - pat[None, :, :]), axis=2)
            total += np.sum(d <= r)
            for j in range(len(cp)):
                idx = i + j
                if idx < n_pat and d[j, idx] <= r:
                    total -= 1.0
        return total
    Bm = _count(m)
    Bm1 = _count(m + 1)
    if Bm1 <= EPS or Bm <= EPS:
        return 0.0
    return float(-np.log(Bm1 / Bm))

def _smooth_envelope(x, fs=25, win_sec=0.25):
    x = np.abs(x)
    w = max(3, int(round(win_sec * fs)))
    if w % 2 == 0:
        w += 1
    return np.convolve(x, np.ones(w) / w, mode="same")

def _emg_frequency_features(bp, fs=1000):
    """Returns (MNF, MDF, PKF, PSR, POW_20_60, POW_60_150, POW_150_450, POW_LH_RATIO, SE95).
    使用 Welch 方法 (nperseg=512)，与 s03 训练代码保持一致。
    """
    if bp is None or len(bp) < 16:
        return (0.0,) * 9
    x = np.asarray(bp, dtype=np.float64)
    try:
        from scipy.signal import welch
        nperseg = min(512, len(x) // 2)
        if nperseg < 16:
            return (0.0,) * 9
        noverlap = nperseg // 2
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        return (0.0,) * 9
    mask = (f >= 20) & (f <= 450)
    if not np.any(mask) or np.sum(Pxx[mask]) < EPS:
        return (0.0,) * 9
    bf, bs = f[mask], Pxx[mask]
    total_p = np.sum(bs) + EPS
    mnf = float(np.sum(bf * bs) / total_p)
    cumsum = np.cumsum(bs)
    mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    mdf = float(bf[min(mdf_idx, len(bf) - 1)])
    pkf = float(bf[np.argmax(bs)])
    low_100 = (bf >= 20) & (bf <= 100)
    high_100 = (bf > 100) & (bf <= 450)
    psr = float(np.sum(bs[low_100]) / (np.sum(bs[high_100]) + EPS))
    pow_20_60 = float(np.sum(bs[(bf >= 20) & (bf <= 60)]) / total_p)
    pow_60_150 = float(np.sum(bs[(bf > 60) & (bf <= 150)]) / total_p)
    pow_150_450 = float(np.sum(bs[(bf > 150) & (bf <= 450)]) / total_p)
    pow_lh = float(np.sum(bs[(bf >= 20) & (bf <= 60)]) / (np.sum(bs[(bf > 150) & (bf <= 450)]) + EPS))
    se95_idx = np.searchsorted(cumsum, cumsum[-1] * 0.95)
    se95 = float(bf[min(se95_idx, len(bf) - 1)])
    return mnf, mdf, pkf, psr, pow_20_60, pow_60_150, pow_150_450, pow_lh, se95

def _resample_poly(data, up, down):
    from scipy.signal import resample_poly as _rp
    return _rp(data.astype(np.float32, copy=False), up, down).astype(np.float64)


# ========== Preprocessing ==========

def _preprocess_ppg(x, fs=100):
    """Returns (raw_clean, bp, dc)."""
    x = np.asarray(x, dtype=np.float64).copy()
    x = _remove_burr(x)
    x = _remove_step(x)
    mf_k = max(3, int(round(0.05 * fs)))
    if mf_k % 2 == 0:
        mf_k += 1
    if len(x) >= mf_k:
        try:
            x = medfilt(x, kernel_size=mf_k)
        except Exception:
            pass
    ma_w = max(2, int(round(0.03 * fs)))
    kernel = np.ones(ma_w) / ma_w
    x = np.convolve(x, kernel, mode="same")
    dc = float(np.median(x))
    bp = _bandpass(x, fs)
    bp = np.convolve(bp, kernel, mode="same")
    return x, bp, dc

def _build_3ch_ppg(ppg):
    """Convert raw six-channel PPG to the same three virtual channels used in training."""
    ppg = np.asarray(ppg, dtype=np.float64)
    if ppg.ndim == 1:
        return ppg.reshape(-1, 1)
    if ppg.ndim != 2:
        raise ValueError(f"ppg must be 1D or 2D, got shape={{ppg.shape}}")
    if ppg.shape[1] == 6:
        raw = ppg
    elif ppg.shape[0] == 6:
        raw = ppg.T
    else:
        return ppg
    ch_A = (raw[:, 0] + raw[:, 1]) / 2.0
    ch_B = (raw[:, 2] + raw[:, 4]) / 2.0
    ch_C = (raw[:, 3] + raw[:, 5]) / 2.0
    return np.column_stack([ch_A, ch_B, ch_C])

def _narrow_notch(x, fs, f0, bw_hz=0.8, order=2):
    nyq = 0.5 * fs
    w0 = f0 / nyq
    wbw = bw_hz / nyq
    lo = max(w0 - wbw, 1e-6)
    hi = min(w0 + wbw, 0.999)
    if not (0 < lo < hi < 1):
        return x
    try:
        b, a = butter(order, [lo, hi], btype="bandstop")
        return filtfilt(b, a, x)
    except Exception:
        return x

def _compute_leak_ratios(bp_ref, fs, leak_freqs, bw_hz=0.8):
    """Compute narrowband leakage ratios on notch-free reference signal.
    使用 Welch 方法 (nperseg=512)，与 s03 extract_emg_leakage_features 一致。
    """
    if bp_ref is None or len(bp_ref) < 16:
        return (0.0,) * len(leak_freqs)
    x = np.asarray(bp_ref, dtype=np.float64)
    try:
        from scipy.signal import welch
        nperseg = min(512, len(x) // 2)
        if nperseg < 16:
            return (0.0,) * len(leak_freqs)
        noverlap = nperseg // 2
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        return (0.0,) * len(leak_freqs)
    mask_total = (f >= 20) & (f <= 450)
    total_pow = float(np.sum(Pxx[mask_total]))
    if total_pow < EPS:
        return (0.0,) * len(leak_freqs)
    ratios = []
    for f0 in leak_freqs:
        mask = (f >= f0 - bw_hz) & (f <= f0 + bw_hz)
        p_band = float(np.sum(Pxx[mask]))
        ratios.append(p_band / total_pow)
    return tuple(ratios)


def _emg_robust_clean(x, mad_k=10.0):
    if x is None or len(x) < 5:
        return x
    try:
        x = medfilt(x, kernel_size=3)
    except Exception:
        pass
    mad = float(np.median(np.abs(x - np.median(x))))
    if mad > EPS:
        clip = mad_k * mad
        np.clip(x, -clip, clip, out=x)
    return x

def _acc_robust_clean(acc, burr_k=6.0):
    if acc is None:
        return acc
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim == 1 or len(acc) < 3:
        return acc.copy()
    out = acc.copy()
    for ax in range(out.shape[1]):
        out[:, ax] = _remove_burr(out[:, ax], k=burr_k)
    return out

def _preprocess_emg(x, fs=1000):
    """Returns (bp_leak_ref, bp_clean, env, x_demean)."""
    if x is None or len(x) < 4:
        return None, None, None, None
    x = np.asarray(x, dtype=np.float64).copy()
    x_demean = x - np.mean(x)
    x_clean = _emg_robust_clean(x_demean.copy())
    bp = _bandpass(x_clean, fs, lowcut=20.0, highcut=450.0, order=4)
    # 保存 notch 前参考信号
    bp_leak_ref = bp.copy()
    # 统一 notch (50/100/150/200/250/300, bw=0.8Hz)
    for f0 in (50.0, 100.0, 150.0, 200.0, 250.0, 300.0):
        bp = _narrow_notch(bp, fs, f0, bw_hz=0.8)
    bp_clean = bp
    env = np.abs(bp_clean)
    return bp_leak_ref, bp_clean, env, x_demean


def _band_power(x, low, high, fs):
    """Welch 功率谱密度中指定频段的总能量（与 s03 _emg_welch_spectrum 一致）。"""
    if x is None or len(x) < 16:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy.signal import welch
        nperseg = min(1024, len(x) // 2)
        if nperseg < 16:
            return 0.0
        noverlap = nperseg // 2
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        return 0.0
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(Pxx[mask]))


def _welch_coherence(x, y, fs, nperseg=None):
    try:
        from scipy.signal import coherence as _coh
    except Exception:
        return None, None
    n = min(len(x), len(y))
    if n < int(2 * fs):
        return None, None
    nps = int(2 * fs) if nperseg is None else min(nperseg, n)
    return _coh(x[:n], y[:n], fs=fs, nperseg=nps)


def _detect_ppg_peaks(ppg_bp, fs=100):
    if ppg_bp is None or len(ppg_bp) < int(fs):
        return np.array([], dtype=int)
    min_dist = max(1, int(0.33 * fs))
    prom = max(float(np.std(ppg_bp)) * 0.3, EPS)
    try:
        peaks, _ = find_peaks(ppg_bp, distance=min_dist, prominence=prom)
    except Exception:
        peaks = np.array([], dtype=int)
    return peaks


def _ppg_morphology_stats(ppg_bp, peaks, fs=100):
    """Returns (dicrotic_ratio, aug_index_mean, pulse_width_cv)."""
    if ppg_bp is None or len(peaks) < 2:
        return 0.0, 0.0, 0.0
    x = np.asarray(ppg_bp, dtype=np.float64)
    dicrotic_hits = 0
    aug_indices = []
    pulse_widths = []
    for i in range(len(peaks) - 1):
        p0, p1 = int(peaks[i]), int(peaks[i+1])
        seg = x[p0:p1]
        if len(seg) < int(0.2 * fs):
            continue
        pulse_widths.append(len(seg) / fs)
        s_start = int(0.25 * len(seg))
        s_end = int(0.75 * len(seg))
        if s_end - s_start < 3:
            continue
        sub = seg[s_start:s_end]
        if len(sub) < 3:
            continue
        sub_peak = float(np.max(sub))
        sub_mean = float(np.mean(seg))
        sub_std = float(np.std(seg)) + EPS
        main_peak = float(np.max(seg[:s_start])) if s_start > 0 else float(seg[0])
        if sub_peak > sub_mean + 0.2 * sub_std and main_peak > EPS:
            dicrotic_hits += 1
            aug_indices.append(sub_peak / main_peak)
    n_seg = max(1, len(peaks) - 1)
    dicrotic_ratio = dicrotic_hits / n_seg
    aug_mean = float(np.mean(aug_indices)) if aug_indices else 0.0
    if len(pulse_widths) >= 2:
        pwm = float(np.mean(pulse_widths))
        pw_cv = float(np.std(pulse_widths) / (pwm + EPS))
    else:
        pw_cv = 0.0
    return dicrotic_ratio, aug_mean, pw_cv


def _ppg_hrv_stats(peaks, fs=100):
    """Returns (rmssd, cv, pnn30)."""
    if len(peaks) < 3:
        return 0.0, 0.0, 0.0
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / fs
    if len(rr) < 2:
        return 0.0, 0.0, 0.0
    drr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(drr * drr)))
    rr_mean = float(np.mean(rr))
    cv = float(np.std(rr) / (rr_mean + EPS))
    pnn30 = float(np.mean(np.abs(drr) > 0.030))
    return rmssd, cv, pnn30


def _acc_ppg_coherence(acc_mag, ppg_bp, fs_acc=100, fs_ppg=100):
    """Returns (coh_micro, coh_hr)."""
    if acc_mag is None or ppg_bp is None:
        return 0.0, 0.0
    m = np.asarray(acc_mag, dtype=np.float64)
    m = m - np.mean(m)
    if abs(fs_acc - fs_ppg) > 1e-3:
        try:
            m = _resample_poly(m, int(round(fs_ppg)), int(round(fs_acc)))
        except Exception:
            return 0.0, 0.0
    f, Cxy = _welch_coherence(m, ppg_bp, fs=fs_ppg)
    if f is None:
        return 0.0, 0.0
    mask_micro = (f >= 0.5) & (f <= 3.0)
    mask_hr = (f >= 0.8) & (f <= 3.0)
    micro = float(np.mean(Cxy[mask_micro])) if np.any(mask_micro) else 0.0
    hr = float(np.mean(Cxy[mask_hr])) if np.any(mask_hr) else 0.0
    return micro, hr


# ========== Main ==========

def extract_features(ppg, emg=None, acc=None, fs=100, fs_emg=1000):
    """Extract {n_features} features from a 3s@100Hz PPG window + EMG + ACC.

    ppg: 2D float array (N,6) raw PPG @100Hz or (N,3) virtual PPG @100Hz.
    """
    # ---- Preprocess PPG (raw 6 channels -> training virtual 3 channels) ----
    ppg = np.asarray(ppg, dtype=np.float64)
    ppg = _build_3ch_ppg(ppg)
    n_ch = ppg.shape[1]
    raw_list, bp_list = [], []
    for ch_idx in range(min(n_ch, 3)):
        ch_raw, ch_bp, _ = _preprocess_ppg(ppg[:, ch_idx], fs)
        raw_list.append(ch_raw)
        bp_list.append(ch_bp)
    ppg_ch_raw = np.column_stack(raw_list)
    ppg_ch_bp = np.column_stack(bp_list)

    # 3-ch mean for single-channel features
    ppg_raw = np.mean(ppg_ch_raw, axis=1)
    ppg_bp = np.mean(ppg_ch_bp, axis=1)
    ppg_dc = float(np.median(ppg_raw))
    ir = ppg_ch_raw[:, 0]  # ch_A for spatial coupling

    # ---- 3ch spatial intermediates ----
    if n_ch >= 3:
        g = ppg_ch_raw.T
        _sp_std = np.std(g, axis=0)
        _sp_mean = np.mean(g, axis=0)
        imb = _sp_std / (np.abs(_sp_mean) + EPS)
        g_max = np.max(g, axis=0)
        g_min = np.min(g, axis=0)
        denom_abs = np.abs(g[0]) + np.abs(g[1]) + np.abs(g[2]) + EPS
        rn = (g_max - g_min) / denom_abs
        vx = g[0] - 0.5 * g[1] - 0.5 * g[2]
        vy = (np.sqrt(3.0) / 2.0) * (g[1] - g[2])
        vmag = np.sqrt(vx * vx + vy * vy) / denom_abs
        dc3 = np.array([float(np.median(g[0])), float(np.median(g[1])), float(np.median(g[2]))])
        # 3ch bandpass consistency
        c0 = ppg_ch_bp[:, 0] - np.mean(ppg_ch_bp[:, 0])
        c1 = ppg_ch_bp[:, 1] - np.mean(ppg_ch_bp[:, 1])
        c2 = ppg_ch_bp[:, 2] - np.mean(ppg_ch_bp[:, 2])
        c01 = _safe_corr(c0, c1)
        c12 = _safe_corr(c1, c2)
        c20 = _safe_corr(c2, c0)
        c3 = [c01, c12, c20]
        xc01 = np.correlate(c0, c1, mode="same")
        xc12 = np.correlate(c1, c2, mode="same")
        l01 = np.argmax(np.abs(xc01)) - len(c0) // 2
        l12 = np.argmax(np.abs(xc12)) - len(c0) // 2
    else:
        imb = rn = vmag = np.zeros(len(ppg_raw))
        dc3 = np.zeros(3)
        c3 = [0.0, 0.0, 0.0]
        l01 = l12 = 0

    # ---- Preprocess EMG ----
    EMG_LEAK_FREQS = (100.0, 150.0, 200.0, 250.0, 300.0)
    if emg is not None and len(emg) >= 4:
        emg = np.asarray(emg, dtype=np.float64)
        if emg.ndim == 1:
            emg = emg.reshape(-1, 1)
        emg0_leak_ref, emg0_bp, emg0_env, emg0_demean = _preprocess_emg(emg[:, 0], fs_emg)
        if emg.shape[1] >= 2:
            emg1_leak_ref, emg1_bp, emg1_env, emg1_demean = _preprocess_emg(emg[:, 1], fs_emg)
        else:
            emg1_leak_ref = emg1_bp = emg1_env = emg1_demean = None
        # leak ratio array per channel
        emg0_leak = _compute_leak_ratios(emg0_leak_ref, fs_emg, EMG_LEAK_FREQS) if emg0_leak_ref is not None else (0,)*5
        emg1_leak = _compute_leak_ratios(emg1_leak_ref, fs_emg, EMG_LEAK_FREQS) if emg1_leak_ref is not None else (0,)*5
        emg0_freq = _emg_frequency_features(emg0_bp, fs_emg)
        emg1_freq = _emg_frequency_features(emg1_bp, fs_emg) if emg1_bp is not None else (0,)*9
        if emg0_env is not None:
            emg0_env_ds = _resample_poly(emg0_env, fs, fs_emg)
            emg0_env_smooth = _smooth_envelope(emg0_env, fs_emg)
            emg0_env_smooth_ds = _resample_poly(emg0_env_smooth, fs, fs_emg)
        else:
            emg0_env_ds = emg0_env_smooth_ds = None
    else:
        emg0_leak_ref = emg0_bp = emg0_env = emg0_demean = None
        emg1_leak_ref = emg1_bp = emg1_env = emg1_demean = None
        emg0_leak = emg1_leak = (0,) * 5
        emg0_freq = emg1_freq = (0,) * 9
        emg0_env_ds = emg0_env_smooth_ds = None

    # ---- ACC (gravity/motion separation) ----
    if acc is not None and len(acc) >= 4:
        acc = _acc_robust_clean(np.asarray(acc, dtype=np.float64))
        # 低通 <0.5Hz 分离重力
        axis_grav = []
        for ax in range(acc.shape[1]):
            ax_raw = acc[:, ax]
            ax_mean = np.mean(ax_raw)
            try:
                ax_lp = _bandpass(ax_raw - ax_mean, fs, 0.1, 0.5, order=2)
            except Exception:
                ax_lp = np.zeros(len(ax_raw))
            axis_grav.append(ax_lp + ax_mean)
        acc_grav = np.column_stack(axis_grav)
        grav_mag = np.sqrt(np.sum(acc_grav**2, axis=1) + EPS)
        gm = np.mean(acc_grav, axis=0)
        # 运动分量 = 原始 - 重力
        acc_motion = acc - acc_grav
        motion_mag = np.sqrt(np.sum(acc_motion**2, axis=1) + EPS)
        acc_mag = np.sqrt(np.sum(acc**2, axis=1) + EPS)
        acc_mag_bp = _bandpass(acc_mag - np.mean(acc_mag), fs, 0.5, 5.0, order=2)
    else:
        grav_mag = motion_mag = acc_mag = None
        acc_mag_bp = None
        gm = np.zeros(3)
    am = acc_mag is not None
    ambp = acc_mag_bp

    # ---- FFT / autocorr caches ----
    fft_p = _fft_features(ppg_bp, fs)
    ac_p = _autocorr_features(ppg_bp, fs)

    # ---- Smooth envelopes for cross-modal ----
    ppg_env = _smooth_envelope(ppg_bp, fs)

    # ---- Temporal peak detection ----
    Np = len(ppg_bp)
    pk = find_peaks(ppg_bp, prominence=0)
    vk = find_peaks(-ppg_bp, prominence=0)

    # ---- Anti-spoof intermediates ----
    ppg_peaks = _detect_ppg_peaks(ppg_bp, fs)
    ppg_dicr, ppg_aug, ppg_pw_cv = _ppg_morphology_stats(ppg_bp, ppg_peaks, fs)
    ppg_rmssd, ppg_cv, ppg_pnn30 = _ppg_hrv_stats(ppg_peaks, fs)
    acc_coh_micro, acc_coh_hr = _acc_ppg_coherence(acc_mag, ppg_bp, fs_acc=fs, fs_ppg=fs)

    # PPG PI sub-window IQR
    _pis = []
    _sub_n = int(fs)
    for _i in range(0, len(ppg_raw) - _sub_n + 1, _sub_n):
        _sr = ppg_raw[_i:_i+_sub_n]
        _sb = ppg_bp[_i:_i+_sub_n]
        if len(_sr) < _sub_n:
            continue
        _sac = float(np.sqrt(np.mean(_sb ** 2)))
        _sdc = float(np.median(_sr))
        _pis.append(_sac / (abs(_sdc) + EPS))
    ppg_pi_sub_iqr = _robust_iqr(np.asarray(_pis)) if len(_pis) >= 2 else 0.0

    # ---- Linear-regression slope ----
    _t = np.arange(Np, dtype=np.float64)
    _t_mean = float(np.mean(_t))
    p_mean = float(np.mean(ppg_bp))
    _num = float(np.sum((_t - _t_mean) * (ppg_bp - p_mean)))
    _den = float(np.sum((_t - _t_mean) ** 2))
    slope = (_num / _den) if _den > EPS else 0.0
    _fitted = p_mean + slope * (_t - _t_mean)
    slope_std = float(np.std(ppg_bp - _fitted))

    # ---- Per-feature computation ----
    f = {{}}
{feat_block}

    # ---- Build output vector with fill values and clip bounds ----
    vec = []
    for name in FEATURE_ORDER:
        v = f.get(name, 0.0)
        if v is None or not np.isfinite(v):
            v = FILL_VALUES.get(name, 0.0)
        # Apply training clip bounds (IQR-based, matches s05 clip_outliers)
        bound = CLIP_BOUNDS.get(name)
        if bound is not None and isinstance(bound, (list, tuple)) and len(bound) == 2:
            lo, hi = float(bound[0]), float(bound[1])
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
        vec.append(float(v))
    return vec


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ppg = rng.normal(0, 1, 300)
    emg = rng.normal(0, 1, (3000, 2))
    acc = rng.normal(0, 1, (300, 3))
    vec = extract_features(ppg, emg, acc)
    print(f"Feature vector: {{len(vec)}} values")
    for i, (name, val) in enumerate(zip(FEATURE_ORDER, vec)):
        print(f"  {{i:2d}} {{name:35s}} = {{val:.6f}}")
'''
    return script


# =========================================================
# 部署配方导出
# =========================================================

def export_deploy_cookbook(artifact_dir):
    """导出完整部署配方: 特征从原始窗口到特征值的完整链路。"""
    bundle_path = os.path.join(artifact_dir, "model_bundle.pkl")
    th_path = os.path.join(artifact_dir, "stage1_threshold.json")

    if not os.path.exists(bundle_path):
        print("[WARN] model_bundle.pkl not found, skip deploy cookbook")
        return

    bundle = joblib.load(bundle_path)
    selected = bundle["feature_names"]
    fill_values = bundle["fill_values"]
    clip_bounds = bundle.get("clip_bounds", {})
    threshold = float(bundle["threshold"])
    model = bundle["model"]
    booster = model.get_booster()
    n_estimators = model.n_estimators
    postprocess_cfg = {
        "alpha": 0.4,
        "median_k": 1,
        "T_on": 0.75,
        "T_off": 0.35,
        "K_on": 5,
        "K_off": 5,
        "cooldown_sec": 5.0,
    }
    config_path = os.path.join(artifact_dir, "final_model_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            if "postprocess" in saved_cfg:
                postprocess_cfg.update(saved_cfg["postprocess"])
        except Exception as e:
            print(f"[WARN] final_model_config.json 读取失败，deploy_cookbook 使用默认后处理参数: {e}")

    # Build feature formulas from s06
    from s06_deploy_eval import build_feature_formula_map
    formula_map = build_feature_formula_map(selected)

    cookbook = {
        "_title": "手表佩戴活体检测 (PPG+EMG+ACC) — 部署配方",
        "_for": "嵌入式/工程化部署工程师。本文件自包含。",
        "_input": "3s@100Hz PPG窗口 (300 samples) + EMG窗口 (3000 samples x 2) + ACC窗口 (300 samples x 3)",
        "A_preprocessing_ppg": {
            "_note": "对单通道 PPG 执行管线: remove_burr → remove_step → medfilt(50ms) → movavg(30ms) → BP(0.4-6Hz, order=4)",
        },
        "A_preprocessing_emg": {
            "_note": "对 EMG 执行: demean → robust_clean(medfilt3+MAD_clip) → BP(20-450Hz) → unified_notch(50/100/150/200/250/300, ±0.8Hz) → envelope; 窄带串扰由 LEAK_* 特征显式建模",
        },
        "B_selected_features": {
            "_note": f"共 {len(selected)} 个特征",
            "feature_order": selected,
            "formulas": {f: formula_map.get(f, {}).get("formula", "[未匹配]") for f in selected},
        },
        "C_xgboost_inference": {
            "fill_values": fill_values,
            "clip_bounds": clip_bounds,
            "preprocess_order": ["select feature_order", "fill NaN/inf with fill_values", "clip by clip_bounds"],
            "fill_rule": "feature_vec[i] 为 NaN/inf 时用 fill_values[feature_name] 替换",
            "clip_rule": "fill 后对每个入选特征执行 clip(lower, upper)，边界来自训练集",
            "model_threshold": threshold,
            "n_estimators": n_estimators,
            "model_json": json.loads(booster.save_config()),
        },
        "D_stage1_gate": {
            "input": "6-ch PPG @ 100Hz -> channel mean, no downsampling",
            "window": "3s window / 1s stride (300 points @100Hz)",
            "rule": "dc > dc_thresh AND ac/|dc| < acdc_thresh",
            "sample_rule": "any(window_pass) → 进入 Stage2",
        },
        "D_stage3_postprocess": {
            "algorithm": "EMA + hysteresis + cooldown",
            "params": {
                "alpha": float(postprocess_cfg.get("alpha", 0.4)),
                "T_on": float(postprocess_cfg.get("T_on", 0.75)),
                "T_off": float(postprocess_cfg.get("T_off", 0.35)),
                "K_on": int(postprocess_cfg.get("K_on", 5)),
                "K_off": int(postprocess_cfg.get("K_off", 5)),
                "cooldown_sec": float(postprocess_cfg.get("cooldown_sec", 5.0)),
                "median_k": int(postprocess_cfg.get("median_k", 1)),
            },
        },
    }

    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            th_data = json.load(f)
        dth = th_data.get("deploy_stage1_threshold", {})
        cookbook["D_stage1_gate"]["thresholds"] = {
            "dc_threshold": float(dth.get("dc_threshold", 0)),
            "ac_dc_threshold": float(dth.get("ac_dc_threshold", 0)),
        }

    out_path = os.path.join(artifact_dir, "deploy_cookbook.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cookbook, f, indent=2, ensure_ascii=False)
    print(f"[OK] deploy_cookbook.json -> {out_path}")

    xgb_path = os.path.join(artifact_dir, "deploy_xgboost.json")
    with open(xgb_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": selected,
            "feature_order": selected,
            "fill_values": fill_values,
            "clip_bounds": clip_bounds,
            "preprocess_order": ["select feature_order", "fill NaN/inf with fill_values", "clip by clip_bounds"],
            "n_estimators": n_estimators,
            "threshold": threshold,
            "model": json.loads(booster.save_config()),
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] deploy_xgboost.json -> {xgb_path}")


# =========================================================
# 评估辅助
# =========================================================

def generate_eval_csv(artifact_dir, split="test", method="state_machine"):
    """生成逐样本 CSV: sample_name, target, total_windows, correct_windows。"""
    details = _load_eval_details(artifact_dir, split, method)
    if not details:
        print("[WARN] 评估结果为空，跳过 CSV")
        return

    csv_path = os.path.join(artifact_dir, "per_sample_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_name,target,total_windows,correct_windows\n")
        for d in details:
            wpreds = d.get("window_preds", [])
            t = d.get("target", 0)
            n_win = len(wpreds)
            n_correct = sum(1 for p in wpreds if p == t) if n_win > 0 else 0
            f.write(f"{d['sample_name']},{t},{n_win},{n_correct}\n")
    print(f"[OK] 逐样本 CSV: {csv_path}")


def plot_error_samples(artifact_dir, split="test", method="state_machine",
                       window_sec=3, stride_sec=1):
    """画预测错误样本图。"""
    import numpy as _np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return

    details = _load_eval_details(artifact_dir, split, method)
    if not details:
        print("[WARN] no eval details found")
        return

    errors = [d for d in details if d["pred"] != d["target"]]
    if not errors:
        print("[OK] all samples correct, no plots needed")
        return

    out_dir = os.path.join(artifact_dir, "error_plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Wrong predictions: {len(errors)}/{len(details)}")

    for d in errors:
        target = d["target"]
        pred = d["pred"]
        probs = d.get("window_probs", [])
        scores = d.get("window_scores", [])
        states = d.get("window_states", [])
        n_win = d.get("n_windows", len(probs))
        t = _np.arange(n_win) * stride_sec + window_sec / 2.0 if n_win > 0 else _np.array([])

        fig, axes = _plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        ax = axes[0]
        ax.set_ylabel("Target", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{d['sample_name']}  target={target}  pred={pred}  "
                     f"s1_pass={d.get('stage1_pass', False)}  fb={d.get('fallback', False)}",
                     fontsize=10)
        if n_win > 0:
            c = "green" if target == 1 else "red"
            ax.axhline(y=target, color=c, linewidth=2, linestyle="--", alpha=0.7)
            ax.fill_between([t[0], t[-1]], target - 0.05, target + 0.05, alpha=0.15, color=c)

        ax = axes[1]
        ax.set_ylabel("Model Probs", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if n_win > 0:
            ax.step(t, probs, where="mid", linewidth=1.5, color="steelblue")
            ax.fill_between(t, 0, _np.array(probs), alpha=0.12, color="steelblue", step="mid")
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

        ax = axes[2]
        ax.set_ylabel("State Machine Score", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if n_win > 0 and len(scores) > 0:
            ax.plot(t, scores, linewidth=2, color="darkorange", marker=".", markersize=3)
            ax.fill_between(t, 0, _np.array(scores), alpha=0.10, color="darkorange")

        ax = axes[3]
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("State Labels", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        if n_win > 0 and len(states) > 0:
            ax.step(t, states, where="mid", linewidth=2, color="crimson")
            is_wrong = pred != target
            ax.text(t[len(t) // 2] if len(t) > 0 else 0, 0.5,
                    "WRONG" if is_wrong else "OK",
                    fontsize=28, color="red" if is_wrong else "green",
                    alpha=0.25, weight="bold", ha="center", va="center")

        _plt.tight_layout()
        safe_name = d["sample_name"].replace("/", "_").replace("\\", "_")
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=120, bbox_inches="tight")
        _plt.close(fig)

    print(f"[OK] {len(errors)} plots -> {out_dir}/")


# =========================================================
# main
# =========================================================

def main():
    p = argparse.ArgumentParser(
        description='手表佩戴活体检测 (PPG+EMG+ACC) — 全流程一键运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dataset_dir dataset --artifact_dir artifacts
  %(prog)s --stop_after s04
  %(prog)s --skip s02,s03
  %(prog)s --artifact_dir artifacts --export_deploy  (仅评估 + 导出)
  %(prog)s --dataset_dir dataset --artifact_dir artifacts --export_window_cache --optimize_postprocess
""")

    # ── 通用参数 ──
    p.add_argument('--dataset_dir', default='dataset', help='原始H5数据目录')
    p.add_argument('--artifact_dir', default='artifacts', help='产物目录')
    p.add_argument('--n_workers', type=int,
                   default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                   help='并行worker数')
    p.add_argument('--dry_run', action='store_true', help='只打印命令不执行')

    # ── 步骤控制 ──
    p.add_argument('--skip', default='', help='跳过的步骤，逗号分隔 (如 s03,s04)')
    p.add_argument('--stop_after', default='s06_cb',
                   help='运行到此步骤后停止；默认 s06_cb')

    # ── s03/s04 参数 ──
    p.add_argument('--window_sec', type=int, default=3, help='Stage2窗口秒数')
    p.add_argument('--stride_sec', type=float, default=1, help='Stage2滑窗步长（秒）')
    p.add_argument('--max_features', type=int, default=15, help='最终入选特征数')

    # ── s05 模型搜参 ──
    p.add_argument('--model_search', action=argparse.BooleanOptionalAction, default=True,
                   help='Enable XGBoost hyperparameter search (default: enabled)')
    p.add_argument('--max_model_nodes', type=int, default=500)
    p.add_argument('--model_search_strategy', default='staged_group_cv',
                   choices=['staged_group_cv', 'staged_valid'])
    p.add_argument('--model_search_max_candidates', type=int, default=600)
    p.add_argument('--model_search_stage2_top_k', type=int, default=80)
    p.add_argument('--model_search_cv_folds', type=int, default=3)
    p.add_argument('--model_search_cv_repeats', type=int, default=2)
    p.add_argument('--model_search_random_state', type=int, default=42)
    p.add_argument('--model_search_accuracy_tolerance', type=float, default=0.0)
    p.add_argument('--model_search_stage1_top_k', type=int, default=4)
    p.add_argument('--model_search_n_estimators', default='20,25,30,35,40,45,50,55,60')
    p.add_argument('--model_search_max_depth', default='2,3,4')
    p.add_argument('--model_search_learning_rate', default='0.025,0.03,0.04,0.05,0.06,0.08,0.10')
    p.add_argument('--model_search_min_child_weight', default='10,15,20,25,30,40,50')
    p.add_argument('--model_search_reg_lambda', default='5,8,10,12,16,20,30')
    p.add_argument('--model_search_reg_alpha', default='0,0.5,1,1.5,2,3')
    p.add_argument('--model_search_subsample', default='0.70,0.75,0.80,0.85,0.90')
    p.add_argument('--model_search_colsample_bytree', default='0.70,0.75,0.80,0.85,0.90')

    # ── s06 / s07 后处理搜参 ──
    p.add_argument('--export_window_cache', action=argparse.BooleanOptionalAction, default=False,
                   help='导出 valid NPZ 缓存，供 s07 后处理搜参使用')
    p.add_argument('--optimize_postprocess', action=argparse.BooleanOptionalAction, default=False,
                   help='运行 s07 FP 敏感后处理搜参')
    p.add_argument('--postprocess_fp_cost', type=float, default=4.0,
                   help='s07 sample false-positive cost')
    p.add_argument('--split', default='test', choices=['train', 'valid', 'test'],
                   help='s06 评估用的数据 split')
    p.add_argument('--model_search_feature_counts', type=str, default='',
                   help='搜参时测试的特征数量，逗号分隔 (如 8,10,12,15)。留空则用 --max_features')

    # ── 向后兼容: --with_postprocess ──
    p.add_argument('--with_postprocess', action='store_true',
                   help='等效于 --export_window_cache --optimize_postprocess')

    args = p.parse_args()

    # ── 步骤定义 ──
    all_steps = _step_list()
    step_keys = [key for key, _, _ in all_steps]

    skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
    stop_after = args.stop_after
    if stop_after not in step_keys:
        print(f'[ERROR] unknown --stop_after={stop_after!r}; choose from: {",".join(step_keys)}')
        sys.exit(2)

    # ── 自动启用可选步骤 ──
    if args.with_postprocess:
        args.export_window_cache = True
        args.optimize_postprocess = True
    if stop_after in {'s06_cache_valid', 's07_post'}:
        if 's06_cache_valid' not in skip_set:
            args.export_window_cache = True
    if stop_after == 's07_post' or args.optimize_postprocess:
        if 's07_post' not in skip_set:
            args.optimize_postprocess = True

    # ── 构建命令 ──
    cmd = build_pipeline_commands(args)

    # ── 运行 ──
    print('=' * 70)
    print(' 手表佩戴活体检测 (PPG+EMG+ACC) — 全流程')
    print('=' * 70)
    print(f'  数据目录:     {args.dataset_dir}')
    print(f'  产物目录:     {args.artifact_dir}')
    print(f'  并行worker:   {args.n_workers}')
    print(f'  入选特征数:   {args.max_features}')
    print(f'  Stage2窗长:   {args.window_sec}s')
    if args.dry_run:
        print('  [DRY RUN]     只打印命令不执行')
    if skip_set:
        print(f'  跳过步骤:     {",".join(sorted(skip_set))}')
    print(f'  停在:         {stop_after}')
    print('=' * 70)

    total_start = time.time()
    for key, display_name, default_enabled in all_steps:
        if key in skip_set:
            print(f'[SKIP] {display_name}')
            continue
        if key not in cmd:
            continue
        # 判断是否执行：默认启用的步骤 或 被显式打开的可选步骤
        enabled = default_enabled
        if key == 's06_opt':
            enabled = getattr(args, 'optimize', False)  # legacy
        if key == 's06_cache_valid':
            enabled = args.export_window_cache
        if key == 's07_post':
            enabled = args.optimize_postprocess
        if not enabled:
            continue

        command = cmd[key]
        if args.dry_run:
            print(f'[DRY] {display_name}: {command}')
            continue

        if command == '__plot__':
            t0 = time.time()
            generate_eval_csv(args.artifact_dir, split='test')
            plot_error_samples(args.artifact_dir, split='test', window_sec=args.window_sec, stride_sec=args.stride_sec)
            dt = time.time() - t0
            print(f'[OK] {display_name}  [{timedelta(seconds=int(dt))}]')
            continue
        if command == '__extractor__':
            t0 = time.time()
            export_feature_extractor_script(args.artifact_dir)
            dt = time.time() - t0
            print(f'[OK] {display_name}  [{timedelta(seconds=int(dt))}]')
            continue
        if command == '__cookbook__':
            t0 = time.time()
            export_deploy_cookbook(args.artifact_dir)
            dt = time.time() - t0
            print(f'[OK] {display_name}  [{timedelta(seconds=int(dt))}]')
            continue

        # 特征数量搜参：对每个 k 独立运行 s05
        if key == 's05' and args.model_search_feature_counts:
            _counts = [int(x.strip()) for x in args.model_search_feature_counts.split(',') if x.strip()]
            _counts = sorted(set(_counts))
            if _counts:
                print(f'\n[特征数量搜参] 测试 k = {_counts}')
                _best_k, _best_acc = None, -1.0
                for _k in _counts:
                    _cmd_k = command.replace(
                        f'--max_features {args.max_features}',
                        f'--max_features {_k}')
                    print(f'\n  --- k={_k} ---')
                    _ok = _run(f'{display_name} (k={_k})', _cmd_k)
                    if not _ok:
                        print(f'\n[FAIL] s05 k={_k} 失败')
                        continue
                    # 读取该 k 的 model_search_records 评估效果
                    _rec_path = os.path.join(args.artifact_dir, 'model_search_records.json')
                    if os.path.exists(_rec_path):
                        try:
                            with open(_rec_path, 'r', encoding='utf-8') as _rf:
                                _records = json.load(_rf)
                            if _records:
                                _best_rec = _records[0]
                                _acc = _best_rec.get('mean_cv_accuracy', 0.0)
                                if _acc > _best_acc:
                                    _best_k, _best_acc = _k, _acc
                        except Exception:
                            pass
                if _best_k and _best_k != _counts[-1]:
                    print(f'\n[特征数量搜参] 最优 k={_best_k} (acc={_best_acc:.4f})，重新运行 s05')
                    _cmd_best = command.replace(
                        f'--max_features {args.max_features}',
                        f'--max_features {_best_k}')
                    ok = _run(f'{display_name} (final k={_best_k})', _cmd_best)
                    if not ok:
                        print(f'\n[FAIL] 流水线中断于: {display_name}')
                        sys.exit(1)
                else:
                    ok = True  # 最后一个 k 已经是最优或唯一
            else:
                ok = _run(display_name, command)
                if not ok:
                    print(f'\n[FAIL] 流水线中断于: {display_name}')
                    sys.exit(1)
        else:
            ok = _run(display_name, command)
            if not ok:
                print(f'\n[FAIL] 流水线中断于: {display_name}')
                sys.exit(1)

        if key == stop_after:
            print(f'\n[STOP] 已运行到 {stop_after}，按 --stop_after 提前结束')
            break

    total_dt = time.time() - total_start
    print(f'\n{"=" * 70}')
    print(f'[OK] 全流程完成  [{timedelta(seconds=int(total_dt))}]')
    print(f'{"=" * 70}')
    pkg = os.path.join(args.artifact_dir, 'deploy_package')
    if os.path.isdir(pkg):
        print(f'\n部署产物: {pkg}/')
        for f in sorted(os.listdir(pkg)):
            print(f'  {f}  ({os.path.getsize(os.path.join(pkg, f)):,} bytes)')


if __name__ == "__main__":
    main()
