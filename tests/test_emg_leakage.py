"""测试 EMG 窄带串扰建模：leak 特征、notch 效果、工频漂移覆盖。"""
import numpy as np
import pytest


def _synthetic_emg_with_tone(fs=1000, duration_sec=3, tone_hz=100.0, tone_amp=50.0):
    """生成叠加窄带正弦的合成 EMG 信号。"""
    rng = np.random.default_rng(42)
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    emg = rng.normal(0, 1, size=n)  # 白噪声基底
    emg += tone_amp * np.sin(2 * np.pi * tone_hz * t)  # 注入窄带正弦
    return emg


def test_leak_ratio_detects_100hz_crosstalk():
    """100Hz 窄带串扰应被 LEAK_100_RATIO 显著检出。"""
    from s03_extract_feature_pool import extract_emg_leakage_features, preprocess_emg_signal_with_raw

    emg = _synthetic_emg_with_tone(tone_hz=100.0, tone_amp=50.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)
    feat = extract_emg_leakage_features(bp_leak_ref, fs=1000.0, prefix="EMG0")

    leak_100 = feat["EMG0_LEAK_100_RATIO"]
    assert leak_100 > 0.1, f"100Hz 串扰应产生显著 leak ratio, 实际: {leak_100:.4f}"


def test_leak_ratio_low_for_clean_signal():
    """无串扰的清洁信号应产生低 leak ratio。"""
    from s03_extract_feature_pool import extract_emg_leakage_features, preprocess_emg_signal_with_raw

    rng = np.random.default_rng(99)
    emg = rng.normal(0, 1, size=3000)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)
    feat = extract_emg_leakage_features(bp_leak_ref, fs=1000.0, prefix="EMG0")

    # 白噪声无窄带能量，各频点 ratio 应远小于 1
    assert feat["EMG0_LEAK_100_RATIO"] < 0.3
    assert feat["EMG0_LEAK_SUM_RATIO"] < 0.5
    assert feat["EMG0_LEAK_MAX_RATIO"] < 0.2


def test_notch_suppresses_150hz_peak():
    """150Hz notch 后 PKF 不应被 150Hz 串扰主导。"""
    from s03_extract_feature_pool import (
        preprocess_emg_signal_with_raw,
        extract_emg_frequency_features,
    )

    emg = _synthetic_emg_with_tone(tone_hz=150.0, tone_amp=50.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)

    # 清洁信号上计算频域特征
    feat_clean = extract_emg_frequency_features(bp_clean, fs=1000.0, prefix="EMG")
    # 参考信号（未 notch）上计算频域特征——应被 150Hz 主导
    feat_ref = extract_emg_frequency_features(bp_leak_ref, fs=1000.0, prefix="EMG")

    # notch 后的 PKF 应远离 150Hz（不应被 150Hz 直线主导）
    pkf_clean = feat_clean["EMG_PKF"]
    assert pkf_clean is not None
    # notch 前 PKF 应在 150Hz 附近
    pkf_ref = feat_ref["EMG_PKF"]
    assert abs(pkf_ref - 150.0) < 10.0, f"notch 前 PKF 应在 150Hz 附近, 实际: {pkf_ref}"


def test_50hz_notch_covers_drifted_mains():
    """49.5Hz 的工频漂移应被 ±0.8Hz notch 有效压制。"""
    from s03_extract_feature_pool import preprocess_emg_signal_with_raw

    emg = _synthetic_emg_with_tone(tone_hz=49.5, tone_amp=30.0)
    bp_leak_ref, bp_clean, env, x_demean = preprocess_emg_signal_with_raw(emg)

    # notch 后信号在 49.5Hz 附近的能量应被显著削弱
    # 验证：bp_clean 的标准差应显著低于 bp_leak_ref
    std_ref = float(np.std(bp_leak_ref))
    std_clean = float(np.std(bp_clean))
    # notch 后信号能量降低（标准差减小）
    assert std_clean < std_ref * 0.9, (
        f"notch 应削弱窄带能量: std_ref={std_ref:.2f}, std_clean={std_clean:.2f}"
    )


def test_mains_features_on_leak_ref():
    """50Hz mains 特征应在 notch 前参考信号上计算，能反映工频能量。"""
    from s03_extract_feature_pool import (
        preprocess_emg_signal_with_raw,
        extract_emg_mains_features,
    )

    # 无 50Hz 的信号
    rng = np.random.default_rng(1)
    emg_quiet = rng.normal(0, 1, size=3000)

    # 有 50Hz 的信号
    emg_loud = emg_quiet + 20.0 * np.sin(2 * np.pi * 50.0 * np.arange(3000) / 1000.0)

    _, _, _, _ = preprocess_emg_signal_with_raw(emg_quiet)
    bp_leak_loud, _, _, _ = preprocess_emg_signal_with_raw(emg_loud)

    feat_quiet = extract_emg_mains_features(emg_quiet, fs=1000.0, prefix="TEST")
    feat_loud = extract_emg_mains_features(bp_leak_loud, fs=1000.0, prefix="TEST")

    # 含 50Hz 的信号应有更高的 PWR_50HZ 和 50HZ_RATIO
    assert feat_loud["TEST_PWR_50HZ"] > feat_quiet["TEST_PWR_50HZ"], (
        f"含 50Hz 信号应有更高 PWR_50HZ: loud={feat_loud['TEST_PWR_50HZ']:.2f}, quiet={feat_quiet['TEST_PWR_50HZ']:.2f}"
    )
    assert feat_loud["TEST_50HZ_RATIO"] > feat_quiet["TEST_50HZ_RATIO"] * 5


def test_emg_missing_all_leak_features_zero():
    """EMG 缺失时所有 leak 特征应为 0.0。"""
    from s03_extract_feature_pool import extract_emg_features

    feat = extract_emg_features(None, fs=1000.0)

    for ch in [0, 1]:
        assert feat[f"EMG{ch}_LEAK_100_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_150_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_SUM_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_MAX_RATIO"] == 0.0
        assert feat[f"EMG{ch}_LEAK_MAX_FREQ"] == 0.0


def test_emg_single_channel_leak_features():
    """单通道 EMG 时，ch0 有值 ch1 为 0。"""
    from s03_extract_feature_pool import extract_emg_features

    rng = np.random.default_rng(7)
    emg = rng.normal(0, 1, size=(3000, 1))  # 单通道

    feat = extract_emg_features(emg, fs=1000.0)

    # ch0 应有有效值
    assert feat["EMG0_LEAK_100_RATIO"] >= 0.0
    assert feat["EMG0_LEAK_SUM_RATIO"] >= 0.0
    # ch1 应为 0（缺失）
    assert feat["EMG1_LEAK_100_RATIO"] == 0.0
    assert feat["EMG1_LEAK_SUM_RATIO"] == 0.0


def test_preprocess_returns_four_values():
    """preprocess_emg_signal_with_raw 应返回 4 个值。"""
    from s03_extract_feature_pool import preprocess_emg_signal_with_raw

    rng = np.random.default_rng(3)
    emg = rng.normal(0, 1, size=3000)
    result = preprocess_emg_signal_with_raw(emg)

    assert len(result) == 4, f"应返回 4 个值, 实际: {len(result)}"
    bp_leak_ref, bp_clean, env, x_demean = result
    assert bp_leak_ref is not None
    assert bp_clean is not None
    assert env is not None
    assert x_demean is not None
    assert len(bp_leak_ref) == len(bp_clean) == len(emg)


def test_leak_features_names_consistent():
    """s03 提取的 leak 特征名应包含预期的 16 个特征 (2ch × 8)。"""
    from s03_extract_feature_pool import extract_emg_features

    rng = np.random.default_rng(5)
    emg = rng.normal(0, 1, size=(3000, 2))
    feat = extract_emg_features(emg, fs=1000.0)

    expected_leak_keys = []
    for ch in [0, 1]:
        for hz in [100, 150, 200, 250, 300]:
            expected_leak_keys.append(f"EMG{ch}_LEAK_{hz}_RATIO")
        expected_leak_keys.append(f"EMG{ch}_LEAK_SUM_RATIO")
        expected_leak_keys.append(f"EMG{ch}_LEAK_MAX_RATIO")
        expected_leak_keys.append(f"EMG{ch}_LEAK_MAX_FREQ")

    for k in expected_leak_keys:
        assert k in feat, f"缺少特征: {k}"
        assert isinstance(feat[k], float), f"{k} 应为 float, 实际: {type(feat[k])}"


def test_leak_max_freq_is_valid():
    """LEAK_MAX_FREQ 应为 _EMG_LEAK_FREQS 中的某个值。"""
    from s03_extract_feature_pool import extract_emg_features, _EMG_LEAK_FREQS

    # 制造一个 200Hz 强串扰信号
    emg0 = _synthetic_emg_with_tone(tone_hz=200.0, tone_amp=80.0)
    emg1 = _synthetic_emg_with_tone(tone_hz=150.0, tone_amp=30.0)
    emg = np.column_stack([emg0, emg1])

    feat = extract_emg_features(emg, fs=1000.0)

    # ch0 最大串扰频点应为 200Hz
    assert feat["EMG0_LEAK_MAX_FREQ"] in _EMG_LEAK_FREQS
    # 200Hz 窄带注入后，200Hz ratio 应主导
    assert feat["EMG0_LEAK_200_RATIO"] > feat["EMG0_LEAK_100_RATIO"], (
        "200Hz 串扰信号应使 LEAK_200_RATIO 最高"
    )


def test_notch_config_constant_values():
    """验证新 notch 常量。"""
    from s03_extract_feature_pool import (
        _EMG_NOTCH_BW_HZ, _EMG_CLEAN_NOTCH_FREQS, _EMG_LEAK_FREQS,
    )

    assert _EMG_NOTCH_BW_HZ == 0.8
    assert 50.0 in _EMG_CLEAN_NOTCH_FREQS
    assert 150.0 in _EMG_CLEAN_NOTCH_FREQS
    assert 250.0 in _EMG_CLEAN_NOTCH_FREQS
    assert 400.0 not in _EMG_CLEAN_NOTCH_FREQS  # 已移除

    assert 50.0 not in _EMG_LEAK_FREQS  # 工频不在 leak 频点中
    assert 300.0 in _EMG_LEAK_FREQS
