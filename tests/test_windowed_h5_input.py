import shutil
import sys
import uuid
import inspect
import pickle
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_windowed_h5():
    root = Path.cwd() / ".test_windowed_h5" / uuid.uuid4().hex
    root.mkdir(parents=True)
    h5_path = root / "windowed.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("sample_1")
        grp.create_dataset("ppg", data=np.ones((2, 6, 300), dtype=np.float64) * 0.30e6)
        grp.create_dataset("emg", data=np.ones((2, 2, 3000), dtype=np.float64))
        grp.create_dataset("acc", data=np.ones((2, 3, 300), dtype=np.float64) * 0.01)
    return root, h5_path


def _make_nested_window_h5():
    root = Path.cwd() / ".test_windowed_h5" / uuid.uuid4().hex
    root.mkdir(parents=True)
    h5_path = root / "nested_windowed.h5"
    with h5py.File(h5_path, "w") as f:
        parent = f.create_group("subjectA_session1")
        for idx in [4, 1, 3, 0, 2]:
            grp = parent.create_group(f"subjectA_session1_w{idx}_1")
            grp.create_dataset("ppg", data=np.ones((6, 300), dtype=np.float64) * (0.30e6 + idx))
            grp.create_dataset("emg", data=np.ones((2, 3000), dtype=np.float64) * idx)
            grp.create_dataset("acc", data=np.ones((3, 300), dtype=np.float64) * (idx + 0.01))
    return root, h5_path


def test_s01_scans_nested_window_groups_as_one_sample_and_drops_first_three():
    import s01_data_split as s01

    root, h5_path = _make_nested_window_h5()
    try:
        samples, filtered = s01._scan_one_h5(str(h5_path))

        assert len(samples) == 1
        assert filtered["emg_channel_count"] == 0
        assert samples[0]["sample_name"] == "subjectA_session1"
        assert samples[0]["target"] == 1
        assert samples[0]["window_indices"] == [3, 4]
        assert samples[0]["window_names"] == [
            "subjectA_session1_w3_1",
            "subjectA_session1_w4_1",
        ]
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s01_hash_split_keeps_existing_samples_stable_when_new_data_is_appended():
    import s01_data_split as s01

    old_samples = [
        {"sample_name": f"old_{i:03d}", "h5_file": "old_file.h5", "target": i % 2}
        for i in range(60)
    ]
    new_samples = [
        {"sample_name": f"new_{i:03d}", "h5_file": "new_file.h5", "target": i % 2}
        for i in range(30)
    ]

    before = s01.split_samples(old_samples, valid_size=0.15, test_size=0.15, random_state=42)
    after = s01.split_samples(old_samples + new_samples, valid_size=0.15, test_size=0.15, random_state=42)

    def assignment(split):
        return {
            sample["sample_name"]: part
            for part, items in split.items()
            for sample in items
            if sample["sample_name"].startswith("old_")
        }

    assert assignment(after) == assignment(before)


def test_s01_hash_split_keeps_existing_samples_stable_when_same_h5_is_appended():
    import s01_data_split as s01

    old_samples = [
        {"sample_name": f"old_{i:03d}", "h5_file": "same_file.h5", "target": i % 2}
        for i in range(30)
    ]
    appended_samples = [
        {"sample_name": f"new_{i:03d}", "h5_file": "same_file.h5", "target": i % 2}
        for i in range(90)
    ]

    before = s01.split_samples(old_samples, valid_size=0.2, test_size=0.2, random_state=42)
    after = s01.split_samples(old_samples + appended_samples, valid_size=0.2, test_size=0.2, random_state=42)

    def assignment(split):
        return {
            sample["sample_name"]: part
            for part, items in split.items()
            for sample in items
            if sample["sample_name"].startswith("old_")
        }

    assert assignment(after) == assignment(before)


def test_s01_hash_split_covers_each_large_h5_file_across_splits():
    import s01_data_split as s01

    samples = [
        {"sample_name": f"sample_{h5_idx}_{i:03d}", "h5_file": f"scene_{h5_idx}.h5", "target": i % 2}
        for h5_idx in range(3)
        for i in range(80)
    ]

    split = s01.split_samples(samples, valid_size=0.15, test_size=0.15, random_state=42)

    by_h5 = {f"scene_{idx}.h5": set() for idx in range(3)}
    for part, items in split.items():
        for sample in items:
            by_h5[sample["h5_file"]].add(part)

    assert by_h5 == {
        "scene_0.h5": {"train", "valid", "test"},
        "scene_1.h5": {"train", "valid", "test"},
        "scene_2.h5": {"train", "valid", "test"},
    }


def test_s01_split_uses_stable_hash_bucket_boundaries():
    import s01_data_split as s01

    samples = [
        {"sample_name": f"sample_{i:03d}", "h5_file": "scene_a.h5", "target": i % 2}
        for i in range(80)
    ]

    split = s01.split_samples(samples, valid_size=0.2, test_size=0.2, random_state=42)

    for part, items in split.items():
        for sample in items:
            score = s01._hash_fraction(s01._stable_sample_key(sample), seed=42)
            if part == "test":
                assert score < 0.2
            elif part == "valid":
                assert 0.2 <= score < 0.4
            else:
                assert score >= 0.4


def test_nested_window_loaders_preserve_name_order_and_window_indices():
    import s01_data_split as s01
    import s02_ir_dc_threshold as s02
    import s03_extract_feature_pool as s03

    root, h5_path = _make_nested_window_h5()
    try:
        samples, _filtered = s01._scan_one_h5(str(h5_path))
        sample = samples[0]

        ppg = s03.load_ppg(sample)
        emg = s03.load_emg(sample)
        acc = s03.load_acc(sample)
        ppg_mean_windows = s02.load_ppg(sample)
        windows = list(s03.iter_sample_windows(
            ppg, emg, acc,
            win_samples=300,
            stride_samples=100,
            window_indices=sample["window_indices"],
        ))

        assert ppg.shape == (2, 300, 6)
        assert emg.shape == (2, 3000, 2)
        assert acc.shape == (2, 300, 3)
        assert ppg[0, 0, 0] == 0.30e6 + 3
        assert ppg[1, 0, 0] == 0.30e6 + 4
        assert ppg_mean_windows.shape == (2, 300)
        assert windows[0]["start_100hz"] == 300
        assert windows[1]["start_100hz"] == 400
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s01_scan_keeps_windowed_emg_when_window_count_is_not_channel_count():
    import s01_data_split as s01

    root = Path.cwd() / ".test_windowed_h5" / uuid.uuid4().hex
    root.mkdir(parents=True)
    h5_path = root / "windowed.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("sample_1")
        grp.create_dataset("ppg", data=np.ones((3, 6, 300), dtype=np.float64))
        grp.create_dataset("emg", data=np.ones((3, 2, 3000), dtype=np.float64))
        grp.create_dataset("acc", data=np.ones((3, 3, 300), dtype=np.float64))
    try:
        samples, filtered = s01._scan_one_h5(str(h5_path))
        assert len(samples) == 1
        assert filtered["emg_channel_count"] == 0
        assert samples[0]["emg_shape"] == [3, 2, 3000]
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_windowed_h5_loaders_normalize_to_window_point_channel():
    import s03_extract_feature_pool as s03

    root, h5_path = _make_windowed_h5()
    try:
        sample = {"h5_file": str(h5_path), "sample_name": "sample_1", "target": 1}

        ppg = s03.load_ppg(sample)
        emg = s03.load_emg(sample)
        acc = s03.load_acc(sample)

        assert ppg.shape == (2, 300, 6)
        assert emg.shape == (2, 3000, 2)
        assert acc.shape == (2, 300, 3)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_windowed_h5_iter_sample_windows_does_not_reslice():
    import s03_extract_feature_pool as s03

    root, h5_path = _make_windowed_h5()
    try:
        sample = {"h5_file": str(h5_path), "sample_name": "sample_1", "target": 1}
        ppg = s03.load_ppg(sample)
        emg = s03.load_emg(sample)
        acc = s03.load_acc(sample)

        windows = list(s03.iter_sample_windows(
            ppg, emg, acc,
            win_samples=300,
            stride_samples=100,
        ))

        assert len(windows) == 2
        assert windows[0]["start_100hz"] == 0
        assert windows[1]["start_100hz"] == 300
        assert windows[0]["ppg_6ch"].shape == (300, 6)
        assert windows[0]["emg"].shape == (3000, 2)
        assert windows[0]["acc"].shape == (300, 3)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s03_extract_rows_uses_existing_h5_windows(monkeypatch):
    import s03_extract_feature_pool as s03

    root, h5_path = _make_windowed_h5()
    try:
        sample = {"h5_file": str(h5_path), "sample_name": "sample_1", "target": 1}
        calls = []

        def fake_extract(ppg_signal, emg_window, acc_window, **kwargs):
            calls.append((ppg_signal.shape, emg_window.shape, acc_window.shape))
            return {"FAKE_FEATURE": float(len(calls))}

        monkeypatch.setattr(s03, "extract_feature_pool_from_window", fake_extract)

        rows = s03._extract_rows_for_sample(
            sample,
            dc_threshold=0.2e6,
            ac_dc_threshold=0.35,
            win_samples=300,
            stride_samples=100,
            fs_ppg_orig=100,
        )

        assert len(rows) == 2
        assert len(calls) == 2
        assert rows[0]["start_100hz"] == 0
        assert rows[1]["start_100hz"] == 300
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s02_stage1_features_use_existing_h5_windows():
    import s02_ir_dc_threshold as s02

    root, h5_path = _make_windowed_h5()
    try:
        sample = {"h5_file": str(h5_path), "sample_name": "sample_1", "target": 1}
        ppg_mean_windows = s02.load_ppg(sample)
        rows = s02.extract_dc_acdc_features(ppg_mean_windows)

        assert ppg_mean_windows.shape == (2, 300)
        assert len(rows) == 2
        assert rows[0]["start_100hz"] == 0
        assert rows[1]["start_100hz"] == 300
        assert rows[0]["dc"] > 0.2e6
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s02_extract_windows_keeps_windowed_samples_shorter_than_min_duration():
    import s02_ir_dc_threshold as s02

    root, h5_path = _make_windowed_h5()
    try:
        sample = {"h5_file": str(h5_path), "sample_name": "sample_1", "target": 1}
        rows = s02._extract_windows_from_sample(sample, min_duration=1000)

        assert len(rows) == 2
        assert rows[0]["start_100hz"] == 0
        assert rows[1]["start_100hz"] == 300
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s02_default_min_duration_matches_three_second_window():
    import s02_ir_dc_threshold as s02

    default_min_duration = inspect.signature(
        s02.extract_stage1_windows
    ).parameters["min_duration_sec"].default

    assert default_min_duration == s02.STAGE1_WINDOW_SEC


def test_s06_inference_uses_existing_h5_windows(monkeypatch):
    import s06_deploy_eval as s06

    root, h5_path = _make_windowed_h5()
    try:
        sample = {"h5_file": str(h5_path), "sample_name": "sample_1", "target": 1}
        calls = []

        def fake_extract(ppg_signal, emg_window, acc_window, **kwargs):
            calls.append((ppg_signal.shape, emg_window.shape, acc_window.shape))
            return {"FAKE_FEATURE": float(len(calls))}, {}

        monkeypatch.setattr(s06, "extract_feature_pool_from_window", fake_extract)
        monkeypatch.setattr(s06, "predict_label_windows", lambda feats, bundle: ([1] * len(feats), [0.9] * len(feats)))

        bundle = {
            "meta": {"fs_ppg": 100, "fs_emg": 1000},
            "threshold": 0.5,
            "feature_names": ["FAKE_FEATURE"],
        }
        out = s06._infer_one_sample(
            sample,
            dc_threshold=0.2e6,
            ac_dc_threshold=0.35,
            window_sec=3.0,
            stride_sec=1.0,
            bundle=bundle,
        )

        assert len(calls) == 2
        assert out["stage1_pass"] is True
        assert out["stage1_frame_results"] == [True, True]
        assert out["window_probs"] == [0.9, 0.9]
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s06_inference_uses_nested_window_group_order(monkeypatch):
    import s01_data_split as s01
    import s06_deploy_eval as s06

    root, h5_path = _make_nested_window_h5()
    try:
        sample = s01._scan_one_h5(str(h5_path))[0][0]
        calls = []

        def fake_extract(ppg_signal, emg_window, acc_window, **kwargs):
            calls.append((float(ppg_signal[0, 0]), emg_window.shape, acc_window.shape))
            return {"FAKE_FEATURE": float(len(calls))}, {}

        monkeypatch.setattr(s06, "extract_feature_pool_from_window", fake_extract)
        monkeypatch.setattr(s06, "predict_label_windows", lambda feats, bundle: ([1] * len(feats), [0.8, 0.9]))

        bundle = {
            "meta": {"fs_ppg": 100, "fs_emg": 1000},
            "threshold": 0.5,
            "feature_names": ["FAKE_FEATURE"],
        }

        out = s06._infer_one_sample(
            sample,
            dc_threshold=0.2e6,
            ac_dc_threshold=0.35,
            window_sec=3.0,
            stride_sec=1.0,
            bundle=bundle,
        )

        assert sample["window_indices"] == [3, 4]
        assert [c[0] for c in calls] == [0.30e6 + 3, 0.30e6 + 4]
        assert out["window_probs"] == [0.8, 0.9]
        assert out["window_start_100hz"] == [300, 400]
        assert len(out["stage1_frame_results"]) == 2
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_window_cache_preserves_nested_window_start_indices(monkeypatch):
    import s01_data_split as s01
    import s06_deploy_eval as s06

    root, h5_path = _make_nested_window_h5()
    try:
        sample = s01._scan_one_h5(str(h5_path))[0][0]

        def fake_extract(ppg_signal, emg_window, acc_window, **kwargs):
            return {"FAKE_FEATURE": 1.0}, {}

        monkeypatch.setattr(s06, "extract_feature_pool_from_window", fake_extract)
        monkeypatch.setattr(s06, "predict_label_windows", lambda feats, bundle: ([1] * len(feats), [0.8, 0.9]))

        bundle = {
            "meta": {"fs_ppg": 100, "fs_emg": 1000},
            "threshold": 0.5,
            "feature_names": ["FAKE_FEATURE"],
        }
        out = s06._infer_one_sample(
            sample,
            dc_threshold=0.2e6,
            ac_dc_threshold=0.35,
            window_sec=3.0,
            stride_sec=1.0,
            bundle=bundle,
        )
        cache_dir = root / "cache"
        cache_path = s06.write_window_cache_npz(
            out,
            str(cache_dir),
            window_sec=3.0,
            stride_sec=1.0,
            model_threshold=0.5,
        )

        with np.load(cache_path, allow_pickle=False) as data:
            assert data["window_start_sec"].tolist() == [3.0, 4.0]
            assert data["window_end_sec"].tolist() == [6.0, 7.0]
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_s03_multiprocess_initializer_is_picklable(monkeypatch):
    import concurrent.futures
    import s03_extract_feature_pool as s03

    class PickleCheckingExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            pickle.dumps(initializer)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable, chunksize=1):
            return [[] for _ in iterable]

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", PickleCheckingExecutor)

    samples = [
        {"h5_file": "dummy.h5", "sample_name": f"sample_{idx}_1", "target": 1}
        for idx in range(3)
    ]

    df = s03.extract_features_for_split(
        samples,
        dc_threshold=0.2e6,
        ac_dc_threshold=0.35,
        n_workers=2,
    )

    assert df.empty
