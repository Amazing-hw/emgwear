# EMG Feature Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deployable EMG subwindow stability, channel-balance, and spectral-shape features while preserving notch-before-model behavior and notch-free leakage features.

**Architecture:** Training feature extraction remains in `s03_extract_feature_pool.py`; standalone deployment formulas remain in `s08_run_pipeline.py`; feature selection group membership remains in `s04_feature_selection.py`. New features use existing preprocessed EMG signals: `bp_clean` for physiological EMG features and `bp_leak_ref` only for explicit mains/leakage features.

**Tech Stack:** Python, NumPy, SciPy signal utilities, pytest, existing deploy feature contract tests.

---

### Task 1: Training Feature Extraction

**Files:**
- Modify: `s03_extract_feature_pool.py`
- Test: `tests/test_emg_feature_additions.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that `extract_emg_features()` emits subwindow, spectral-shape, and channel-balance features with finite values, and that burst-style features are not emitted.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_emg_feature_additions.py -v`
Expected: FAIL because the new feature names are absent.

- [ ] **Step 3: Implement minimal feature extraction**

Add helpers in `s03_extract_feature_pool.py`:
- `extract_emg_subwindow_features(emg_bp, emg_env, fs, prefix)`
- `extract_emg_spectral_shape_features(emg_bp, fs, prefix)`
- `extract_emg_channel_balance_features(ch0_env, ch1_env, ch0_bp, ch1_bp, fs)`

- [ ] **Step 4: Verify tests pass**

Run: `python -m pytest tests/test_emg_feature_additions.py -v`
Expected: PASS.

### Task 2: Deployment Formula Parity

**Files:**
- Modify: `s08_run_pipeline.py`
- Test: `tests/test_pipeline_commands.py`

- [ ] **Step 1: Write failing deploy parity test**

Extend deploy parity coverage to include a representative subset:
`EMG0_RMS_SUBWIN_CV`, `EMG0_SPEC_ENTROPY`, `EMG_ENV_CORR`, `EMG_CONTACT_IMBALANCE`.

- [ ] **Step 2: Verify test fails**

Run: `python -m pytest tests/test_pipeline_commands.py::test_deploy_feature_extractor_matches_training_for_all_deployable_features -v`
Expected: FAIL because deploy formula map does not yet include new feature names.

- [ ] **Step 3: Implement deploy formulas**

Add standalone helpers to the generated deploy extractor script for subwindow, spectral shape, and channel-balance values. Add the new feature names to `_build_feature_code_map()`.

- [ ] **Step 4: Verify deploy parity passes**

Run: `python -m pytest tests/test_pipeline_commands.py::test_deploy_feature_extractor_matches_training_for_all_deployable_features -v`
Expected: PASS.

### Task 3: Feature Selection Groups

**Files:**
- Modify: `s04_feature_selection.py`
- Test: `tests/test_feature_selection_vif.py`

- [ ] **Step 1: Write failing grouping test**

Assert new features map to existing groups:
- subwindow features -> `emg_activity`
- spectral-shape features -> `emg_frequency`
- channel-balance features -> `emg_cross`

- [ ] **Step 2: Verify test fails**

Run: `python -m pytest tests/test_feature_selection_vif.py -v`
Expected: FAIL because new features currently map to `other`.

- [ ] **Step 3: Update groups**

Add new feature names to existing `FEATURE_GROUPS` without raising group limits initially.

- [ ] **Step 4: Verify grouping test passes**

Run: `python -m pytest tests/test_feature_selection_vif.py -v`
Expected: PASS.

### Task 4: Full Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Syntax check**

Run: `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in ['s03_extract_feature_pool.py','s04_feature_selection.py','s08_run_pipeline.py']]; print('ast ok')"`
Expected: `ast ok`.

- [ ] **Step 2: Targeted test suite**

Run: `python -m pytest tests/test_emg_feature_additions.py tests/test_emg_leakage.py tests/test_pipeline_commands.py::test_deploy_feature_extractor_matches_training_for_all_deployable_features tests/test_feature_selection_vif.py -v`
Expected: PASS.

- [ ] **Step 3: Review diff**

Run: `git diff -- s03_extract_feature_pool.py s04_feature_selection.py s08_run_pipeline.py tests/test_emg_feature_additions.py tests/test_pipeline_commands.py tests/test_feature_selection_vif.py`
Expected: only scoped EMG feature and test changes.
