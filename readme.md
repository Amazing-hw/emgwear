# Wearing Liveness Detection

本项目用于手表佩戴活体检测。当前主流程是：

1. `s01` 扫描 H5，并按“原始数据条目”切分 `train/valid/test`。
2. `s02` 用固定工程阈值做 Stage1 PPG DC/ACDC 粗筛。
3. `s03` 对通过 Stage1 的 3s 窗口提取 PPG/EMG/ACC 特征。
4. `s04` 在 `train` 上做特征筛选，`valid` 只用于检查。
5. `s05` 训练 XGBoost；默认用 `train` 内部 group CV 做模型参数搜索，再用 `valid` 固化窗口概率阈值。
6. `s06` 在 `test` 上做端到端评估，并导出部署产物。
7. `s07` 是独立的后处理搜参脚本，默认全流程不会自动运行。

## 一条命令

如果你在项目根目录 `D:\wearing_liveness` 下运行：

```bash
# 默认流程（不含后处理搜参）
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts

# 完整流程（含 NPZ 缓存 + 后处理状态机搜参）
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess
```

如果你已经在 `D:\wearing_liveness\new_new` 目录下运行：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess
```

这条默认命令会运行：

```text
s01 -> s02 -> s03 -> s04 -> s05 -> s06_eval -> s06_xpt -> s06_feat -> s06_plot -> s06_cb
```

这条默认命令不会运行：

```text
s06_cache_valid    # 导出 valid 的逐窗 NPZ 缓存
s07_post           # 基于 NPZ 缓存做后处理参数搜索
s06_opt            # s06 内置的状态机参数网格搜索
```

也就是说，默认命令不会存储 NPZ，也不会重新搜索后处理状态机参数。

明确结论：默认命令不会重新搜索后处理状态机参数；只有手动运行 `s07_postprocess_optimize.py` 或 `s06_deploy_eval.py --optimize` 才会做后处理搜参。

## 为什么默认命令仍然有端到端准确率

`s06_deploy_eval.py` 的评估本身就包含 Stage3 状态机后处理，因此即使没有运行 `s07` 搜参，也会输出：

```text
指标1: 端到端评估 (Stage1 -> Stage2 -> Stage3)
指标2: Stage2 模型评估 (仅通过 Stage1 的数据)
参考: Stage2+3 流式状态
```

这里的区别是：

```text
端到端准确率
  样本级指标。完整模拟部署链路：
  Stage1 先粗筛样本；通过后，Stage2 输出逐窗概率；Stage3 状态机给出最终样本预测。

Stage2 模型准确率
  窗口级指标。只看通过 Stage1 的窗口，直接按 XGBoost 的窗口概率阈值判断，不看状态机最终状态。

Stage2+3 状态机准确率
  窗口/流式参考指标。把 Stage2 的逐窗概率输入状态机，观察状态机逐窗状态，不等同于最终样本级端到端准确率。
```

默认评估使用的状态机参数来源：

```text
1. 如果 artifacts/final_model_config.json 里已经有 postprocess 字段：
   s06 会读取这个已保存的状态机参数。
   这通常来自之前手动跑过 s07_postprocess_optimize.py 或 s06_deploy_eval.py --optimize。

2. 如果没有 postprocess 字段：
   s06 使用代码里的默认状态机参数：
   alpha = 0.4
   median_k = 1
   T_on = 0.75
   T_off = 0.35
   K_on = 5
   K_off = 5
   cooldown_sec = 5
```

所以：默认命令会“使用状态机做评估”，但不会“搜索状态机参数”。这两个动作不是一回事。

## 当前 H5 格式

当前推荐格式是一个 H5 包含多条原始数据；每条原始数据下面包含多个窗口 group：

```text
file.h5
  sample_A/
    sample_A_w0_1/
      ppg
      emg
      acc
    sample_A_w1_1/
      ppg
      emg
      acc
  sample_B/
    sample_B_w0_0/
      ppg
      emg
      acc
```

窗口 group 名称规则：

```text
任意前缀_w窗口编号_label
```

例子：

```text
subject01_session02_w20_1
```

含义：

```text
label = 1
窗口编号 = 20
```

注意：

```text
每个窗口都是 3s 数据。
原始信号按 stride=1s 截取，所以 w20 表示原始时间轴上的第 20 个 1s 起点窗口。
H5 中窗口 group 的存储顺序不可信，程序会按 w 后面的数字排序。
s01 会按窗口编号排序后去掉前三个窗口。
同一条原始数据下面所有窗口的 label 必须一致。
```

每个窗口内的数据形状：

```text
ppg: (6, 300)      # 6 通道 PPG，3s @100Hz
emg: (2, 3000)     # 2 通道 EMG，3s @1000Hz
acc: (3, 300)      # 3 通道 ACC，3s @100Hz
```

旧的预切窗数组格式也仍然支持：

```text
ppg: (num_windows, 6, 300)
emg: (num_windows, 2, 3000)
acc: (num_windows, 3, 300)
```

## Stage1

Stage1 使用 6 通道 PPG 的平均信号：

```text
signal = mean(ppg_ch0 ... ppg_ch5)
fs = 100Hz
window = 3s = 300 samples
stride = 1s = 100 samples
```

计算：

```text
DC = min((x[i] + x[i+1]) / 2)
AC = median(abs(diff(x)))
AC_DC_RATIO = AC / (abs(DC) + eps)
```

部署粗筛规则：

```text
DC > dc_threshold AND AC_DC_RATIO < ac_dc_threshold
```

默认部署阈值：

```text
dc_threshold = 2.2e6
ac_dc_threshold = 0.35
```

训练/特征提取阶段会用一个更宽松的 Stage1 gate，让 Stage2 能看到更多训练窗口；最终端到端 test 评估仍使用部署阈值。

## Stage2

Stage2 输入：

```text
PPG: 6ch -> build_3ch_ppg(...) -> 3ch @100Hz
EMG: 2ch @1000Hz
ACC: 3ch @100Hz
```

默认窗口参数：

```text
window_sec = 3
stride_sec = 1
```

对于当前已经切好的 3s H5 窗口，`s02`、`s03`、`s06` 不会再次滑窗切分，而是直接按窗口 group 读取。

EMG 使用两条滤波分支：

```text
emg_bp_leak_ref:
  20-450Hz bandpass，notch 前，用于 LEAK_* 特征。

emg_bp_clean:
  20-450Hz bandpass + notch(50/100/150/200/250/300Hz, +/-0.8Hz)，用于常规 EMG 特征。
```

## 默认全流程参数

`s08_run_pipeline.py` 常用参数：

```text
--dataset_dir
  H5 数据目录。
  如果在 D:\wearing_liveness 下运行，通常传 dataset。

--artifact_dir
  所有中间产物、模型、评估结果、部署包的输出目录。
  默认建议 artifacts。

--n_workers
  并行 worker 数。
  默认是 CPU 数的一半，上限 4。
  如果你怀疑多进程卡住，可以先设为 1 验证链路。

--max_features
  最终选入模型的特征数量。
  默认 15。

--window_sec
  Stage2 窗长。
  当前 H5 已经是 3s 窗口，默认 3。

--stride_sec
  窗口步长。
  当前 H5 是按 stride=1s 截取，默认 1。

--model_search / --no-model_search
  默认开启 XGBoost 参数搜索。
  如果要跳过模型参数搜索，用 --no-model_search。

--max_model_nodes
  XGBoost 模型最大节点数约束。
  这是部署侧唯一硬约束；搜参时超过该节点数的模型不会被选为最优。

--model_search_strategy
  默认 staged_group_cv。
  含义是先从高预算细粒度空间确定性采样候选，再用 train 内部 group-aware repeated CV 复评。

--model_search_max_candidates
  Stage A 最多采样多少个候选。
  默认 600。

--model_search_stage2_top_k
  Stage B 进入 group CV 复评的候选数。
  默认 80。

--model_search_cv_folds
  group CV 折数。
  默认 3。

--model_search_cv_repeats
  group CV 重复次数。
  默认 2。

--model_search_random_state
  候选采样和 CV 分组的随机种子。
  默认 42，保证可复现。

--model_search_accuracy_tolerance
  CV mean accuracy 容忍范围。
  默认 0.0，不主动牺牲 accuracy 换更小模型。

--model_search_stage1_top_k
  两阶段模型搜索中，第一阶段保留多少个结构组合进入第二阶段细搜。
  越大越慢，可能找到更优参数；越小越快。

--model_search_n_estimators
  XGBoost 树数量候选。
  默认 20,25,30,35,40,45,50,55,60,70,80。

--model_search_max_depth
  XGBoost 单棵树最大深度候选。
  默认 2,3,4。

--model_search_learning_rate
  学习率候选。
  默认 0.025,0.03,0.04,0.05,0.06,0.08,0.10。

--model_search_min_child_weight
  子节点最小权重候选。
  越大模型越保守，通常也更小。
  默认 10,15,20,25,30,40,50。

--model_search_reg_lambda
  L2 正则候选。
  默认 5,8,10,12,16,20,30。

--model_search_reg_alpha
  L1 正则候选。
  默认 0,0.5,1,1.5,2,3。

--model_search_subsample
  行采样比例候选。
  默认 0.70,0.75,0.80,0.85,0.90。

--model_search_colsample_bytree
  列采样比例候选。
  默认 0.70,0.75,0.80,0.85,0.90。

--model_search_feature_counts
  搜索最优特征数量。逗号分隔的候选值（如 "8,10,12,15"）。
  需要先跑过一次 s04 生成 ranked_features.json。
  留空则使用 --max_features 固定值。

--skip
  跳过指定步骤，逗号分隔（如 s03,s04）。
  用于复用已有产物，跳过不需要重跑的步骤。

--stop_after
  运行到指定步骤后停止（如 s04 只跑到特征筛选）。
  默认 s06_cb。

--dry_run
  只打印命令不执行，用于预览流水线步骤。

--export_window_cache / --no-export_window_cache
  是否导出 valid 的逐窗 NPZ 缓存（供 s07 后处理搜参用）。
  默认关闭。

--optimize_postprocess / --no-optimize_postprocess
  是否运行 s07 FP 敏感后处理搜参。
  默认关闭。

--with_postprocess
  等效于 --export_window_cache --optimize_postprocess，一条命令启用完整后处理搜参。

--postprocess_fp_cost
  s07 假阳性惩罚权重。
  默认 4.0。越大越倾向于减少负样本误判为正。

--split
  s06 评估用的数据 split。
  可选 train / valid / test，默认 test。
```

默认模型搜索选择逻辑：

```text
1. 只用 train 内部 group CV 选择模型参数，不用 valid，也不用 test。
2. group 默认使用 `sample_name`，防止同一条原始数据的窗口跨 fold 泄漏；缺失时退回分层 CV，并在 summary 记录 fallback。
3. 默认参数 baseline 强制加入候选，即使用户自定义 grid 没覆盖默认参数。
4. 只在 `total_nodes <= max_model_nodes` 的候选里选择。
5. 主排序是 `mean_cv_accuracy` 最高。
6. 次排序是 `std_cv_accuracy` 更低。
7. 再排序是 `mean_cv_fp_rate` 更低。
8. 最后排序是 `final_total_nodes` 更少。
9. 如果搜索候选的 mean CV accuracy 没超过默认参数，默认参数胜出。
10. valid 只用于窗口概率阈值固化；test 只用于最终报告。
```

## 默认产物

默认命令会生成：

```text
artifacts/splits.json
artifacts/stage1_threshold.json
artifacts/stage1_train_windows.csv
artifacts/stage1_valid_windows.csv
artifacts/feature_pool_train.csv
artifacts/feature_pool_valid.csv
artifacts/feature_pool_test.csv
artifacts/selected_features.json
artifacts/model_bundle.pkl
artifacts/final_model.json
artifacts/final_model_config.json
artifacts/model_search_records.csv
artifacts/model_search_records.json
artifacts/model_search_results.csv
artifacts/end_to_end_eval_test_state_machine.json
artifacts/deploy_package/
artifacts/deploy_feature_extractor.py  ← 独立部署脚本，包含 FILL_VALUES + CLIP_BOUNDS
artifacts/deploy_cookbook.json         ← 自包含部署配方（信号预处理+特征公式+模型推理+Stage1 阈值）
artifacts/deploy_xgboost.json          ← 模型结构 JSON + fill_values + clip_bounds + preprocess_order
artifacts/per_sample_summary.csv
artifacts/error_plots/
```

默认命令不会生成：

```text
artifacts/window_outputs/
artifacts/postprocess_opt/
```

除非你手动运行 NPZ 缓存导出和后处理搜参。

## 手动导出 NPZ

如果要为后处理搜参或窗口诊断导出 valid 的逐窗结果：

```bash
python new_new/s06_deploy_eval.py --artifact_dir artifacts --split valid --export_window_cache --window_output_root window_outputs
```

如果在 `new_new` 目录下：

```bash
python s06_deploy_eval.py --artifact_dir artifacts --split valid --export_window_cache --window_output_root window_outputs
```

输出：

```text
artifacts/window_outputs/valid/*.npz
artifacts/window_outputs/valid/manifest.csv
```

NPZ 里包含：

```text
sample_name
target
prob_raw
pred_raw
stage1_enabled
quality
ood_rate
stage1_dc
stage1_acdc
stage1_dc_margin
stage1_acdc_margin
window_start_sec
window_end_sec
model_threshold
window_sec
stride_sec
model_fingerprint_json
feature_names_json
```

对于新 H5 格式，`window_start_sec/window_end_sec` 会按窗口名里的 `w编号` 写入，而不是按 H5 存储顺序猜。

## 手动后处理搜参

基于上一步 NPZ 缓存搜索状态机参数：

```bash
python new_new/s07_postprocess_optimize.py --artifact_dir artifacts --split valid --cache_root window_outputs --fp_cost 4.0
```

如果在 `new_new` 目录下：

```bash
python s07_postprocess_optimize.py --artifact_dir artifacts --split valid --cache_root window_outputs --fp_cost 4.0
```

可选：

```bash
python new_new/s07_postprocess_optimize.py --artifact_dir artifacts --split valid --cache_root window_outputs --fp_cost 4.0 --skip_initial_windows 1 --thresholds 0.3,0.4,0.5,0.6,0.7,0.8
```

参数含义：

```text
--split
  用哪个 split 的缓存做搜参。
  推荐 valid，不建议用 test 搜参。

--cache_root
  NPZ 缓存目录名。
  实际读取路径是 artifacts/<cache_root>/<split>/。

--fp_cost
  假阳性惩罚权重。
  越大越倾向于减少负样本误判为正。

--skip_initial_windows
  搜参/诊断时跳过每条样本开头的几个窗口。
  只影响 s07 的后处理搜参，不会改变原始 H5 或 s01 的前三窗剔除。

--thresholds
  额外扫描窗口级概率阈值，用于诊断窗口阈值敏感性。
```

输出：

```text
artifacts/postprocess_opt/postprocess_search_valid.csv
artifacts/postprocess_opt/window_threshold_scan_valid.csv
artifacts/postprocess_opt/window_error_report_valid.csv
artifacts/postprocess_opt/window_error_summary_valid.csv
artifacts/postprocess_opt/postprocess_optimized_valid.json
```

同时，`s07` 会把最优 `postprocess` 写入：

```text
artifacts/final_model_config.json
```

后续再运行 `s06` 或默认 `s08` 时，会读取这个已保存的 `postprocess` 参数做评估；但这不代表默认命令重新进行了搜参。

## 部署一致性检查

最终交付工程化时，Stage2 窗口级识别必须和训练/测试流程使用同一套输入顺序和预处理：

```text
1. PPG 输入可以是 6 通道原始窗口，也可以是已经配对后的 3 通道虚拟窗口。
2. 6 通道原始 PPG 必须按训练侧 build_3ch_ppg 的规则配成 3 通道：
   ch_A = avg(ch0, ch1)
   ch_B = avg(ch2, ch4)
   ch_C = avg(ch3, ch5)
3. deploy_feature_extractor.py 生成的 FEATURE_ORDER 必须等于 selected_features。
4. 模型输入必须按 FEATURE_ORDER 排列。
5. 非有限值 inf/-inf 先按缺失值处理。
6. NaN/inf 使用 fill_values 填充。
7. fill 后再按 clip_bounds 对每个入选特征裁剪。
8. XGBoost 输出概率后，窗口级 0/1 判断使用 window_threshold。
```

### 训练与推理的预处理管道

训练侧（s05）和推理侧（s06 / deploy_feature_extractor.py）执行相同的预处理语义，仅在执行顺序上有细微差异：

```text
训练侧 (s05):
  raw features (from s03 CSV)
    → ① clip_outliers(k=1.5, IQR-based)
        用 train 的 Q1-1.5*IQR / Q3+1.5*IQR 裁剪极端值
        valid 用 train 的裁剪边界，避免 valid IQR 泄漏
    → ② prepare_fill_values
        对裁剪后的 train 计算每个特征的中位数
    → ③ apply_fill: inf → NaN → fillna(train median)
    → ④ XGBoost 训练

推理侧 (s06 apply_preprocess):
  raw features (from s03 extract_feature_pool_from_window)
    → ① inf → NaN
    → ② fillna(fill_values)
    → ③ clip(clip_bounds)
    → ④ XGBoost 推理

推理侧 (deploy_feature_extractor.py, 独立部署脚本):
  raw features (standalone extraction)
    → ① inf/None → fill_values (FILL_VALUES dict)
    → ② clip(clip_bounds) (CLIP_BOUNDS dict)
    → ③ 返回特征向量
```

关键差异说明：

```text
训练侧先 clip 再 fill，推理侧先 fill 再 clip。
fill_values 是 train median（通常落在 clip 范围内），因此顺序差异在绝大多数情况下不影响结果。

训练侧的 fill_values 是在 clip_outliers 之后重新计算的（s05 prepare_fill_values），
而非复用 s04 clean_features_by_train 的 fill_values。
s04 的 fill_values 保存在 selected_features.json 的 train_fill_values 字段中，
仅供诊断参考，实际部署使用的是 model_bundle.pkl 中的 fill_values。
```

### clip_outliers 参数

```text
方法: IQR-based 异常值裁剪
k = 1.5 (标准 Tukey 参数)
训练时从 train 学边界 → 用 train 边界裁剪 train 和 valid
valid 不自算 IQR（防止数据泄漏）

当前 s03 特征提取未直接调用 clip_outliers；
裁剪只发生在 s05 训练前的 DataFrame 层面。
推理侧通过 clip_bounds JSON 应用相同的裁剪逻辑。
```

### s03 特征提取中的 NaN/inf 处理

```text
s03 extract_feature_pool_from_window 末尾（第 1799 行）：
  if v is None or not np.isfinite(v):
      feat[k] = 0.0

这是特征提取层面的"兜底"处理，所有无法计算的值（除零、log(0)、空信号等）
在写入特征池 CSV 之前就被替换为 0.0。

这意味着：
  - s04/s05 从 CSV 读取特征时已经看不到 NaN/inf（大部分已被 s03 消除）
  - 下游的 inf→NaN→fill 链是防御性的额外保护（处理 s03 未覆盖的边界情况）
  - 0.0 可能偏离特征分布的合理默认值，但会被 s05 clip_outliers 收到 IQR 下界

训练和推理使用相同的 s03 代码，因此这一行为在两端一致，
不构成训练-部署 gap。
```

### 部署产物的预处理信息

三个部署产物都包含完整的 fill + clip 信息，但形式不同：

```text
1. deploy_package/model_params.json
   {
     "fill_values": {...},          ← 每个特征的 train median
     "clip_bounds": {...},          ← 每个特征的 [lower, upper] IQR 边界
     "preprocess_order": [
       "select selected_features in order",
       "fill NaN/inf with fill_values",
       "clip each selected feature by clip_bounds"
     ]
   }

2. deploy_xgboost.json
   {
     "feature_names": [...],
     "feature_order": [...],
     "fill_values": {...},
     "clip_bounds": {...},
     "preprocess_order": ["select feature_order", "fill NaN/inf with fill_values", "clip by clip_bounds"],
     "n_estimators": ...,
     "threshold": ...,
     "model": {...}                 ← XGBoost booster JSON
   }

3. deploy_feature_extractor.py（独立 Python 脚本）
   - FILL_VALUES = {...}            ← 硬编码字典
   - CLIP_BOUNDS = {...}            ← 硬编码字典
   - 输出向量构建时先 fill 再 clip
   - 零外部依赖（仅 numpy + scipy）
```

### 部署 bundle 完整内容

`model_bundle.pkl`（joblib 序列化）包含以下字段：

```text
version: "v2"
feature_names: [...]               ← 入选特征名列表（FEATURE_ORDER）
fill_values: {feat: median}        ← train median（clip_outliers 之后计算）
clip_bounds: {feat: [lo, hi]}      ← IQR 裁剪边界（train 学得）
scaler: null                       ← 不使用标准化
model: XGBClassifier               ← 训练好的 XGBoost 模型
threshold: float                   ← valid 固化的窗口概率阈值
threshold_policy: {...}            ← 阈值选择策略
quality_thresholds: {...}          ← 质量评分阈值
feature_quantiles: {...}           ← OOD 监控分位数
fingerprint: {...}                 ← 数据/代码版本指纹
model_search: {...}                ← 模型搜索记录
xgboost_complexity: {total_nodes, avg_nodes_per_tree, max_model_nodes}
meta: {
  fs_ppg, fs_emg, fs_acc           ← 采样率
  win_sec, step_sec                ← 窗口参数
  n_ppg_channels, n_emg_channels, n_acc_channels
  ppg_mode: "6ch_avg_single_channel"
  emg_notch_config: {
    notch_freqs_hz: [50,100,150,200,250,300]
    notch_bw_hz: 0.8
    leak_freqs_hz: [100,150,200,250,300]
  }
}
```

### 部署步骤

如果先运行 `s07_postprocess_optimize.py` 做了后处理搜参，再要交付 Stage3 状态机参数，必须重新运行一次 `s06_deploy_eval.py --export_deploy` 或默认 `s08` 的部署导出步骤，让：

```text
artifacts/deploy_package/postprocess_config.json
```

同步 `artifacts/final_model_config.json` 里的最新 `postprocess`。

## 各脚本说明

```text
s01_data_split.py
  扫描 H5，识别每条原始数据，按 sample 级别切分 train/valid/test。
  对当前嵌套 H5，会按 w编号 排序并去掉前三个窗口。

s02_ir_dc_threshold.py
  生成 Stage1 阈值配置和 train/valid 诊断 CSV。
  当前默认使用固定工程阈值，不在 train/valid 上搜索 Stage1 阈值。

s03_extract_feature_pool.py
  提取 Stage2 特征池。
  对预切好的 3s 窗口不再二次切分。
  特征提取末尾会将 None / NaN / inf 替换为 0.0（兜底处理）。
  下游 s05/s06 的 fill+clip 链会在此基础上进一步处理极端值。

s04_feature_selection.py
  做特征清洗、相关性/VIF 去冗余、稳定性选择，输出 selected_features.json。

s05_train_final_model.py
  训练 XGBoost。
  默认开启模型参数搜索，使用 train 内部 group CV 选择模型参数，使用 valid 固化窗口阈值。

s06_deploy_eval.py
  端到端评估、部署产物导出、可选 NPZ 缓存导出、可选内置状态机参数优化。

s07_postprocess_optimize.py
  独立后处理搜参脚本。
  必须先有 s06 导出的 NPZ 缓存。

s08_run_pipeline.py
  一条命令串起默认全流程。
  额外生成 deploy_feature_extractor.py（独立部署特征提取脚本，
  包含 FILL_VALUES + CLIP_BOUNDS + 完整预处理逻辑）、
  deploy_cookbook.json（部署配方）和 deploy_xgboost.json（模型+预处理参数）。
```

## 输出指标字段

`end_to_end_eval_test_state_machine.json` 中主要字段：

```text
summary
  样本级端到端指标。

details
  每条样本的预测细节。

window_model_summary
  Stage2 逐窗模型指标。

window_stream_summary
  Stage2+3 状态机逐窗/流式参考指标。

ood_summary
  特征 OOD 比例统计。
```

端到端 summary 常见字段：

```text
total_samples
  当前 split 的样本数。

stage1_pass_samples
  通过 Stage1 的样本数。

fallback_samples
  加载、特征提取或推理失败后走 fallback 的样本数。

confusion_matrix
  样本级 TN/FP/FN/TP。

accuracy / precision / recall / f1
  样本级端到端指标。

postprocess
  本次评估使用的状态机参数。

model_threshold
  Stage2 窗口概率阈值。
```

Stage2 模型指标常见字段：

```text
total_input_samples
  输入 s06 的样本数。

stage1_pass_samples
  通过 Stage1、有机会进入 Stage2 的样本数。

samples_with_no_windows
  没有可用 Stage2 窗口的样本数。

total_windows
  参与 Stage2 模型评估的窗口数。

accuracy / precision / recall / f1
  窗口级模型指标。
```

Stage2+3 状态机指标常见字段：

```text
warmup_frames
  评估流式窗口状态时跳过的开头窗口数。
  s06 默认 3。

skipped_warmup_windows
  所有样本合计跳过的窗口数。

total_windows
  warmup 后参与统计的窗口数。

accuracy / precision / recall / f1
  状态机逐窗状态的参考指标。
```

## 推荐使用方式

第一次完整训练和评估（仅 Stage2 模型 + 默认后处理参数）：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

完整流程（含 NPZ 缓存 + 后处理状态机搜参），一条命令：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess
# 等效于: --export_window_cache --optimize_postprocess
```

跳过某些步骤（复用已有产物）：

```bash
python new_new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s05
```

只跑到特征筛选后停止：

```bash
python new_new/s08_run_pipeline.py --stop_after s04
```

预览命令不执行（dry run）：

```bash
python new_new/s08_run_pipeline.py --dry_run
```

如果要确认不是读取旧的后处理参数，换一个新的输出目录：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts_fresh
```

如果只想快速验证链路，减少并行和模型搜索耗时：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts_debug --n_workers 1 --no-model_search
```

如果默认模型搜索太慢：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --model_search_max_candidates 300 --model_search_stage2_top_k 40
```

如果需要手动分步做后处理搜参（等价于 `--with_postprocess`）：

```bash
python new_new/s06_deploy_eval.py --artifact_dir artifacts --split valid --export_window_cache --window_output_root window_outputs
python new_new/s07_postprocess_optimize.py --artifact_dir artifacts --split valid --cache_root window_outputs --fp_cost 4.0
python new_new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s05
```

评估其他 split（如 train 或 test）：

```bash
python new_new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s05 --split valid
```

调整后处理搜参的 FP 惩罚权重：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess --postprocess_fp_cost 2.0
```

如果需要测试不同特征数量（需先跑过一次 s04 生成 ranked_features.json）：

```bash
python new_new/s05_train_final_model.py --artifact_dir artifacts --max_features 10
python new_new/s05_train_final_model.py --artifact_dir artifacts --max_features 20
```

## 验证命令

```bash
python -m pytest tests -q
python -m py_compile s01_data_split.py s02_ir_dc_threshold.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py
python s06_deploy_eval.py --help
python s07_postprocess_optimize.py --help
python s08_run_pipeline.py --help
```
