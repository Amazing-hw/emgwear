# Wearing Liveness Detection

手表佩戴活体检测项目，输入 H5 格式的 PPG、EMG、ACC 窗口数据，训练 XGBoost 窗口级模型，并用 Stage1 门控和 Stage3 状态机给出样本级端到端结果。当前推荐入口是 `s08_run_pipeline.py`。

## 一条命令跑完整功能

如果你已经在 `D:\wearing_liveness\new_new` 目录下：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess
```

如果你在上一级项目目录 `D:\wearing_liveness` 下：

```bash
python new_new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess
```

这条命令会完成：

```text
s01 数据扫描与 train/valid/test 固定切分
s02 Stage1 PPG DC/ACDC 工程阈值配置
s03 PPG/EMG/ACC 特征池提取
s04 特征清洗、相关性/VIF/稳定性筛选
s05 XGBoost 训练、train 内部 group CV 搜参、valid 窗口阈值固化
s06 导出 train/valid 逐窗 NPZ 缓存
s07 基于 train+valid hard samples 做后处理状态机和阈值 offset 联合搜参
s06 在 test 上做最终端到端评估
s06 导出部署包
s08 导出独立特征提取脚本、错误样本图、部署 cookbook
```

对应的 `s08` 内部步骤键名：

```text
s01 -> s02 -> s03 -> s04 -> s05 -> s06_cache_train -> s06_cache_valid -> s07_post -> s06_eval -> s06_xpt -> s06_feat -> s06_plot -> s06_cb
```

`--with_postprocess` 等效于：

```text
--accuracy_first_optimize --export_window_cache --optimize_postprocess
```

它会让 Stage2 模型按窗口准确率优先选择，再让 `s07_postprocess_optimize.py` 搜索 Stage3 状态机参数。

## 快速预览

只打印将要执行的命令，不真正运行：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess --dry_run
```

只跑默认流程，不做后处理搜参：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

快速调试链路，减少模型搜索耗时：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts_debug --n_workers 1 --no-model_search
```

如果模型搜索太慢：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --search_budget fast
```

如果希望更充分搜索：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess --search_budget accuracy
```

## 数据使用原则

本项目把 `test` 当作最终封闭评估集，不用于调参。

```text
train:
  特征筛选
  XGBoost 参数/特征数搜索的内部 group CV
  hard negative mining

valid:
  Stage2 窗口概率阈值选择
  与 train 缓存一起参与 Stage3 后处理搜参

train+valid:
  s07 默认只取窗口级不是全对的样本参与后处理搜参
  窗口全对样本只作为 guardrail 回放检查

test:
  最终端到端报告
  部署产物导出验证
  不参与模型搜参、阈值选择或后处理搜参
```

代码会拒绝下面这类高风险命令：

```bash
python s06_deploy_eval.py --optimize --optimize_split test
python s07_postprocess_optimize.py --split test
python s07_postprocess_optimize.py --search_splits test
```

## 输入 H5 格式

推荐格式是一个 H5 文件包含多条原始样本，每条样本下包含多个窗口 group：

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

示例：

```text
subject01_session02_w20_1
```

含义：

```text
窗口编号 = 20
label = 1
```

每个窗口内的数据形状：

```text
ppg: (6, 300)      # 6 通道 PPG，3s @100Hz
emg: (2, 3000)     # 2 通道 EMG，3s @1000Hz
acc: (3, 300)      # 3 通道 ACC，3s @100Hz
```

旧的预切数组格式仍支持：

```text
ppg: (num_windows, 6, 300)
emg: (num_windows, 2, 3000)
acc: (num_windows, 3, 300)
```

注意：

```text
同一条原始样本下的窗口 label 必须一致。
H5 中 group 存储顺序不可信，程序按名称里的 w 编号排序。
s01 会按窗口编号排序后去掉每条样本开头前三个窗口。
```

## 流程脚本

| 脚本 | 作用 |
| --- | --- |
| `s01_data_split.py` | 扫描 H5，按样本级固定切分 `train/valid/test` |
| `s02_ir_dc_threshold.py` | 生成 Stage1 PPG DC/ACDC 工程阈值配置 |
| `s03_extract_feature_pool.py` | 提取 PPG/EMG/ACC 窗口级特征池 |
| `s04_feature_selection.py` | 特征清洗、相关性/VIF、稳定性筛选 |
| `s05_train_final_model.py` | 训练 XGBoost，train 内部 group CV 搜参，valid 固化窗口阈值 |
| `s06_deploy_eval.py` | 端到端评估、部署包导出、逐窗 NPZ 缓存导出 |
| `s07_postprocess_optimize.py` | 基于 NPZ 缓存搜索 Stage3 状态机后处理参数 |
| `s08_run_pipeline.py` | 一键串联全流程 |
| `test_feature_report.py` | 生成测试集特征嵌入与诊断图 |

## Stage 说明

### Stage1: PPG 工程门控

Stage1 使用 6 通道 PPG 的均值信号：

```text
signal = mean(ppg_ch0 ... ppg_ch5)
fs = 100Hz
window = 3s = 300 samples
stride = 1s = 100 samples
```

部署门控规则：

```text
DC > dc_threshold AND AC_DC_RATIO < ac_dc_threshold
```

默认部署阈值：

```text
dc_threshold = 0.2e6
ac_dc_threshold = 1.0
```

### Stage2: XGBoost 窗口模型

Stage2 输入：

```text
PPG: 6ch -> build_3ch_ppg(...) -> 3ch @100Hz
EMG: 2ch @1000Hz
ACC: 3ch @100Hz
```

6 通道 PPG 到 3 通道虚拟 PPG 的规则：

```text
ch_A = avg(ch0, ch1)
ch_B = avg(ch2, ch4)
ch_C = avg(ch3, ch5)
```

`s05` 会在 valid 上选择窗口概率阈值，并保存到：

```text
artifacts/model_bundle.pkl
artifacts/final_model_config.json
```

### Stage3: 状态机后处理

Stage3 对窗口概率做时序后处理。默认参数来自代码，若 `final_model_config.json` 中有 `postprocess` 字段，则优先使用保存的参数。

`s07` 当前默认策略：

```text
search_splits = train,valid
hard_samples_only = true
threshold_offsets = -0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3
max_all_correct_regressions = 0
```

`s08` 会把 `--n_workers` 传给 `s07` 的 `--workers`。`s07` 的 exact 搜参会预计算 `(threshold_offset, median_k)` 窗口序列，避免每个候选重复做阈值变换和中值滤波；运行时会打印候选数、样本数、worker 数、预计算序列数、进度、耗时、ETA 和当前最优分数。

hard sample 定义：

```text
pred_window = prob_raw >= model_threshold AND stage1_enabled
如果某样本任一参与窗口 pred_window != sample_target，则该样本参与 s07 搜参。
窗口全对样本不参与搜索，只用于 guardrail 回放。
```

阈值 offset 进入状态机的方式：

```text
p_state = clip(prob_raw - (model_threshold + threshold_offset) + 0.5, 0, 1)
```

如果 guardrail 发现窗口全对样本被新状态机参数改错，`s07` 会保存诊断结果，但不会写回 `final_model_config.json`。

## 常用 s08 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset_dir` | `dataset` | H5 数据目录 |
| `--artifact_dir` | `artifacts` | 输出产物目录 |
| `--n_workers` | CPU 半数，上限 4 | 并行 worker 数 |
| `--max_features` | `15` | 最终入选特征数 |
| `--window_sec` | `3` | Stage2 窗口秒数 |
| `--stride_sec` | `1` | Stage2 窗口步长秒数 |
| `--model_search / --no-model_search` | 开启 | 是否启用 XGBoost 搜参 |
| `--search_budget` | `balanced` | `fast`、`balanced`、`accuracy` |
| `--with_postprocess` | 关闭 | 开启完整后处理搜参流程 |
| `--postprocess_fp_cost` | `1.5` | s07 假阳性惩罚权重 |
| `--postprocess_threshold_offsets` | `-0.3,...,0.3` | s07 联合搜索的阈值偏移 |
| `--split` | `test` | s06 最终评估 split |
| `--skip` | 空 | 跳过步骤，逗号分隔 |
| `--stop_after` | `s06_cb` | 跑到指定步骤后停止 |
| `--dry_run` | 关闭 | 只打印命令 |

搜索预算：

```text
fast:
  model_search_max_candidates = 150
  model_search_stage2_top_k = 20
  model_search_cv_repeats = 1

balanced:
  model_search_max_candidates = 300
  model_search_stage2_top_k = 40
  model_search_cv_repeats = 3

accuracy:
  model_search_max_candidates = 600
  model_search_stage2_top_k = 80
  model_search_cv_repeats = 5
```

## 手动分步命令

完整训练到模型：

```bash
python s01_data_split.py --dataset_dir dataset --artifact_dir artifacts
python s02_ir_dc_threshold.py --artifact_dir artifacts
python s03_extract_feature_pool.py --artifact_dir artifacts
python s04_feature_selection.py --artifact_dir artifacts --max_features 15
python s05_train_final_model.py --artifact_dir artifacts --model_search
```

导出 train/valid 逐窗缓存：

```bash
python s06_deploy_eval.py --artifact_dir artifacts --split train --export_window_cache --window_output_root window_outputs
python s06_deploy_eval.py --artifact_dir artifacts --split valid --export_window_cache --window_output_root window_outputs
```

后处理搜参：

```bash
python s07_postprocess_optimize.py --artifact_dir artifacts --search_splits train,valid --cache_root window_outputs --fp_cost 1.5 --workers 4 --hard_samples_only --threshold_offsets=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3
```

临时调试搜参速度时可以加 `--max_candidates 200 --progress_interval 20`，正式结果不要限制 `--max_candidates`。

最终 test 评估和部署导出：

```bash
python s06_deploy_eval.py --artifact_dir artifacts --split test
python s06_deploy_eval.py --artifact_dir artifacts --split test --export_deploy
python s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s05 --split test
```

## 主要输出产物

默认流程会生成：

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
artifacts/deploy_feature_extractor.py
artifacts/deploy_cookbook.json
artifacts/deploy_xgboost.json
artifacts/per_sample_summary.csv
artifacts/error_plots/
```

使用 `--with_postprocess` 还会生成：

```text
artifacts/window_outputs/train/*.npz
artifacts/window_outputs/train/manifest.csv
artifacts/window_outputs/valid/*.npz
artifacts/window_outputs/valid/manifest.csv
artifacts/postprocess_opt/postprocess_search_train_valid.csv
artifacts/postprocess_opt/window_threshold_scan_train_valid.csv
artifacts/postprocess_opt/window_error_report_train_valid.csv
artifacts/postprocess_opt/window_error_summary_train_valid.csv
artifacts/postprocess_opt/postprocess_optimized_train_valid.json
```

部署相关产物：

```text
artifacts/deploy_package/
artifacts/deploy_package/stage1_config.json
artifacts/deploy_package/model_params.json
artifacts/deploy_package/postprocess_config.json
artifacts/deploy_feature_extractor.py
artifacts/deploy_cookbook.json
artifacts/deploy_xgboost.json
```

## 评估 JSON 字段

`end_to_end_eval_test_state_machine.json` 主要包含：

```text
summary
  样本级端到端指标

details
  每条样本的预测细节

window_model_summary
  Stage2 窗口级模型指标

window_stream_summary
  Stage2+3 流式窗口状态指标

ood_summary
  特征 OOD 比例统计
```

常用样本级字段：

```text
total_samples
stage1_pass_samples
fallback_samples
confusion_matrix
accuracy / precision / recall / f1
postprocess
model_threshold
```

## 测试集特征可视化

```bash
python test_feature_report.py --artifact_dir artifacts
python test_feature_report.py --artifact_dir artifacts --methods pca,tsne --max_points 500 --dpi 200
```

输出默认写入：

```text
artifacts/test_feature_report/
```

## 验证命令

开发或交付前建议运行：

```bash
python -m pytest -q
python -m py_compile deploy_feature_contract.py s01_data_split.py s02_ir_dc_threshold.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py test_feature_report.py
python s01_data_split.py --help
python s02_ir_dc_threshold.py --help
python s03_extract_feature_pool.py --help
python s04_feature_selection.py --help
python s05_train_final_model.py --help
python s06_deploy_eval.py --help
python s07_postprocess_optimize.py --help
python s08_run_pipeline.py --help
python test_feature_report.py --help
```

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

```text
numpy
scipy
pandas
scikit-learn
xgboost
h5py
matplotlib
joblib
pytest
```

## 常见操作

只跑到特征筛选：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --stop_after s04
```

复用已有模型，只重新评估和导出：

```bash
python s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s05 --split test
```

评估 valid：

```bash
python s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s05 --split valid
```

调整后处理 FP 惩罚：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess --postprocess_fp_cost 2.0
```

调整后处理阈值 offset 搜索范围：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --with_postprocess --postprocess_threshold_offsets=-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4
```

换一个全新输出目录，避免读取旧配置：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts_fresh --with_postprocess
```

## 注意事项

```text
1. 不要用 test 做模型参数、窗口阈值或后处理参数选择。
2. 如果已经根据 test 结果反复调参，当前 test 应视为开发集，最终报告需要新 lockbox test。
3. s07 的 test split 会被拒绝。
4. s06 默认会读取 final_model_config.json 中已有 postprocess 参数。
5. s07 guardrail 未通过时不会写回 final_model_config.json，需要查看 postprocess_optimized_train_valid.json 的 write_skipped_reason。
6. 部署前若重新跑过 s07，需要重新跑 s06 --export_deploy 或 s08 默认部署导出步骤，让 deploy_package/postprocess_config.json 同步最新参数。
```
