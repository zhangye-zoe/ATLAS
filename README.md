# ATLAS Table 3: HSPC T3/T4 复现扩展（独立模块）

## 先读这个：复现范围与真实限制

- 数据集：`HSPC_2M`、`HSPC_RNA`、`HSPC_MAC`。
- 原始数据路径与旧 `VelocityBench/ALTAS/scripts_V4/` 三个 notebook 一致；**代码包不包含 h5ad 数据**。
- 输入三模态：RNA `layers["Mu"]`（unspliced）、RNA `layers["Ms"]`（spliced）、ATAC `layers["Mc"]`（chromatin）；按 cell barcode 和 gene name 对齐。
- **新模型是当前仓库 `atlas.model.ATLAS`**（三模态初始化），不是旧 `tide_velocity_benchmark_model_v3_t4optimized.py`。`src/atlas/hspc_table3.py` 是独立训练/评估器，旧 Table 2 和 E1–E4 不会受影响。
- **T3**：Graph/Diffusion/Attribution teacher 在训练细胞中计算（最多 4000 landmarks），蒸馏训练 TimeHead，评估 held-out 细胞 POA/POAUC。时间方向仅使用 **训练集的** stage 标签决定，模型损失不使用 stage 标签。
- **T4**：Phase III 训练潜变量局部位移预测头（latent velocity proxy），构建训练集参考细胞之间的 forward-time kNN 马尔可夫转移图；从训练集终末细胞吸收得到未来命运得分，再预测独立测试细胞，输出 macro-F1/AUROC 及 time-only 消融。
- **重要：T4 的 `future_map` 来自当前细胞类型对应的命运类别，这是 fate proxy，不是实验观测的真实未来状态！** 不能将此处生成的数字直接作为论文 Table 3 的正式原始复现结果；需要原始未来命运标注/实验划分和 T4 目标定义才能严格对齐。旧脚本某些计算使用测试真实细胞类型构建终末参考，存在泄漏；此版本不会如此做。

## 安装：不会覆盖原来的 `trainer.py`

在本地下载 zip 文件后传送到服务器（例如使用 `scp`），再于服务器执行：

```bash
cd /data5/zhangye/ATLAS
unzip /tmp/ATLAS_Table3_HSPC_T3_T4.zip -d .
python tests/test_protocol_static.py
python tests/test_pure_metrics.py
python -c 'from atlas.hspc_table3 import HSPCConfig; print(HSPCConfig())'
```

注意：最后一行需要先 `pip install -e .` 或 `export PYTHONPATH="$PWD/src:$PYTHONPATH"`。

推荐：

```bash
cd /data5/zhangye/ATLAS
export PYTHONPATH="$PWD/src:$PYTHONPATH"
pip install anndata scipy scikit-learn torch pandas matplotlib umap-learn nbformat
```

如你的 E1–E4 实验仍在运行，请**先只检查文件和 notebook，不要同时在相同 GPU 运行 Table 3 正式训练**。

## 数据文件

保持源文件在 `/data5/zhangye/VelocityBench/data/`：

| 数据集 | RNA 文件 | ATAC 文件 |
|---|---|---|
| HSPC_2M | `3423-MV-2-8489-MV-1_adata_postpro_concat.h5ad` | `3423-MV-2-8489-MV-1_adata_atac_postpro_concat.h5ad` |
| HSPC_RNA | `3423-MV-2-8489-MV-1-GSM5460410_adata_postpro_concat.h5ad` | `3423-MV-2-8489-MV-1-GSM5460410_adata_atac_postpro_concat.h5ad` |
| HSPC_MAC | `8489-MV-1-9060-MV-3_adata_postpro_concat.h5ad` | `8489-MV-1-9060-MV-3_adata_atac_postpro_concat.h5ad` |

如文件在其他路径，在 notebook 修改 `DATA_ROOT` 或 CLI 使用 `--data-root`。

## 配套 annotation 配置，务必核对

- `configs/HSPC_2M_annotations.json`：取自旧 V2 notebook 的通用 HSPC stage/fate mapping；V4 的 HSPC_2M notebook 仅引用外部 JSON，**这些旧 JSON 不在你本次 ZIP 里**，此文件不保证与原始实验逐项相同。
- `configs/HSPC_RNA_annotations.json`：参照旧 V4 RNA notebook cell 7 的内联映射。
- `configs/HSPC_MAC_annotations.json`：参照旧 V4 MAC notebook cell 7 的内联映射（含 Monocyte / macrophage / MDP）。

运行 notebook 第二节审查 `celltype` 值、stage/fate 映射是否齐全；无法映射的细胞会从相关指标中排除，`data_audit.json` 会显式记录。不要默默把未知 label 当作某一 stage。

## 三个 notebooks（按数据集分别训练）

```text
notebooks/table3_hspc/HSPC_2M_T3_T4.ipynb
notebooks/table3_hspc/HSPC_RNA_T3_T4.ipynb
notebooks/table3_hspc/HSPC_MAC_T3_T4.ipynb
```

每个 notebook 包含：环境/数据检查 → H5AD 和 annotations 检查 → 模型训练 → T3/T4 指标 → T3 可视化 → T4 预测与诊断 → 全数据汇总。

默认 `QUICK = True` 是小规模功能测试（2/1/2 epoch）；确认通过后改成 `False`，开始完整 80/40/60 epochs。

## 不用 notebook 的 CLI（训练逻辑完全一致）

```bash
cd /data5/zhangye/ATLAS
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 仅功能检查：
python experiments/run_table3_hspc.py --dataset HSPC_2M --device cuda:0 --quick

# 正式训练：
python experiments/run_table3_hspc.py --dataset HSPC_2M --device cuda:0
python experiments/run_table3_hspc.py --dataset HSPC_RNA --device cuda:0
python experiments/run_table3_hspc.py --dataset HSPC_MAC --device cuda:0

python experiments/summarize_table3_hspc.py
```

结果分别输出到 `results/table3_hspc/{DATASET}/{quick|full}/seed_1234`。每个数据集会保存：

- `config.json`, `data_audit.json`, `split_manifest.csv`：配置、标签覆盖及训练/测试样本列表。
- `checkpoint_phase1_last.pt`, `checkpoint_phase2_last.pt`, `checkpoint_phase3_last.pt`, `training_history.csv`：训练阶段检查点/损失记录。
- `T3_metrics.json`, `T3_test_predictions.npz`, `T3_trajectory.png`：T3 指标与可视化。
- `T4_metrics.json`, `T4_heldout_predictions.csv`, `T4_transition_diagnostics.json`：T4 和 time-only 消融。
- `summary.json`：最终指标及协议声明。

最后生成 `results/table3_hspc/Table3_reproduced_summary.csv`。

## 进一步改善方向

1. 通过 `--quick` 和完整阶段验证三模态数据与 loss 接口。不要用 quick 指标对照论文。
2. 比较使用 `TimeHead` 与 `graph_auto`、DPT 等估计器；选择估计器不得使用报告集标签。
3. 添加明确的实际未来命运标注、配对时间点或独立 lineage tracing 数据，从 fate proxy 升级为真正的 T4 future-state 评估。
4. 旧 V4 的 TIDE 模型还包含 batch-adversarial、DPT-anchored temporal teacher、特定优化的 transition；当前 ATLAS 版本**没有原样移植**，不同结果不应归因于单个模块。
5. 保持 test split、标签解释、future fate 类集合和 feature selection 规则固定，并进行 3+ seeds。

## 版本与验证

已检查 3 个 notebook 的所有 code cell 的 Python 语法、模块编译、合成样本 T3 ordering 与 T4 terminal absorption 数值，未在真实服务器数据/GPU 上完整训练。HSPC 的实际 `h5ad`、batch 与 celltype 字段必须在服务器上运行 notebook 第二节检查。
