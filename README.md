# CapePart

CapePart 是一个针对零样本/小样本部件分割（few-shot part segmentation）的训练与推理框架。核心流程包括 episode 级别的数据加载、CapeFormer 主干网络以及推理可视化工具。

## 模块结构与执行顺序

1. **配置加载**（`Config/default.py` 或自定义配置）
   - 关键参数：
     - `DATASET.ROOT`：数据根目录。
     - `DATASET.TRAIN/VAL/TEST`：episode 标注 JSON 文件名（或 `AUTO_BUILD` 自动生成）。
     - `DATASET.NUM_SHOTS / NUM_QUERIES`：每个 episode 的 support/query 样本数。
     - `DATASET.IMG_SIZE`、`DATASET.MEAN`、`DATASET.STD`：图像与掩码预处理尺寸与归一化参数。
     - `TRAIN.BATCH_SIZE`、`TRAIN.NUM_WORKERS`：训练 DataLoader 的批大小与加载线程数。
     - `MODEL.*`：CapeFormer 网络结构参数（`D_MODEL`、`NHEAD`、`NUM_ENCODER_LAYERS` 等）。
     - `MODEL.JOINT_ENCODER_LAYERS`：默认值为 3，表示开启 Query-Support 联合精炼编码层；若设为 0 则完全关闭该模块。
     - `MODEL.USE_INFO_FUSION`：是否在第二阶段微调时启用 InformationFusion 特征融合模块。
     - `EVAL.THRESHOLD`：推理阶段掩码二值化阈值。
     - `SYSTEM.DEVICE`：训练/推理设备。

2. **数据构造**（`Myprogram/Dataloader/Dataloader.py`）
   - `build_dataloader` 根据配置选择对应的注释文件并实例化 `FewShotSegmentationDataset`。
   - `FewShotSegmentationDataset.__getitem__` 的运行顺序：
     1. 解析 episode 中 support/query 的图像与多通道 PNG 掩码路径。
     2. 使用 `decode_multi_channel_mask` 将单个 PNG 拆分为多通道二值图。
     3. 调用 `_sample_entries` 随机或顺序抽取 support/query 样本。
     4. `EpisodeAugmentor` 负责图像增广与掩码 dropout。
     5. `_align_support_query_masks` 将 support/query 掩码通道按照部件名称或索引对齐，缺失通道补零，保证输出一致。
     6. 输出对齐后的张量、原图路径及对齐信息字典，供模型与调试使用。

3. **模型前向**（`Myprogram/Backbone/model.py`）
   - `CapeFormerSegmentation` 的主要流程：
     1. `CapeFormerBackbone` 提取 support/query 特征，可选 `MultiScaleFeatureFusion` 融合多尺度特征。
     2. `input_proj` 将 Backbone 输出映射到 Transformer 维度，`PositionEmbeddingSine` 生成位置编码。
     3. `_compute_prototypes` 利用对齐后的 support 掩码计算每个部件的原型向量。
    4. `CapeFormerTransformer`（默认启用的 `QuerySupportJointEncoder` 以及可选 `InformationFusion`）交互 support/query 表征；其中 `QuerySupportJointEncoder` 负责联合精炼 support/query token，而 `InformationFusion` 仅在配置 `MODEL.USE_INFO_FUSION=True` 的二阶段训练/微调流程中启用，用于进一步融合查询编码与记忆特征。
     5. `CapeFormerMaskHead` 将 Transformer 输出恢复为查询掩码，并插值回原图大小。

4. **训练脚本**（`Myprogram/Train.py`）
   - 加载配置 → 构建 DataLoader → 实例化 `CapeFormerSegmentation` → 迭代 episode 计算损失、反向传播并保存 checkpoint。

5. **评估脚本**（`Myprogram/Test.py`）
   - 复用同一配置与数据加载逻辑，加载训练好的模型进行指标评估。

6. **单 episode 推理**（`Myprogram/TestSingleEpisode.py`）
   - 接受显式的 support/query 图像与多通道 PNG 掩码路径。
   - 输出原尺寸的 per-part 概率图、二值掩码以及与查询图叠加的可视化结果。

## 运行示例

### 安装依赖

```bash
pip install -r Myprogram/requirements.txt
```

### 训练

```bash
PYTHONPATH=Myprogram python Myprogram/Train.py \
  --config Myprogram/Config/default.py \
  --model-dir ./Checkpoints \
  --log-dir ./Logs
```

### 验证或测试

```bash
PYTHONPATH=Myprogram python Myprogram/Test.py \
  --config Myprogram/Config/default.py \
  --checkpoint ./Checkpoints/best.pth
```

### 单 episode 推理与可视化

```bash
PYTHONPATH=Myprogram python Myprogram/TestSingleEpisode.py \
  --config Myprogram/Config/default.py \
  --checkpoint ./Checkpoints/best.pth \
  --support-image /path/to/support.jpg \
  --support-mask /path/to/support_mask.png \
  --query-image /path/to/query.jpg \
  --output-dir ./single_episode_output
```

执行完成后，`single_episode_output/` 将包含：

- `pred_masks.npz`：查询图上每个部件的概率图。
- `*_mask.png`：阈值化后的单部件掩码。
- `overlay.png`：与原始查询图融合的部件可视化。

## 数据格式提醒

- 每张 support/query 图像必须配套一个“单 PNG，多通道”掩码文件；各通道分别对应一个部件。
- 若 episode 中 support 与 query 的部件集合不一致，数据集会自动补零/裁剪并对齐通道，确保模型输入始终匹配。
- 若缺少 episode JSON，可使用 `python -m tools.build_episode_json` 依据文件夹结构自动生成。
- 数据根目录既可以直接按 `<ROOT>/<超类>/<子类>/images|masks` 排列，也可以包含显式的划分层级（例如 `<ROOT>/train/<超类>/<子类>/images|masks`）；加载器会优先进入与 `split` 对应的子目录收集子类。
- 当 JSON 中混入了不同子类的 support/query 图像时，DataLoader 会自动过滤并仅保留与 support 同属一个子类的样本，防止训练时跨类配对。

## 常见问题排查

- **报错 `default_collate: batch must contain tensors ... found <class 'NoneType'>`**：
  - 该错误来自 PyTorch `DataLoader` 的默认 collate 逻辑，表示批次中存在 `None` 值。早期版本在记录 support/query 部件对齐信息时，如果某个部件缺失会写入 `None` 占位符，从而在多进程加载时触发此异常。
  - 当前版本已将此类占位符统一替换为 `-1`，既能保留对齐调试信息，也不会被 collate 视为非法类型。如仍遇到该报错，请检查是否加载了旧的缓存或本地修改，确保使用最新的 `Myprogram/Dataloader/Dataloader.py`。
  - 若日志中伴随 `WARN zero-area mask after downsample`，说明该部件在当前分辨率下被完全滤除。这类样本会自动补零，不影响训练流程，但建议复核原始掩码以确认是否存在标注缺失。
- **日志反复提示 `WARN zero-area mask after downsample`**：
  - DataLoader 会先读取原始掩码 → 最近邻缩放至 `DATASET.IMG_SIZE` → 触发 `EpisodeAugmentor.refine_mask_tensor` 的最小面积/随机擦除逻辑 → 最后检查像素面积。
  - 警告信息会区分三类来源：
    - `zero-area mask after downsample`：源掩码在读取阶段即为空，多半意味着该部件在原图中缺失或 JSON 中声明了多余通道。
    - `mask below min-area threshold after downsample`：掩码存在但像素少于 `DATASET.MIN_MASK_AREA` 设定；查询样本会被清零以避免噪声，可通过下调阈值（设为 0 即完全关闭）或增大输入尺寸缓解。
    - `min-area threshold cleared support mask; restored original`：support 掩码像素虽低但已自动恢复为缩放前的结果，不会影响后续对齐，只是提示阈值过严。
  - 若需要进一步保留细节，可适当提高 `DATASET.IMG_SIZE` 或在数据准备阶段对关键部件进行上采样/膨胀处理。
