# GEO 营销场景下的实时深度伪造检测（最小可跑通版）

按《GEO 营销场景下的实时深度伪造检测》开题报告落地的工程骨架。当前版本已完成**图像/视频帧级检测 + 流式音频检测 + 多模态融合 + 评估脚本 + REST API**；任一模态权重缺失时融合层自动降级。

## 与开题报告的对应关系

| 开题报告 3.3 模块 | 实现位置 | 状态 |
| --- | --- | --- |
| 预处理模块（人脸裁剪、抽帧、多维增强） | `geo_deepfake/data/preprocess.py` | 完成（Haar 级联，可选 MTCNN） |
| 视觉检测模块（帧级识别 + 时序聚合） | `geo_deepfake/models/visual.py` | 完成（timm 轻量骨干，mean/attn 聚合） |
| 音频检测模块（流式语音克隆检测） | `geo_deepfake/models/audio.py`、`data/audio.py` | 完成（log-Mel + 60.5k 参数 CNN，1.5 s 窗 / 0.5 s 跳步） |
| 多模态融合模块（跨模态特征融合 + GEO 可信评分） | `geo_deepfake/models/fusion.py` | 概率级加权融合 + 单模态降级 |
| 实时推理引擎（量化与加速推理） | `engine/predictor.py`、`engine/onnx_engine.py`、`scripts/export_onnx.py` | 完成（ONNX 导出 + INT8 静态 QDQ，1.46×） |
| 评估与接口层（指标、可视化、REST API） | `engine/metrics.py`、`api/server.py`、`scripts/visualize.py` | 完成（指标 + API + 离线图表） |

> 注：开题报告图 1 把 GEO 写成"地理空间 (GEO) 特征提取"，与正文的 Generative Engine Optimization 口径冲突。本实现按正文口径，把 GEO 信号实现为营销内容的**来源凭证完备度 / 内容重复度**等可信性元数据，不涉及地理坐标。

## 目录结构

```
configs/default.yaml           # 全部超参（对应 3.4 环境配置）
geo_deepfake/
  config.py                    # dataclass 配置加载
  data/preprocess.py           # 人脸裁剪 / 抽帧 / JPEG·模糊·亮度增强
  data/dataset.py              # manifest.csv 读写 + 帧级 Dataset
  data/audio.py                # 音频 IO / log-Mel / 流式分块
  data/audio_dataset.py        # 音频清单 + 分块级 Dataset
  models/visual.py             # 轻量骨干 + 帧级分类 + 时序聚合
  models/audio.py              # 流式音频检测（log-Mel + 轻量 CNN）
  models/fusion.py             # 加权融合 + GEO 可信评分
  engine/metrics.py            # AUC / EER / ACC / P / R / F1
  engine/predictor.py          # 端到端推理
  engine/onnx_engine.py        # ONNX Runtime 推理封装
  api/server.py                # FastAPI：/healthz、/predict/image|video|audio
scripts/
  make_dummy_data.py           # 合成冒烟数据（无需数据集）
  build_manifest.py            # FF++ / Celeb-DF / 通用目录 → manifest.csv
  prepare_wilddeepfake.py      # HuggingFace WildDeepfake 分片 → 帧清单
  prepare_asvspoof.py          # ASVspoof 2019 → 音频清单
  train.py                     # 视觉训练（AMP / 平衡采样 / 时序聚合）
  train_audio.py               # 音频训练（LFCC 或 log-Mel、类别权重）
  evaluate.py                  # 视频级评测（AUC/EER/F1 + 延迟）
  evaluate_audio.py            # 音频评测（EER/AUC + 流式延迟/RTF）
  benchmark.py                 # 三后端延迟 / FPS / 参数量 / FLOPs / 显存
  export_onnx.py               # ONNX 导出 + INT8 静态/动态量化
  robustness.py                # JPEG 压缩 / 分辨率衰减曲线
  ablation.py                  # 时序聚合 / 融合权重 / GEO 信号消融
  visualize.py                 # ROC / 分数分布 / 延迟 / 鲁棒性曲线
tests/test_smoke.py            # 预处理、音频特征、指标、融合、配置的单测
```

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 或按 README 顶部注释装 CUDA 版 torch

# 1) 合成冒烟数据（无需下载数据集）
python scripts/make_dummy_data.py --out data/dummy

# 2) 训练（3 个 epoch，几分钟内完成）
python scripts/train.py --config configs/default.yaml

# 3) 视频级评测：AUC / EER / F1
python scripts/evaluate.py --config configs/default.yaml --split val

# 4) 实时性基准：单帧延迟 / FPS / 参数量
python scripts/benchmark.py --ckpt runs/exp1/best.pt --frames 8 --iters 30

# 5) 启动 API
python -m geo_deepfake.api.server --config configs/default.yaml --ckpt runs/exp1/best.pt
curl -X POST "http://127.0.0.1:8000/predict/image?provenance=0.9&duplication=0.1" -F "file=@data/dummy/frames/fake/fake_000/frame_00.jpg"

# 6) 单测
python -m unittest discover -s tests -v
```

## 使用真实数据集

当前盘上可用的数据：

| 数据集 | 模态 | 规模 | 清单 | 状态 |
| --- | --- | --- | --- | --- |
| ASVspoof 2019 LA | 音频 | 121,461 条 / 7.4 GB | `data/asvspoof/manifest.csv` | ✅ 已训（eval EER 7.80%） |
| **FaceForensics++ c23** | 视频帧 | 1,588 个视频 → 12,704 帧 / 3.1 GB | `data/ffpp/manifest.csv` | ✅ 已训（视频级 AUC 0.793） |
| ASVspoof 2019 LA（平衡子集） | 音频 | 15,448 条 | `data/asvspoof_bal/manifest.csv` | ✅ 最优配置 |
| WildDeepfake（全量） | 视频帧 | 57,348 帧 / 66 GB 分片 | `data/wild/manifest.csv` | ✅ 已训（视频级 AUC 0.883） |
| 合成 dummy | 图/视频 | 64 帧 + 8 mp4 | `data/dummy/manifest.csv` | 仅冒烟 |

缺：FaceForensics++、Celeb-DF v2、DFDC（均需签署许可后下载）。拿到视频后：

```bash
# FaceForensics++（c23 压缩版），自动抽帧到 frames_cache
python scripts/build_manifest.py --root /data/FaceForensics++/c23 --layout ff++ \
    --out data/ffpp/manifest.csv --frames-per-video 8

# Celeb-DF v2
python scripts/build_manifest.py --root /data/Celeb-DF-v2 --layout celebdf \
    --out data/celebdf/manifest.csv --frames-per-video 8
```

manifest 中 `split` 按 video_id 哈希划分为 8:1:1，避免同一视频的帧跨集合造成泄漏。

### WildDeepfake（HuggingFace，可直接下载，无需申请）

`xingjunm/WildDeepfake`，Apache-2.0，共 72.8 GB，webdataset 分片：
`deepfake_in_the_wild/{real,fake}_{train,test}/<n>.tar.gz`，分片内为 `./<n>/<label>/<seq>/<frame>.png`。

```bash
# 全量（1,120 个分片 / 72.8 GB）：并发下载 + 断点续传 + 抽帧 + 清单
python scripts/prepare_wilddeepfake.py --out data/wild --train-shards -1 --test-shards -1 \
    --max-seqs 0 --max-frames 8 --workers 12
python scripts/train.py --manifest data/wild/manifest.csv --out runs/wild_full \
    --epochs 5 --pretrained --batch-size 64 --lr 1e-4
python scripts/evaluate.py --ckpt runs/wild_full/best.pt --manifest data/wild/manifest.csv --split val
```

全量实测（57,348 帧：train 51,068 / val 6,280，1,120 个分片）：

| 层级 | AUC | EER | ACC | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 视频级（785 个序列） | **0.8826** | **17.02%** | 0.8166 | 0.8818 | 0.7482 | 0.8095 |
| 帧级 | 0.8661 | 19.91% | 0.7963 | 0.8550 | 0.7335 | 0.7896 |

延迟：单帧 1.9 ms，单个序列（8 帧）15.5 ms。

> 小样本（1,432 帧 / 21 个分片）时 val AUC 只有 0.27，扩到全量后升到 0.88 —— 视觉侧的问题从头到尾是数据量，不是代码。
>
> 注意：可直接用 tar.gz 分片，不必走 Croissant/parquet 视图（后者总计 9.9 GB 且单分片 500 MB）。

### FaceForensics++（官方 EU2 服务器，c23）

```bash
# 并发下载（原 official/Deepfakes，各 800 个源视频），支持断点续传
python scripts/download_ffpp.py --root /home/ddl/data/ffpp --methods original,Deepfakes --limit 800 --workers 16
# 官方目录结构 → 抽帧 → 清单
python scripts/build_manifest.py --root /home/ddl/data/ffpp --layout ff++v2 \
    --out data/ffpp/manifest.csv --frames-per-video 8
python scripts/train.py --manifest data/ffpp/manifest.csv --out runs/ffpp --epochs 5 --pretrained
```

### 跨库泛化（开题报告 3.5 的"跨数据集泛化"，样本均为视频级）

| 训练集 | 测试集 | AUC | EER |
| --- | --- | --- | --- |
| FF++ c23 | FF++ c23（域内） | 0.8380 | 24.42% |
| WildDeepfake | WildDeepfake（域内） | **0.8826** | 17.02% |
| FF++ | WildDeepfake | 0.5905 | 46.54% |
| WildDeepfake | FF++ | 0.5382 | 45.35% |

双向跨库都掉到 0.54–0.59（接近随机），这是该领域的典型域偏移现象，也说明只靠帧级监督
+ ImageNet 预训练不足以泛化；报告里应把它作为"待解决问题"而不是主结果。

两套跨域补救都实测过，均未奏效：

| 方法 | 命令 | 目标域 AUC | 源域 AUC |
| --- | --- | --- | --- |
| 基线（FF++→Wild） | — | 0.5905 | 0.8380 |
| 测试时 BN 自适应（仅目标域，60 step） | `adapt_bn.py --steps 60` | 0.6203（+0.030） | **0.5563（−0.282，灾难性遗忘）** |
| 测试时 BN 自适应（源域混合 50%，动量 0.01） | `adapt_bn.py --source-mix 0.5 --momentum 0.01` | 0.6044（+0.014） | 0.8469（+0.009） |
| 跨域增强重训（高频扰动 + 二次转码 + 色度噪声） | `train.py --aug` | **0.5181（−0.072）** | 0.8111（−0.027） |

结论：仅靠 BN 统计更新或通用图像增强无法弥合 FF++→WildDeepfake 的域差；
增强反而破坏了模型依赖的高频线索。跨域泛化需要专门的域泛化方法（如频域解耦、攻击方法增广、
更强的自监督前端），不是调参能解决的。

### ASVspoof 2019（音频侧，Kaggle 公开，ODC-By）

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('awsaf49/asvpoof-2019-dataset'))"
# 解压后（约 23.6 GB 压缩包 → LA/PA 若干子集）：
python scripts/prepare_asvspoof.py --subset LA --out data/asvspoof          # 全量
python scripts/prepare_asvspoof.py --subset LA --max-per-class 4000         # 快速子集
python scripts/train_audio.py --manifest data/asvspoof/manifest.csv --out runs/audio --epochs 5
python scripts/evaluate_audio.py --ckpt runs/audio/best.pt --split test     # EER/AUC + RTF
```

`prepare_asvspoof.py` 按协议文件里的 utt_id 反查 flac，因此不依赖镜像的具体目录层级。
训练裁剪时长必须与流式窗口（`--window-s`，默认 1.5 s）一致，否则训练/推理分布不匹配会导致评分漂移。

### 消融实测（LA，RTX 3050，eval 统一取 10,000 条随机样本）

| 配置 | dev EER | eval AUC | eval EER |
| --- | --- | --- | --- |
| 平衡子集 5,160 + logmel + 1.5 s 窗 | 1.10% | 0.9460 | 12.03% |
| 全量 25,380 + logmel + pos_weight | 2.55% | 0.9076 | 15.59% |
| 平衡子集 + **LFCC** + 1.5 s 窗 | 1.57% | 0.8912 | 19.44% |
| 全量 + LFCC + pos_weight | 1.37% | 0.8943 | 17.81% |
| 平衡子集 + logmel + **4 s 窗** | 2.08% | **0.9744** | **7.80%** |
| 全量 25,380 + logmel + 4 s 窗（5 epoch） | **1.49%** | 0.9534 | 11.45% |
| 全量 25,380 + logmel + 4 s 窗（早停 2 epoch） | 2.16% | 0.9691 | 8.37% |

结论（与直觉相反，但数据一致）：
- **4 s 分析窗收益最大**：eval EER 12.03% → 7.80%，且 RTF 仅 0.0016，仍远低于 3.5 的 50–200 ms；
- **LFCC 反而更差**（19.44%），线性频标 + CMVN 丢掉了判别性较强的低频包络；换回 64 维 log-Mel；
- **pos_weight 无益**：spoof 是多数类（1:8.8），按 `n_neg/n_pos` 降权正类会削弱对攻击类的学习，eval EER 退化到 15.59%；
- 全量训练未提升 eval，说明瓶颈是**未见攻击的泛化**，不是数据量。

默认配置已固化为 `logmel + 4 s 窗`，推理：单条 6.7 ms（p95 12.4 ms），RTF 0.0016。

## 推理加速：ONNX + INT8（开题报告 3.2）

```bash
python scripts/export_onnx.py --kind visual --ckpt runs/exp1/best.pt --out runs/exp1/model.onnx
python scripts/benchmark.py --modality visual --ckpt runs/exp1/best.pt \
    --onnx runs/exp1/model.onnx --onnx-int8 runs/exp1/model.int8.onnx --out runs/exp1/bench.json
```

视觉模型（efficientnet_b0，224×224，8 帧/次，RTX 3050 + 20 核 CPU）：

| 后端 | p50 | 相对 onnx-fp32 | 模型体积 | 设备 |
| --- | --- | --- | --- | --- |
| PyTorch fp32 | **9.4 ms** | — | — | CUDA |
| ONNX fp32 | 47.1 ms | 1.00× | 16.02 MB | CPU EP |
| ONNX INT8（静态 QDQ） | 35.3 ms | **1.34×** | **4.85 MB** | CPU EP |

端到端（含裁剪/归一化/聚合）8 帧 14.9 ms ≈ 538 FPS；参数 4.01 M，FLOPs 0.385 G，峰值显存 112 MB。
输出一致性：ONNX fp32 与 PyTorch 最大偏差 1.6e-6。
INT8 加速比在 1.3–1.5× 间波动（同一 CPU 复用多次测量，受机器负载影响）。

> 音频模型（0.24 MB）上 INT8 **没有收益**：CPU 上 0.42 ms → 1.94 ms（动态量化产生 ConvInteger，本机比 fp32 慢）。小模型不适合权重量化，这是实测结论。

## 消融与鲁棒性（开题报告 3.5）

```bash
python scripts/ablation.py --exp all --manifest data/dummy/manifest.csv --out runs/ablation
python scripts/robustness.py --ckpt runs/wild/best.pt --manifest data/wild/manifest.csv --split val
python scripts/visualize.py --eval-json runs/audio/eval_test.json --bench-json runs/audio/bench.json --out runs/figures/audio
```

- **时序聚合**：mean AUC 0.8047 / EER 0.25 vs attn AUC 0.7578 / EER 0.25（dummy 数据，仅验证实现，不作为结论）
- **融合权重**：0.5/0.5 时视觉 0.9 + 音频 0.85 → 0.875；视觉 0.9 + 音频 0.1（冲突）→ 0.5；单模态缺失时权重自动归一化
- **GEO 信号**：同一伪造概率下，来源凭证从"完备"变为"缺失+高重复"可让 trust 从 0.93 降到 0.63（最大降幅 0.30）
- **鲁棒性**（WildDeepfake 全量模型，200 个序列分层抽样，基线 AUC 0.8831）：

| JPEG 质量 | 95 | 75 | 50 | 30 | 10 |
| --- | --- | --- | --- | --- | --- |
| AUC | 0.8804 | 0.8798 | 0.8819 | 0.8760 | **0.7796** |

| 分辨率 | ×1.0 | ×0.75 | ×0.5 | ×0.35 | ×0.25 |
| --- | --- | --- | --- | --- | --- |
| AUC | 0.8831 | 0.8795 | 0.8536 | 0.8210 | **0.8040** |

即：JPEG 在 q≥30 时几乎无损（降幅 <0.01），q=10 才明显掉（−0.10）；分辨率降到 1/4 时 AUC 仍有 0.80。

## 参考论文的改进尝试（LaDeDa / SENTRY / SFMFNet）

| 论文 | 借鉴点 | 实现 | 实测结果 | 结论 |
| --- | --- | --- | --- | --- |
| LaDeDa (Cavia 2024) | 小感受野 patch 打分 + 均值池化 | `models/patch_visual.py`（32×32 patch、stride 16、4 层卷积、0.09 M 参数） | 视频级 AUC **0.683** / EER 36.4%（基线 0.883 / 17.0%） | 20× 更小但明显更差；与全局模型平均集成后 0.874，仍不如单用基线 |
| LaDeDa 蒸馏 | 教师软标签 → patch 模型 | `scripts/make_soft_labels.py` + `train.py --soft-labels` | 蒸馏后 AUC 0.693（仅 +0.010） | 教师输出 94% 饱和在 <0.01 或 >0.99，软标签退化为硬标签，无"暗知识"可迁移 |
| SENTRY (Banerjee 2026) | 整数提升小波 + WPT 包络 + WHT 序列谱 + 微统计 + GBDT/Mahalanobis | `data/wavelet_features.py`（49 维）、`models/sentry_audio.py` | val EER 11.5%，**eval EER 16.65%**（基线 7.80%） | 特征维度低 100×、纯 CPU 可跑，但未见攻击上明显劣于 CNN |
| SFMFNet | 小波高低频分离 + 全局频域分支 + 双向交叉注意力 | 未实现 | — | **抓错论文**：工作区这份是超声心动图左心室分割（Dang Li 2026），不是开题报告引用的实时深度伪造检测 SFMFNet |

复现命令：

```bash
# LaDeDa 式 patch 检测器
python scripts/train.py --manifest data/wild/manifest.csv --out runs/wild_patch --arch patch --epochs 5 --lr 1e-3
# SENTRY 式音频检测器
python scripts/train_sentry_audio.py --manifest data/asvspoof_bal/manifest.csv --out runs/audio_sentry \
    --eval-manifest data/asvspoof/manifest.csv --eval-sample 10000
```

结论：这两条思路在**边缘部署成本**上有优势（patch 模型 0.09 M / SENTRY 纯整数运算无 GPU），
但在我们的数据上没有带来精度提升。若要在开题报告里写"改进"，应表述为
"以精度换部署成本的可选分支"，而不是"精度提升"。

## 评测指标（对应开题报告 3.5）

- 检测性能：AUC、EER（含 EER 阈值）、Accuracy、Precision、Recall、F1（视频级与帧级各一份）
- 实时指标：单帧延迟 mean/p50/p95、FPS、参数量、FLOPs、峰值显存（`benchmark.py`）
- 鲁棒性：JPEG 压缩 / 分辨率衰减曲线（`robustness.py`）
- 消融：时序聚合、融合权重、GEO 信号（`ablation.py`）
- 可视化：ROC、分数分布、延迟对比、鲁棒性曲线（`visualize.py`）

## 当前版本的限制（后续工作）

1. 融合为**概率级加权融合**，尚未实现跨模态特征级对齐；`fusion_weights` 目前是人工设定。
2. FF++ 只取了 c23 的 original + Deepfakes 各 800 个源视频（约 12.7k 帧），未含 Face2Face /
   FaceSwap / NeuralTextures / FaceShifter；Celeb-DF、DFDC 仍缺。
   跨库泛化已测但结果很弱（FF++↔WildDeepfake 双向 AUC≈0.55–0.58），是目前最大的技术缺口。
   另外视觉与音频**不是同一批素材**，融合链路只能在接口层联调，无法给出真实的多模态联合指标。
3. 音频端未实现整数小波特征与 SENTRY 式极低延迟分类器；当前为 log-Mel + CNN（60.5 k 参数）。
4. ONNX Runtime 本机无 CUDA EP，加速对比是 GPU-PyTorch vs CPU-ONNX，不是同设备对比；TensorRT 未接入。
5. 可视化是离线图表（matplotlib），未做 Web 前端。
