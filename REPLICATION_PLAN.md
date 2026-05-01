# WHOLE 复现计划

论文：[WHOLE: World-Grounded Hand-Object Lifted from Egocentric Videos](https://arxiv.org/abs/2602.22209)  
项目页：https://judyye.github.io/whole-www  
状态：官方代码未公开，完全从头复现

---

## 一、方法总览

### 输入 / 输出

```
输入：
  - Metric-SLAMed 自我中心视频（Aria / Quest3）
  - 3D 物体模板 mesh

输出：
  - H^{1:T}  双手 MANO 轨迹（world frame）
  - T^{1:T}  物体 SE(3) 轨迹（world frame）
  - C^{1:T}  双手接触 binary label
```

### 整体流程

```
Egocentric Video
    ├── SLAM poses          ← HOT3D 提供，gravity-aware metric SLAM
    ├── Hand Estimator      ← HaWoR [73] → 粗糙手部轨迹 H̃
    ├── VLM Contact         ← GPT-4V + visual prompt → 接触 binary C̃ @ 3fps
    └── Object Template O   ← HOT3D object_models/*.glb → BPS 编码
              ↓
    Diffusion Prior Dψ      ← 训练阶段离线学习
              ↓
    Guided Generation       ← 测试时：扩散步 ↔ guidance 步交替
              ↓
    Output: H, T, C（world space，120帧窗口）
```

---

## 二、核心数据结构

### 73D Diffusion Variable x

| 字段 | 来源 | 维度 |
|------|------|------|
| 物体 SE(3) 9D representation | `objects.json` → quaternion+t → 9D | 9 |
| 双手接触 binary | VLM / GT contact label | 2 |
| 手部 Γ global orientation | `wrist_xform[0:3]` axis-angle | 3 |
| 手部 Λ translation | `wrist_xform[3:6]` world coords | 3 |
| 手部 Θ articulation PCA | `hands.json` → `thetas` 15D | 15 |
| 手部 β shape | `__hand_shapes.json__` 10D | 10 |
| **单手小计** | | **31** |
| **双手（左+右）** | | **62** |
| **总计** | | **73** |

### HOT3D 数据字段映射

每帧 tar 内文件：

| 文件 | 内容 | 用途 |
|------|------|------|
| `{idx}.cameras.json` | `T_world_from_camera`（quaternion + t） | SLAM pose |
| `{idx}.hands.json` | `mano_pose.thetas`（15D），`wrist_xform`（6D），2D boxes | 手部 GT |
| `{idx}.objects.json` | `T_world_from_object`（quaternion + t），object_bop_id | 物体 GT |
| `{idx}.image_214-1.jpg` | RGB 主摄 1408×1408 fisheye | 视觉观测 |
| `{idx}.image_1201-*.jpg` | RGB SLAM 摄像头 640×480 | 视觉观测 |
| `{idx}.info.json` | device, sequence_id, timestamps | 元数据 |
| `__hand_shapes.json__` | MANO beta 10D（clip 级别，per person） | 手部形状 |

---

## 三、项目结构

```
whole/
├── configs/
│   └── default.yaml             # 所有超参数
│
├── data/
│   ├── hot3d_loader.py          # WebDataset tar 流式加载
│   ├── mano_converter.py        # PCA thetas → full MANO，FK 算 joints
│   ├── preprocessing.py         # Gravity 对齐，BPS，训练噪声增强
│   └── collate.py               # batch 拼装，padding
│
├── models/
│   ├── denoiser.py              # 4-layer Transformer decoder（12.35M）
│   ├── diffusion.py             # DDPM schedule，采样，训练 step
│   ├── bps.py                   # Basis Point Set 编码
│   └── ambient_sensor.py        # BPS_J(T_i[O]) hand-object 近邻特征
│
├── losses/
│   ├── ddpm.py                  # 标准 DDPM 去噪 loss
│   ├── interaction.py           # L_inter：接触距离 + near-rigid transport
│   ├── consistency.py           # L_const：J_ψ vs MANO FK 一致性
│   └── smoothness.py            # L_smooth：加速度惩罚
│
├── guidance/
│   ├── reprojection.py          # g_reproj：one-way Chamfer 到 2D mask
│   ├── interaction.py           # g_inter：同训练 interaction loss
│   ├── temporal.py              # g_temp：时序平滑
│   └── vlm_contact.py           # GPT-4V contact 标注（3fps 采样）
│
├── hand_estimator/
│   └── hawor_wrapper.py         # HaWoR 推理接口，输出 H̃
│
├── eval/
│   ├── metrics_hand.py          # W-MPJPE, WA-MPJPE, ACC-NORM, PA-MPJPE
│   ├── metrics_object.py        # AUC of ADD / ADD-S（阈值 0.3）
│   └── metrics_hoi.py           # 全局对齐手后算物体相对误差
│
├── utils/
│   ├── rotation.py              # 9D repr, quaternion, axis-angle 互转
│   ├── geometry.py              # SE(3) ops, projection, Chamfer
│   └── visualization.py         # 3D 可视化（allocentric view）
│
├── train.py                     # 训练主入口
├── inference.py                 # 测试时 guided generation
└── evaluate.py                  # 跑 Table 1-3 指标
```

---

## 四、分阶段实现计划

### 阶段 1：数据管线 ✅ 当前阶段

**目标**：能读出一个 batch，可视化手+物体轨迹

- [ ] `data/hot3d_loader.py`
  - 用 `webdataset` 流式解 tar（避免全部解压）
  - 解析 hands/objects/cameras json
  - 支持 Aria（3 摄像头）和 Quest3（2 摄像头）
  - 输出固定长度 T=120 帧窗口

- [ ] `data/mano_converter.py`
  - 从 HOT3D toolkit 提取 PCA basis 矩阵（`thetas_pca_basis`）
  - `pca_to_mano(thetas_15d, betas_10d, wrist_xform_6d)` → MANO 参数
  - MANO FK → joint positions J（21×3），joint velocities J̇

- [ ] `data/preprocessing.py`
  - **Gravity alignment**：每个 window 起始帧，从 `T_world_from_camera` 提取 gravity vector，旋转 z 轴对齐
  - **训练噪声增强**：对 GT MANO 参数注入 trajectory-level 噪声 ς^g + per-frame 噪声 ς^t，random frame drop 模拟遮挡
  - **SE(3) → 9D**：quaternion + t → 9D rotation representation（Zhou et al. 2019）

- [ ] 可视化脚本：从一个 clip 渲染出 allocentric view 的手+物体轨迹

### 阶段 2：核心模型

**目标**：能跑一次 forward pass（随机噪声输入 → 去噪输出）

- [ ] `models/bps.py`
  - 预计算 4096 basis points（单位球面上均匀采样）
  - `BPS(mesh)` → O（object geometry descriptor）
  - `BPS_J(T_i[O])` → Ambient Sensor feature（以手关节为 basis）

- [ ] `models/denoiser.py`
  - 输入：x_n（73D × T）+ condition c（H̃ encoded, O encoded）
  - 4-layer Transformer decoder，4 attention heads，hidden dim 512
  - Non-autoregressive（整段 120 帧并行处理）
  - 参考 MDM (Motion Diffusion Model) 架构
  - 目标参数量：~12.35M

- [ ] `models/diffusion.py`
  - Cosine noise schedule，1000 steps
  - `w_n` variance schedule weight（DDPM loss 加权）
  - Conditional sampling：`sample(condition=c)`

### 阶段 3：训练

**目标**：loss 正常下降，1M iterations 后生成合理的 HOI 运动

- [ ] `losses/ddpm.py` — `‖x̃₀ - Dφ(xₙ|n, H̃, O)‖²` 加权
- [ ] `losses/interaction.py` — 接触时手-物距离 + contact point near-rigid transport
- [ ] `losses/consistency.py` — `‖J_ψ - MANO(Γ_ψ, Λ_ψ, Θ_ψ)‖₂`
- [ ] `losses/smoothness.py` — 加速度 L2 惩罚
- [ ] `train.py`
  - Curriculum：前 10k steps 只用 L_DDPM，之后加辅助 loss
  - AdamW, lr=2e-4，1M iterations
  - 数据增强：随机旋转 object template，translation jitter
  - 多卡（8×H200）DDP

### 阶段 4：测试时 Guided Generation

**目标**：给定视频，输出世界坐标系下 HOI 轨迹

- [ ] `guidance/vlm_contact.py`
  - 渲染手部（绿/红点）+ 物体彩色 mask 覆盖图像
  - GPT-4V API 调用（使用 Appendix 的 system + user prompt）
  - JSON 解析，one-out-of-k 约束验证

- [ ] `guidance/reprojection.py`
  - MANO 和物体 mesh 投影到 2D（fisheye 畸变模型）
  - One-way Chamfer：只算预测投影点 → 观测 mask 的距离（不反向）

- [ ] `inference.py`
  - 主循环：for n in N→1: diffusion step → guidance gradient → update
  - 梯度 clipping（防止 NaN）
  - **长视频滑动窗口**：步长 < 120，overlap 区域 blend MANO shape params，per-window 分别去噪

### 阶段 5：评估

**目标**：复现 Table 1-3 数字

- [ ] `eval/metrics_hand.py`
  - W-MPJPE：全局对齐所有帧关节
  - WA-MPJPE：用前两帧仿射变换对齐
  - PA-MPJPE：per-frame Procrustes
  - ACC-NORM：每帧关节加速度误差

- [ ] `eval/metrics_object.py`
  - ADD：平均顶点位移
  - ADD-S：对称物体取最近点
  - AUC @ 阈值 0.3（论文使用 0.3 而非常见的 0.1）

- [ ] `eval/metrics_hoi.py`
  - 先全局对齐手轨迹（WA-MPJPE 方式）
  - 再在对齐空间中算物体 ADD/ADD-S

---

## 五、关键超参数（论文原文）

| 参数 | 值 |
|------|----|
| 窗口长度 T | 120 frames |
| Diffusion steps N | 1000 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Hidden dim | 512 |
| 模型参数量 | 12.35M |
| Batch size | 未明确（估计 32-64） |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Training iterations | 1,000,000 |
| Curriculum | 前 10k 步只用 L_DDPM |
| Guidance weight w | 未明确（需要调参） |
| VLM 采样频率 | 3 fps（每隔 ~11 帧） |
| Peak memory | 14 GB（单卡 RTX 6000） |
| Inference time | ~59s / clip（RTX 6000 Blackwell） |

---

## 六、依赖与环境

### 待安装

```bash
pip install webdataset          # tar 流式加载
pip install smplx               # MANO 模型
pip install pytorch3d           # Chamfer loss, mesh ops
pip install einops              # Transformer 维度操作
pip install bps-torch           # Basis Point Set
pip install openai              # GPT-4V API
pip install trimesh             # mesh 读取（已安装）
```

### 已有

- PyTorch 2.10.0 + CUDA 12.8
- 8× NVIDIA H200 (139 GB VRAM each)
- trimesh 4.11.3
- OpenCV 4.13.0
- HaMeR（`hamer/` 目录，注意与 HaWoR 不同）

### 外部模型

| 模型 | 用途 | 获取方式 |
|------|------|---------|
| MANO | 手部参数化模型 | mano.is.tue.mpg.de（申请） |
| HaWoR [73] | 世界坐标系手部估计（测试时用） | 论文主页下载 |
| GPT-4V / GPT-5 | VLM 接触标注 | OpenAI API |

---

## 七、已知卡点与风险

| 风险 | 严重度 | 对策 |
|------|--------|------|
| HOT3D PCA basis 矩阵不公开 | 中 | 从 HOT3D Python toolkit 代码里提取，或直接在 15D 空间操作 |
| HaWoR 坐标系与 HOT3D 不对齐 | 高 | 仔细对比两者世界坐标系定义，写对齐脚本 |
| Gravity alignment per-window 拼接 | 中 | 保存每个 window 的对齐矩阵，推理时逆变换 |
| Guidance 梯度不稳定（NaN/爆炸） | 高 | 梯度 clipping + 小 guidance weight w 热身 |
| VLM contact F1 从 57% → 81% 的 5-shot 校准 | 低 | 按 Appendix prompt 严格复现，5 条 annotated examples |
| 评估阈值 AUC@0.3 非标准 | 中 | 与 BOP 默认 AUC@0.1 区分，单独实现 |
| test 集无 GT 标注（BOP 规则） | 中 | 从 train 集中自行切分验证集，或用 HOT3D 完整集 |

---

## 八、数据下载状态

| 数据 | 大小 | 状态 |
|------|------|------|
| metadata + object_models | 0.3 GB | ✅ 完成 |
| test_aria（467 clips） | 44 GB | ✅ 完成 |
| train_aria（1,516 clips） | 163 GB | 🔄 下载中 |
| test_quest3（561 clips） | 57 GB | 🔄 下载中 |
| train_quest3（1,288 clips） | 131 GB | 排队中 |

数据目录：`/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/`

---

## 九、参考实现

| 组件 | 参考代码 |
|------|---------|
| Transformer denoiser | [MDM: Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model) |
| DDPM framework | [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) |
| BPS encoding | [bps-torch](https://github.com/sergeyprokudin/bps) |
| 9D rotation | [Zhou et al. 2019 ref impl](https://github.com/papagina/RotationContinuity) |
| MANO / smplx | [smplx](https://github.com/vchoutas/smplx) |
| Classifier guidance | [DiffusionCLIP / DDPM guidance](https://github.com/gwang-kim/DiffusionCLIP) |

---

*最后更新：2026-05-01*
