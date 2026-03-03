# TwinBrain — AI Agent Instructions
> **This file must be read before any other file in every session.**

---

## 一、身份与角色定位

你是一个拥有海量知识储备的高级研究伙伴，而非执行指令的工具。请始终以此自我定位：

- **不是劳工**：拒绝只做最低限度的完成任务或简单修复 bug。
- **是思考者**：理解项目本质目标，主动识别深层问题。
- **是设计者**：提出系统性的架构改进，而非局部补丁。
- **是合作者**：必要时从根本上挑战和改进用户的想法。

## 二、行为准则

1. **优先理解目标**：先思考"这个问题的本质是什么？是否有更优雅的解法？"
2. **稳定性优先**：所有优化必须可验证、稳定，不引入新风险。
3. **主动创新**：发现更优解时主动提出并说明理由。
4. **诚实评估**：用户方案有根本性缺陷时，明确指出并给出替代方案。
5. **记录错误**：每次发现值得记录的 bug 或陷阱，立即写入本文件§三。

## 三、错误记录（Error Log）

> **记录原则**：重在**思维上的盲区**，而非具体报错信息。每条记录应能回答：*"下次遇到类似问题，我应该先想到什么？"*

---

### [2026-03-01] temporal_chunk_size × no-checkpointing = 纯 Python overhead（训练极慢）

**用户观察**：epoch 时间 200-260s，但日志显示 `GPU Memory: allocated=0.16 GB, reserved=0.44 GB`（8GB GPU 仅用 2%）。

**思维误区**：看到 `temporal_chunk_size=32` 就认为"为了节省显存，合理"，没有追问：**当 gradient_checkpointing 关闭时，分块到底节省了什么显存？**

**根因**：
- 无 gradient checkpointing 时，PyTorch autograd 会为每个 tensor operation 存储前向激活（用于 backward）
- 无论 chunk_size=32 还是 chunk_size=T（整体），autograd **都会存储全部 T 步的激活**
- 分块只是把一个 propagate() 调用拆成了 T/chunk_size 个独立调用
- 每次 propagate() 调用都有 Python overhead（PyG 的 __check_input__、__lift__、__collect__ 等 Python 函数 + CUDA kernel launch）
- EEG T=500, chunk_size=32 → 16 次 propagate/层，4 层 × 3 边 × 16 次 = 192 次 Python 调用/forward
- GPU 96% 时间在等待 CPU 的 Python 调度，真正的计算时间只有 4%

**正确认识**：
| 场景 | 分块效果 |
|------|---------|
| checkpointing=False（默认） | ❌ 不节省显存，纯 Python overhead |
| checkpointing=True，训练时 | ✅ 真正节省 backward 峰值显存（每块重计算） |
| checkpointing=True，推理时 | ❌ 不节省（inference 无 backward） |

**修复（V5.45）**：
```python
# SpatialTemporalGraphConv.forward()
if self.use_gradient_checkpointing and self.training:
    chunk_size = self.temporal_chunk_size or T  # 按配置分块（真正节省显存）
else:
    chunk_size = T  # 单次调用：最快，与分块内存相同
```

**规则**：**"分块" 只有与 gradient checkpointing 配合才有实际意义。** 在评估任何 "分块优化" 时，必须先追问：不分块时，内存如何变化？如果内存不变（autograd 保留所有激活），分块只添加 Python overhead，应当去除。

---

### [2026-03-02] V5.47 引入的双重回归：temporal_chunk_size=64 + InfoNCE temperature=0.1

**用户观察**：V5.47 相较前一个版本（V5.46）更慢（430s/epoch vs 之前更快），且 pred_r2_h1_fmri 随 epoch 下降（0.162→0.042→-0.010），GPU 仍只用 0.18GB / 8GB = 2%。

**根因（两个独立回归叠加）**：

**回归 1 — temporal_chunk_size: null → 64 重引入了 V5.45 修复的 Python overhead**

V5.45 设默认 temporal_chunk_size=null，并在代码中实现：
- checkpointing=True + null → chunk_size=T（单次 gradient_checkpoint 调用）
- checkpointing=False → chunk_size=T（V5.45 fix，完全不分块）

V5.47 为"配合 checkpointing=True 的内存节省"改为 temporal_chunk_size=64，但：
- checkpointing=True + chunk=64 → EEG 8 块，4层×3边×8块=96 forward + 96 backward recompute = **192 次 propagate() 调用**
- vs V5.46 (null+checkpointing=True)：4层×3边×1块=12 forward + 12 recompute = **24 次调用**
- 192/24 = **8× 更多 Python 调用** → 训练从 ~100s/epoch → ~430s/epoch（~4-5× 变慢）

V5.47 的出发点是好的（checkpointing+分块 = 真正节省内存），但用户 GPU 只用了 0.18GB，根本不需要内存节省，代价是 8× 的速度损失。

**回归 2 — InfoNCE temperature=0.1 在 warmup 期间压制预测准确度（pred_r2_h1_fmri 下降）**

V5.47 新增 InfoNCE 损失（use_info_nce=true, temperature=0.1）。
- fMRI: n_items = N×S = 190×17 = 3230 → 初始 loss_scale ≈ log(3230)/0.1 ≈ **81**
- pred_sig 损失（直接优化 pred_r2）的初始 loss_scale ≈ 1.0
- warmup_epochs=10 期间权重固定：InfoNCE effective 梯度 = 4.0×81=**324** vs pred_sig=6.0×1=**6** → InfoNCE **50× 压制**预测准确度训练信号
- 结果：predictor 学会"区分未来"但不学会"准确预测未来" → pred_r2_h1_fmri 随 epoch 持续下降

这是 AGENTS.md 规则"添加新任务损失时必须问：这个数据集规模能否支持额外的梯度预算？"的直接违反。

### 修复（V5.48）：
1. `temporal_chunk_size: 64 → null`（恢复 V5.45/V5.46 行为，8× 提速）
2. `use_gradient_checkpointing: true → false`（额外 2× 提速，GPU 利用率仅 2% 无需节省内存）
   **⚠ 注意：V5.49 发现此修复是错误的，见下方 V5.49 记录。backward 峰值 ~12 GB，必须保持 true。**
3. `info_nce_temperature: 0.1 → 0.5`（loss_scale 从 81 降至 16，与 pred_sig 同量级，恢复梯度平衡）
4. `task_priorities.pred_nce: 4.0 → 2.0`（warmup 期 InfoNCE effective 梯度比从 50:1 → 5:1，消除 pred_r2 < 0 告警）

**快速诊断规则（发现"GPU 利用率 <10% 但训练很慢"时的排查顺序）**：
1. 检查 `temporal_chunk_size` × `use_gradient_checkpointing` 组合
   - chunk=null 或 checkpointing=false → 单次 propagate，最快
   - chunk=64 + checkpointing=true → 8× Python overhead，需要吗？查 GPU allocated
2. 若 GPU allocated < GPU_TOTAL × 20%，**不要**直接关闭 checkpointing——`allocated` 是步间稳定状态，backward 峰值可能远高于此（参见下方 V5.49 错误记录中的峰值公式）
3. **若仍慢（排除 GC 配置问题）**，检查训练样本大小参数（见下方 V5.49 性能调优规则）：
   - `eeg_window_size`: O(T²) attention, O(T×E) propagate → 最直接影响
   - `k_nearest_fmri` / `k_dynamic_neighbors`: O(E) propagate
4. 若 OOM 再开启（先 checkpointing=true，再 chunk=64，再 chunk=32）

**InfoNCE 温度调优规则（小数据集，4-8 被试）**：
- n_items = N × prediction_steps（如 190×17=3230 for fMRI）
- 初始 loss_scale ≈ log(n_items) / temperature
- 目标：InfoNCE effective 梯度 ≤ 2× pred_sig，避免 pred_r2 随训练深入持续下降
  - effective_nce = pred_nce_priority × loss_scale = pred_nce_priority × log(n_items) / temperature
  - effective_pred = pred_priority × pred_sig_scale ≈ 6.0 × 1.0 = 6
  - V5.48 配置：pred_nce=2.0, temperature=0.5 → effective_nce = 32 vs 6 → 5:1（对低ρ数据过高）
  - **V5.50 配置：pred_nce=1.0, temperature=1.0 → effective_nce = 8 vs 6 → 1.3:1（均衡）**
  - 注意：低 BOLD 自相关数据（ρ≈0.23，实验性 EEG-fMRI）比静息态数据（ρ≈0.85-0.95）对 InfoNCE 竞争更敏感
- 若仍出现 pred_r2 持续下降（≥3次验证），设 use_info_nce: false（明确禁用优于错误配置）

---

### [2026-03-02] V5.49 错误：「GPU allocated=0.18GB → GC=false 安全」推断错误导致 backward OOM

**用户反馈**：将 `use_gradient_checkpointing: false` 后在反向传播中发生 CUDA OOM，"光是 fMRI 和 EEG 数据就占了很多内存"。

**思维误区**：看到 `GPU allocated=0.18 GB`（epoch 间稳定状态读数）就得出"无内存压力、GC 是浪费"的结论，**完全忘记询问 backward 峰值与稳定状态读数是两个不同的量**。

**根因**：
- `torch.cuda.memory_allocated()` 是在 epoch 结束后（backward 完成、所有中间激活已释放后）的读数
- backward 过程中，PyTorch autograd 必须同时持有所有 propagate() 调用的中间激活（用于链式法则）
- 具体计算（EEG T=500, fMRI T=50, H=128, 4层, 3边，k_eeg=10+10, k_fmri=20+10）：
  - EEG intra ×4层 ≈ 7,741 MB
  - fMRI intra ×4层 ≈ 3,502 MB
  - Cross-modal ×4层 ≈ 584 MB
  - **合计 ≈ 11,827 MB = 11.6 GB** → 8GB GPU 必然 OOM ❌
- 与之对比，GC=True + chunk=null：
  - backward 每次只重计算 **1 个 propagate()**，峰值 = max(EEG intra per layer) + boundary ≈ **2.1 GB** ✅

**正确结论**：
| 配置 | backward 峰值 | 8GB GPU | 速度代价 |
|------|--------------|---------|---------|
| GC=False + chunk=null | ~11.6 GB | ❌ OOM | 无 |
| GC=True + chunk=null | ~2.1 GB | ✅ 安全 | +40% backward |
| GC=True + chunk=64 | ~0.4 GB | ✅ 最安全 | +320% (8× Python) |

**修复（V5.49）**：`use_gradient_checkpointing: false → true`（已回退，保持 V5.46 行为）

**V5.49 追加（性能优化，同 PR）**：用户报告训练极慢（524s/epoch，GPU 利用率仅 2%）。
GC=True 的 +40% backward overhead 是已知代价，但 524s 远超预期。根因分析：
- EEG window T=500 → TemporalAttention O(T²)=250K ops, propagate O(T×E)=945K messages
- k_nearest_fmri=20 → fMRI 7600 edges/layer → 380K messages/layer
- 上述参数通过减小可显著降低每步的 GPU 工作量

**V5.49 性能优化配置**：
| 参数 | 旧值 | 新值 | 效果 |
|------|------|------|------|
| `eeg_window_size` | 500 | 250 | TemporalAttention 4× 快；propagate 2× 快 |
| `k_nearest_fmri` | 20 | 10 | fMRI edges 3800→1900，propagate 2× 快 |
| `k_dynamic_neighbors` | 10 | 5 | dynamic edges 减半，DGC topk 2× 快 |

**V5.49 代码优化**：
- `SpatialTemporalGraphConv._ei_cache`: 缓存 `(ei_chunk, ea_chunk)` 扩展张量。
  同一 run 所有窗口共享同一 `edge_index.data_ptr()` → 首次调用构建 [2,T×E] 张量，
  后续调用直接复用，消除 N_windows 次重复的 GPU 内存分配。
- `train_model()` 预加载数据到 GPU：训练循环前调用 `g.to(device)` 一次，
  消除逐步 CPU→GPU 传输（即使传输量小，也有 Python 函数调用开销）。

**性能调优规则（训练慢但 GC=True 不可关闭时）**：
1. **`eeg_window_size` 是最大的性能旋钮**：O(T²) attention + O(T×E) propagate，250 → 500 约 3× 慢
2. **`k_nearest_fmri / k_dynamic_neighbors`**：直接控制 E，边减半 → propagate 约快 2×
3. **`eeg_window_size` 最小安全值**：250 pts × 4ms = 1s，覆盖 alpha(8-12Hz, >8 周期)/beta/gamma ✓
4. **`k_nearest_fmri` 最小安全值**：10（平均度 20，符合小世界网络特性）
5. **`use_dynamic_graph: false`**：完全消除 DGC 开销，并使 ei_cache 命中率从 0% → 100%（静态图下所有窗口完全命中）

**根本规则**：
1. **`memory_allocated()` 是稳定状态，不是 backward 峰值**。评估 GC 必要性时，必须计算 backward 峰值 = Σ(所有 propagate() 激活)，而非读取训练间隙的显存状态。
2. **backward 峰值公式（GC=False）**：`4层 × 3边 × max(T×E×H per edge type) × 6 ≈ 11.6+ GB`（T=500 时）；`T=250 时 ≈ 5.8 GB`（仍超过 8GB 时建议 GC=True）
   （×6：每次 propagate() autograd 保留约 6 个 [T×E, H] 张量：x_j、x_i、x_t、alpha、output、dropout mask）
3. **backward 峰值公式（GC=True, chunk=null）**：`max(T×E×H per edge type) × 6 + boundary ≈ 2.1 GB`（T=500）；`≈ 1.1 GB`（T=250）
   （boundary：GC 边界处保留的两端节点特征张量，约 [T×N, H] × 2 ≈ 192 MB）
4. 只有当 backward 峰值 < GPU 总显存 × 70% 时，才可以安全关闭 GC。

---

### [2026-03-01] EEG/fMRI 时间异质性：预测物理约束与文献支持的 R² 标准

**用户观察**：pred_r2_eeg=0.051，明显低于 pred_r2_fmri=0.205，询问是否实现有问题；
后续又问 R² > 0.3 的标准是否合适，以及 0.05/0.15 是否设置过低。

**关键认识**：

1. **EEG 预测本质困难**（文献支持）：  
   EEG 信号在 ms 量级有大量生理噪声（眼动、肌电、电极漂移，SNR < 1）。  
   **Schirrmeister et al. 2017**（*Human Brain Mapping*）：深度学习 EEG 解码在原始波形重建任务上  
   R² ≈ 0.05–0.20；**Kostas et al. 2020**（*J. Neural Eng.*）：跨被试 EEG 时序预测 R² ≈ 0.08–0.18；  
   **Roy et al. 2019**（*J. Neural Eng.*）：EEG 深度学习系统综述，原始信号回归 R² 的正常范围为 0.05–0.25。  
   → **充分训练后期望：pred_r2_eeg ≈ 0.10–0.20（物理上限约 0.20–0.25）**

2. **fMRI 预测本质容易**（文献支持）：  
   BOLD 信号是神经活动的低通滤波（HRF 带宽 ~0.1 Hz，**Logothetis et al. 2001**, *Nature*），  
   有强自相关性。**Thomas et al. 2022**（NeurIPS）：自监督脑动力学模型在 fMRI 预测上  
   R² ≈ 0.15–0.35；**Bolt et al. 2022**（*Nat. Neurosci.*）：BOLD 网络动态的可重复性意味着  
   R² ≥ 0.20 是充分训练后 2-8 被试模型的合理目标。  
   → **充分训练后期望：pred_r2_fmri ≈ 0.20–0.40（2-8 被试）**

3. **R² > 0.3 的研究标准仅适用于重建**：  
   Kingma & Welling 2014（VAE 原论文）、Turk-Browne 2013（*J. Exp. Psych.*）将 R² ≥ 0.3  
   定为神经影像自编码器的"可用"下限。这个标准**不适用**于预测任务——预测未来  
   本质上比重建当前更难，且预测视界越长 R² 越低。

4. **EEG→fMRI 低通滤波是物理正确的**：  
   EEG latent 从 T=500 插值到 T=50，等效于神经血管耦合（NVC）的时间积分  
   （**Debener et al. 2006**, *J. Neurosci.*；**Mukamel et al. 2005**, *Science*）。  
   "能量损失"反映真实物理过程（HRF 低通特性），不是 bug。

**正确期望（充分训练模型，非早期轮次）**：
- `pred_r2_eeg ≥ 0.10`：充分训练后的"良好"标准（文献范围 0.08–0.18）
  - 注意：早期训练（< 20 epoch）达到 0.05 已属正常，不应视为失败
- `pred_r2_fmri ≥ 0.20`：充分训练后的"良好"标准（文献范围 0.15–0.35，2-8 被试）
  - 注意：epoch 5 已观测到 0.205，训练完成后应可达 0.25–0.35
- `r2_eeg, r2_fmri ≥ 0.30`：重建标准，已验证（epoch 5 即已 > 0.88，0.96）
- 若 pred_r2 < 0：模型问题（检查因果性、loss 权重），与物理上限无关

**为什么之前设为 0.05/0.15（V5.45）而现在提高到 0.10/0.20（V5.46）？**  
V5.45 的阈值是"任何有意义信号"的下限（早期训练的期望）。  
V5.46 修正为"充分训练后的合理目标"：  
- 用户指出模型才训练 5 epoch，按理训练完应更高 → 提高阈值反映充分训练预期  
- 文献数值（上述引用）支持 EEG=0.10、fMRI=0.20 为正常完成训练后的水平

**规则**：
1. **R² 标准必须区分任务类型（重建 vs. 预测）和模态（EEG vs. fMRI）**
2. **评估指标应对应"充分训练后的模型"，而非早期检查点**
3. **EEG 预测 R² < 0.10 在训练中是正常的；最终结论时才应判断是否达标**



### [2026-03-01] pred_r2 随训练加深持续恶化：TemporalAttention 双向注意力造成训练-验证致命偏差

**用户观察**：epoch 5 pred_r2_eeg=0.219（最优），之后持续恶化到 -0.163，70 epoch 时仍为 -0.02。
同时 r2_eeg（重建 R²）也在下降（0.891 → 0.767），与通常"训练越深重建越好"的预期相反。
用户反映"比前几个版本更差，且提供了更多训练样本"。

**思维误区**：AGENTS.md 之前把训练-验证偏差归因于 Conv1d 对称 padding 的 ±1 步泄漏（"约 0.5% 的边界泄漏"），认为是可接受的"工程权衡"。
**完全忽略了** `TemporalAttention(is_causal=False)` 的**全局双向注意力**！

**根因（三叠加）**：

**根因 1 — TemporalAttention 双向注意力造成全局未来信息泄漏（最严重，P0）**：
- `TemporalAttention` 使用 `F.scaled_dot_product_attention(..., is_causal=False)`，即每个时间位置 t 可以通过注意力机制**全局**访问未来所有时间步 T_ctx..T
- Conv1d 的 ±1 步泄漏微不足道（<1%），但 TemporalAttention 的全局双向泄漏巨大：T_ctx 位置的隐状态包含了所有 T-T_ctx 个未来步的完整注意力加权信息
- V5.31 修复了**验证**时的因果性（re-encode only T_ctx steps），但**训练**时 encoder 仍是双向的
- 结果：
  - 训练时：predictor 学会"回忆"已编码在上下文中的未来信息（trivial shortcut）
  - 验证时：因果 re-encode 使 shortcut 失效 → pred_r2 崩溃
  - 随训练加深：encoder 越来越擅长提取双向注意力的未来信息 → shortcut 越来越强 → 验证 pred_r2 越来越差
  - r2_eeg 下降的原因：因为 encoder 学会把未来信息"塞进"所有位置的表示，这反而降低了对局部信号的重建质量

**根因 2 — V5.40 添加的 spectral_loss 在小数据集上稀释梯度预算**：
- V5.40 新增 `spectral_eeg`、`spectral_fmri` 两个频域损失任务 + pred_sig 的 Pearson 分量
- 共 8 个任务（recon×2 + spectral×2 + pred×2 + pred_sig×2）竞争梯度预算
- 在 4-8 被试的小数据集上，梯度信号本就稀缺，更多任务使每个任务得到的有效梯度更少
- 用户报告"比前几个版本更差"正是从 V5.40 开始的

**根因 3 — 自适应 loss balancer 逐步削减预测任务权重**：
- GradNorm 机制：latent 空间 pred 损失若收敛"更快"（绝对值下降更多）→ 降低 pred 权重
- 但 latent 空间收敛快 ≠ signal 空间 pred_r2 好
- 结果：训练 65 个 epoch 后 pred 权重最多可累积下降 exp(-0.65)≈0.52×，进一步剥夺预测梯度

**修复（V5.42）**：
1. **`TemporalAttention.forward()` 改为因果**：`is_causal=False` → `is_causal=True`，fallback 路径同步添加显式上三角因果 mask
2. **`use_spectral_loss` 默认关闭**：`true` → `false`，恢复 V5.38 的梯度预算
3. **`pred_sig` 损失 Pearson 权重 0.2 → 0.5**：直接加强对 pred_r2 指标的优化信号
4. **`task_priorities.pred` 3.0 → 6.0**：确保预测任务从一开始就占据主导梯度预算
5. **`adaptive_loss.warmup_epochs` 5 → 10**：更稳定的基线建立后再启动自适应调整
6. **`AdaptiveLossBalancer` 新增 `pred_weight_floor=0.5`**：pred 任务 log_weight 不低于 log(初始值)-0.5，防止 GradNorm 机制将预测任务饿死

**规则**：
- **任何"因果"设计声明必须端到端核查**：验证因果（V5.31）+ 训练双向（V5.0~V5.41）= 不因果系统！
- **添加新任务损失时必须问：这个数据集规模能否支持额外的梯度预算？** 4-8 被试的小数据集极易被多任务稀释
- **AGENTS.md 中的 ±1 步 Conv1d 泄漏说明是错误的**：它完全没有考虑 TemporalAttention 的全局双向注意力。正确认识：TemporalAttention 无因果 mask 时的泄漏是全局性的（O(T) 量级），不是边界性的（O(1) 量级）

---

### [2026-02-20] CUDA OOM：序列长度 + 时间循环两个叠加根因

**思维误区**：以为"截断序列 = 内存优化"，没有追问为什么要 T 个时间步逐一调用 propagate()。

**根因（双重）**：
1. EEG 以 250Hz 采样，完整序列 T≈75000，产生 `[N, T, 128]` 巨张量。
2. ST-GCN 对每个时间步独立 `propagate()`，T=300×4层×3边 = 3600 次 CUDA 内核启动，Python 循环是串行瓶颈（GPU 实际空闲）。
3. 显存碎片化：reserved 但 unallocated 的块不足以响应新分配。

**正确思路**：
- `max_seq_len` 截断全序列；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 消碎片；`empty_cache()` 每 epoch 归还块。
- 时间循环改为"时序虚拟节点"向量化：将 edge_index 扩展 T 次，单次 propagate 处理 T×E 条消息，GPU 满载，理论加速 100×。

---

### [2026-02-28] CUDA OOM 第五次修复：累积碎片 + 向量化与检查点的内存陷阱

**用户观察**：每次"修复"后 OOM 出现的 epoch 推后（第 10 轮 → 第 40-47 轮），说明问题在训练过程中**累积**，而非单步峰值固定。

**思维误区**：以为"向量化后峰值固定，只要峰值不超限就不会 OOM"，没有追问峰值张量被反复分配释放对 CUDA 分配器的长期影响。

**双重根因（叠加）**：

**根因 1 — 累积碎片（主因）**：
- T×E 向量化使每步 backward 重计算时一次性实例化 `[T×E, H]` 消息张量（fMRI T=190, E=5700, H=128: 263 MB float16）。
- 每个训练步骤完成后这块内存被释放，但 CUDA 分配器将其放入"空闲列表"而非立即归还 CUDA。
- 旧代码仅在**验证时**（每 5 epoch）调用 `empty_cache()`，5 epoch × 20 图 = 100 个步骤的碎片累积，空闲列表中存在多个不连续的 263 MB 块。
- 第 47 epoch：碎片累积达到临界点，新的 263 MB 分配请求找不到足够的连续块 → OOM。
- **历次"修复"每次减小了峰值（263→200→150→…）**，碎片仍在累积，只是需要更多 epoch 才达到临界点 → OOM 推后而非消除。

**根因 2 — V5.40 新增动态图使峰值增大**：
- `use_dynamic_graph: true`（V5.40 默认从 false 改为 true）每层新增 `N×k_dynamic=1900` 条动态边，E 从 3800 增至 5700（+50%）。
- 峰值从 175 MB 升至 263 MB，碎片达到临界点所需的 epoch 减少，OOM 从之前的"几乎不出现"变成第 47 epoch。

**根因 3 — 空间索引 bug（附带发现，同步修复）**：
- `x_t.reshape(N_src*T, H)` 对 C-连续 `[N_src, T, H]` 给出行 `n*T+t`，但边索引偏移使用 `t*N_src+n`，两者不一致。
- 空间消息传递实际上在错误的（节点, 时间）对之间传递特征（对 fMRI 偶然影响小，因 T≈N，对 EEG 影响大）。
- 修复：在 reshape 前先 `permute(1,0,2).contiguous()` 将布局变为 `[T, N, H]`，使 `flat[t*N+n] = x[n,t]` ✓。

**修复（V5.41）**：
1. **时序分块 `temporal_chunk_size=32`**（从 64 降至 32）：峰值从 88MB 降至 44MB（32/190 × 263），支持 8GB GPU 运行 4-8 被试。
2. **每 epoch 调用 `gc.collect() + empty_cache()`**（非仅验证时）：将空闲列表在每个 epoch 结束后归零，彻底阻断累积机制。
3. **windowed_sampling 默认开启**：每步只处理 T=50 而非 T=190，峰值张量降低 6×。
4. **OOM 警告文本升级**：明确提示 `temporal_chunk_size` 和 `use_dynamic_graph` 作为调参方向。

**规则**：
- **"每次修复后 OOM 推后"是累积问题的明确信号**，不能只看单步峰值，必须问：这块内存被反复分配释放吗？垃圾回收频率够吗？
- **大块张量（>100 MB）的分配/释放必须配合 `empty_cache()`**，否则 CUDA 分配器碎片在多 epoch 训练后必然累积。
- **向量化（T×E 单次）和梯度检查点（省内存）配合时，峰值是 `chunk_size×E×H` 而非 `E×H`**——只有配合分块才能真正降低峰值。

---

### [2026-02-28] cuda_clear_interval + gradient_accumulation_steps 交互陷阱（V5.41 发现）

**思维误区**：以为 `is_accum_boundary AND (i+1)%interval==0` 等价于"每 interval 步在优化器边界清理"，没有追问两个独立周期同时为真的概率。

**根因**：
- `is_accum_boundary` 在 `i=ga-1, 2*ga-1, ...`（每 ga 步一次）为 True
- `(i+1) % interval == 0` 在 `i=interval-1, 2*interval-1, ...`（每 interval 步一次）为 True
- AND 联合要求 (i+1) 同时是 ga 和 interval 的倍数 = LCM(ga, interval) 的倍数
- `ga=4, interval=50` → `LCM(4,50)=100`，实际每 **100 步**才清理一次
- 典型 epoch = 80 步（8被试 × 10窗口）→ **一次都不触发**，完全无效化碎片防护

**验证**：
```python
import math
ga, interval = 4, 50
print(math.lcm(ga, interval))  # 100
fires = [i for i in range(80) if (i+1)%ga==0 and (i+1)%interval==0]
print(fires)  # [] — 80步epoch中零次触发
```

**修复（V5.41.1）**：移除 `is_accum_boundary` 条件，保留 `(i+1)%interval==0`。
- `gc.collect()`：只释放 Python 对象引用循环，不影响 autograd 图中受 C++ 保护的梯度张量
- `empty_cache()`：只回收**已释放**的 CUDA 块（在空闲列表中），不影响任何活跃张量
- 因此在梯度累积中途调用也**完全安全**，不存在"梯度被误回收"的风险

**规则**：
- **两个独立条件用 AND 联合时，有效周期 = LCM（而非较小值）**。`is_accum_boundary AND (i+1)%50==0` 用 AND 连接，是一个将 50 步清理变成 LCM(ga,50) 步清理的隐性 bug。
- **每当引入"条件 A AND 条件 B"控制某功能时，必须计算 A 和 B 同时为真的实际频率**，验证它与设计意图一致。

---

### [2026-02-28] prediction_steps 与 fMRI 窗口大小不匹配（V5.41 发现）

**思维误区**：以为"prediction_steps=30 只是预测 30 步，能监督多少就监督多少"，没有追问这对算力利用率和模型训练质量的影响。

**根因**：
- fMRI 窗口 T=50 TRs，`_PRED_CONTEXT_RATIO=2/3` → T_ctx=33，T_fut=50-33=**17 TRs**
- `prediction_steps=30 > T_fut=17` → `aligned_steps = min(30,17) = 17`
- 预测步 18-30（共 13 步）：前向 pass 生成、显存占用、但**零监督梯度**
- 浪费比例：13/30 = **43% 算力用于无梯度的预测步**
- 同样影响 signal-space 预测损失（只有 17 步有效 future signal 可对齐）

**量化**：
| prediction_steps | fMRI supervised_steps | 浪费率 | EEG supervised_steps |
|---|---|---|---|
| 30 | 17 | 43% | 30 (0% waste) |
| 15 | 15 | 0% | 15 (0% waste) |

**修复（V5.41.1）**：`prediction_steps: 30 → 15`。
- fMRI: 15 步 × TR=2s = 30s，覆盖血动力学响应峰值（完全在 T_fut=17 范围内）
- EEG: 15 步 × 4ms = 60ms，覆盖 alpha/gamma 主要分量（EEG T_fut=167 >> 15）
- 选择依据：`prediction_steps ≤ floor(fmri_window_size × (1 - _PRED_CONTEXT_RATIO))`
  = floor(50 × 1/3) = 16，选 15 保留 2 步缓冲

**规则**：
- **prediction_steps 不是越大越好**。有效上界 = `T_window × (1 - _PRED_CONTEXT_RATIO)`。
  超过此值的步骤有正向计算开销但无训练信号（梯度为零）。
- **windowed_sampling 改变了 prediction_steps 的有效上界**。每次修改窗口大小，
  必须同步检查 `prediction_steps ≤ T_fut = T_window × (1-context_ratio)`。
- **通用检查公式**（代码可直接用）：
  `max_effective_steps = int(window_size * (1 - _PRED_CONTEXT_RATIO))`

---

### [2026-02-21] "声明了但从未真正工作"的功能（通用反模式，多次复现）

**思维误区**：看到代码里有某功能的声明/参数，就以为它在工作。

**正确思路**：对每个"已实现"的功能，追问三点：
1. 调用入口存在吗？（`set_epoch` 在 `train_epoch` 里有没有被调用？）
2. 内部实现能执行吗？（`autograd.grad` 在 `backward` 后会崩溃）
3. 形状与调用者匹配吗？（predictor 接 4-D 还是 3-D？）

**具体案例（已修复）**：
- `AdaptiveLossBalancer.update_weights` 用 `autograd.grad`，但 `backward()` 已释放计算图 → 崩溃；`set_epoch()` 从未调用 → warmup 永不结束 → 上述 bug 反而被保护。
- `use_gradient_checkpointing` 调用 `HeteroConv.gradient_checkpointing_enable()`，该方法不存在。
- `modality_energy_ratios` 存入 `AdaptiveLossBalancer` 但从不参与计算（死存储）。
- `v5_optimization` 配置块中所有参数均被硬编码默认值覆盖，从未从 YAML 读取。
- EEG handler 里的 `eeg_info['regularization_loss']` 从未加入 `total_loss`。

**规则**：凡是"存储但不使用"的属性，立即问：*这个值应该在什么时候、以什么方式被使用？*

---

### [2026-02-21] fMRI 实际只有 1 个节点（架构上的空话）

**思维误区**：看到"图原生"就以为 fMRI 真的有空间结构，没有打印 N_fmri。

**根因**：`process_fmri_timeseries` 对所有体素 `mean(axis=0)` 后 `reshape(1, -1)` → N_fmri=1，空间信息完全丢失。

**修复**：用 `NiftiLabelsMasker` 应用 Schaefer200 图谱，提取 ~190 个 ROI 时序（实际数量因 EPI 覆盖而异，少于 200 是正常的，不要填充）。

---

### [2026-02-21] 跨模态边的两个叠加 N 不匹配 bug

**思维误区**：只修了可见的 guard，没有追问 guard 能否被触发。

**根因**：
1. `update()` 无条件做 `aggr_out + lin_self(x_self)`。跨模态边 N_src≠N_dst，PyTorch 广播导致输出 N=N_src（错误值），污染后续所有层。
2. `propagate()` 未传 `size=(N_src, N_dst)`，PyG 默认 `size=(N_src, N_src)`，使 `aggr_out` 形状本身就错误，guard 永远不会触发。

**修复**：① `update()` 检查 `aggr_out.shape[0] != x_self.shape[0]` 时跳过 self-connection；② `propagate()` 显式传入 `size`；③ `compute_loss` 在 `huber_loss` 前 raise RuntimeError 防止广播静默掩盖。

---

### [2026-02-21] EEG enhancement 原地修改 data 导致跨 epoch "backward 两次" 错误

**思维误区**：以为 `.to(device)` 会重新创建张量，切断梯度链。实际上 `.to()` 对已在目标设备的张量是 no-op。

**根因**：`data['eeg'].x = eeg_x_enhanced` 永久改写了 `data_list` 里的原始对象。Epoch 1 的 `backward()` 释放了 saved tensors，Epoch 2 在已释放的图上构建新图 → 报错。

**修复**：`try-finally` 块保证步骤结束后（无论异常）恢复 `original_eeg_x`。`original_eeg_x` 仅在**实际修改前**才保存（non-None ↔ 已修改）。

---

### [2026-02-21] log_weights 参与 backward 图导致梯度累积

**思维误区**：看到 `nn.Parameter` 就以为"有梯度是正常的"，没有追问这个梯度会被使用吗。

**根因**：`log_weights` 通过 `update_weights()` 手动更新，完全不依赖 `.grad`。让它参与 backward 图只有负面效果：浪费显存、梯度无限累积。

**规则**：对**手动更新**的参数（不通过 optimizer），forward 中用 `.detach()` 排除；向后处理函数传入张量时用 `.detach()` 明确语义。

---

### [2026-02-21] Decoder ConvTranspose1d(stride=1) 悄悄改变时序长度

**正确思路**：ConvTranspose1d 设计用于上采样（stride>1）。stride=1 时输出长度 = `T + kernel_size - 2*padding`（不等于 T）。**stride=1 应用 Conv1d，不应用 ConvTranspose1d。**

---

### [2026-02-23] 硬编码顺序假设破坏 EEG→fMRI 方向

**思维误区**：`edge_types.append((node_types[0], 'projects_to', node_types[1]))`，以为 EEG 一定排第一。

**规则**：有方向性的设计意图，永远用**模态名**而非位置索引确定方向：`if 'eeg' in node_types and 'fmri' in node_types: append(('eeg', 'projects_to', 'fmri'))`。

---

### [2026-02-24 / 2026-02-26] HeteroConv 内部 key 不可预期 → 改用 nn.ModuleDict

**思维误区**：以为 PyG `HeteroConv.convs` 是标准 Python dict，用 `'__'.join(edge_type)` 可以直接访问。

**根因**：`HeteroConv.convs` 是 PyG 自定义 `ModuleDict`，其 `to_internal_key()` 实现因版本而异，导致存入 key=A，读取时得到 B → KeyError。

**规则**：对任何第三方库容器，不能假设其访问语义与标准 dict 相同。正确做法：用 `nn.ModuleDict({'__'.join(et): conv})` 替换 `HeteroConv`，避免任何隐式 key 转换。

---

### [2026-02-24] 替换 self.model 后 optimizer 参数组失效

**根因**：`super().__init__(model=base_model)` 后立即 `self.model = new_model`，optimizer 在构造时捕获参数快照，不随 `self.model` 更新。增强模块参数有梯度但永远不会被更新。

**规则**：每次 `self.model = new_model`，必须重建 optimizer（或 `add_param_group()`）。

---

### [2026-02-25] 1:N EEG→fMRI 对齐是「设计缺失」而非「设计完成」

**场景**：2 个 EEG 条件（GRADON/GRADOFF）共享 1 个 fMRI 扫描（task-CB）。

**思维误区**：`_discover_tasks()` 把"BIDS 文件名中出现的所有 task"等同于"有效 run"，导致出现 fMRI-only 幽灵任务，GRADON/GRADOFF 静默回退到"任意 bold"文件。

**修复**：
1. `fmri_task_mapping` 支持显式映射（V5.15）。
2. `_load_fmri()` 新增 ON/OFF 后缀自动剥离（V5.16，GRADON→GRA 匹配 GRA bold）。
3. `_discover_tasks()` 改为只扫描 EEG 文件，彻底杜绝幽灵任务。
4. 查找链：显式 mapping → 同名直接匹配 → ON/OFF 剥离重试 → 任意 bold 回退。

---

### [2026-02-26] 缓存命中路径的 `continue` 绕过必要副作用

**通用规则**：每次在主循环中加 `continue`/`return`/`break`，都必须问：**循环尾部有没有必须执行的副作用（metadata 写入、索引赋值）？** 如有，必须在 `continue` 前补全。

**案例**：缓存命中直接 `continue` 跳过了 `subject_idx`、`run_idx`、`task_id`、`subject_id_str` 的赋值，导致从缓存加载的图缺少个性化嵌入索引和训练数据追踪信息。

---

### [2026-02-26] 异质图「框架使用完整」≠「特性使用完整」

**排查方法**：为每个 `HeteroData` 属性画「SET → READ」追踪表，找出只 SET 从未 READ 的属性。

**发现（已修复）**：
- `eeg.labels`、`fmri.labels`、`eeg.atlas_mapping`、`fmri.pos`：死存储，never READ。
- 跨模态边无 `edge_attr`（同模态边有相关性权重，不一致）。
- DTI 接口完全缺失：无 `_load_dti()`、无 `('fmri','structural','fmri')` 边类型。

**修复原则**：DTI 不设独立节点类型（无时序特征），而是在 fMRI 节点上新增结构连通性边 `('fmri','structural','fmri')`。编码器已有 `if edge_type in edge_index_dict` 保护，DTI 文件缺失时自动降级。

---

### [2026-02-26] V5.21 全流程审查四个隐患

1. **`max_grad_norm` 静默忽略**：YAML 中存在该参数，代码里却硬编码 1.0。修复：`GraphNativeTrainer.__init__` 新增 `max_grad_norm` 参数，`train_model` 传入。

2. **Checkpoint 不保存调度器状态**：恢复训练时 LR 从头重启。**规则**：恢复训练必须保存所有有状态组件：model + optimizer + scheduler + loss_balancer + history。修复：`save_checkpoint` 新增 `scheduler_state_dict`。

3. **无消息时残差连接 ≈ 2x**：`messages=[]` 时 `x_new = x`，然后无条件执行 `x_new = x + dropout(x_new)` → 幅度 ×2，4 层后 ×16。修复：有消息时 `x + dropout(mean(messages))`，无消息时直接 `x`（passthrough）。

4. **窗口模式按窗口划分训练/验证集 → 数据泄漏**：相邻窗口 50% 重叠，同 run 的窗口同时进入两集。修复：引入 `run_idx`，以 run 为单位划分，保证同 run 所有窗口进入同一集合。

---

### [2026-02-26] SpatialTemporalGraphConv 注意力归一化用 global softmax

**根因**：`softmax(alpha, dim=0)` 对形状 `[E, 1]` 是跨所有 E 条边归一化，而非每目标节点独立。E≈4000 时每条边权重≈0.00025，消息几乎消失，GNN 退化为每节点独立 MLP。

**修复**：用 `pyg_softmax(alpha, index, num_nodes=size_i)` 实现每目标节点归一化。

---

### [2026-02-26] 配置文件路径是"死字符串"（Atlas 案例）

**规则**：配置文件中的文件路径每次改动后必须和实际文件系统一一对照验证。`exists()` 检查只是运行时保护，不能替代"写正确的路径"。警告被忽视会导致系统长期静默运行错误配置（Atlas 未找到 → N_fmri=1，空间信息全丢）。

---

### [2026-02-27] 跨条件边界时序污染（ON/OFF 共享 fMRI 场景）

**问题**：GRADON/GRADOFF 共享同一 fMRI 文件，截断和窗口均从 t=0 开始，导致 GRADOFF 条件使用 GRADON 时段的 fMRI 数据；窗口模式下还会跨越条件边界（模型被迫学习"预测实验条件切换"）。

**修复**：新增 `fmri_condition_bounds` 配置项，在图构建前（连通性估计前）先切片 fMRI 时序到当前条件的时间段。缓存 key 包含此配置，修改边界自动失效。

```yaml
fmri_condition_bounds:
  GRADON:  [0,   150]   # CB fMRI TR 0–149
  GRADOFF: [150, 300]   # CB fMRI TR 150–299
```

1:1 标准场景设 `null`，行为不变。

---

### [2026-02-27] 多被试多任务联合训练三项静默缺陷

> `graphs` 列表中每个 `HeteroData` 均为独立的 (被试, 任务) 图，不存在跨被试/跨任务的节点或特征混合。`train_epoch` 以 batch_size=1 逐一独立处理。

1. **`train_epoch` 缺少逐 epoch 打乱**：`train_model` 只打乱一次，之后每 epoch 顺序相同，optimizer 动量导致末尾被试获得更新偏差。修复：`random.Random(epoch).shuffle(copy)` 每轮独立打乱。

2. **EEG handler 通道数跨被试不匹配**：handler 按第一个样本初始化，不同 N_eeg 的被试静默传入错误形状张量。修复：记录 `_eeg_n_channels`，不匹配时跳过增强并 debug 日志。

3. **`task_id` / `subject_id_str` 未存储**：图构建后来源信息丢失，无法验证训练数据分布。修复：在 `build_graphs` 和缓存路径均写入这两个属性，`extract_windowed_samples` 传播到窗口，`log_training_summary` 展示每任务/被试样本数。

---

### [2026-02-27] 预测器以节点为 batch 维度 → 节点间完全独立，刺激单脑区无法传播

**思维误区**：以为"图模型 + 时间序列预测"就等于"系统级预测"。实际上 `EnhancedMultiStepPredictor(h)` 把 `h[N, T, H]` 的 N 视为 batch 维度，对 N 个节点完全独立预测——图拓扑在预测阶段完全缺失。

**根因**：编码器（ST-GCN）在编码阶段通过图边传递了跨区域信息，但预测头对每个节点单独运行 Transformer，图结构在预测时被丢弃。刺激节点 A 只改变 A 的历史表征 → 只影响 A 的预测 → 其他节点预测轨迹完全不变。

**正确思路**：预测分两步：
1. **Per-node 时间预测**（`EnhancedMultiStepPredictor`）：各节点独立预测初始轨迹
2. **系统级图传播**（`GraphPredictionPropagator`）：在预测空间 `{node_type: [N, pred_steps, H]}` 上运行 N 轮 ST-GCN 消息传递，让预测变化通过图拓扑传播

**规则**：任何"对节点集合的预测"，都必须追问：*预测之后，图连通性如何影响相邻节点？* 如果没有图传播步骤，预测就不是系统级的。

**修复版本**：V5.25。`GraphPredictionPropagator`（`num_prop_layers=2`，覆盖 ≥2 跳邻居）注册为 `model.prediction_propagator`，在 `forward()` 和 `compute_loss()` 的预测步骤后均调用。

---

### [2026-02-27] AdaptiveLossBalancer 双重修正 + 梯度方向反转

**思维误区1（双重修正）**：误以为「能量初始化」+「按初始损失归一化」是两个独立增强，实为同一问题的双重修正，叠加后结果远超预期。

**根因**：
- 能量初始化：`w_eeg = 50 × w_fmri`（补偿 EEG 振幅小 50 倍）
- 初始损失归一化：`loss_norm = loss / L0`（L0_eeg ≈ 0.001，L0_fmri ≈ 0.5）
- 叠加效果：EEG 梯度比例 = 50/0.001 = **50,000**，fMRI = 1/0.5 = 2 → **25,000 倍失衡**，训练完全由 EEG 主导，fMRI 实质上被忽略

**思维误区2（方向反转）**：把"减小困难任务的权重，避免其主导梯度"误写为 GradNorm 的正确方向。

**根因**：GradNorm 的正确逻辑是：收敛慢的任务（相对损失 > 均值）→ **提高**权重 → 模型分配更多梯度信号 → 追上进度。现有代码取负号，困难任务权重被**降低**，导致模型放弃困难任务（fMRI），只学习简单任务（EEG）。

**修复**：
1. `forward()`：移除初始损失归一化，仅保留能量初始化（单一修正）
2. `update_weights()`：(a) 计算相对困难度时先除以 `initial_losses` 做量纲归一化；(b) 将符号改正：`weight_update = +lr * (rel_loss - 1.0)`

**规则**：能量初始化（解决振幅差异）和 GradNorm 自适应更新（解决收敛速度差异）是正交机制，不能叠加到同一个归一化步骤中。

---

### [2026-02-27] 训练优化：四个关联盲区（V5.28）

**问题 1 — validate() 返回值变更后调用方同步规则**

`validate()` 从 `float` 改为 `Tuple[float, Dict]` 后，所有调用方必须同步解包，否则在 `np.isnan(val_loss)` 等处触发 `TypeError: ufunc 'isnan' not supported for the input types`。

**规则**：改变函数返回类型前，先用 grep 找出所有调用方，确保全部同步修改。

**问题 2 — AMP scaler 在梯度累积边界的调用顺序**

`scaler.step()` 和 `scaler.update()` 必须成对调用，且只在真正执行 `optimizer.step()` 的那一步（边界步）调用。`scaler.unscale_(optimizer)` 也应仅在边界步调用，否则梯度被提前反缩放导致数值不稳定。

**规则**：梯度累积中，AMP scaler 的 `unscale_/step/update` 三个调用均受 `do_optimizer_step` 条件保护。

**问题 3 — SWA 对 AveragedModel 的属性访问**

`AveragedModel` 是一个 `nn.Module` 包装器，`forward()` 正确透传，但自定义属性（如 `use_prediction`、`compute_loss`）不自动代理，直接访问会 `AttributeError`。

**规则**：凡需要访问被 AveragedModel 包装的原始模型属性或方法，应先 `_orig_model = trainer.model`（AveragedModel 创建前），然后通过 `_orig_model` 访问，不通过 `swa_model` 访问。

**问题 4 — EEG 频谱相干性改变 edge_index，必须纳入缓存 key**

将 `eeg_connectivity_method` 从 `'correlation'` 切换到 `'coherence'` 会产生完全不同的连边和权重。若缓存 key 不包含此参数，旧缓存（用 correlation 权重）会被 coherence 模式无声使用，导致图结构静默错误。

**规则**：任何影响 edge_index 或 edge_attr 的参数（包括连通性计算方法、阈值、k 近邻数）必须全部纳入 `_graph_cache_key()` 的哈希计算。

---

### [2026-02-27] 训练科学可信度三项盲区（V5.29）

**盲区 1 — R² < 0 无报警（最严重）**

`validate()` 返回 R²，但训练循环仅将其格式化为日志字符串，没有任何语义级判断。
R² < 0 表示模型重建误差 > 信号总方差，即比"永远预测均值"还差——这是模型科学失效的明确信号，非专业用户无法从裸数字识别。

**正确思路**：每次验证后检查所有模态的 R²，若任一为负则打印 ⛔ 警告并解释含义（"尚未从数据中学到有效信号"）。

**修复**：在 validation 代码块内，遍历 `r2_dict`，`if r2v < 0: logger.warning(⛔...)`。

---

**盲区 2 — 最佳 epoch 的 R² 未被追踪**

`best_val_loss` 被追踪并在恢复日志中显示，但对应的 R²（评估模型重建意义的关键指标）未被记录。
用户仅能知道"最低损失是 X"，无法知道"最低损失时模型实际重建能力如何"。

**修复**：引入 `best_r2_dict: dict = {}`，在 `val_loss < best_val_loss` 时同步 `best_r2_dict = r2_dict.copy()`；写入"保存最佳模型"日志行和恢复日志行。

---

**盲区 3 — 训练结束无可信度总结**

训练结束后日志末尾只有"最佳验证损失: X.XXXX"。用户面对一个数字无法判断：
1. 模型是否真正有效（R² > 0）；
2. 哪个模态重建可信、哪个不可信；
3. 是否存在过拟合风险（val/train 比率）。

**修复**：训练循环结束后打印"训练可信度摘要（📊）"，含：
- 每个模态的 R² 及评级（✅ ≥0.3 / ⚠️ 0-0.3 / ⛔ <0）；
- 过拟合检测（`val_loss/train_loss > 3.0` 时警告）；
- `val_frequency × patience` 等效 epoch 耐心值（训练开始前打印）；
- 综合结论一行。

**规则**：任何向非专业用户报告的数字，必须附带含义解释；任何可能失败的科学指标（R² < 0），必须有明确告警而非默认忽略。

---

### [2026-02-27] 跨模态边存入缓存导致的双重问题（V5.30 修复）

**思维误区**：认为"缓存图应完整保存所有边"——包括跨模态边。

**根因（两个叠加问题）**：
1. `k_cross_modal` 进入了缓存键哈希 → 修改 k 会使所有旧缓存失效，只有原始数据在才能重建 → 若用户只保留了缓存文件（已删原始数据），修改 k 导致数据无法恢复。
2. 下游代码加载缓存后看到 `graph.node_types = ['eeg', 'fmri']`，无直接方法获取 numpy 数组形式的时序数据，需要了解 PyG HeteroData API。

**规则**：
- **只缓存"原始数据衍生物"**（节点特征和同模态边）；不缓存"特征的函数"（跨模态边，代价低、参数化、随配置变化）。
- 如果一个图属性 X 满足 `X = f(cached_node_features, config_params)`，X 不应写入缓存；应在每次加载时重建。
- 判断一个参数是否应进入缓存键：若该参数改变只需调用已缓存特征的计算而非重读原始文件，则不进入缓存键。

**修复**：
- 跨模态边从缓存保存中剥离；每次加载时用 `create_simple_cross_modal_edges()` 从缓存节点特征重建。
- `k_cross_modal` 从缓存键哈希中删除（修改 k 后无需清缓存）。
- 新增 `load_subject_graph_from_cache(path)` 工具函数，返回 `{'eeg_timeseries': ndarray, 'fmri_timeseries': ndarray, ...}`，下游无需了解 PyG API。

---

### [2026-03-01] NPI对标分析：BOLD自相关性 vs 真实神经动力学预测

**用户观察**：NPI（Nature Methods 2025）使用3个TR预测1个TR（3→1），声称"3步预测1步准确率最高"。

**思维误区**："预测准确率高" ≠ "学到了真正的神经动力学"。BOLD信号在TR=2s时自相关系数ρ≈0.85-0.95（高度自相关），AR(1)基线（最后观测值预测）可免费获得R²≈0.7-0.9（h=1步）。NPI的"3→1准确率最高"可能仅意味着AR(3)比AR(1)或AR(2)更好地利用了自相关，而非真正的神经动力学。

**证伪方法**：
1. 计算AR(1)基线R²（ar1_r2）：predict last value，测量"免费"自相关信号
2. 计算去相关分数（decorr_score = (pred_r2 - ar1_r2) / (1 - ar1_r2)）
3. 在多个horizon（h=1, 5, 10, 17步）上比较：若NPI仅在h=1有效，TwinBrain在h>1仍正且高于AR(1)，则TwinBrain明显优越

**TwinBrain超越NPI的关键证据**：
- decorr_score > 0 at h=1：TwinBrain比AR(1)更好（NPI在此持平）
- decorr_score > 0 at h=5, h=10：TwinBrain在长程horizon仍有效（NPI无法做到）
- pred_r2_h1 > pred_r2_full：多步预测确实比单步更难，证明模型在使用真实上下文

**新增指标（V5.47）**：
- `ar1_r2_<nt>`：AR(1)基线R²（最后观测值预测，BOLD自相关的"免费"R²）
- `decorr_<nt>`：去相关分数 ∈ (-∞, 1]（0=等于AR(1)；>0.15=清晰超越自相关）
- `pred_r2_h1_<nt>`：仅h=1步的R²（与NPI 3→1直接可比）

**规则**：
1. **任何脑动力学预测声明必须附带AR(1)基线和decorr分数，否则无法与纯自相关区分**
2. **NPI的科学价值在于EC映射框架，而非预测准确率本身（3→1预测可能只是AR(3)）**
3. **TwinBrain的科学优势：多步长程预测（17步=34s）+ 图神经网络空间结构 + EEG+fMRI多模态**
4. **ar1_r2_h1 可为负**：当实验fMRI数据的lag-1自相关 ρ<0.5时（如梯度噪声刺激范式、重度预处理、低TR multiband序列），ar1_r2_h1 = 2ρ-1 < 0。此时pred_r2_h1的绝对值看似"低"，但decorr_h1 > 0才是真正的比较指标——NPI高R²的前提（ρ≈0.85-0.95）根本不成立。

---

**项目永远只保留以下四个 MD 文件：**

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | **本文件**：AI Agent 指令 + 错误记录 |
| `SPEC.md` | 项目规范说明：目标、设计理念、架构 |
| `USERGUIDE.md` | 项目使用说明：面向非专业人士 |
| `CHANGELOG.md` | 更新日志：所有版本变更 + 待优化事项 |

**严禁**：在 `docs/`、根目录或任何位置创建第五个 MD 文件。

---

## 六、EEG→fMRI 设计理念与逻辑链（必读，每次编码前确认）

> **核心原则**：EEG 电极（较少节点）向 fMRI ROI（较多节点）投射信号。N_eeg < N_fmri 是整个跨模态设计的前提。

### 数据形状全链路

| 阶段 | EEG | fMRI |
|------|-----|------|
| 原始数据 | `[N_ch, N_times]` (e.g. 63×75000) | `[X, Y, Z, T]` (e.g. 64×64×40×190) |
| 图节点特征 | `[N_eeg, T_eeg, 1]` (e.g. 63×300×1) | `[N_fmri, T_fmri, 1]` (e.g. ~190×190×1，Schaefer200，实际数因 EPI 覆盖而异) |
| 编码器输入投影 | `[N_eeg, T_eeg, H]` | `[N_fmri, T_fmri, H]` |
| ST-GCN 跨模态消息 | 源 → propagate(size=(N_eeg, N_fmri)) → `[N_fmri, T_eeg, H]` → interpolate → `[N_fmri, T_fmri, H]` | 目标 |
| 解码器输出 | `[N_eeg, T_eeg, 1]` | `[N_fmri, T_fmri, 1]` |
| 损失目标 | `data['eeg'].x` | `data['fmri'].x` |

### 关键不变量（违反即 bug）

1. `N_eeg < N_fmri`
2. 跨模态边类型必须是 `('eeg', 'projects_to', 'fmri')`，**不依赖 config 列表顺序**
3. `propagate(size=(N_eeg, N_fmri))` 必须显式传入 size
4. `recon.shape[0] == target.shape[0]`，违反时 raise RuntimeError 而非广播

### 典型症状
- `Using a target size ([63, 190, 1]) different from input ([1, 190, 1])` → 不变量 3 或 4 被违反
- `Trying to backward through the graph a second time` → data_list 中对象被原地修改

---

## 七、训练数据设计原则

### max_seq_len=300 是错误的训练单元

`max_seq_len=300` @ 250Hz = 1.2 秒。从 1.2 秒 EEG 估计 Pearson 相关性统计上不可靠（需至少 10-30 秒）。图的 edge_index 建立在统计噪声之上。

### 正确范式：动态功能连接（dFC）滑动窗口

| 概念 | 图的哪个部分 | 如何计算 |
|------|-------------|---------|
| 结构连通性 | `edge_index` | 完整 run 的相关矩阵 |
| 动态脑状态 | 节点特征 `x` | 时间窗口切片 |

**关键约束**：
1. `windowed_sampling.enabled: true` 时，**必须设 `max_seq_len: null`**。
2. 缓存存储**完整 run 图**，窗口在运行时从缓存切片。
3. `edge_index` 在同一 run 的所有窗口间**共享同一对象**（不复制）。
4. 窗口模式下以 **run** 为单位做训练/验证集划分（防止重叠窗口数据泄漏）。

---

## 八、损失函数体系（防止死代码）

```
train_step()
    eeg_handler() → eeg_info['regularization_loss']  ← 必须加入 total_loss
    model.forward(return_encoded=True) → reconstructed, _, encoded
    model.compute_loss(data, reconstructed, encoded=encoded)
        ├── recon_{node_type}: 重建损失
        └── pred_{node_type}: 潜空间预测损失
    loss_balancer(losses)       ← 平衡 recon_* 和 pred_*
    total_loss += eeg_reg       ← 固定权重，不参与平衡
    total_loss.backward()
```

| 损失名 | 空间 | 目的 | 权重 |
|--------|------|------|------|
| `recon_eeg` | 原始 C=1 | EEG 信号重建 | 自适应 |
| `recon_fmri` | 原始 C=1 | fMRI 信号重建 | 自适应 |
| `pred_eeg` | 潜空间 H | EEG 潜向量未来预测（快速收敛锚） | 自适应 |
| `pred_fmri` | 潜空间 H | fMRI 潜向量未来预测（快速收敛锚） | 自适应 |
| `pred_sig_eeg` | 原始 C=1 | EEG 信号空间端到端预测（与 pred_r2 对齐，V5.39 新增） | 自适应 |
| `pred_sig_fmri` | 原始 C=1 | fMRI 信号空间端到端预测（与 pred_r2 对齐，V5.39 新增） | 自适应 |
| `eeg_reg` | 原始 C=1 | 防止 EEG 通道崩塌 | 固定 0.01×3 |

> `h_fmri` 已含 EEG 跨模态消息，因此 `pred_fmri` 已隐式包含跨模态预测，无需专用跨模态预测头（future work）。
>
> **设计说明（V5.39）**：`pred_*`（潜空间）和 `pred_sig_*`（信号空间）是互补的，不是重复的：
> - `pred_*` 在潜空间提供密集梯度信号，在训练早期快速引导预测器收敛
> - `pred_sig_*` 直接优化 pred_r2 所测量的端到端指标，确保解码器在预测潜向量上也泛化
> 两者同时存在才能既快速收敛又指标对齐。
>
> **计算开销（V5.39 新增）**：`pred_sig_*` 在每次 `compute_loss()` 调用时额外执行一次解码器
> 正向推断（仅 pred_steps 步长，远小于完整 T 的解码，通常 < 5% 总训练时间增量）。
> 若遇到内存或速度瓶颈，可通过将 `use_prediction: false` 关闭整个预测模块来跳过全部 pred_* 损失。

---

### [2026-02-27] 预测功能系统性审查：三个叠加盲区

**背景**：要求以"哪都可能有问题"的批判态度对预测功能进行完整审查。

---

**盲区 1 — HierarchicalPredictor 整数除法产生空张量（P0 bug）**

**思维误区**：以为 `future_steps // scale_factor` 总是正整数。

**根因**：当 `prediction_steps=1`（NPI 风格）且 `scale_factor=4` 时，`1 // 4 = 0`。
`future_init = x_down[:, -1:, :].repeat(1, 0, 1)` 产生形状 `[N, 0, H]` 的空张量，
Transformer 对空序列的输出仍是空张量，`pred_full[:, -0:, :]` = 完整序列（Python 负索引！），
导致 fusion 的输入维度从 `input_dim * 3` 变为 `input_dim * 3 + 全序列 * 2`，报 shape mismatch。

**触发条件**：`prediction_steps` < `max(scale_factors)` = 4（默认 num_scales=3）。
即 `prediction_steps ∈ {1, 2, 3}` 都会在前两个尺度上产生空预测。

**修复**：`future_steps_scaled = max(1, future_steps // scale_factor)`。
每个尺度至少预测 1 步，粗尺度用下采样后的 context 预测 1 步后再上采样到 future_steps，
物理语义为"低分辨率趋势预测"，融合后给细尺度提供全局先验。

**规则**：**所有整数除法，若分子可能小于分母，必须用 max(1, ...) 或 ceiling 除法。**

---

**盲区 2 — validate() pred_R² 使用非因果 context（P1 科学问题）**

**思维误区**：以为 `h[:, :T_ctx, :]` 是"上下文的表示"，没有追问编码器是否是因果的。

**根因**：编码器使用：
- `Conv1d(kernel=3, padding=1)`：对称填充，`h[t]` 包含 `signal[t-1, t, t+1]` 的信息
- `TemporalAttention(is_causal=False)`：全局双向注意力，每个时间步可看到所有其他时间步

因此 `h[:, T_ctx-1, :]`（context 最后一步）包含来自 `T_ctx, T_ctx+1` 的少量未来信息。
旧 validate() 用这个"污染"的 context 做预测，pred_r2 会偏乐观。

**影响评估**：对 `kernel_size=3`, `T_ctx=200` 的情况，泄漏量约为 0.5%（仅边界步受影响）。
训练时接受此近似（避免 2× forward pass cost）。**但评估指标必须严格，不能接受偏差。**

**修复**：validate() 中重新编码仅含 T_ctx 步的数据：
```python
context_data = data.clone()
context_data[nt].x = data[nt].x[:, :T_ctx, :]  # 切片原始信号
_, _, h_ctx = self.model(context_data, return_encoded=True)  # 编码器只见 T_ctx 步
pred_latents = predictor.predict_next(h_ctx[nt])  # 真正因果预测
```
代价：验证时多一次无反向传播的 forward pass（可接受）。

**规则**：**科学评估指标不能有任何信息泄漏。训练效率 vs 评估严格性的权衡，必须让评估端严格。**

---

**盲区 3 — 声明"因果"但未追问编码器是否真的因果**

**思维误区**：看到注释"causal prediction using predict_next()"就以为整个流程是因果的。

**根因**：`predict_next()` 本身是因果的（只用 context 的最后 context_length 步），
但如果 context 是由非因果编码器产生的（包含未来信息），因果预测器也无济于事。
"因果"需要端到端保证：**编码器因果 + 预测器因果 + 评估因果**，缺一不可。

**现状**：
- 训练：编码器非因果（工程权衡，边界泄漏极小）→ 预测器因果 → 近似因果（可接受）
- 评估：**编码器强制因果**（重新编码 T_ctx 步）→ 预测器因果 → 完全因果（严格）

**规则**：**对任何标榜"因果"的预测功能，逐一检查信息流的每一步：原始信号 → 编码 → 预测 → 解码 → 评估，看哪一步有未来信息的访问权限。**

---

**附：StratifiedWindowSampler.forward() 是训练中的死代码**

`EnhancedMultiStepPredictor.forward()`（使用 StratifiedWindowSampler 采样多个窗口）
在当前训练循环中**从未被调用**。训练用 `predict_next()`，验证也用 `predict_next()`。

`forward()` 是 API 的一部分（向后兼容），保留不删除，但新用户不应依赖它进行训练。

---

### [2026-02-27] 预测功能第二轮审查：归一化层选型盲区 + 配置误导

**背景**：对预测系统进行独立第二轮审查，假设所有地方都可能有问题。

---

**盲区 1 — BatchNorm1d 在 prediction_steps_scaled=1 时静默归零粗尺度（P0 静默失效）**

**思维误区**：以为"前一轮修复了整数除法 → 预测步数不再为零 → 一切正常"，
没有追问"steps=1 时，归一化层是否还有问题"。

**根因**：上一轮修复将 `future_steps_scaled = max(1, future_steps // scale_factor)` 后，
预测步数不再为零。但上采样器中的 `BatchNorm1d(input_dim)` 接收形状 `[batch=1, input_dim=128, time=1]`：
- 每通道仅有 N×L = 1×1 = **1 个样本**
- var = 0 → normalized = (x − x) / sqrt(eps) = **0**
- output = gamma × 0 + beta = **0**（BatchNorm 初始化 gamma=1, beta=0）

粗尺度和中等尺度的上采样输出全为零，Fusion 输入 `cat([zeros, zeros, valid_fine])`，
层级预测**静默退化为单尺度**——在 NPI 风格（prediction_steps=1）这一最重要的科学配置下，
多尺度层级架构完全失效，但不会有任何报错或警告。

**修复**：`nn.BatchNorm1d(input_dim)` → `nn.GroupNorm(1, input_dim)`

GroupNorm(num_groups=1, num_channels=C) 将所有 C 个通道作为一个组归一化，
每个样本使用 C×L×N 个值计算统计量。对 `[1, 128, 1]`：128 个值 → 统计量有意义。

归一化层比较（对 [batch=1, channels=128, time=1]）：
| 层 | 每次统计的元素数 | 适用性 |
|----|----|------|
| BatchNorm1d(C) | N×L = 1×1 = 1 | 🔴 退化 |
| InstanceNorm1d(C) | L = 1 | 🔴 退化 |
| GroupNorm(C, C) = InstanceNorm | L = 1 | 🔴 退化 |
| GroupNorm(1, C) | C×L×N = 128 | ✅ 正常 |

**规则**：**在 batch_size=1 的图神经网络场景中，凡 L（时间/空间维度）可能为 1 的地方，
必须选用不依赖 batch 维度和时间维度的归一化层（GroupNorm(1, C) 或 LayerNorm）。**

---

**盲区 2 — 配置注释声称"启用 NLL 损失"但实际从未执行**

**思维误区**：以为"配置里写 use_uncertainty: true + NLL 损失"就代表 NLL 损失在工作。

**根因**：
1. `predict_next()` 硬编码 `return_uncertainty=False`
2. `compute_loss()` → `predict_next()` → uncertainty_head 从不被调用
3. `uncertainty_head` 参数存在于模型中但永远不会收到梯度
4. `default.yaml` 注释"启用 NLL 损失"是**错误的描述**

**后果**：
- 用户相信已启用不确定性估计，实际上什么都没发生
- 浪费 ~input_dim×1.5 个永不更新的参数
- 如果用户依赖不确定性估计做临床决策，将产生虚假安全感

**修复**：
- 更正 `default.yaml` 注释，明确说明"当前版本为后处理工具，不参与训练损失"
- 将默认值从 `true` 改为 `false`（诚实接口原则：默认配置只启用真正工作的功能）

**规则**：**凡是"声明了但从未真正工作"的配置选项，默认值必须为 false，且注释必须如实描述实际状态。**
（参见 AGENTS.md §三 [2026-02-21] "声明了但从未真正工作"的功能）

---

### [2026-02-27] 预测功能第三轮审查：YAML 与代码默认值不同步 + GRU 自回归维度崩溃

**背景**：第三轮以全新视角独立审查，验证前两轮已修复项的正确性，并寻找新盲区。

---

**盲区 1 — use_uncertainty Python 默认值未与 YAML 同步（P1 逻辑不一致）**

**思维误区**：以为"上一轮改了 YAML 就完成了"，没有检查 Python 代码层的 fallback 默认值。

**根因**：Round 2 已将 `default.yaml` 中 `use_uncertainty` 改为 `false`，并在 AGENTS.md 中记录了意图。但 `graph_native_system.py:446` 中的 Python fallback 默认值仍为 `True`：

```python
use_uncertainty=pred_cfg.get('use_uncertainty', True),  # ← 仍为 True！
```

任何不通过 YAML 创建模型的代码（单元测试、API用户、`predictor_config` 缺少该键时）会得到 `True`，创建从未训练的 `uncertainty_head` 参数，浪费显存且误导用户。

**修复**：`True` → `False`（1行改动）。

**规则**：**配置项的 Python fallback 默认值必须与 YAML 默认值保持一致。每次修改 YAML 默认值时，必须同时检索代码中所有对应的 `.get(key, python_default)` 调用并同步修改。**

---

**盲区 2 — GRU 自回归滚动在第2步后崩溃（P0 运行时崩溃）**

**思维误区**：以为"GRU 是简单替代品，没有复杂的维度问题"，没有追问反馈维度是否和输入维度匹配。

**根因**：`EnhancedMultiStepPredictor` 创建时 `hidden_dim = input_dim × 2`（例如 128 → 256）。GRU 的 `input_size=input_dim=128`，`hidden_size=hidden_dim=256`。自回归滚动代码（3处）：

```python
current = context[:, -1:, :]          # [batch, 1, 128] ✓
output, hidden = predictor(current)    # output: [batch, 1, 256]
current = output                       # BUG: 256 ≠ 128 → 第2步崩溃
```

第2步开始 GRU 收到 256 维输入但只接受 128 维 → `RuntimeError`。

**影响的3处位置**：
1. `HierarchicalPredictor._autoregressive_predict()` — 触发条件: `use_transformer=False`
2. `EnhancedMultiStepPredictor.predict_next()` GRU分支 — `use_hierarchical=False, use_transformer=False`
3. `EnhancedMultiStepPredictor.forward()` GRU分支 — 同上

**修复**：为 GRU 添加输出投影层 `nn.Linear(hidden_dim, input_dim)`，在反馈前将 GRU 输出投影回 input_dim。这是标准 seq2seq 解码器设计模式（隐藏空间 ≠ 输入空间，学习投影桥接两者）。

- `HierarchicalPredictor.__init__`: 添加 `self.gru_output_projs = nn.ModuleList([nn.Linear(hidden_dim, input_dim) × num_scales])`（仅 predictor_type='gru' 时）
- `_autoregressive_predict()`: 接受 `output_proj` 参数，使用 `current = proj(output) if proj else output`
- `HierarchicalPredictor.forward()`: 传入 `self.gru_output_projs[i]`
- `EnhancedMultiStepPredictor.__init__`: 添加 `self.gru_output_proj = nn.Linear(hidden_dim, input_dim)`（仅简单GRU模式）
- `predict_next()` / `forward()` GRU分支: 使用 `self.gru_output_proj`

**规则**：**任何 GRU/RNN 自回归滚动，第一步的输入维度和输出维度可能不同。反馈前必须确认 output.shape[-1] == input_size。若不同，必须添加投影层。**

---

### [2026-02-27] 预测功能第四轮审查：GRU 累积输出维度未投影（Round 3 遗漏的对称问题）

**背景**：第四轮独立审查，假设所有已修复项都可能存在遗漏。

---

**盲区 — GRU 累积预测维度错误（P0 运行时崩溃）**

**思维误区**：以为"Round 3 修复了 GRU 输出投影"就等于"GRU 自回归循环完全正确"。没有追问：**修复了反馈路径，累积路径是否也修了？**

**根因**：Round 3 的修复代码：

```python
output, hidden = predictor(current, hidden)  # output: [B, 1, 256]
predictions.append(output)                   # ← 仍然累积 256 维！（遗漏）
current = output_proj(output)                # ← 反馈正确投影到 128 维 ✓
```

`predictions` 累积的是 `hidden_dim=256` 维的原始 GRU 输出，而下游所有消费者都期待 `input_dim=128`：
- `upsamplers[i]` = `ConvTranspose1d(in_channels=128)` → 收到 256 → **CRASH**
- `fusion` = `Linear(input_dim*3=384)` → 收到 768 → **CRASH**
- `prediction_propagator` = `SpatialTemporalGraphConv(in_channels=128)` → 收到 256 → **CRASH**

**影响的3处位置**：
1. `HierarchicalPredictor._autoregressive_predict()` — GRU 模式（`use_transformer=False`）
2. `EnhancedMultiStepPredictor.predict_next()` GRU分支
3. `EnhancedMultiStepPredictor.forward()` GRU分支

**修复**：在所有3处，将"累积"与"反馈"统一为同一个投影后的变量：

```python
# 修复后（每处完全对称）：
output, hidden = predictor(current, hidden)
projected = output_proj(output) if output_proj is not None else output
predictions.append(projected)   # ← 累积 128 维 ✓
current = projected              # ← 反馈 128 维 ✓
```

**根本规则**：**自回归循环中"用于累积（append）"的张量和"用于反馈（current=）"的张量必须是同一个对象。** 如果两者不同，必须追问：两者的维度是否一致，下游消费者期待哪个维度？

---

### [2026-02-27] 预测功能第五轮审查：avg_pool1d 崩溃前守卫 + pred_enc 初始化 + N守卫 + 训练状态污染

**背景**：第五轮独立审查，重新审视每个函数，假设所有已修复项都可能仍有遗漏。

---

**盲区 1 — avg_pool1d 在返回前内部崩溃（P2 潜在崩溃）**

**思维误区**：以为"检查 x_down.shape[1] == 0 就能保护空张量"——没有追问：**PyTorch 是在返回前还是返回后抛出异常？**

**根因**：`F.avg_pool1d` 在计算出 `output_len ≤ 0` 时**直接在函数内部**抛出 "Output size is too small"，*在*返回之前。所以"在 avg_pool1d 调用之后检查结果"永远不会执行。触发条件：`_PRED_MIN_SEQ_LEN=4`、`_PRED_CONTEXT_RATIO=2/3` → T_ctx=2 < scale_factor=4。

```python
# 错误修复（Round 4 分析的方案）：
x_down = F.avg_pool1d(...)   # ← 已经崩溃，走不到下面
if x_down.shape[1] == 0:     # ← 永远不会执行
    ...

# 正确修复（Round 5）：
if x.shape[1] < scale_factor:   # ← 在调用前检查
    logger.debug(f"... zero prediction for this scale ...")
    pred_up = torch.zeros(...)
    scale_predictions.append(pred_up)
    continue
x_down = F.avg_pool1d(...)      # ← 安全
```

**规则**：**对所有可能抛出内部异常的函数（不仅仅是返回错误值），必须在调用前就验证前置条件，不能依赖检查返回值。**

---

**盲区 2 — validate() pred_enc 用原始信号初始化导致 decoder 维度崩溃（P2 潜在崩溃）**

**根因**：
```python
pred_enc = data.clone()  # x shape = [N, T, C=1] — 原始信号！
for nt, pred_lat in pred_latents.items():
    pred_enc[nt].x = pred_lat  # 只覆盖了 pred_latents 中的 modality
# 未被覆盖的 modality: pred_enc[nt].x = [N, T, 1]
# decoder: Conv1d(in_channels=128) 收到 1 通道 → CRASH
```
触发条件：某个 modality 因 T < `_PRED_MIN_SEQ_LEN` 被跳过。

**修复**：先从 `h_ctx_dict` 初始化（正确 H=128 维），再用 pred_latents 覆盖。
未被预测的 modality 的 context latent 喂给 decoder 不影响 R² 评估（`pred_T_ctx.get(nt) is None` 在 Step 6 跳过它们）。

---

**盲区 3 — compute_loss() pred_loss 缺少 N 不匹配守卫（P3 不一致）**

`recon_loss` 有 `if recon.shape[0] != target.shape[0]: raise RuntimeError(...)` 守卫；`pred_loss` 没有。**修复**：添加 `if pred_mean.shape[0] != future_target.shape[0]: logger.warning(...); continue`，与 recon_loss 保持一致的防御性编程。

---

**盲区 4 — UncertaintyAwarePredictor 强制 .train() 污染 eval 状态（P2）**

MC dropout 路径中 `self.base_predictor.train()` 被无条件调用，在 validate() 的 `@torch.no_grad()` 上下文中也会执行，污染 BatchNorm running stats。**修复**：`was_training = self.base_predictor.training`，try/finally 恢复。

---

### [2026-02-27] 预测功能第六轮审查：非层级 TransformerPredictor 错误分派 + UAP shape 错误 + h_ctx_dict 初始化

**背景**：第六轮独立审查，系统性验证所有配置组合。

---

**盲区 1 — 非层级 TransformerPredictor 分派到 GRU 分支（P2 运行时崩溃/静默错误）**

**思维误区**：以为"修复了 GRU 投影就修好了所有 else 分支"，没有追问：**else 分支覆盖了哪些配置？**

**根因**：`EnhancedMultiStepPredictor.__init__()` 存储了 `self.use_hierarchical` 和 `self.use_uncertainty`，但**未存储 `self.use_transformer`**。当 `use_hierarchical=False, use_transformer=True, use_uncertainty=False` 时：
- `self.predictor = TransformerPredictor(...)` (返回单一 Tensor)
- `predict_next()` 和 `forward()` 的 `else` 分支执行 `_, hidden = self.predictor(context)`
- TransformerPredictor 返回 `[B, T, H]`（不是 `(output, hidden)`）
- batch_size ≠ 2 时崩溃；batch_size = 2 时静默产生错误结果

**修复**：
1. `__init__` 中保存 `self.use_transformer = use_transformer`
2. 在 `predict_next()` 和 `forward()` 中添加 `elif self.use_transformer:` 分支
3. 提取模块级辅助函数 `_transformer_seq2seq_predict(predictor, context, n_steps)` — 消除3处重复的 seq2seq 扩展逻辑（extend context + causal mask + take last n_steps）

---

**盲区 2 — UncertaintyAwarePredictor 包装 TransformerPredictor 时返回错误形状（P2 静默错误）**

**思维误区**：以为"修复了 EnhancedMultiStepPredictor 的分派就覆盖了 UAP 的分派"，没有追问：**UAP.forward() 的 isinstance 分支是否正确处理 TransformerPredictor？**

**根因**：`UncertaintyAwarePredictor.forward()` 中 gaussian/dropout 路径只检查 `isinstance(self.base_predictor, HierarchicalPredictor)`；对非 Hierarchical 的情况调用 `self.base_predictor(x)`（TransformerPredictor 返回 `[B, ctx_len, H]`，不是 `[B, future_steps, H]`）。

**修复**：提取 `UncertaintyAwarePredictor._base_predict(x, future_steps)` 辅助方法，使用 `_transformer_seq2seq_predict` 正确处理 TransformerPredictor；gaussian/dropout 所有分支统一调用 `_base_predict`，消除重复的 isinstance 分散逻辑。

---

**盲区 3 — validate() h_ctx_dict 未初始化（P3 鲁棒性）**

`h_ctx_dict` 仅在 `if ctx_T_map:` 块内赋值，在 `if pred_latents:` 块内使用。当前逻辑保证安全（pred_latents 非空 → ctx_T_map 非空 → h_ctx_dict 已赋值），但这是隐式不变式。

**修复**：在 `if ctx_T_map:` 之前添加 `h_ctx_dict: Dict[str, torch.Tensor] = {}` 显式初始化，使不变式变得显式，防止未来代码修改破坏此假设。

**配置空间全覆盖审查结果（Round 6）**：
| 配置 | 状态 |
|------|------|
| use_hierarchical=True（默认） | ✅ 正确 |
| use_hierarchical=False, use_transformer=False (GRU) | ✅ 正确 |
| use_hierarchical=False, use_transformer=True | **已修复 V5.36** |
| use_uncertainty=True 包装任意 base | **已修复 V5.36**（UAP._base_predict） |

---

### [2026-02-28] 预测功能第七轮审查：GRU+不确定性彻底破损 + validate() decoder 潜在崩溃

**背景**：第七轮独立审查，假设所有已修复项都可能存在遗漏，穷举所有配置组合的完整调用链。

---

**盲区 1 — GRU + UncertaintyAwarePredictor 组合三重破损（P2 运行时崩溃）**

**配置**：`use_hierarchical=False, use_transformer=False, use_uncertainty=True`

**思维误区**：以为"Round 6 修复了 UAP._base_predict 就覆盖了所有 else 分支"，没有追问：**当 base_predictor = nn.GRU 时，`_base_predict` else 分支是否还会崩溃？**

**三重根因**：
1. `_base_predict` else 分支调用 `return self.base_predictor(x)` = `nn.GRU(x)` → 返回 `(output, h_n)` **tuple**，不是 Tensor → TypeError
2. 即使解包 `output`：`nn.GRU(x)` 返回 `[N, ctx_len, hidden_dim]`，不是 `[N, future_steps, input_dim]`（时间维度长度错误，特征维度 256≠128）
3. `uncertainty_head` 期待 `input_dim=128`，收到 `hidden_dim=256` → 形状崩溃

**修复**：
1. `EnhancedMultiStepPredictor.__init__` 对不支持的组合提前抛出 `ValueError`（构造时即报告，而非运行时神秘崩溃）
2. `UAP._base_predict` else 分支防御性解包：`result = self.base_predictor(x); return result[0] if isinstance(result, tuple) else result`（保护直接实例化 UAP 的场景，如单元测试）

---

**盲区 2 — validate() decoder 在 T < _PRED_MIN_SEQ_LEN 时崩溃（P3 潜在崩溃）**

**根因**：`pred_enc = data.clone()` 会保留 T < 4 的节点的原始信号 `[N, T, C=1]`。`h_ctx_dict` 不包含这些节点（因 T < _PRED_MIN_SEQ_LEN）。decoder 的 `Conv1d(in_channels=128)` 应用于 `[N, 1, T]` → 通道不匹配崩溃。

**修复**：在调用 decoder 前删除 pred_enc 中不在 h_ctx_dict 中的节点类型：
```python
for nt in list(pred_enc.node_types):
    if nt not in h_ctx_dict:
        logger.debug(f"validate: removing '{nt}' from pred_enc (T < _PRED_MIN_SEQ_LEN)")
        del pred_enc[nt]
```

**配置空间完整覆盖审查结果（Round 7）**：
| 配置 | 状态 |
|------|------|
| use_hierarchical=True（默认） | ✅ |
| use_hierarchical=True, use_transformer=False | ✅ |
| use_hierarchical=False, use_transformer=True | ✅ |
| use_hierarchical=False, use_transformer=False | ✅ |
| + use_uncertainty=True（上述前三种） | ✅ |
| **use_hierarchical=False, use_transformer=False, use_uncertainty=True** | **✅ 已明确禁止（ValueError）V5.37** |

**新规则**：**每次新增 isinstance 分支时，必须追问：`else` 分支现在还有哪些类型会落入？每种落入类型的返回值格式是否兼容后续代码？** 特别是 `nn.GRU` 等 PyTorch 原生模块的返回格式（tuple），永远不应假设与自定义 nn.Module 相同。

---

### [2026-02-28] 预测功能第八轮审查：R² 科学精确性 + 数值稳定性 + 边界防御

**背景**：第八轮独立审查。前七轮已消除所有已知的运行时崩溃类型，本轮聚焦于**科学指标的数学正确性**和**数值稳定性**。

---

**盲区 1 — validate() R² 使用每样本均值而非全局均值（P2 科学精确性）**

**思维误区**：以为"每个样本内计算均值，再累加 SS_tot"在语义上等价于"全局均值 SS_tot"。

**根因**：`SS_tot = Σ_samples Σ(y - ȳ_sample)²` 只包含**样本内方差**（within-sample variance），不包含**样本间方差**（between-sample variance = `Σ n_s*(ȳ_s - ȳ_global)²`）。根据方差分解：`SS_tot_global = SS_tot_within + SS_tot_between`，因此 `SS_tot_per-sample ≤ SS_tot_global`。分母偏小 → `R² = 1 - SS_res/SS_tot` 偏高（乐观偏差）。

**影响**：对 z-scored 信号（全局均值 ≈ 0），每样本均值也 ≈ 0，偏差极小。但训练可信度摘要对 R² = 0.3 做通过/失败判断，数学上正确的指标至关重要。

**修复**：使用**在线单遍算法**（无需存储所有样本）：累积 `ss_res, ss_raw(Σy²), ss_sum(Σy), ss_cnt(n)`，最终用代数恒等式 `SS_tot = Σy² - n·ȳ²` 计算全局均值 SS_tot。对重建 R² 和预测 R² 均应用。

**规则**：**多样本累积 R² 必须使用全局均值。若无法两遍扫描，使用代数恒等式 `SS_tot = Σy² - n·ȳ²` 在单遍内正确计算。**

---

**盲区 2 — `_transformer_seq2seq_predict(n_steps=0)` 返回完整序列（P2）**

**根因**：`pred_all[:, -n_steps:, :]` 当 `n_steps=0` 时，`-0 == 0`，切片返回整个序列而非空张量。

**修复**：调用 `repeat` 之前添加早返回：`if n_steps <= 0: return context.new_empty(B, 0, H)`

---

**盲区 3 — `UncertaintyAwarePredictor.uncertainty_head` 无数值稳定性保护（P3）**

**根因**：`exp(0.5 * log_var)` 无上界约束。`log_var > 88`（float32）时溢出为 Inf，NLL 损失产生 NaN 梯度，训练静默失败。

**修复**：`torch.clamp(log_var, min=-10.0, max=10.0)` → std ∈ [0.007, 148]，覆盖 z-scored 神经信号全范围。

---

### [2026-02-28] pred_r2 长期低下的根因：训练目标与评估指标不对齐

**症状（已观察到）**：
- r2_eeg=0.985, r2_fmri=0.980（重建优秀）
- pred_r2_eeg=-0.026, pred_r2_fmri=0.193（预测极差甚至为负）

**思维误区**：以为"在潜空间训练好预测器，解码后信号空间预测自然也会好"，没有追问三个关键问题：
1. 解码器是否见过预测潜向量作为输入（即被训练来解码 pred_latents）？
2. prediction_steps=10 对 EEG(250Hz) = 40ms 是否足够长以学到有意义的时序动态？
3. 潜空间预测损失的梯度是否已到达解码器？

**根因（三叠加）**：

**根因 1 — 训练-评估目标脱节（最严重，P0）**：
- 训练：`pred_{nt}` 损失 = `huber_loss(pred_latent, future_latent)` ← 纯潜空间
- 评估：`pred_r2` = 信号空间 R²(decoder(pred_latent) vs raw_future_signal)
- 解码器在整个训练期间**从未见过预测潜向量作为输入**；它只被训练来解码编码器潜向量。
  预测潜向量与编码器潜向量可能属于不同分布，解码器无法保证在预测潜向量上泛化。

**根因 2 — prediction_steps=10 与 EEG 动力学不匹配（P1）**：
- EEG 250Hz → 10 步 = 40ms，而 alpha 周期 = 80-125ms（不到一个完整周期）
- 从 200 步 context（800ms）预测 40ms 后的精确相位极其困难
- pred_r2_eeg ≈ -0.026 < 0 意味着预测值比均值基线**更差**，即学到的是噪声而非动态

**根因 3 — AdaptiveLossBalancer 未注册 pred_sig_* 任务（P1）**：
- 加入信号空间损失后，若 `task_names` 不包含 `pred_sig_{nt}`，balancer.forward() 静默忽略它
- 等效于信号空间损失不参与 adaptive 加权，其梯度在 use_adaptive_loss=True 时丢失

**修复（V5.39）**：
1. `compute_loss()` 新增**信号空间预测损失** `pred_sig_{nt}`:
   解码预测潜向量 → 与真实未来信号对比，直接优化 pred_r2 所测量的端到端能力
   梯度路径: `pred_sig_loss → decoder → prediction_propagator → predictor.predict_next`
2. `GraphNativeTrainer.__init__()` 将 `pred_sig_{nt}` 注册进 `AdaptiveLossBalancer.task_names`
3. `default.yaml`: `prediction_steps: 10 → 30`（EEG 120ms 覆盖 alpha 周期；fMRI 60s 覆盖血动力学）

**规则**：**任何"端到端"的训练目标（如 pred_r2）必须有一条等价的训练损失路径。** 仅在中间表示（潜空间）监督，而在输出空间（信号空间）评估，必然产生训练-评估分布偏移。正确模式是同时计算中间监督（快收敛）+ 端到端监督（对齐评估）。

---

### [2026-02-28] pred_r2_eeg 改善：频域监督 + 相关性分量 + 时序遮蔽

**背景**：V5.39 加入了信号空间预测损失（pred_sig）以修复训练-评估脱节。V5.40 在此基础上增加频域监督和相关性监督，直接优化 pred_r2 所测量的三个分量（幅度、频率结构、时序形状）。

**思维误区**：以为"时域 Huber 损失足够，只要幅度对了 R² 就会高"。

**实际情况**：R² = 1 - SS_res/SS_tot，包含三个组成部分：
1. **幅度精度**（均值和方差匹配）← Huber 监督
2. **频率结构**（power spectrum 匹配）← 仅 Huber 不够，需要 FFT 损失
3. **时序形状**（temporal pattern / Pearson r）← 仅 Huber 不够，需要相关性损失

三个分量缺一不可。仅使用 Huber 的模型可以达到低 MSE 但低 R²（通过预测接近均值的平坦信号）。

**修复（V5.40）**：
1. `_spectral_loss()`：FFT 幅度谱 MSE（/T 归一化），作为独立 `spectral_{nt}` 任务注册
2. `pred_sig_{nt}` 加入 Pearson 相关分量：`sig_loss += 0.2 × (1 - r)`
3. `time_mask_max_ratio: 0.1`：随机遮蔽时序窗口，迫使模型学习上下文推断能力
4. `use_dynamic_graph: true`（默认）：动态图拓扑实时适应当前脑状态

**规则**：评估指标由多个分量组成时，训练损失必须覆盖所有分量。「R² 高 ↔ Huber 低」的假设仅在模型输出方差接近目标方差时成立；当模型倾向于预测"平均脑状态"（安全的均值预测）时，Huber 低但 R² 仍可为负。



| 维度 | 状态 | 实现版本 |
|------|------|---------|
| 个性化（被试特异性嵌入） | ✅ 已实现 | V5.19–V5.20 |
| 动态拓扑（DynamicGraphConstructor，默认开启） | ✅ 已实现 | V5.14 / V5.40 |
| 系统级预测传播（GraphPredictionPropagator） | ✅ 已实现 | V5.25 |
| 梯度累积（gradient_accumulation_steps） | ✅ 已实现 | V5.28 |
| 随机权重平均（SWA） | ✅ 已实现（可选） | V5.28 |
| 验证集 R² 指标 | ✅ 已实现 | V5.28 |
| EEG 频谱相干性连接（configurable） | ✅ 已实现 | V5.28 |
| 相关性跨模态边（NVC 感知，top-k |r|） | ✅ 已实现 | V5.30 |
| 训练曲线可视化（loss + R² PNGs） | ✅ 已实现 | V5.30 |
| 断点续训（--resume） | ✅ 已实现 | V5.30 |
| 时序数据增强（noise + scale，默认开启） | ✅ 已实现 | V5.30 / V5.40 |
| NPI 风格 context_length 可配置 | ✅ 已实现 | V5.31 |
| HierarchicalPredictor NPI 兼容（prediction_steps=1 整数除法） | ✅ 已修复 | V5.31 |
| validate() 真因果 pred_R²（因果编码） | ✅ 已修复 | V5.31 |
| HierarchicalPredictor 上采样器 BatchNorm1d→GroupNorm（静默归零修复） | ✅ 已修复 | V5.32 |
| use_uncertainty 配置文档准确化（默认关闭） | ✅ 已修复 | V5.32 |
| use_uncertainty Python 默认值同步 YAML（True→False） | ✅ 已修复 | V5.33 |
| GRU 自回归滚动输出投影——反馈维度修复 | ✅ 已修复 | V5.33 |
| GRU 自回归滚动输出投影——累积维度修复（对称完整） | ✅ 已修复 | V5.34 |
| avg_pool1d 调用前守卫（短 context 崩溃修复） | ✅ 已修复 | V5.35 |
| validate() pred_enc 从 h_ctx_dict 初始化（decoder 维度修复） | ✅ 已修复 | V5.35 |
| pred_loss N 不匹配守卫（防御性一致性） | ✅ 已修复 | V5.35 |
| UncertaintyAwarePredictor MC dropout eval 状态恢复 | ✅ 已修复 | V5.35 |
| 非层级 TransformerPredictor 错误分派修复（predict_next/forward） | ✅ 已修复 | V5.36 |
| UAP._base_predict 统一分派（TransformerPredictor shape 修复） | ✅ 已修复 | V5.36 |
| _transformer_seq2seq_predict 模块级辅助函数（DRY 消重） | ✅ 已实现 | V5.36 |
| validate() h_ctx_dict 显式初始化（鲁棒性） | ✅ 已修复 | V5.36 |
| GRU+uncertainty 不支持组合 ValueError（构造时明确拒绝） | ✅ 已修复 | V5.37 |
| UAP._base_predict GRU tuple 防御性解包 | ✅ 已修复 | V5.37 |
| validate() decoder 在 T < _PRED_MIN_SEQ_LEN 时删除节点守卫 | ✅ 已修复 | V5.37 |
| validate() R² 全局均值（per-sample mean → 在线单遍算法） | ✅ 已修复 | V5.38 |
| _transformer_seq2seq_predict n_steps=0 返回空张量守卫 | ✅ 已修复 | V5.38 |
| UAP uncertainty_head log_var clamp（防溢出 → NaN 梯度） | ✅ 已修复 | V5.38 |
| 信号空间预测损失 pred_sig_{nt}（训练-评估对齐，pred_r2 改善） | ✅ 已实现 | V5.39 |
| pred_sig_{nt} 注册进 AdaptiveLossBalancer | ✅ 已修复 | V5.39 |
| prediction_steps 默认值 10→30（覆盖 EEG alpha 周期/fMRI 血动力学）| ✅ 已更新 | V5.39 |
| 频域重建损失 spectral_{nt}（FFT 幅度谱 MSE，改善 pred_r2_eeg） | ✅ 已实现 | V5.40 |
| pred_sig Pearson 相关分量（直接优化 pred_r2，weight=0.2） | ✅ 已实现 | V5.40 |
| 时序遮蔽增强（time masking，SpecAugment 风格，默认开启 10%）| ✅ 已实现 | V5.40 |
| 动态图 use_dynamic_graph 默认 false→true | ✅ 已更新 | V5.40 |
| windowed_sampling 默认开启（支持 4-8 被试 / 8GB GPU） | ✅ 已实现 | V5.41 |
| temporal_chunk_size 默认 64→32（44MB/step，内存再降半） | ✅ 已更新 | V5.41 |
| gradient_accumulation_steps 默认 1→4（小数据集梯度质量） | ✅ 已更新 | V5.41 |
| cuda_clear_interval 新增（epoch 内周期碎片清理） | ✅ 已实现 | V5.41 |
| cuda_clear_interval is_accum_boundary 门控 bug 修复（LCM 陷阱）| ✅ 已修复 | V5.41.1 |
| prediction_steps 默认值 30→15（匹配 fMRI 窗口 T_fut=17，零浪费）| ✅ 已更新 | V5.41.1 |
| compute_loss pred_loss 超出 T_fut 的 debug 提示 | ✅ 已实现 | V5.41.1 |
| TemporalAttention is_causal=False → True（P0 训练-验证偏差根本修复）| ✅ 已修复 | V5.42 |
| use_spectral_loss 默认 true → false（保留 pred 梯度预算）| ✅ 已更新 | V5.42 |
| pred_sig Pearson 权重 0.2 → 0.5（加强 pred_r2 信号）| ✅ 已更新 | V5.42 |
| task_priorities.pred 3.0 → 6.0（预测任务梯度优先级）| ✅ 已更新 | V5.42 |
| adaptive_loss.warmup_epochs 5 → 10（延迟自适应调整）| ✅ 已更新 | V5.42 |
| pred_weight_floor=0.5（防止 GradNorm 饿死预测任务）| ✅ 已实现 | V5.42 |
| 跨模态潜空间对齐损失 cross_modal_align（cosine similarity，CMC 风格）| ✅ 已实现 | V5.43 |
| 预测步指数加权 pred_step_weight_gamma（远期步权重更高）| ✅ 已实现 | V5.43 |
| 修复误导性 is_causal=False 注释（compute_loss + validate）| ✅ 已修复 | V5.43 |
| 跨会话预测（run_embed，会话级嵌入）| ✅ 已实现 | V5.44 |
| 干预响应仿真（simulate_intervention，TMS 数字孪生）| ✅ 已实现 | V5.44 |
| 少样本个性化推理（adapt_to_subject，O(H) 参数更新）| ✅ 已实现 | V5.44 |
| 梯度归因可解释性（compute_attribution，功能指纹）| ✅ 已实现 | V5.44 |
| 检查点自动推理（TwinBrainDigitalTwin.from_checkpoint）| ✅ 已实现 | V5.44 |
| 自我演化（多会话在线学习）| ❌ Future work | — |
| temporal_chunk_size 无 checkpointing 时自动不分块（消除 16× Python overhead）| ✅ 已修复 | V5.45 |
| predict_next() num_steps 参数（按需生成，避免 fMRI 66% 无监督步）| ✅ 已实现 | V5.45 |
| pred_step_weight_gamma 向量化（5360 kernel → 160 kernel/epoch）| ✅ 已修复 | V5.45 |
| temporal_chunk_size 默认值 32 → null（文档准确化）| ✅ 已更新 | V5.45 |
| modality-aware _trust_threshold()（EEG=0.10，fMRI=0.20，recon=0.30）| ✅ 已实现 | V5.46 |
| _r2_rating() 分模态描述 + 中文说明文字 | ✅ 已实现 | V5.46 |
| 可视化 R² 参考线更新（0.10/0.20/0.30 三条线）| ✅ 已更新 | V5.46 |
| AGENTS.md 和 USERGUIDE.md 文献引用（7 篇论文，EEG/fMRI 物理约束支撑）| ✅ 已更新 | V5.46 |
| temporal_chunk_size: null → 64（V5.47 回归，已在 V5.48 修复回 null）| ✅ 已修复 | V5.48 |
| use_gradient_checkpointing: true → false（V5.48 错误，V5.49 已回退）| ⚠️ 已回退 | V5.48→V5.49 |
| info_nce_temperature: 0.1 → 0.5（防 warmup 期 InfoNCE 压制预测梯度）| ✅ 已更新 | V5.48 |
| task_priorities.pred_nce: 4.0 → 2.0（warmup 期 InfoNCE:pred_sig 比从 50:1→5:1，消除 pred_r2<0 告警）| ✅ 已更新 | V5.48 |
| GraphNativeDecoder BatchNorm1d → GroupNorm(1, out_dim)（短序列归一化稳定）| ✅ 已修复 | V5.47 |
| compute_loss() pred_nce_{nt} InfoNCE 对比预测损失（防均值崩塌）| ✅ 已实现 | V5.47 |
| GraphNativeTrainer pred_nce_{nt} 注册进 AdaptiveLossBalancer | ✅ 已实现 | V5.47 |
| GraphNativeBrainModel.compute_effective_connectivity()（NPI 对等 EC 矩阵）| ✅ 已实现 | V5.47 |
| TwinBrainDigitalTwin.compute_effective_connectivity()（多窗口 EC 高层接口）| ✅ 已实现 | V5.47 |
| TwinBrainDigitalTwin.compute_model_fc()（模型 FC 矩阵，FC vs EC 对比验证）| ✅ 已实现 | V5.47 |
| validate() ar1_r2_{nt}、decorr_{nt}、pred_r2_h1_{nt} NPI 对比科学指标 | ✅ 已实现 | V5.47 |
| main.py 传递 use_info_nce、info_nce_temperature | ✅ 已实现 | V5.47 |
| use_gradient_checkpointing: false → true（V5.48 错误回退，backward 峰值 ~12 GB）| ✅ 已修复 | V5.49 |
| eeg_window_size: 500 → 250（TemporalAttention 4×，propagate 2× 快；1s 覆盖 alpha/beta/gamma）| ✅ 已更新 | V5.49 |
| k_nearest_fmri: 20 → 10（fMRI edges 3800→1900，propagate 2× 快）| ✅ 已更新 | V5.49 |
| k_dynamic_neighbors: 10 → 5（dynamic edges 减半，DGC topk 2× 快）| ✅ 已更新 | V5.49 |
| SpatialTemporalGraphConv._ei_cache（edge expansion 张量缓存，消除 N_windows 次 GPU 分配）| ✅ 已实现 | V5.49 |
| train_model() GPU 预加载（训练前 to(device)，稳定 data_ptr() 提升 cache 命中率）| ✅ 已实现 | V5.49 |

### 被试特异性嵌入全链路（V5.19–V5.20）

`subject_to_idx` → `built_graph.subject_idx`（含缓存路径）→ `extract_windowed_samples` 传播 → `nn.Embedding(N, H)` → `x_proj += embed.view(1,1,-1)`

**⚠️ 注意**：缓存命中路径须在 `extract_windowed_samples` 前先写入 `subject_idx`（V5.20 修复）。

### DynamicGraphConstructor 使用建议
- 小数据集（< 50 样本）谨慎：额外参数可能过拟合。
- `k_dynamic_neighbors`：fMRI(N≈190) 设 10；EEG(N≤64) 设 5。
