

# DexPilotOptimizer 优化笔记

## 1. 目标函数

优化器最小化以下损失函数（NLopt 内部优化）：

\[
\mathcal{L}(\mathbf{x}) = \frac{1}{N_{\text{vec}}} \sum_{k=1}^{N_{\text{vec}}} w_k \cdot \text{Huber}_{\delta}\big( d_k(\mathbf{x}) \big) \;+\; \alpha \|\mathbf{x} - \mathbf{x}_{\text{prev}}\|^2
\]

- **\(\mathbf{x}\)**：当前优化的关节角度向量（长度 \(d\)）。
- **\(\mathbf{x}_{\text{prev}}\)**：上一帧优化结果（用于正则化）。
- **\(N_{\text{vec}}\)**：约束向量总数 = 手指间向量数 + 手腕‑手指向量数。
- **\(d_k(\mathbf{x})\)**：第 \(k\) 个机器人向量与目标向量的欧氏距离：
  \[
  d_k(\mathbf{x}) = \|\mathbf{v}_k^{\text{robot}}(\mathbf{x}) - \mathbf{v}_k^{\text{target}}\|
  \]
- **\(w_k\)**：动态权重（投影状态高权重，否则为 1；手腕‑手指向量权重更大）。
- **\(\text{Huber}_{\delta}(d)\)**：Huber 损失（Smooth L1），阈值为 \(\delta\)（`huber_delta`）。
- **\(\alpha\)**：正则化强度（`norm_delta`）。

### 代码对应
- 损失计算：`huber_distance` 是加权 Huber 损失之和（不含正则项）。
- 正则项梯度在梯度计算时加入，不在损失值中体现（优化器根据梯度更新）。



## 2. 梯度计算

我们需要 \(\nabla_{\mathbf{x}} \mathcal{L}\)，它由两部分组成：

\[
\nabla_{\mathbf{x}} \mathcal{L} = \nabla_{\mathbf{x}} L_{\text{huber}} \;+\; 2\alpha (\mathbf{x} - \mathbf{x}_{\text{prev}})
\]

其中 \(L_{\text{huber}}\) 是加权 Huber 损失项。

### 2.1 计算 \(\nabla_{\mathbf{x}} L_{\text{huber}}\)

通过链式法则：
\[
\nabla_{\mathbf{x}} L_{\text{huber}} = \sum_{i=1}^{N_{\text{links}}} \left( \frac{\partial L_{\text{huber}}}{\partial \mathbf{p}_i} \right)^{\!T} \cdot \frac{\partial \mathbf{p}_i}{\partial \mathbf{x}}
\]

- **\(\mathbf{p}_i\)**：第 \(i\) 个连杆在相机坐标系中的 3D 位置。
- **\(\frac{\partial \mathbf{p}_i}{\partial \mathbf{x}}\)**：连杆位置对关节角度的雅可比矩阵 \(\mathbf{J}_{\mathbf{p}_i}\)（3×d）。
- **\(\frac{\partial L_{\text{huber}}}{\partial \mathbf{p}_i}\)**：损失对连杆位置的梯度（3 维向量），由自动微分获得。

#### 代码步骤

**1. 获取 \(\frac{\partial L_{\text{huber}}}{\partial \mathbf{p}_i}\)**  
```python
huber_distance.backward()
grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]   # shape (n_links, 1, 3)
```
- `huber_distance` 是标量损失 \(L_{\text{huber}}\)。
- `torch_body_pos` 是形状 `(n_links, 3)` 的张量，记录了所有相关连杆的位置，且 `requires_grad=True`。
- 反向传播后，`torch_body_pos.grad` 即为每个连杆的梯度向量（3 维），我们将其扩展为 `(n_links, 1, 3)` 以便后续矩阵乘法。

**2. 计算雅可比矩阵 \(\mathbf{J}_{\mathbf{p}_i}\)**  
```python
jacobians = []  # 每个元素形状 (3, dof)
for i, index in enumerate(self.computed_link_indices):
    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]   # 局部线速度雅可比 (3, dof)
    link_rot = target_link_poses[i][:3, :3]   # 旋转矩阵 R (3,3)
    link_kinematics_jacobian = link_rot @ link_body_jacobian   # 世界坐标系下位置雅可比 (3, dof)
    jacobians.append(link_kinematics_jacobian)
jacobians = np.stack(jacobians, axis=0)   # 形状 (n_links, 3, dof)
```
- 局部雅可比给出在连杆局部坐标系中的线速度与关节速度的关系；通过旋转矩阵变换到世界坐标系，得到 \(\frac{\partial \mathbf{p}_i}{\partial \mathbf{x}}\)。

**3. 将位置梯度转换为关节梯度**  
```python
grad_qpos = np.matmul(grad_pos, np.array(jacobians))   # 形状 (n_links, 1, dof)
grad_qpos = grad_qpos.mean(1).sum(0)                  # 形状 (dof,)
```
- 矩阵乘法：`grad_pos[i]` (1×3) 与 `jacobians[i]` (3×dof) 相乘，得到该连杆对梯度的贡献 (1×dof)。
- 对连杆维度求和，即 \(\sum_i \left( \frac{\partial L}{\partial \mathbf{p}_i} \right)^{\!T} \mathbf{J}_{\mathbf{p}_i}\)，得到 \(\nabla_{\mathbf{x}} L_{\text{huber}}\)。

### 2.2 添加正则项梯度
```python
grad_qpos += 2 * self.norm_delta * (x - last_qpos)   # 2α (x - x_prev)
```

### 2.3 输出梯度
```python
grad[:] = grad_qpos[:]   # 填充 NLopt 的梯度缓冲区
```

---

## 3. 链式法则的完整推导（自动微分对应）

我们详细写出从 \(L_{\text{huber}}\) 到 \(\mathbf{x}\) 的导数链，并说明代码中的每一步。

### 3.1 \(L_{\text{huber}}\) 对 \(\mathbf{v}_k\) 的梯度
\[
L_{\text{huber}} = \frac{1}{N} \sum_{k} w_k \cdot \text{Huber}(d_k), \quad d_k = \|\mathbf{v}_k - \mathbf{v}_k^{\text{target}}\|
\]
\[
\frac{\partial L}{\partial \mathbf{v}_k} = \frac{w_k}{N} \cdot \text{Huber}'(d_k) \cdot \frac{\partial d_k}{\partial \mathbf{v}_k}
\]
其中：
- \(\text{Huber}'(d) = \begin{cases} d/\delta, & |d| \le \delta \\ \operatorname{sign}(d), & |d| > \delta \end{cases}\)，\(\text{Huber}(d) = \begin{cases} 0.5d^2/\delta, & |d| \le \delta \\ |d|-0.5\delta, & |d| > \delta \end{cases}\)
- \(\frac{\partial d_k}{\partial \mathbf{v}_k} = \frac{(\mathbf{v}_k - \mathbf{v}_k^{\text{target}})^{\!T}}{d_k + \epsilon}\)，其中$\epsilon$是代码中为了放置梯度保证加入的，数学表达中没有

### 3.2 \(\mathbf{v}_k\) 对连杆位置 \(\mathbf{p}_i\) 的梯度
向量 \(\mathbf{v}_k = \mathbf{p}_{\text{task}} - \mathbf{p}_{\text{origin}}\)，因此：
\[
\frac{\partial \mathbf{v}_k}{\partial \mathbf{p}_i} = 
\begin{cases}
+ \mathbf{I}_3, & \text{if } \mathbf{p}_i = \mathbf{p}_{\text{task}} \\
- \mathbf{I}_3, & \text{if } \mathbf{p}_i = \mathbf{p}_{\text{origin}} \\
0, & \text{otherwise}
\end{cases}
\]
所以 \(\frac{\partial L}{\partial \mathbf{p}_i} = \sum_k \frac{\partial L}{\partial \mathbf{v}_k} \cdot \frac{\partial \mathbf{v}_k}{\partial \mathbf{p}_i}\)。PyTorch 的自动微分正是通过构建计算图，自动完成这些求和与链式乘法。$I_3$ 表示 3×3 的单位矩阵

### 3.3 \(\mathbf{p}_i\) 对 \(\mathbf{x}\) 的梯度（雅可比）
\[
\mathbf{J}_{\mathbf{p}_i} = \frac{\partial \mathbf{p}_i}{\partial \mathbf{x}} \quad (\text{3×d})
\]
代码中通过 `link_rot @ link_body_jacobian` 得到。

### 3.4 最终关节梯度
\[
\frac{\partial L}{\partial \mathbf{x}} = \sum_i \left( \frac{\partial L}{\partial \mathbf{p}_i} \right)^{\!T} \cdot \mathbf{J}_{\mathbf{p}_i}
\]
代码中 `np.matmul(grad_pos, jacobians).mean(1).sum(0)` 实现。

---

## 4. 关键代码段索引（文件 `optimizer.py`）

| 操作           | 代码位置（近似行号）                                         | 说明                 |
| -------------- | ------------------------------------------------------------ | -------------------- |
| 构造权重       | `weight = np.where(self.projected, high_weight, normal_weight)` | 投影状态权重 200/400 |
| 构造目标向量   | `reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])` | 投影后固定长度       |
| Huber 损失     | `huber_distance = (self.huber_loss(vec_dist, zeros) * weight / len(robot_vec)).sum()` | 加权平均             |
| 反向传播       | `huber_distance.backward()`                                  | 计算位置梯度         |
| 提取位置梯度   | `grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]`   | (n_links,1,3)        |
| 计算世界雅可比 | `link_kinematics_jacobian = link_rot @ link_body_jacobian`   | (3, dof)             |
| 转换关节梯度   | `grad_qpos = np.matmul(grad_pos, np.array(jacobians)).mean(1).sum(0)` | (dof,)               |
| 正则项梯度     | `grad_qpos += 2 * self.norm_delta * (x - last_qpos)`         |                      |
| 输出梯度       | `grad[:] = grad_qpos[:]`                                     |                      |

---

## 5. 总结

- 优化目标由**加权 Huber 损失**（跟踪人类手部向量）和**时序正则项**（平滑）组成。
- 梯度计算利用 PyTorch 自动微分获得损失对连杆位置的梯度，再通过手动机器人雅可比转换为关节角度梯度。
- 链式法则在自动微分中隐式实现，无需手动展开 \( \frac{\partial d_k}{\partial \mathbf{x}} \)。
- 正则项仅以梯度形式加入，避免修改损失值，符合 NLopt 接口。

这份笔记可以帮助你理解代码的数学原理，并在以后查阅时快速对应。如果需要添加更多细节（如具体变量形状、调试技巧），可以继续补充。



-- --



# 代码中的疑问：

### 1. 为什么返回的 `result` 没有包含正则项？

在代码中，`result = huber_distance.cpu().detach().item()` 只返回了加权 Huber 损失（即向量跟踪误差），而正则项 α∥x−xprev∥2α∥x−xprev∥2 **并未加入这个返回值**。然而，正则项的梯度 2α(x−xprev)2α(x−xprev) 已经在 `grad_qpos` 中被加上了（见 `grad_qpos += 2 * self.norm_delta * (x - last_qpos)`），因此优化器在更新参数时会受到正则项的约束。

**为什么可以这样？**
NLopt 优化器在求解时，主要依赖目标函数的梯度信息，函数值只用于判断收敛（如绝对/相对容差）。如果函数值不包含正则项，优化器实际优化的函数是 L(x)=Lhuber(x)L(x)=Lhuber​(x)，而梯度中却包含了正则项的梯度，这会导致优化器在“错误”的函数值下使用“正确”的梯度，本质上破坏了梯度与函数值的一致性。这种不一致可能会影响收敛性判断，但实际中如果正则项很小或只是用于平滑，优化器依然可能正常工作。作者可能认为正则项的影响主要在于梯度，而函数值只是用于监控，因此没有加入。更严谨的做法是让函数值也包含正则项，但代码中未这么做。

------

### 2. 为什么要判断 `if grad.size > 0`？

NLopt 的 `opt.optimize()` 会调用目标函数，并传入一个 `grad` 数组。如果该数组的长度大于 0，说明优化器需要梯度；如果长度为 0，说明当前只需要计算函数值（例如在仅进行函数评估或某些无导数优化中）。
这个判断允许代码在不需要梯度时跳过昂贵的雅可比计算和反向传播，提高效率。代码中只有在 `grad.size > 0` 时才计算雅可比、反向传播、梯度转换等，否则只返回函数值。这是 NLopt 接口的常见用法。



-- --

 # NLOPT的使用

是的，**`self.opt.set_min_objective(objective_fn)` 中的 `objective_fn` 必须是一个回调函数，其签名固定为 `objective(x, grad) -> float`**，这是 NLopt 库的接口要求。

---

## 1. 为什么必须包含 `grad` 参数？

NLopt 的 `set_min_objective` 期望的回调函数形式为：
```python
def myfunc(x, grad):
    if grad.size > 0:
        # 计算梯度，填充到 grad 数组
        grad[:] = ... 
    # 返回目标函数值
    return value
```

- 当 NLopt 调用该函数时，会根据当前使用的优化算法决定是否需要梯度：
  - **基于梯度的算法**（如 `LD_SLSQP`）会在每次迭代时要求计算梯度，此时 `grad` 数组非空（`size > 0`），函数必须将梯度写入 `grad` 数组。
  - **无梯度算法**（如 `LN_COBYLA`）在调用时 `grad` 数组长度可能为 0，函数只需返回值。

在你的代码中，优化器被设置为 `nlopt.LD_SLSQP`，这是一个**基于梯度的算法**，因此 NLopt 每次调用 `objective` 时都会传入一个非空的 `grad` 数组，要求计算梯度。

---

## 2. `objective` 函数内部如何工作？

在 `DexPilotOptimizer` 中，`get_objective_function` 返回的闭包 `objective` 正是这样一个符合 NLopt 要求的函数：
```python
def objective(x: np.ndarray, grad: np.ndarray) -> float:
    # ... 前向计算，得到损失值 result ...
    if grad.size > 0:
        # 计算梯度并填充到 grad 数组
        grad[:] = grad_qpos[:]
    return result
```
- 无论是否计算梯度，`objective` 都必须返回目标函数值。
- 当 `grad.size > 0` 时，它才计算梯度；否则跳过（节省计算量）。

---

## 3. 这个设计的作用

- **统一接口**：同一个 `objective` 函数既可以用于需要梯度的优化算法，也可以用于不需要梯度的算法（只需在 `grad.size == 0` 时跳过梯度计算）。
- **效率**：避免在不需要梯度时执行昂贵的雅可比计算和反向传播。

---

## 4. 总结

- `set_min_objective` 传入的回调函数**必须**具有 `(x, grad) -> float` 的签名。
- 回调函数**应当**根据 `grad.size > 0` 判断是否需要计算梯度。
- 在 DexPilot 代码中，`objective` 函数通过这种方式实现了与 NLopt 的无缝集成，并利用梯度信息加速优化。



## 5 扩展

### 5.1 关于约束的加入



-- --



# PositionOptimizer 优化笔记

## 1. 方法概述

PositionOptimizer 是一种**多目标 IK（逆运动学）**方法，用于离线数据处理场景（如 DexYCB 数据集）。

- **输入**：人手 N 个关键点的 3D 世界坐标（从 MANO 模型 forward kinematics 得到）
- **输出**：机器人关节角 `q`
- **本质**：最小化机器人 N 个 link 的 FK 位置与人手对应关节位置之间的距离

与 DexPilotOptimizer 的区别：
- Position：直接匹配 3D 位置，适合离线重定向（有完整手部姿态）
- DexPilot/Vector：匹配相对向量，适合实时遥操作（对绝对位置漂移更鲁棒）

---

## 2. 数据输入

**来源**：DexYCB 数据集，`pose["pose_m"]` 经 MANO 模型 FK 得到 21 个关节的 3D 坐标。

MANO 21 个关节索引：

| 索引 | 关节 | 索引 | 关节 |
|------|------|------|------|
| 0 | wrist | 11 | middle_dip |
| 1 | thumb_mcp | 12 | middle_tip |
| 2 | thumb_pip | 13 | ring_mcp |
| 3 | thumb_dip | 14 | ring_pip |
| 4 | thumb_tip | 15 | ring_dip |
| 5 | index_mcp | 16 | ring_tip |
| 6 | index_pip | 17 | little_mcp |
| 7 | index_dip | 18 | little_pip |
| 8 | index_tip | 19 | little_dip |
| 9 | middle_mcp | 20 | little_tip |
| 10 | middle_pip | | |

配置文件中 `target_link_human_indices` 选取其中 N 个作为优化目标，例如 Allegro（offline）取 N=8：
```
[4, 8, 12, 16, 2, 6, 10, 14]  # 四指指尖 + 四指 DIP 关节
```

---

## 3. 目标函数

\[
\mathcal{L}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \text{Huber}_{\delta}\big(\|\mathbf{p}_i^{\text{robot}}(\mathbf{x}) - \mathbf{p}_i^{\text{target}}\|\big) \;+\; \alpha \|\mathbf{x} - \mathbf{x}_{\text{prev}}\|^2
\]

- **\(\mathbf{x}\)**：机器人关节角向量
- **\(\mathbf{p}_i^{\text{robot}}(\mathbf{x})\)**：第 \(i\) 个机器人 link 的 FK 位置（3D）
- **\(\mathbf{p}_i^{\text{target}}\)**：人手对应关节的 3D 位置（`ref_value[i]`）
- **\(\text{Huber}_{\delta}\)**：Huber 损失，`huber_delta=0.02`
- **\(\alpha\)**：时序正则化强度，`norm_delta=4e-3`

与 DexPilotOptimizer 相比，PositionOptimizer **没有动态权重**，所有 link 权重均等。

---

## 4. 梯度计算

与 DexPilotOptimizer 完全一致，链式法则：

\[
\nabla_{\mathbf{x}} \mathcal{L} = \sum_i \left( \frac{\partial L_{\text{huber}}}{\partial \mathbf{p}_i} \right)^{\!T} \cdot \mathbf{J}_{\mathbf{p}_i} \;+\; 2\alpha (\mathbf{x} - \mathbf{x}_{\text{prev}})
\]

代码步骤（`optimizer.py: PositionOptimizer.get_objective_function`）：

1. FK 计算所有目标 link 位置 `body_pos`，shape `(N, 3)`
2. PyTorch 计算 Huber 损失并 `backward()`，得到 `grad_pos` shape `(N, 1, 3)`
3. 计算每个 link 的世界坐标系雅可比：`link_rot @ link_body_jacobian`，shape `(N, 3, dof)`
4. `np.matmul(grad_pos, jacobians).mean(1).sum(0)` 得到关节梯度，shape `(dof,)`
5. 加正则项梯度 `2 * norm_delta * (x - last_qpos)`

---

## 5. Warm Start

PositionOptimizer 配合 `add_dummy_free_joint=True` 使用时，机器人根部有 6-DoF 自由关节，允许手在空间中自由移动。

`SeqRetargeting.warm_start()` 在第一帧通过解析计算初始化腕部 6D 姿态，避免优化从零开始陷入局部最优：

```python
retargeting.warm_start(
    joint[0, :],      # 人手腕部 3D 位置
    wrist_quat,       # 人手腕部四元数
    hand_type=self.hand_type,
    is_mano_convention=True,
)
```

---

# VectorOptimizer 优化笔记

VectorOptimizer 是遥操作场景下的基础向量匹配方法，比 DexPilot 更简单，无投影机制。

## 1 目标函数

\[
\mathcal{L}(\mathbf{x}) = \frac{1}{N} \sum_{k=1}^{N} \text{Huber}_{\delta}\big(\|\mathbf{v}_k^{\text{robot}}(\mathbf{x}) - s \cdot \mathbf{v}_k^{\text{target}}\|\big) \;+\; \alpha \|\mathbf{x} - \mathbf{x}_{\text{prev}}\|^2
\]

- **\(\mathbf{v}_k = \mathbf{p}_{\text{task}} - \mathbf{p}_{\text{origin}}\)**：机器人 link 对之间的相对向量
- **\(s\)**：`scaling_factor`，补偿机器人与人手尺寸差异（如 Allegro 为 1.6）
- 配置中 `target_link_human_indices` shape 为 `(2, N)`，第 0 行是 origin 索引，第 1 行是 task 索引

## 2 与 DexPilotOptimizer 的关系

| 特性 | VectorOptimizer | DexPilotOptimizer |
|------|----------------|-------------------|
| 向量构造 | 任意 link 对 | 固定：手腕-指尖 + 指尖间 |
| 动态权重 | 无（均等） | 有（投影状态高权重） |
| 投影机制 | 无 | 有（防止手指交叉） |
| 适用场景 | 通用遥操作 | 精细抓取遥操作 |