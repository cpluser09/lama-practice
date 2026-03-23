# LaMa 推理流程数据精度分析报告

> 分析对象：[advimman/lama](https://github.com/advimman/lama)
> 分析范围：推理（Inference）全链路数据精度类型

---

## 1. 推理流程总览

LaMa 的推理入口为 `bin/predict.py`，完整链路如下：

```
磁盘图像文件 (uint8)
  → PIL 读取 (uint8)
  → NumPy 转换 + 归一化 (float32, [0,1])
  → PyTorch Tensor (float32)
  → 模型前向传播 (float32)
  → Sigmoid 激活输出 (float32, [0,1])
  → 融合 inpainted (float32, [0,1])
  → 反归一化 (uint8, [0,255])
  → cv2 写入磁盘 (uint8)
```

---

## 2. 各阶段精度详细分析

### 2.1 输入读取阶段

#### 图像读取

| 属性 | 值 |
|------|-----|
| **源文件** | `saicinpainting/evaluation/data.py` |
| **函数** | `load_image()` (第 12–21 行) |
| **读取方式** | `PIL.Image.open(fname).convert('RGB')` |
| **原始精度** | **uint8**（PIL 默认以 8 位无符号整数读取） |
| **支持位深** | 仅支持 8 位图像。PIL `convert('RGB')` 将任何输入统一转为 8 位 RGB |

关键代码（`saicinpainting/evaluation/data.py` 第 12–21 行）：

```python
def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))   # → numpy uint8, shape (H,W,3)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))             # → (3,H,W)
    out_img = img.astype('float32') / 255               # → float32, 值域 [0,1]
    if return_orig:
        return out_img, img
    else:
        return out_img
```

#### Mask 读取

| 属性 | 值 |
|------|-----|
| **源文件** | `saicinpainting/evaluation/data.py` |
| **函数** | `load_image()` 以 `mode='L'` 调用（第 73 行） |
| **原始精度** | **uint8**（灰度图，单通道） |
| **转换后精度** | **float32**，值域 [0, 1] |

调用位置（`saicinpainting/evaluation/data.py` 第 70–74 行，`InpaintingDataset.__getitem__`）：

```python
def __getitem__(self, i):
    image = load_image(self.img_filenames[i], mode='RGB')   # float32, (3,H,W)
    mask = load_image(self.mask_filenames[i], mode='L')      # float32, (H,W)
    result = dict(image=image, mask=mask[None, ...])         # mask → (1,H,W)
```

### 2.2 预处理阶段

#### 填充（Padding）

| 属性 | 值 |
|------|-----|
| **源文件** | `saicinpainting/evaluation/data.py` |
| **函数** | `pad_img_to_modulo()` (第 30–35 行) |
| **数据类型** | 保持 **float32** 不变 |
| **填充方式** | `np.pad(..., mode='symmetric')`，对称填充 |

```python
def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')
```

#### Mask 二值化

在推理时，`bin/predict.py` 第 80 行对 mask 进行了显式二值化：

```python
batch['mask'] = (batch['mask'] > 0) * 1
```

此操作将 mask 从连续 float32 [0,1] 转为 **二值 int（0 或 1）**，然后在后续乘法运算中被自动提升回 **float32**。

#### Collate（批量化）

| 属性 | 值 |
|------|-----|
| **源文件** | `bin/predict.py` 第 72 行 |
| **函数** | `torch.utils.data._utils.collate.default_collate()` |
| **输入** | NumPy float32 数组 |
| **输出** | **torch.Tensor (float32)**，增加 batch 维度 |

```python
batch = default_collate([dataset[img_i]])
```

`default_collate` 将 NumPy float32 数组自动转换为 `torch.FloatTensor`（即 `torch.float32`）。

### 2.3 模型前向传播阶段

#### 模型入口

| 属性 | 值 |
|------|-----|
| **源文件** | `saicinpainting/training/trainers/default.py` |
| **类** | `DefaultInpaintingTrainingModule` |
| **方法** | `forward()` (第 52–87 行) |

```python
def forward(self, batch):
    img = batch['image']           # float32, (B,3,H,W), [0,1]
    mask = batch['mask']           # float32, (B,1,H,W), {0,1}

    masked_img = img * (1 - mask)  # float32, 被 mask 区域置零

    if self.concat_mask:
        masked_img = torch.cat([masked_img, mask], dim=1)  # float32, (B,4,H,W)

    batch['predicted_image'] = self.generator(masked_img)   # float32, (B,3,H,W)
    batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

    return batch
```

**关键数据类型变化：**

| 张量 | 形状 | 精度 | 值域 |
|------|------|------|------|
| `img` | (B, 3, H, W) | float32 | [0, 1] |
| `mask` | (B, 1, H, W) | float32 | {0, 1}（二值） |
| `masked_img` | (B, 4, H, W) | float32 | [0, 1] |
| `predicted_image` | (B, 3, H, W) | float32 | [0, 1]（Sigmoid 输出） |
| `inpainted` | (B, 3, H, W) | float32 | [0, 1] |

#### 生成器架构（FFCResNetGenerator）

| 属性 | 值 |
|------|-----|
| **源文件** | `saicinpainting/training/modules/ffc.py` |
| **类** | `FFCResNetGenerator` (第 305–370 行) |
| **输入通道** | 4（3 通道图像 + 1 通道 mask） |
| **输出通道** | 3 |
| **输出激活函数** | **Sigmoid**（由 `big-lama/config.yaml` 第 113 行 `add_out_act: sigmoid` 控制） |
| **全部计算精度** | **float32** |

输出激活函数的设置逻辑（`saicinpainting/training/modules/ffc.py` 第 365–367 行）：

```python
if add_out_act:
    model.append(get_activation('tanh' if add_out_act is True else add_out_act))
```

当 `add_out_act` 为字符串 `'sigmoid'`（而非布尔 `True`）时，调用 `get_activation('sigmoid')`，返回 `nn.Sigmoid()`。

激活函数注册（`saicinpainting/training/modules/base.py` 第 43–50 行）：

```python
def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')
```

> **重要结论**：big-lama 模型使用 **Sigmoid** 激活函数，生成器输出值域为 **[0, 1]**，而非 [-1, 1]。这与后处理中直接 `× 255` 的操作一致。

#### 融合操作

```python
batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
```

- mask 区域：使用模型预测结果
- 非 mask 区域：保留原始图像
- 精度：始终 **float32**，值域 **[0, 1]**

### 2.4 后处理阶段

| 属性 | 值 |
|------|-----|
| **源文件** | `bin/predict.py` 第 86–89 行 |
| **操作** | 反归一化 → 颜色空间转换 → 写入 |

```python
cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
# → float32 NumPy, (H,W,3), [0,1]

unpad_to_size = batch.get('unpad_to_size', None)
if unpad_to_size is not None:
    orig_height, orig_width = unpad_to_size
    cur_res = cur_res[:orig_height, :orig_width]  # 裁剪到原始尺寸

cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')  # float32 → uint8
cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)         # RGB → BGR
cv2.imwrite(cur_out_fname, cur_res)                        # 写入磁盘
```

---

## 3. 完整推理链路精度变化总表

| 阶段 | 操作 | 数据精度 | 值域 | 形状 | 代码位置 |
|------|------|---------|------|------|---------|
| 磁盘读取 | `Image.open().convert('RGB')` | uint8 | [0, 255] | (H, W, 3) | `evaluation/data.py` L13 |
| NumPy 转换 | `np.array(...)` | uint8 | [0, 255] | (H, W, 3) | `evaluation/data.py` L13 |
| 转置 | `np.transpose(img, (2,0,1))` | uint8 | [0, 255] | (3, H, W) | `evaluation/data.py` L15 |
| 归一化 | `.astype('float32') / 255` | **float32** | **[0, 1]** | (3, H, W) | `evaluation/data.py` L16 |
| Mask 读取 | `Image.open().convert('L')` → 归一化 | **float32** | **[0, 1]** | (H, W) → (1, H, W) | `evaluation/data.py` L13–16, L73–74 |
| 填充 | `np.pad(..., mode='symmetric')` | float32 | [0, 1] | (C, H', W') | `evaluation/data.py` L30–35 |
| Collate | `default_collate()` | **torch.float32** | [0, 1] | (1, C, H', W') | `bin/predict.py` L72 |
| Mask 二值化 | `(mask > 0) * 1` | torch.float32 | {0, 1} | (1, 1, H', W') | `bin/predict.py` L80 |
| 遮蔽拼接 | `cat([img*(1-mask), mask])` | torch.float32 | [0, 1] | (1, 4, H', W') | `trainers/default.py` L60–63 |
| 生成器前向 | `FFCResNetGenerator(input)` | torch.float32 | [0, 1] | (1, 3, H', W') | `modules/ffc.py` L369 |
| Sigmoid 激活 | `nn.Sigmoid()` | torch.float32 | **(0, 1)** | (1, 3, H', W') | `modules/ffc.py` L367 |
| 融合 | `mask * pred + (1-mask) * img` | torch.float32 | [0, 1] | (1, 3, H', W') | `trainers/default.py` L65 |
| Tensor→NumPy | `.permute(1,2,0).cpu().numpy()` | float32 | [0, 1] | (H', W', 3) | `bin/predict.py` L82 |
| 去填充 | `cur_res[:H,:W]` | float32 | [0, 1] | (H, W, 3) | `bin/predict.py` L84–86 |
| 反归一化 | `np.clip(x*255, 0, 255).astype('uint8')` | **uint8** | **[0, 255]** | (H, W, 3) | `bin/predict.py` L88 |
| 颜色转换 | `cv2.cvtColor(RGB→BGR)` | uint8 | [0, 255] | (H, W, 3) | `bin/predict.py` L89 |
| 写入磁盘 | `cv2.imwrite()` | uint8 | [0, 255] | (H, W, 3) | `bin/predict.py` L90 |

---

## 4. 输入/输出精度总结

### 4.1 输入精度

| 输入类型 | 磁盘格式 | 读取精度 | 模型输入精度 | 模型输入值域 |
|---------|---------|---------|------------|------------|
| **图像** (RGB) | 8-bit PNG/JPG | uint8 | torch.float32 | [0, 1] |
| **Mask** (灰度) | 8-bit PNG | uint8 | torch.float32 | {0, 1}（二值化） |

> **注意**：代码仅支持 **8 位** 输入图像。`PIL.Image.convert('RGB')` 会将 16 位图像降至 8 位；不支持 32 位浮点输入格式。

### 4.2 输出精度

| 属性 | 值 |
|------|-----|
| **模型输出精度** | torch.float32 |
| **模型输出值域** | [0, 1]（经 Sigmoid 激活） |
| **最终输出精度** | **uint8** |
| **最终输出值域** | [0, 255] |
| **输出格式** | PNG 或 JPG（由 `out_ext` 配置项决定，默认 `.png`） |
| **输出颜色空间** | BGR（cv2 格式） |

---

## 5. Refinement 路径的精度分析

当 `predict_config.refine = True` 时，走 refinement 路径（`saicinpainting/evaluation/refinement.py`）。

### 5.1 Refinement 流程精度

| 阶段 | 操作 | 精度 | 值域 | 代码位置 |
|------|------|-----|------|---------|
| 图像金字塔构建 | `_pyrdown()` (高斯模糊 + 双线性下采样) | float32 | [0, 1] | `refinement.py` L19–25 |
| Mask 金字塔构建 | `_pyrdown_mask()` (下采样 + 二值化) | float32 | {0, 1} | `refinement.py` L27–52 |
| Tensor 填充 | `pad_tensor_to_modulo()` + 二值化 | float32 | {0, 1} | `refinement.py` L333–335 |
| 遮蔽拼接 | `image * (1-mask)` + `cat([..., mask])` | float32 | [0, 1] | `refinement.py` L116–117 |
| 特征提取 | `forward_front(masked_image)` (no_grad) | float32 | 任意 | `refinement.py` L121 |
| 迭代优化 | Adam 优化器更新 z1, z2 特征 | float32 | 任意 | `refinement.py` L127–156 |
| 融合输出 | `mask * pred + (1-mask) * image` | float32 | [0, 1] | `refinement.py` L163 |

### 5.2 Mask 二值化策略

在 refinement 路径中，mask 的二值化有两种模式（`_pyrdown_mask()`，第 27–52 行）：

- **`round_up=True`**：`mask[mask >= eps] = 1`，`mask[mask < eps] = 0`（更激进，默认）
- **`round_up=False`**：`mask[mask >= 1-eps] = 1`，`mask[mask < 1-eps] = 0`（更保守）

---

## 6. JIT 导出路径的精度分析

| 属性 | 值 |
|------|-----|
| **源文件** | `bin/to_jit.py` 第 14–27 行 |
| **Trace 输入** | `torch.rand(1, 3, 120, 120)` — float32, [0, 1) |
| **输入精度** | 与推理一致：image float32 [0,1], mask float32 [0,1] |
| **输出** | `out["inpainted"]` — float32, [0, 1] |

```python
class JITWrapper(nn.Module):
    def forward(self, image, mask):
        batch = {"image": image, "mask": mask}
        out = self.model(batch)
        return out["inpainted"]
```

---

## 7. 关键结论

1. **输入仅支持 8 位精度**：通过 `PIL.Image.open().convert()` 读取，不支持 16 位或 32 位输入图像。
2. **模型内部计算全程 float32**：从归一化开始到输出融合，所有张量运算均为 `torch.float32`。
3. **生成器输出值域为 [0, 1]**：big-lama 配置使用 **Sigmoid** 激活函数（`add_out_act: sigmoid`），而非 Tanh。这一点与直接 `× 255` 反归一化的后处理逻辑一致。
4. **输出为 8 位精度**：最终输出通过 `np.clip(x * 255, 0, 255).astype('uint8')` 转回 uint8，保存为 PNG/JPG。
5. **Mask 被二值化处理**：推理主路径通过 `(mask > 0) * 1` 二值化；refinement 路径有更精细的阈值控制。
6. **不存在 float16/bfloat16 推理路径**：代码中未使用 `torch.half()`、`torch.autocast` 或混合精度推理。

---

## 8. 精度对比总表

| 数据 | 磁盘精度 | 预处理后 | 模型计算 | 后处理后 | 磁盘输出 |
|------|---------|---------|---------|---------|---------|
| **输入图像** | uint8 (8-bit) | float32 [0,1] | float32 | — | — |
| **输入 Mask** | uint8 (8-bit) | float32 {0,1} | float32 | — | — |
| **predicted_image** | — | — | float32 [0,1] | — | — |
| **inpainted** | — | — | float32 [0,1] | uint8 [0,255] | uint8 (8-bit) |

---

*本报告基于 [advimman/lama](https://github.com/advimman/lama) 源码及 `big-lama/config.yaml` 配置文件分析生成。所有结论均附有具体源码定位。*
