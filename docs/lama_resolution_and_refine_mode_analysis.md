# LaMa 训练分辨率、推理最优分辨率及 Refine 模式深度分析

> 分析对象：[advimman/lama](https://github.com/advimman/lama)
>
> 分析范围：训练分辨率、推理最优分辨率、Refine 模式原理、推理模式对比

---

## 目录

1. [训练时的图片分辨率](#1-训练时的图片分辨率)
2. [推理阶段最优分辨率](#2-推理阶段最优分辨率)
3. [Refine 模式深度解析](#3-refine-模式深度解析)
4. [推理模式对比](#4-推理模式对比)

---

## 1. 训练时的图片分辨率

### 1.1 训练分辨率确定

LaMa 模型在训练时使用的是 **256×256** 分辨率。

### 1.2 关键证据

所有主要训练配置文件都明确指定 `out_size: 256`：

| 配置文件 | 位置 | out_size 值 |
|---------|------|------------|
| `abl-04-256-mh-dist.yaml` | `configs/training/data/` | **256** |
| `abl-04-256-mh-dist-web.yaml` | `configs/training/data/` | **256** |
| `abl-04-256-mh-dist-celeba.yaml` | `configs/training/data/` | **256** |

代码位置示例 (`configs/training/data/abl-04-256-mh-dist.yaml:9`)：
```yaml
train:
  indir: ${location.data_root_dir}/train
  out_size: 256          # ← 训练输出尺寸
  mask_gen_kwargs:
    irregular_proba: 1
    ...
```

### 1.3 训练数据加载流程

训练时的数据处理流程 (`saicinpainting/training/data/datasets.py`)：

```python
def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),      # +/- 20% 随机缩放
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),  # 裁剪到 out_size
            ...
        ])
    ...
```

训练数据增强包含：
- 随机缩放 (+/- 20%)
- 填充到目标尺寸
- 随机裁剪到 **256×256**
- 水平翻转、颜色抖动等

### 1.4 训练配置引用关系

主要训练配置都引用 256 分辨率的数据配置：

| 训练配置 | 引用的数据配置 |
|---------|--------------|
| `lama-regular.yaml` | `abl-04-256-mh-dist` |
| `lama-fourier.yaml` | `abl-04-256-mh-dist` |
| `big-lama.yaml` | `abl-04-256-mh-dist` |
| `big-lama-regular.yaml` | `abl-04-256-mh-dist` |

---

## 2. 推理阶段最优分辨率

### 2.1 推理配置关键参数

推理配置位于 `configs/prediction/default.yaml`：

```yaml
dataset:
  kind: default
  img_suffix: .png
  pad_out_to_modulo: 8          # 填充到 8 的倍数

refine: False
refiner:
  gpu_ids: 0,1
  modulo: ${dataset.pad_out_to_modulo}
  n_iters: 15
  lr: 0.002
  min_side: 512                 # 最小边长
  max_scales: 3                  # 最大尺度数
  px_budget: 1800000            # 像素预算
```

### 2.2 各参数详细解析

#### 2.2.1 像素预算 (px_budget: 1,800,000)

**定义**：图像的 `height × width ≤ 1,800,000` 像素

这是推理时的硬性限制，超过此尺寸的图像会被自动缩放。

代码位置 (`saicinpainting/evaluation/refinement.py:203-211`)：
```python
if h*w > px_budget:
    # resize
    ratio = np.sqrt(px_budget / float(h*w))
    h_orig, w_orig = h, w
    h,w = int(h*ratio), int(w*ratio)
    print(f"Original image too large for refinement! Resizing {(h_orig,w_orig)} to {(h,w)}...")
    image = resize(image, (h,w), interpolation='bilinear', align_corners=False)
    mask = resize(mask, (h,w), interpolation='bilinear', align_corners=False)
    mask[mask>1e-8] = 1
```

**常见尺寸示例**：
- 1340 × 1340 ≈ 180万像素（正方形）
- 1920 × 937 ≈ 180万像素（16:9）
- 1600 × 1125 = 180万像素（手机竖屏）
- 1024 × 1024 = 105万像素（在预算内）

#### 2.2.2 最小边长 (min_side: 512)

**定义**：图像金字塔中最小尺度的边长至少为 `512 / √2 ≈ 362` 像素

代码位置 (`saicinpainting/evaluation/refinement.py:212-214`)：
```python
breadth = min(h,w)
n_scales = min(1 + int(round(max(0,np.log2(breadth / min_side)))), max_scales)
```

这个参数确保多尺度金字塔的最低分辨率不会太小，保证细节生成质量。

#### 2.2.3 填充要求 (pad_out_to_modulo: 8)

**定义**：图像尺寸需要是 8 的倍数

代码位置 (`saicinpainting/evaluation/data.py:29-33`)：
```python
def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')
```

这是因为网络结构中有 3 次下采样（2³ = 8），需要保证尺寸可以被 8 整除。

#### 2.2.4 多尺度金字塔 (max_scales: 3)

**定义**：使用最多 3 个尺度的图像金字塔进行渐进式细化

### 2.3 推荐的最优推理分辨率

基于上述参数分析，推荐以下最优推理分辨率：

| 分辨率 | 像素数 | 说明 |
|--------|--------|------|
| **512 × 512** | 262,144 | 接近训练分辨率，效果稳定快速，适合测试 |
| **1024 × 1024** | 1,048,576 | 在预算内，能获得较好的细节平衡 |
| **1340 × 1340** | 1,795,600 | 接近 180 万像素预算上限，质量最高 |
| **1920 × 937** | 1,799,040 | 16:9 宽屏，接近预算上限 |

**推荐方案**：
- **快速测试**：512×512
- **通用用途**：1024×1024
- **最佳质量**：1340×1340（或同像素数的其他比例）

---

## 3. Refine 模式深度解析

### 3.1 什么是 Refine 模式？

**Refine 模式**（也称为 *Plug-n-Play* 或 *Multi-Scale Refinement*）是 LaMa 的一种高级推理模式，通过**多尺度金字塔** + **迭代优化**来提升 inpainting 质量。

与标准单次前向推理不同，Refine 模式从低分辨率开始，逐步细化到高分辨率，并通过优化特征空间来提升一致性和细节质量。

### 3.2 Refine 模式的意义

#### 3.2.1 解决的问题

1. **大尺度一致性**：普通单次推理在处理大图像时容易出现局部一致性问题
2. **细节缺失**：高分辨率下的细节生成质量往往不如人意
3. **多尺度语义**：低分辨率捕获全局结构，高分辨率补充细节

#### 3.2.2 优势

| 特性 | 说明 |
|------|------|
| **多尺度融合** | 从低到高逐步细化，兼顾全局结构和局部细节 |
| **迭代优化** | 通过梯度下降优化特征，而非单次前向 |
| **更好的一致性** | 利用低分辨率结果作为高分辨率的参考 |

### 3.3 Refine 模式原理详解

#### 3.3.1 整体流程

Refine 模式的核心流程 (`saicinpainting/evaluation/refinement.py:refine_predict`)：

```
输入图像 + Mask
    ↓
[步骤 1] 构建图像金字塔 (3个尺度)
    ↓
[步骤 2] 从最低分辨率开始
    ↓
[步骤 3] 提取特征 z1, z2 (梯度不更新)
    ↓
[步骤 4] 迭代优化 z1, z2 (15次 Adam 优化)
    ↓
[步骤 5] 上采样到下一尺度，作为参考
    ↓
[步骤 6] 重复步骤 3-5，直到最高分辨率
    ↓
输出最终结果
```

#### 3.3.2 关键组件详解

##### 组件 1：图像金字塔构建 (`_get_image_mask_pyramid`)

代码位置 (`saicinpainting/evaluation/refinement.py:176-226`)：

```python
def _get_image_mask_pyramid(batch, min_side, max_scales, px_budget):
    # 1. 检查并应用像素预算
    h, w = batch['unpad_to_size']
    if h*w > px_budget:
        ratio = np.sqrt(px_budget / float(h*w))
        h, w = int(h*ratio), int(w*ratio)
        image = resize(image, (h,w), ...)
        mask = resize(mask, (h,w), ...)

    # 2. 计算尺度数量
    breadth = min(h,w)
    n_scales = min(1 + int(round(max(0,np.log2(breadth / min_side)))), max_scales)

    # 3. 构建金字塔（从高到低）
    ls_images = [image]
    ls_masks = [mask]
    for _ in range(n_scales - 1):
        image_p = _pyrdown(ls_images[-1])      # 高斯模糊 + 下采样
        mask_p = _pyrdown_mask(ls_masks[-1])    # 下采样 mask
        ls_images.append(image_p)
        ls_masks.append(mask_p)

    # 4. 反转，让最低分辨率在最前面
    return ls_images[::-1], ls_masks[::-1]
```

**金字塔构建示例**（假设输入 1024×1024，max_scales=3）：
- 尺度 0（最低）：256×256
- 尺度 1：512×512
- 尺度 2（最高）：1024×1024

##### 组件 2：下采样 (`_pyrdown`)

代码位置 (`saicinpainting/evaluation/refinement.py:19-26`)：

```python
def _pyrdown(im, downsize=None):
    """downscale the image"""
    if downsize is None:
        downsize = (im.shape[2]//2, im.shape[3]//2)
    # 步骤 1：高斯模糊
    im = gaussian_blur2d(im, kernel_size=(5,5), sigma=(1.0,1.0))
    # 步骤 2：双线性插值下采样
    im = F.interpolate(im, size=downsize, mode='bilinear', align_corners=False)
    return im
```

##### 组件 3：Mask 下采样 (`_pyrdown_mask`)

代码位置 (`saicinpainting/evaluation/refinement.py:28-64`)：

```python
def _pyrdown_mask(mask, downsize=None, eps=1e-8, blur_mask=True, round_up=True):
    if blur_mask:
        mask = gaussian_blur2d(mask, kernel_size=(5,5), sigma=(1.0,1.0))
    mask = F.interpolate(mask, size=downsize, mode='bilinear', align_corners=False)

    # 二值化
    if round_up:
        mask[mask>=eps] = 1      # 激进：≥eps 设为 1
        mask[mask<eps] = 0
    else:
        mask[mask>=1.0-eps] = 1  # 保守：≥1-eps 设为 1
        mask[mask<1.0-eps] = 0
    return mask
```

##### 组件 4：核心推理 (`_infer`)

这是 Refine 模式最核心的部分，代码位置 (`saicinpainting/evaluation/refinement.py:86-174`)：

```python
def _infer(image, mask, forward_front, forward_rears,
           ref_lower_res, orig_shape, devices, scale_ind,
           n_iters=15, lr=0.002):
    """Performs inference with refinement at a given scale."""

    # 1. 准备输入
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)
    mask = mask.repeat(1,3,1,1)

    # 2. 提取特征 (torch.no_grad，梯度不回传这部分)
    with torch.no_grad():
        z1, z2 = forward_front(masked_image)

    # 3. 设置优化器：只优化 z1, z2 特征，不优化网络权重
    z1, z2 = z1.detach().to(devices[0]), z2.detach().to(devices[0])
    z1.requires_grad, z2.requires_grad = True, True
    optimizer = Adam([z1, z2], lr=lr)

    # 4. 迭代优化 (15次)
    for idi in range(n_iters):
        optimizer.zero_grad()
        input_feat = (z1, z2)

        # 前向传播生成图像
        for idd, forward_rear in enumerate(forward_rears):
            output_feat = forward_rear(input_feat)
            if idd < len(devices) - 1:
                midz1, midz2 = output_feat
                input_feat = (midz1, midz2)
            else:
                pred = output_feat

        # 如果是最低尺度，不需要优化损失
        if ref_lower_res is None:
            break

        # 5. 计算多尺度损失
        losses = {}
        # 下采样预测结果
        pred_downscaled = _pyrdown(pred[:,:,:orig_shape[0],:orig_shape[1]])
        mask_downscaled = _pyrdown_mask(mask[:,:1,:orig_shape[0],:orig_shape[1]],
                                         blur_mask=False, round_up=False)
        mask_downscaled = _erode_mask(mask_downscaled, ekernel=ekernel)
        mask_downscaled = mask_downscaled.repeat(1,3,1,1)

        # L1 损失：预测结果的下采样应与低尺度参考一致
        losses["ms_l1"] = _l1_loss(pred, pred_downscaled, ref_lower_res,
                                     mask, mask_downscaled, image, on_pred=True)

        loss = sum(losses.values())

        # 6. 反向传播更新 z1, z2
        if idi < n_iters - 1:
            loss.backward()
            optimizer.step()

    # 7. 融合输出
    inpainted = mask * pred + (1 - mask) * image
    return inpainted
```

**关键点**：
- 不优化网络权重，只优化中间特征 `z1, z2`
- 使用低尺度结果作为高尺度的参考
- 损失函数鼓励高尺度预测的下采样与低尺度参考一致

##### 组件 5：损失函数 (`_l1_loss`)

代码位置 (`saicinpainting/evaluation/refinement.py:75-84`)：

```python
def _l1_loss(pred, pred_downscaled, ref, mask, mask_downscaled,
             image, on_pred=True):
    """l1 loss on src pixels, and downscaled predictions if on_pred=True"""
    # 非 mask 区域：保持与原图一致
    loss = torch.mean(torch.abs(pred[mask<1e-8] - image[mask<1e-8]))
    if on_pred:
        # mask 区域：下采样后应与低分辨率参考一致
        loss += torch.mean(torch.abs(pred_downscaled[mask_downscaled>=1e-8]
                                      - ref[mask_downscaled>=1e-8]))
    return loss
```

#### 3.3.3 网络拆分

Refine 模式将生成器网络拆分为两部分 (`refinement.py:280-289`)：

```python
# 找到第一个 ResNet 块的位置
first_resblock_ind = 0
found_first_resblock = False
for idl in range(len(inpainter.generator.model)):
    if isinstance(inpainter.generator.model[idl], FFCResnetBlock):
        n_resnet_blocks += 1
        found_first_resblock = True
    elif not found_first_resblock:
        first_resblock_ind += 1

# 拆分网络
forward_front = inpainter.generator.model[0:first_resblock_ind]  # 前端
forward_rears = [...]  # 后端（可能分到多个 GPU）
```

**拆分意义**：
- `forward_front`：下采样 + 初始卷积 → 输出特征 `z1, z2`
- `forward_rears`：ResNet 块 + 上采样 → 从特征重建图像

### 3.4 Refine 模式完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入 (H,W)                               │
│                    Image + Mask                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  步骤 1: 像素预算检查 & 缩放 (如果 H*W > 1,800,000)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  步骤 2: 构建图像金字塔 (max_scales=3)                            │
│                                                                   │
│  尺度 0: 256×256 (最低分辨率) ←─┐                               │
│  尺度 1: 512×512                 │  从低到高处理                 │
│  尺度 2: 1024×1024 (最高) ──────┘                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │尺度 0   │         │尺度 1   │         │尺度 2   │
   │256×256  │         │512×512  │         │1024×1024│
   └────┬────┘         └────┬────┘         └────┬────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │提取特征 │         │提取特征 │         │提取特征 │
   │z1, z2   │         │z1, z2   │         │z1, z2   │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │Adam优化 │         │Adam优化 │         │Adam优化 │
   │15次迭代 │         │15次迭代 │         │15次迭代 │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │生成分辨 │         │生成分辨 │         │生成分辨 │
   │率结果   │────────▶│率结果   │────────▶│率结果   │
   └─────────┘  作为   └─────────┘  作为   └────┬────┘
               参考                 参考             │
                                                      ▼
                                          ┌──────────────────┐
                                          │   最终输出结果     │
                                          └──────────────────┘
```

---

## 4. 推理模式对比

LaMa 有 **两种主要推理模式**，下面详细对比。

### 4.1 模式 1：标准模式 (Standard Mode)

**配置**：`refine: False`

#### 4.1.1 流程

代码位置 (`bin/predict.py:80-90`)：

```python
else:
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = model(batch)                    # 单次前向传播
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
```

**关键特点**：
- 单次前向传播 (`model(batch)`)
- `torch.no_grad()`：无梯度计算
- 直接输出结果

#### 4.1.2 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 速度快（单次前向） | ❌ 大图像一致性可能较差 |
| ✅ 内存占用低 | ❌ 细节质量一般 |
| ✅ 实现简单 | ❌ 没有多尺度融合 |

#### 4.1.3 适用场景

- 快速预览
- 小图像（≤ 512×512）
- 批量处理
- 实时应用

---

### 4.2 模式 2：Refine 模式 (Refine Mode)

**配置**：`refine: True`

#### 4.2.1 流程

代码位置 (`bin/predict.py:75-79`)：

```python
if predict_config.get('refine', False):
    assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
    # 图像 unpadding is taken care of in the refiner
    cur_res = refine_predict(batch, model, **predict_config.refiner)
    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
```

**关键特点**（详见第 3 节）：
- 多尺度金字塔（3个尺度）
- 迭代优化（15次 Adam）
- 从低到高渐进式细化
- 特征空间优化（不优化网络权重）

#### 4.2.2 Refine 模式配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gpu_ids` | `0,1` | 使用的 GPU ID，单 GPU 用 `"0,"` |
| `modulo` | `8` | 填充到的模数（与网络下采样次数对应） |
| `n_iters` | `15` | 每个尺度的迭代优化次数 |
| `lr` | `0.002` | Adam 优化器学习率 |
| `min_side` | `512` | 金字塔最小边长 |
| `max_scales` | `3` | 最大尺度数 |
| `px_budget` | `1800000` | 像素预算上限 |

#### 4.2.3 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 更好的全局一致性 | ❌ 速度慢（3个尺度 × 15次迭代） |
| ✅ 更高的细节质量 | ❌ 内存占用高 |
| ✅ 多尺度融合 | ❌ 实现复杂 |
| ✅ 对大图像效果好 | ❌ 需要更多计算资源 |

#### 4.2.4 适用场景

- 高质量输出需求
- 大图像（≥ 512×512）
- 复杂结构的 inpainting
- 最终结果生成

---

### 4.3 两种模式详细对比表

| 对比项 | 标准模式 (refine=False) | Refine 模式 (refine=True) |
|--------|------------------------|---------------------------|
| **前向传播次数** | 1 次 | 3 个尺度 × (1 + 15) 次 ≈ 48 次 |
| **梯度计算** | 无 (`torch.no_grad()`) | 有（优化特征 z1, z2） |
| **优化对象** | 无 | 中间特征 z1, z2 |
| **多尺度** | 无 | 有（最多 3 个尺度） |
| **图像金字塔** | 无 | 有 |
| **内存占用** | 低 | 高（需存储多个尺度） |
| **速度** | 快 | 慢（约慢 10-50 倍） |
| **小图像质量** | 好 | 好 |
| **大图像质量** | 一般 | **更好** |
| **全局一致性** | 一般 | **更好** |
| **细节质量** | 一般 | **更好** |
| **像素预算** | 无硬性限制 | **有** (≤ 1,800,000 像素) |

---

### 4.4 如何选择推理模式？

决策树：

```
开始
  │
  ├─ 图像尺寸 ≤ 512×512?
  │   ├─ 是 → 标准模式 (快速且质量足够)
  │   └─ 否 → 继续判断
  │
  ├─ 追求速度还是质量?
  │   ├─ 速度 → 标准模式
  │   └─ 质量 → 继续判断
  │
  ├─ 图像内容复杂度?
  │   ├─ 简单纹理 → 标准模式
  │   └─ 复杂结构/语义 → Refine 模式
  │
  └─ 计算资源?
      ├─ 有限 (CPU/小内存) → 标准模式
      └─ 充足 (GPU/大内存) → Refine 模式
```

---

## 总结

### 训练分辨率
- **256×256** - 所有主要配置都使用此分辨率

### 推理最优分辨率
- **512×512** - 快速测试
- **1024×1024** - 通用用途（推荐）
- **1340×1340** - 最佳质量（接近 180 万像素预算）

### Refine 模式
- **原理**：多尺度金字塔 + 迭代特征优化
- **意义**：提升大图像一致性和细节质量
- **代价**：速度慢 10-50 倍，内存占用高

### 推理模式
- **标准模式**：快速、简单，适合小图像/预览
- **Refine 模式**：高质量，适合大图像/最终输出

---

*本分析基于 LaMa 源码 commit 分析生成，所有结论均附有具体代码位置。*
