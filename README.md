# LaMa Image Inpainting Service

基于 LaMa (Resolution-robust Large Mask Inpainting with Fourier Convolutions) 的图像修复 Web 服务。

## Quick Start

### 本地快速启动

```bash
# 克隆项目
git clone --recurse-submodules https://github.com/cpluser09/lama-practice.git
cd lama-practice

# 使用 Docker Compose 启动
docker-compose up --build

# 服务将在 http://localhost:5000 启动
```

### 服务器部署

#### 1. 首次部署到服务器

```bash
# SSH 连接到服务器
ssh user@your-server.com

# 克隆项目（包含子模块）
git clone --recurse-submodules https://github.com/cpluser09/lama-practice.git
cd lama-practice

# 启动服务（后台运行）
docker-compose up -d --build

# 查看日志
docker-compose logs -f
```

#### 2. 更新服务器代码

**快速更新（仅修改网页/服务代码）：**

```bash
# 拉取最新代码
git pull

# 重启容器加载新代码（无需重新构建，只需几秒）
docker-compose restart lama-service

# 或者使用一键脚本
./reload.sh
```

**完整更新（依赖/模型变化时）：**

```bash
# 拉取最新代码（包括子模块更新）
git pull
git submodule update --remote --merge

# 重新构建并启动服务（首次部署或依赖变化时使用）
docker-compose up -d --build
```

**开发模式（代码热加载）：**

```bash
# 启动开发模式，代码通过 volume 挂载，修改后自动生效
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 修改代码后只需重启容器即可（无需重建）
docker-compose restart lama-service
```

#### 一键更新脚本（可选）

创建 `update.sh` 脚本实现一键更新：

```bash
#!/bin/bash
# update.sh - 服务器更新脚本

cd /path/to/lama-practice
git pull
git submodule update --remote --merge
docker-compose down
docker-compose up -d --build
echo "Service updated successfully!"
```

使用方式：
```bash
chmod +x update.sh
./update.sh
```

## 部署方式

### 使用 Docker Compose (推荐)

```bash
# 构建并启动服务
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 停止服务
docker-compose down
```

### 使用 Docker

```bash
# 构建镜像
docker build -t lama-inpainting .

# 运行容器
docker run -p 5000:5000 lama-inpainting
```

## API 使用

### 健康检查

```bash
curl http://localhost:5000/health
```

### 图像修复

#### 使用 curl

```bash
# 不带掩码 (使用默认中心方块掩码)
curl -X POST http://localhost:5000/inpaint \
     -F "image=@input.jpg" \
     -o output.png

# 带掩码 (掩码中白色像素表示需要修复的区域)
curl -X POST http://localhost:5000/inpaint \
     -F "image=@input.jpg" \
     -F "mask=@mask.png" \
     -o output.png
```

#### 使用 Python

```python
import requests

# 不带掩码
with open('input.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/inpaint',
        files={'image': f}
    )

with open('output.png', 'wb') as f:
    f.write(response.content)

# 带掩码
with open('input.jpg', 'rb') as img_f, open('mask.png', 'rb') as mask_f:
    response = requests.post(
        'http://localhost:5000/inpaint',
        files={'image': img_f, 'mask': mask_f}
    )

with open('output.png', 'wb') as f:
    f.write(response.content)
```

## Web 界面功能

### 图像对比查看器
- **滑动对比**：拖动滑块或点击图像任意位置查看修复前后对比
- **一键切换**：通过按钮快速切换查看原图、结果或滑动对比模式
- **响应式设计**：支持桌面和移动端访问

### 测试图片
内置 6 种测试场景，覆盖常见修复需求：
- 文字去除 - 移除图片上的文字水印
- 物体移除 - 删除不需要的物体
- 划痕修复 - 修复照片划痕和斑点
- 人脸修复 - 修复人脸缺失部分
- 水印移除 - 去除预览水印
- 旧照片修复 - 修复破损的老照片

## 测试

运行测试脚本：

```bash
python test_client.py
```

## 项目结构

```
.
├── lama/                    # LaMa 原始项目 (git submodule)
├── lama_service.py          # Flask Web 服务
├── templates/               # Web 界面模板
│   └── index.html          # 主页面（含对比查看器）
├── generate_test_images.py  # 测试图片生成脚本
├── Dockerfile               # Docker 镜像配置
├── docker-compose.yml       # Docker Compose 配置
├── requirements-service.txt # Python 依赖
├── test_client.py           # 测试客户端
└── README.md                # 本文档
```

## 常见问题

### 为什么 `docker-compose up -d --build` 花费很长时间？

完整重建会重新下载所有依赖和模型（约 200MB），需要几分钟。如果只是修改了网页或服务代码，使用快速更新：

```bash
# 快速更新（几秒钟）
docker-compose restart lama-service
```

### 如何区分什么时候需要完整重建？

| 变更内容 | 更新方式 |
|---------|---------|
| 网页 HTML/CSS | `docker-compose restart` |
| 服务代码 lama_service.py | `docker-compose restart` |
| 添加新的 Python 依赖 | `docker-compose up -d --build` |
| 修改 Dockerfile | `docker-compose up -d --build` |
| 模型更新 | `docker-compose up -d --build --no-cache` |

## 注意事项

- 服务运行在 CPU 模式下，适合 macOS
- 首次启动需要下载预训练模型 (约 200MB)
- 建议图像分辨率不超过 2000x2000

## 参考

- [LaMa GitHub](https://github.com/advimman/lama)
- [LaMa 论文](https://arxiv.org/abs/2109.07161)
