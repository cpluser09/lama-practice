# LaMa 图像修复服务 / LaMa Image Inpainting Service

基于 LaMa (Resolution-robust Large Mask Inpainting with Fourier Convolutions) 的图像修复 Web 服务。

Image inpainting web service based on LaMa (Resolution-robust Large Mask Inpainting with Fourier Convolutions).

## 快速开始 / Quick Start

### macOS Apple Silicon (GPU 加速 / GPU Accelerated)

```bash
git clone --recurse-submodules https://github.com/cpluser09/lama-practice.git
cd lama-practice

# 创建虚拟环境并安装依赖 (依赖安装到项目目录 venv/ 下)
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements-service.txt

# 启动 GPU 服务
./launch_gpu_service_mac.sh
# Service: http://localhost:5002
```

### Docker (CPU)

```bash
git clone --recurse-submodules https://github.com/cpluser09/lama-practice.git
cd lama-practice
docker-compose up -d --build
# Service: http://localhost:5001
```

> **详细部署指南**: [DEPLOYMENT.md](DEPLOYMENT.md)

## 部署选项 / Deployment Options

| 选项 | 硬件 | 加速 | 端口 | 性能 (512x384) |
|------|------|------|------|-----------------|
| macOS MPS | Apple Silicon (M1/M2/M3/M4) | Metal GPU | 5002 | ~2s (3.5x) |
| Docker CPU | 任意 / Any | CPU only | 5001 | ~7s |

**选择建议**:
- **Apple Silicon Mac** → 使用 MPS GPU 获得 3-4x 加速
- **Intel Mac / 其他** → 使用 Docker CPU

## API 使用 / API Usage

### 健康检查 / Health Check

```bash
# MPS GPU
curl http://localhost:5002/health

# Docker CPU
curl http://localhost:5001/health
```

### 图像修复 / Inpaint Image

```bash
# 不带掩码 (自动生成中心方块掩码)
curl -X POST http://localhost:5002/inpaint \
  -F "image=@input.jpg" \
  -o output.png

# 带掩码 (白色像素表示需要修复的区域)
curl -X POST http://localhost:5002/inpaint \
  -F "image=@input.jpg" \
  -F "mask=@mask.png" \
  -o output.png
```

### 响应头 / Response Headers

```
X-Processing-Time: 2.15      # 总耗时 (秒) / Total time (seconds)
X-Inference-Time: 1.93       # 推理耗时 (秒) / Inference time (seconds)
X-Input-Resolution: 512x384   # 输入分辨率 / Input resolution
X-Output-Resolution: 512x384  # 输出分辨率 / Output resolution
```

## 命令行接口 / CLI

### PyTorch 模式 / PyTorch Mode

直接使用 Python 命令行进行图像修复，支持 GPU 加速：

```bash
# 基本用法
python inpaint_cli.py --image input.jpg --mask mask.png --output result.jpg

# 简写形式
python inpaint_cli.py -i photo.jpg -m mask.png -o result.jpg

# 指定模型路径
python inpaint_cli.py -i photo.jpg -m mask.png -o result.jpg --model /path/to/big-lama
```

**参数说明**:
| 参数 | 简写 | 说明 | 必需 |
|------|------|------|------|
| `--image` | `-i` | 输入图片路径 | ✓ |
| `--mask` | `-m` | 掩码图片路径 (白色=修复区域) | ✓ |
| `--output` | `-o` | 输出图片路径 | ✓ |
| `--model` | - | 模型目录 (默认: `big-lama`) | - |

**设备自动检测**:
- Apple Silicon (M1/M2/M3/M4) → MPS GPU
- NVIDIA GPU → CUDA
- 其他 → CPU

## Web 界面功能 / Web Interface Features

### 图像对比查看器 / Image Comparison Viewer
- **滑动对比**: 拖动滑块或点击图像任意位置查看修复前后对比
- **一键切换**: 快速切换查看原图、结果或滑动对比模式
- **响应式设计**: 支持桌面和移动端访问

### 绘图编辑器 / Drawing Editor
- **矩形工具**: 拖拽绘制矩形修复区域
- **画笔工具**: 手动涂抹需要修复的区域
- **实时预览**: 绘制时实时显示修复区域
- **预设测试**: 内置 6 种测试场景

### 测试场景 / Test Scenarios
1. 文字去除 - 移除图片上的文字水印
2. 物体移除 - 删除不需要的物体
3. 划痕修复 - 修复照片划痕和斑点
4. 人脸修复 - 修复人脸缺失部分
5. 水印移除 - 去除预览水印
6. 旧照片修复 - 修复破损的老照片

## 项目结构 / Project Structure

```
.
├── lama/                    # LaMa 原始项目 (git submodule)
├── inpaint_cli.py           # PyTorch 命令行接口
├── launch_gpu_service_mac.sh              # MPS GPU 服务 (Apple Silicon)
├── launch-cpu-service_docker.py              # Docker CPU 服务
├── templates/               # Web 界面模板
│   └── index.html          # 主页面
├── static/test/             # 测试图片
├── Dockerfile               # Docker 镜像配置
├── docker-compose.yml       # Docker Compose 配置
├── docker-compose.dev.yml   # 开发模式 (热加载)
├── requirements-service.txt # Python 依赖
├── reload-docker.sh         # Docker 快速重载
├── DEPLOYMENT.md            # 部署指南
└── README.md                # 本文档
```

## 更新代码 / Update Code

### 快速更新 (仅代码变更 / Code Only)

```bash
git pull
./reload-docker.sh
```

### 完整重建 (依赖/模型变化 / Dependencies Changed)

```bash
git pull
git submodule update --remote --merge
docker-compose down
docker-compose up -d --build
```

## 模型文件 / Model Files

### 预训练模型下载

模型大小：~363MB (big-lama.zip)

**Docker CPU 模式** - 自动下载：
- 首次构建 Docker 镜像时自动从 HuggingFace 下载
- 模型路径：`/app/big-lama/` (容器内)
- 缓存在 Docker 层中，重建时无需重新下载

**MPS GPU 模式** - 手动下载：
```bash
# 首次运行前需要下载模型
cd /path/to/lama-practice
curl -L -o big-lama.zip "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
unzip big-lama.zip
# 模型保存到 ./big-lama/ 目录
```

### 模型路径配置

| 模式 | 代码中的路径 | 实际位置 |
|------|------------|----------|
| MPS GPU | `./big-lama/` | 项目根目录的 `big-lama/` |
| Docker CPU | `/app/big-lama/` | 容器内的 `/app/big-lama/` |

## 常见问题 / FAQ

### 首次运行 MPS GPU 服务出错？

确保已下载模型文件：
```bash
ls -la big-lama/models/best.ckpt
# 应该存在此文件
```

### macOS 端口 5000 被占用？

macOS 的 AirPlay Receiver 占用端口 5000。
- **MPS GPU 服务** 使用端口 5002
- **Docker 服务** 使用端口 5001

### 为什么 docker-compose 重建很慢？

完整重建会重新下载所有依赖和模型 (~200MB)。如果只修改了代码，使用快速更新：
```bash
./reload-docker.sh  # 只需几秒
```

### MPS GPU 不工作？

检查 MPS 支持：
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# 应该返回: True
```

### GPU vs CPU 性能对比

| 分辨率 | CPU | MPS (GPU) | 加速比 |
|--------|-----|-----------|--------|
| 512x384 | ~7s | ~2s | 3.5x |
| 1024x768 | ~15s | ~4s | 3.8x |
| 1500x2000 | ~111s | ~25s | 4.4x |

## 参考 / References

- [LaMa GitHub](https://github.com/advimman/lama)
- [LaMa 论文 / Paper](https://arxiv.org/abs/2109.07161)
- [详细部署指南 / Deployment Guide](DEPLOYMENT.md)
