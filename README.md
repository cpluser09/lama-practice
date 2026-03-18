# LaMa Image Inpainting Service

基于 LaMa (Resolution-robust Large Mask Inpainting with Fourier Convolutions) 的图像修复 Web 服务。

## 部署方式

### 使用 Docker Compose (推荐)

```bash
# 构建并启动服务
docker-compose up --build

# 服务将在 http://localhost:5000 启动
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

## 测试

运行测试脚本：

```bash
python test_client.py
```

## 项目结构

```
.
├── lama/                    # LaMa 原始项目
├── lama_service.py          # Flask Web 服务
├── Dockerfile               # Docker 镜像配置
├── docker-compose.yml       # Docker Compose 配置
├── requirements-service.txt # Python 依赖
├── test_client.py           # 测试客户端
└── README.md                # 本文档
```

## 注意事项

- 服务运行在 CPU 模式下，适合 macOS
- 首次启动需要下载预训练模型 (约 200MB)
- 建议图像分辨率不超过 2000x2000

## 参考

- [LaMa GitHub](https://github.com/advimman/lama)
- [LaMa 论文](https://arxiv.org/abs/2109.07161)
