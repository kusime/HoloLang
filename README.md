# HoloLang - 多语言 TTS 管道

支持中文、英文、日文等多种语言的 TTS 合成与字符级时间戳对齐。

## ✨ 特性

- 🌍 **多语言支持** - 自动检测并处理中文、英文、日文混合文本
- 🎵 **TTS 合成** - 支持 gptsovits、vits、bark 等多种 TTS 引擎
- ⏰ **字符级时间戳** - 使用 WhisperX 生成精确的字符级时间对齐
- ☁️ **云存储集成** - 自动上传到 S3/MinIO 并生成预签名 URL
- 🚀 **高性能** - 异步处理、批量合成、并行推理
- 📊 **装饰器日志** - 非侵入式流程追踪日志系统

---

## 📖 目录

- [快速开始](#-快速开始)
- [API 使用](#-api-使用)
- [架构设计](#-架构设计)
- [WhisperX 配置](#-whisperx-配置)
- [开发指南](#-开发指南)

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda (推荐) 或 pip
- CUDA（可选，用于加速 WhisperX）

### 安装依赖

```bash
# 激活环境
conda activate hololang

# 或从头创建
conda create -n hololang python=3.10 -y
conda activate hololang
pip install -r requirements.txt
```

### 配置环境变量

复制并编辑 `.env` 文件：

```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=8000

# TTS 服务配置
TTS_BASE_URL=http://localhost:9880
TTS_TIMEOUT=120

# WhisperX 配置
WHISPERX_DEVICE=auto  # auto/cpu/cuda

# S3/MinIO 配置
S3_ENDPOINT=localhost:9000
S3_ACCESS_KEY=admin
S3_SECRET_KEY=change_this_strong_password
S3_BUCKET=tts-pipeline
S3_SECURE=false
S3_PRESIGN_TTL=3600
S3_KEY_PREFIX=tts
```

### 启动服务

```bash
python -m app.api.routes
# 或
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

访问 API 文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🎯 API 使用

### 请求示例

```bash
curl -X POST http://localhost:8000/v2/tts/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "text": "黄昏の駅で、私は歩幅を小さく整えた。Next, we switch to English.",
    "ref_audio_path": "z.refs/main.wav",
    "prompt_text": "这是参考文本",
    "prompt_lang": "zh"
  }'
```

**注意**：系统会自动检测语言，无需手动指定 `text_lang`。

### 响应示例

```json
{
  "job_id": "20260131-xxx",
  "created_at": "2026-01-31T07:45:00Z",
  "contain_lang": ["ja", "en"],
  "duration": 5.234,
  "urls": {
    "audio_presigned_url": "http://...",
    "chars_presigned_url": "http://...",
    "presign_ttl_sec": 3600
  }
}
```

### 日志输出示例

装饰器日志系统会自动记录流程：

```
2026-01-31 18:05:56 | INFO | app.services.segmentation | segment_text 开始
2026-01-31 18:05:56 | INFO | app.services.segmentation | segment_text 完成 (0.01s)
2026-01-31 18:05:56 | INFO | app.clients.tts | get_tts_wav 开始
2026-01-31 18:05:56 | INFO | app.clients.tts | get_tts_wav 完成 (0.25s)
2026-01-31 18:05:58 | INFO | app.clients.whisperx | align 完成 (2.00s)
2026-01-31 18:05:58 | INFO | app.clients.s3 | upload 完成 (0.00s)
```

---

## 🏗️ 架构设计

### 分层架构

```
┌─────────────────────────────────────┐
│       API 层 (app/api/)            │  ← FastAPI 路由和依赖注入
├─────────────────────────────────────┤
│     服务层 (app/services/)         │  ← 业务逻辑编排
├─────────────────────────────────────┤
│     客户端层 (app/clients/)        │  ← 外部服务封装
├─────────────────────────────────────┤
│     工具层 (app/utils/)            │  ← 通用辅助函数
├─────────────────────────────────────┤
│     模型层 (app/models/)           │  ← 数据模型定义
└─────────────────────────────────────┘
```

### 数据流

```
文本输入 → 语言检测 → 文本分段 → TTS 合成 → WhisperX 对齐 → 音频合并 → S3 上传 → 返回 Manifest
```

### 项目结构

```
app/
├── config.py                # 统一配置管理
├── models/                  # 数据模型（Pydantic）
│   ├── segment.py          # 分段相关模型
│   ├── tts.py              # TTS 相关模型
│   ├── whisperx.py         # WhisperX 相关模型
│   └── pipeline.py         # 管道输入/输出模型
├── clients/                # 外部服务客户端
│   ├── s3.py               # S3 客户端
│   ├── tts.py              # TTS 客户端（装饰器日志）
│   └── whisperx.py         # WhisperX 客户端（装饰器日志）
├── services/               # 业务逻辑服务
│   ├── segmentation.py     # 文本分段服务（装饰器日志）
│   └── pipeline.py         # 管道编排服务（装饰器日志）
├── api/                    # FastAPI 端点
│   ├── dependencies.py     # 依赖注入 + 日志初始化
│   └── routes.py           # API 路由
└── utils/                  # 工具函数
    ├── audio.py            # WAV 处理工具
    └── logging_decorator.py # 装饰器日志模块
```

### 核心特性

#### 1. 自动语言检测
使用 Lingua 库检测中/英/日三种语言，无需手动指定。

#### 2. 装饰器日志系统
非侵入式日志记录，业务代码 0 改动：
```python
@log_function()  # ← 只需一行
def segment_text(payload):
    # 业务逻辑完全干净
    ...
```

#### 3. WAV 参数一致性检查
确保合并音频的采样率、位宽、声道数一致。

#### 4. 依赖注入
使用 FastAPI lifespan 管理客户端生命周期，避免全局状态。

---

## 🔧 WhisperX 配置

### 常见问题

运行时可能遇到模型下载错误：
```
"The chosen align_model could not be found in huggingface"
```

### 解决方案

#### 方案 1：预下载模型（推荐）

```python
import whisperx

languages = ["zh", "en", "ja"]
for lang in languages:
    print(f"下载 {lang} 模型...")
    model, metadata = whisperx.load_align_model(language_code=lang, device="cpu")
    print(f"✅ {lang} 模型下载成功")
```

#### 方案 2：配置网络代理

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像站
```

#### 方案 3：离线部署（生产环境）

预先下载所有模型到 Docker 镜像：
```dockerfile
RUN python -c "import whisperx; \
    whisperx.load_align_model('zh', 'cpu'); \
    whisperx.load_align_model('en', 'cpu'); \
    whisperx.load_align_model('ja', 'cpu')"
```

### 模型缓存位置

- Linux/Mac: `~/.cache/whisperx/`
- Windows: `%USERPROFILE%\.cache\whisperx\`

### 支持的语言模型

- `zh` (中文): jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
- `en` (英文): WAV2VEC2_ASR_BASE_960H
- `ja` (日文): jonatasgrosman/wav2vec2-large-xlsr-53-japanese

---

## 🧪 开发指南

### 运行测试

```bash
# 单元测试
pytest tests/ -v

# 非侵入性验证
python test_non_invasive.py

# API 测试
python tests/test_api_local.py
```

### 代码格式化

```bash
black app/
isort app/
```

### 扩展性

#### 添加新的 TTS 引擎

1. 在 `app/clients/` 创建新客户端
2. 实现 `get_tts_wav(text, lang, config)` 接口
3. 在 `dependencies.py` 注册
4. 添加 `@log_function()` 装饰器

#### 添加新的语言

1. 更新 `app/models/segment.py` 的 `LangCode`
2. 在 `segmentation.py` 添加检测规则
3. 确保 WhisperX 支持该语言

---

## 🚢 部署

### Docker 部署

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 环境建议

- **开发**：本地 TTS + MinIO
- **测试**：Docker Compose
- **生产**：K8s + 外部 S3

---

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题，请通过 Issue 联系我们。

---

**版本**: 2.0.0  
**最后更新**: 2026-01-31
