# WhisperX 模型配置说明

## 问题描述

在运行完整的 TTS 管道时，可能会遇到以下错误：

```
"The chosen align_model \"jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn\" could not be found in huggingface"
```

## 原因

WhisperX 需要从 HuggingFace 下载语言特定的对齐模型。常见原因：

1. **网络问题** - 无法访问 HuggingFace
2. **模型已移除或重命名** - HuggingFace 上的模型路径已更改
3. **首次运行** - 模型尚未缓存到本地

## 解决方案

### 方案 1：预下载模型（推荐）

使用以下 Python 脚本预下载模型：

```python
import whisperx

# 预下载模型
languages = ["zh", "en", "ja"]  # 根据需要调整
for lang in languages:
    print(f"下载 {lang} 模型...")
    try:
        model, metadata = whisperx.load_align_model(language_code=lang, device="cpu")
        print(f"✅ {lang} 模型下载成功")
    except Exception as e:
        print(f"❌ {lang} 模型下载失败: {e}")
```

### 方案 2：配置网络代理

如果网络受限，配置代理：

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像站
```

### 方案 3：手动下载并缓存

1. 访问 HuggingFace 模型页面
2. 下载模型文件到 `~/.cache/huggingface/hub/`
3. 重新运行服务

### 方案 4：仅测试核心功能

如果只想测试 API 和分段功能（不需要字符级时间戳）：

```bash
# 使用简化测试脚本
PYTHONPATH=/home/kusime/Desktop/HoloReader/HoloLang \
  conda run -n hololang python tests/test_local_simple.py
```

## 模型缓存位置

WhisperX 模型默认缓存在：
- Linux/Mac: `~/.cache/whisperx/`
- Windows: `%USERPROFILE%\.cache\whisperx\`

## 支持的语言和模型

WhisperX 支持的语言（截至 v3.7.2）：
- `zh` (中文): jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
- `en` (英文): WAV2VEC2_ASR_BASE_960H
- `ja` (日文): jonatasgrosman/wav2vec2-large-xlsr-53-japanese

如果默认模型失效，可以在 WhisperX 源码或文档中查找替代模型。

## 生产环境建议

1. **离线部署** - 预先下载所有需要的模型到 Docker 镜像中
2. **健康检查** - 在启动时验证模型是否可用
3. **降级策略** - 如果 WhisperX 不可用，提供简化的时间戳估算

## 相关资源

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [HuggingFace Models](https://huggingface.co/models)
- [WhisperX 文档](https://github.com/m-bain/whisperX#readme)
