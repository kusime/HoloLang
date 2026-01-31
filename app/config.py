# -*- coding utf-8 -*-
"""
统一配置管理

从环境变量读取所有配置项，提供默认值和类型转换。
"""

import os


def _env(name: str, default: str) -> str:
    """读取字符串环境变量"""
    return os.getenv(name, default)


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量"""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量"""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ========== API 配置 ==========
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = _int_env("API_PORT", 8000)

# ========== TTS 配置 ==========
TTS_BASE_URL = _env("TTS_BASE_URL", "http://localhost:9880")
TTS_TIMEOUT = _int_env("TTS_TIMEOUT", 120)

# ========== WhisperX 配置 ==========
WHISPERX_DEVICE = _env("WHISPERX_DEVICE", "auto")  # auto/cpu/cuda

# ========== S3 配置 ==========
S3_ENDPOINT = _env("S3_ENDPOINT", "localhost:9000")
S3_ACCESS_KEY = _env("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY = _env("S3_SECRET_KEY", "change_this_strong_password")
S3_BUCKET = _env("S3_BUCKET", "tts-pipeline")
S3_SECURE = _bool_env("S3_SECURE", False)
S3_PRESIGN_TTL = _int_env("S3_PRESIGN_TTL", 3600)
S3_KEY_PREFIX = _env("S3_KEY_PREFIX", "tts")
