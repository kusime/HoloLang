# -*- coding: utf-8 -*-
"""
客户端模块
"""

from app.clients.s3 import S3Client, S3ContentType
from app.clients.tts import TTSClient

# WhisperX 需要 torch，使用延迟导入以避免依赖问题
# 使用时: from app.clients.whisperx import WhisperXSingleClient

__all__ = [
    "S3Client",
    "S3ContentType",
    "TTSClient",
    # "WhisperXSingleClient",  # 需要手动导入
]
