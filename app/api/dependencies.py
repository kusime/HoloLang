# -*- coding: utf-8 -*-
"""
FastAPI 依赖注入

管理客户端实例的生命周期
"""

from contextlib import asynccontextmanager
from typing import Optional
import logging
import sys

from fastapi import FastAPI

from app import config
from app.clients.s3 import S3Client
from app.clients.tts import TTSClient
from app.clients.whisperx import WhisperXSingleClient


def _init_logging():
    """初始化基础日志配置（轻量级）"""
    # 只为 app.* 的 logger 设置，不影响其他库
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True  # 覆盖已有配置
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理"""
    # 初始化日志
    _init_logging()
    
    # 启动时初始化客户端
    app.state.tts = _make_tts_client()
    app.state.align = _make_aligner()
    app.state.s3 = _make_s3_client()
    app.state.key_prefix = config.S3_KEY_PREFIX
    
    try:
        yield
    finally:
        # 关闭时清理资源
        tts = getattr(app.state, "tts", None)
        try:
            if tts is not None and hasattr(tts, "_session"):
                tts._session.close()
        except Exception:
            pass


def _make_tts_client() -> TTSClient:
    """创建 TTS 客户端"""
    return TTSClient(base_url=config.TTS_BASE_URL, timeout=config.TTS_TIMEOUT)


def _make_aligner() -> WhisperXSingleClient:
    """创建 WhisperX 对齐客户端"""
    device: Optional[str] = None if config.WHISPERX_DEVICE == "auto" else config.WHISPERX_DEVICE
    return WhisperXSingleClient(device=device)


def _make_s3_client() -> S3Client:
    """创建 S3 客户端"""
    return S3Client()
