# -*- coding: utf-8 -*-
"""
S3 存储客户端（基于 minio-py）
"""

import io
import os
from datetime import timedelta
from enum import Enum
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

from minio import Minio
from pydantic import BaseModel, ConfigDict, Field

from app.utils.logging_decorator import log_function


# ---------- Content-Type 枚举 ----------
class S3ContentType(Enum):
    """S3 对象内容类型"""

    JSON = "application/json; charset=utf-8"
    OCTET = "application/octet-stream"
    TEXT = "text/plain; charset=utf-8"
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    FLAC = "audio/flac"
    OGG = "audio/ogg"

    @classmethod
    def from_audio_ext(cls, ext: str) -> "S3ContentType":
        """根据音频文件扩展名返回对应的 ContentType"""
        e = ext.lower().lstrip(".")
        return {
            "wav": cls.WAV,
            "mp3": cls.MP3,
            "flac": cls.FLAC,
            "ogg": cls.OGG,
        }.get(e, cls.OCTET)


# ---------- Pydantic I/O ----------
class UploadRequest(BaseModel):
    """上传请求"""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="对象键（如 tts/<job_id>/final.wav）")
    data: bytes = Field(..., description="要上传的字节数据")
    content_type: Union[S3ContentType, str] = Field(..., description="MIME 类型")
    cache_control: Optional[str] = Field(None, description="可选：缓存策略")


class UploadResult(BaseModel):
    """上传结果"""

    model_config = ConfigDict(extra="forbid")

    key: str
    etag: str


class PresignRequest(BaseModel):
    """预签名 URL 请求"""

    model_config = ConfigDict(extra="forbid")

    key: str
    ttl_sec: Optional[int] = Field(
        None, ge=1, description="URL 有效期秒数；None=用环境变量默认"
    )


class PresignResult(BaseModel):
    """预签名 URL 结果"""

    model_config = ConfigDict(extra="forbid")

    key: str
    url: str
    ttl_sec: int


# ---------- 内部工具 ----------
def _parse_endpoint(ep_env: str | None, secure_env: Optional[str]) -> Tuple[str, bool]:
    """
    解析 endpoint 环境变量
    
    Args:
        ep_env: endpoint 字符串（可能包含协议）
        secure_env: secure 标志字符串
        
    Returns:
        (host[:port], secure)
    """
    ep = (ep_env or "localhost:9000").strip()
    secure_flag = (secure_env or "false").strip().lower() in ("1", "true", "yes", "on")
    if "://" in ep:
        p = urlparse(ep)
        host = p.netloc or p.path
        return host, (p.scheme == "https")
    return ep, secure_flag


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量"""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


# ---------- S3 Client ----------
class S3Client:
    """
    S3 存储客户端
    
    环境变量配置：
      S3_ENDPOINT=localhost:9000 或 https://s3.example.com
      S3_ACCESS_KEY=admin
      S3_SECRET_KEY=change_this_strong_password
      S3_BUCKET=tts-pipeline
      S3_SECURE=false
      S3_PRESIGN_TTL=3600
    """

    def __init__(self) -> None:
        endpoint_raw = os.getenv("S3_ENDPOINT", "localhost:9000")
        secure_raw = os.getenv("S3_SECURE", "false")
        self._endpoint, self._secure = _parse_endpoint(endpoint_raw, secure_raw)

        self._access_key = os.getenv("S3_ACCESS_KEY", "admin")
        self._secret_key = os.getenv("S3_SECRET_KEY", "change_this_strong_password")
        self._bucket = os.getenv("S3_BUCKET", "tts-pipeline")
        self._presign_ttl = _int_env("S3_PRESIGN_TTL", 3600)

        self._cli = Minio(
            self._endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )

        # 若桶不存在则创建（幂等）
        if not self._cli.bucket_exists(self._bucket):
            self._cli.make_bucket(self._bucket)

    @log_function()
    def upload(self, req: UploadRequest) -> UploadResult:
        """
        上传文件到 S3
        
        Args:
            req: 上传请求
            
        Returns:
            上传结果（包含 etag）
        """
        ctype = (
            req.content_type.value
            if isinstance(req.content_type, S3ContentType)
            else str(req.content_type)
        )

        metadata = {"Cache-Control": req.cache_control} if req.cache_control else None

        r = self._cli.put_object(
            self._bucket,
            req.key,
            io.BytesIO(req.data),
            length=len(req.data),
            content_type=ctype,
            metadata=metadata,
        )
        return UploadResult(key=req.key, etag=r.etag)

    def presign(self, req: PresignRequest) -> PresignResult:
        """
        生成预签名 GET URL
        
        Args:
            req: 预签名请求
            
        Returns:
            预签名结果（包含 URL）
        """
        ttl = int(req.ttl_sec or self._presign_ttl)
        url = self._cli.presigned_get_object(
            self._bucket, req.key, expires=timedelta(seconds=ttl)
        )
        return PresignResult(key=req.key, url=url, ttl_sec=ttl)
