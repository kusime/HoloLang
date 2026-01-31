# s3_client.py
# -*- coding: utf-8 -*-
"""
极简 S3 Client（minio-py）
- 只做两件事：上传 bytes、生成下载预签名 URL
- 配置全部读取环境变量
- 用 Pydantic v2 封装调用参数/返回值
- 提供内容类型枚举 S3ContentType
- __main__：包含“流水线产物（bytes）→ 上传→ 预签名”的最小测试

依赖：
  pip install minio pydantic
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from minio import Minio


# ---------- Content-Type 枚举 ----------
class S3ContentType(Enum):
    JSON = "application/json; charset=utf-8"
    OCTET = "application/octet-stream"
    TEXT = "text/plain; charset=utf-8"
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    FLAC = "audio/flac"
    OGG = "audio/ogg"

    @classmethod
    def from_audio_ext(cls, ext: str) -> "S3ContentType":
        e = ext.lower().lstrip(".")
        return {
            "wav": cls.WAV,
            "mp3": cls.MP3,
            "flac": cls.FLAC,
            "ogg": cls.OGG,
        }.get(e, cls.OCTET)


# ---------- Pydantic I/O ----------
class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="对象键（如 tts/<job_id>/final.wav）")
    data: bytes = Field(..., description="要上传的字节数据")
    content_type: Union[S3ContentType, str] = Field(..., description="MIME 类型")
    cache_control: Optional[str] = Field(None, description="可选：缓存策略")


class UploadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    etag: str


class PresignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    ttl_sec: Optional[int] = Field(
        None, ge=1, description="URL 有效期秒数；None=用环境变量默认"
    )


class PresignResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    url: str
    ttl_sec: int


# ---------- 内部工具 ----------
def _parse_endpoint(ep_env: str | None, secure_env: Optional[str]) -> Tuple[str, bool]:
    """
    把环境变量里的 endpoint 解析成 (host[:port], secure)
    - 允许 'localhost:9000' 或 'https://s3.example.com'
    - 若带协议则以协议为准；否则用 S3_SECURE 判断（默认 False）
    """
    ep = (ep_env or "localhost:9000").strip()
    secure_flag = (secure_env or "false").strip().lower() in ("1", "true", "yes", "on")
    if "://" in ep:
        p = urlparse(ep)
        host = p.netloc or p.path
        return host, (p.scheme == "https")
    return ep, secure_flag


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


# ---------- S3 Client（职责单一） ----------
class S3Client:
    """
    只负责对象上传与下载签名；不混入其他业务。
    读取环境变量：
      S3_ENDPOINT=localhost:9000 或 https://s3.example.com
      S3_ACCESS_KEY=...
      S3_SECRET_KEY=...
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

    def upload(self, req: UploadRequest) -> UploadResult:
        """
        上传任意 bytes 到 S3。
        返回对象的 ETag。
        """
        ctype = (
            req.content_type.value
            if isinstance(req.content_type, S3ContentType)
            else str(req.content_type)
        )

        # minio-py 的 put_object 没有单独 cache_control 参数；如果你需要严格的响应头，
        # 建议在边缘层（Nginx/CDN）改写；这里作为 user metadata 透传。
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
        生成 GET 预签名 URL。
        """
        ttl = int(req.ttl_sec or self._presign_ttl)
        url = self._cli.presigned_get_object(
            self._bucket, req.key, expires=timedelta(seconds=ttl)
        )
        return PresignResult(key=req.key, url=url, ttl_sec=ttl)


# ========== 最小“单元测试式”示例（含流水线集成） ==========
if __name__ == "__main__":
    """
    优先路径：直接跑我们的多语种流水线，拿到 bytes 后上传并生成预签名 URL。
      - 需要项目内存在：
        multiple_pipeline_api.PipeLineOutput, multiple_pipeline_api.multiple_pipeline_api
        segment_api.TextIn
        tts_client.TTSClient, tts_client.TTSConfig
        whisperx_single_client.WhisperXSingleClient
      - TTS 服务运行中（默认 http://localhost:9880）
    回退路径：
      - 若导入失败或流水线异常：尝试上传本地 final.wav / chars_merged.json
      - 再不行：上传 hello.txt
    """
    import json
    from uuid import uuid4

    s3 = S3Client()
    job_id = (
        os.getenv("JOB_ID")
        or datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + str(uuid4())[:8]
    )

    def _upload_and_sign(key: str, data: bytes, ctype: S3ContentType) -> dict:
        up = s3.upload(UploadRequest(key=key, data=data, content_type=ctype))
        url = s3.presign(PresignRequest(key=key)).url
        return {"key": up.key, "etag": up.etag, "url": url}

    # 尝试：直接跑流水线 → 上传
    tried_pipeline = False
    try:
        from multiple_pipeline_api import (  # type: ignore
            PipeLineOutput,
            multiple_pipeline_api,
        )
        from segment_api import TextIn  # type: ignore
        from tts_client import TTSClient, TTSConfig  # type: ignore
        from whisperx_single_client import WhisperXSingleClient  # type: ignore

        tried_pipeline = True

        # 输入文本：优先读取 ./text.txt，否则用内置示例
        TEXT_PATH = "text.txt"
        if os.path.exists(TEXT_PATH):
            input_text = open(TEXT_PATH, "r", encoding="utf-8").read()
        else:
            input_text = "今天我们测试一下,WhisperX alignment。Next, we switch to English.最後に日本語でも話してみます。"

        # TTS 配置（与你之前验证一致；确保 media_type='wav'）
        cfg = TTSConfig(
            ref_audio_path="z.refs/main.wav",
            aux_ref_audio_paths=[],
            prompt_text="这是最后一件了吧？嗯，这里确实有七十件。",
            prompt_lang="zh",
            text_split_method="cut5",
            batch_size=60,
            batch_threshold=0.75,
            split_bucket=True,
            parallel_infer=True,
            fragment_interval=0.27,
            speed_factor=1.05,
            top_k=6,
            top_p=1.0,
            temperature=0.65,
            repetition_penalty=1.25,
            sample_steps=32,
            super_sampling=False,
            media_type="wav",
            streaming_mode=False,
            seed=-1,
        )

        # 子模块客户端
        tts_base = os.getenv("TTS_BASE_URL", "http://localhost:9880")
        tts_timeout = _int_env("TTS_TIMEOUT", 120)
        tts = TTSClient(base_url=tts_base, timeout=tts_timeout)
        align = WhisperXSingleClient(device=None)

        # 跑流水线（返回 bytes）
        out = multiple_pipeline_api(
            TextIn(text=input_text),
            tts_client=tts,
            tts_cfg=cfg,
            align_client=align,
        )

        # 上传到 S3：tts/<job_id>/
        audio_info = _upload_and_sign(
            f"tts/{job_id}/final.wav", out.audio, S3ContentType.WAV
        )
        json_info = _upload_and_sign(
            f"tts/{job_id}/chars_merged.json", out.chars_time, S3ContentType.JSON
        )

        manifest = {
            "job_id": job_id,
            "audio": audio_info,
            "chars": json_info,
        }
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

    except Exception as e:
        # 仅打印提示，不影响下面的回退逻辑
        if tried_pipeline:
            print(f"[pipeline] 跑流水线失败，回退本地文件/最小示例。原因: {e}")

        # 回退1：若有本地产物就上传
        if os.path.exists("final.wav") and os.path.exists("chars_merged.json"):
            audio_bytes = open("final.wav", "rb").read()
            chars_bytes = open("chars_merged.json", "rb").read()
            audio_info = _upload_and_sign(
                f"tts/{job_id}/final.wav", audio_bytes, S3ContentType.WAV
            )
            json_info = _upload_and_sign(
                f"tts/{job_id}/chars_merged.json", chars_bytes, S3ContentType.JSON
            )
            print(
                json.dumps(
                    {"job_id": job_id, "audio": audio_info, "chars": json_info},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            # 回退2：最小 hello.txt
            hello = UploadRequest(
                key=f"test/{job_id}/hello.txt",
                data="hello minio\n".encode("utf-8"),
                content_type=S3ContentType.TEXT,
            )
            r = s3.upload(hello)
            u = s3.presign(PresignRequest(key=hello.key))
            print(
                json.dumps(
                    {"hello": {"key": r.key, "etag": r.etag, "url": u.url}},
                    ensure_ascii=False,
                    indent=2,
                )
            )
