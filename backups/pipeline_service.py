# pipeline_service.py
# -*- coding: utf-8 -*-
"""
FastAPI · Pipeline Service (lifespan 版)
- 兼容原始 TTS 调用参数（text, text_lang, ref_audio_path, ...）
- 流程：segment_api → 逐段 TTS → WhisperX 单语种对齐 → 合并（bytes）
- 存储：S3Client（minio-py），上传 final.wav 与 chars_merged.json
- 返回：方案 A manifest（含预签名 URL）

运行：
  python pipeline_service.py
或：
  uvicorn pipeline_service:app --reload
"""

from __future__ import annotations

import io
import json
import os
import time
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from multiple_pipeline_api import PipeLineOutput, multiple_pipeline_api  # type: ignore
from s3_client import (  # type: ignore
    PresignRequest,
    S3Client,
    S3ContentType,
    UploadRequest,
)

# --- 你已有的模块（按需调整导入路径） ---
from segment_api import TextIn, segment_api  # type: ignore
from tts_client import TTSClient, TTSConfig  # type: ignore
from whisperx_single_client import WhisperXSingleClient  # type: ignore


# ========== 兼容原始 TTS 调用的输入模型 ==========
class PipelineIn(BaseModel):
    """
    与原始 TTS 请求参数保持兼容（同名同义）：
    text + text_lang + （其余全部为 TTSConfig 字段）
    """

    model_config = ConfigDict(extra="forbid")

    # 必填
    text: str = Field(..., description="要合成的文本")
    text_lang: str = Field(..., description="文本语言代码，如 zh/en/ja/yue/ko/auto 等")

    # 参考/提示
    ref_audio_path: Optional[str] = Field(
        None, description="参考音频路径（服务器可访问的路径）"
    )
    aux_ref_audio_paths: List[str] = Field(
        default_factory=list, description="辅助参考音频路径列表"
    )
    prompt_text: Optional[str] = Field(
        None, description="提示文本（用于控制音色/情绪等）"
    )
    prompt_lang: Optional[str] = Field(None, description="提示文本语言代码")

    # 切分 & 推理
    text_split_method: str = Field("cut5", description="文本切分方法")
    batch_size: int = Field(60, ge=1, description="批大小")
    batch_threshold: float = Field(0.75, ge=0.0, le=1.0, description="切分/并发阈值")
    split_bucket: bool = Field(True, description="是否按桶划分切分片段")
    parallel_infer: bool = Field(True, description="是否并行推理")
    fragment_interval: float = Field(
        0.27, ge=0.0, description="片段间插入的静音间隔（秒）"
    )

    # 采样/解码
    speed_factor: float = Field(1.05, gt=0.0, description="语速因子 > 0")
    top_k: int = Field(6, ge=0, description="采样 top_k")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="采样 top_p (0,1]")
    temperature: float = Field(0.65, gt=0.0, description="采样温度 > 0")
    repetition_penalty: float = Field(1.25, gt=0.0, description="重复惩罚 > 0")
    sample_steps: int = Field(32, ge=1, description="解码步数 >= 1")
    super_sampling: bool = Field(False, description="是否开启超采样")

    # 输出控制
    media_type: str = Field("wav", description="输出音频格式（pipeline 强制为 wav）")
    streaming_mode: bool = Field(False, description="是否以流式返回音频（二进制分块）")
    seed: int = Field(-1, description="随机种子；-1 表示随机")

    # 便捷：可外部传 job_id；否则自动生成
    job_id: Optional[str] = Field(
        None, description="可选自定义任务 ID；未提供时自动生成"
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def default_prompt_lang_and_media(self):
        # 若给了 prompt_text 但没给 prompt_lang，则回退到 text_lang
        if self.prompt_text and not self.prompt_lang:
            self.prompt_lang = self.text_lang
        # pipeline 只支持 wav 合并，统一强制为 wav
        self.media_type = "wav"
        return self


# ========== Manifest 输出 ==========
class ManifestOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    created_at: str
    contain_lang: List[str]
    duration: float
    keys: Dict[str, str]  # {"audio": "...", "chars": "..."}
    etag: Dict[str, str]  # {"audio": "...", "chars": "..."}
    urls: Dict[str, Any]  # {"audio_presigned_url": "...", "chars_presigned_url": "..."}
    version: str = "2.0.0"


# ========== 工具 & 依赖工厂 ==========
def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _make_tts_client() -> TTSClient:
    base_url = _env("TTS_BASE_URL", "http://localhost:9880")
    timeout = int(_env("TTS_TIMEOUT", "120"))
    return TTSClient(base_url=base_url, timeout=timeout)


def _make_aligner() -> WhisperXSingleClient:
    device = _env("WHISPERX_DEVICE", "auto")
    device_final = None if device == "auto" else device
    return WhisperXSingleClient(device=device_final)


def _make_s3_client() -> S3Client:
    # S3Client 自身读取环境变量：S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET, S3_SECURE, S3_PRESIGN_TTL
    return S3Client()


def _wav_duration_sec(wav_bytes: bytes) -> float:
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        nframes = wf.getnframes()
        fr = wf.getframerate() or 1
        return nframes / float(fr)


def _gen_job_id() -> str:
    # 时间戳 + 短随机，简洁可读
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = hex(int(time.time() * 1e9))[-6:]
    return f"{ts}-{rnd}"


# ========== 应用（lifespan 注入依赖） ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    app.state.tts = _make_tts_client()
    app.state.align = _make_aligner()
    app.state.s3 = _make_s3_client()
    app.state.key_prefix = _env("S3_KEY_PREFIX", "tts")
    try:
        yield
    finally:
        # shutdown：关闭 TTSClient 的会话（若存在）
        tts = getattr(app.state, "tts", None)
        try:
            if tts is not None and hasattr(tts, "_session"):
                tts._session.close()
        except Exception:
            pass


app = FastAPI(title="Pipeline Service", version="2.0.0", lifespan=lifespan)


# ========== 健康检查 ==========
@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "pipeline", "version": "2.0.0"}


# ========== 主接口 ==========
@app.post("/v2/tts/pipeline", response_model=ManifestOut)
def run_pipeline(req: PipelineIn) -> ManifestOut:
    """
    兼容原始 TTS 参数的升级版接口：
    - 自动分段、多语种逐段合成与对齐
    - 上传合并后的 final.wav 与 chars_merged.json
    - 返回包含预签名 URL 的 manifest
    """
    # 0) 读取依赖
    tts: TTSClient = app.state.tts
    align: WhisperXSingleClient = app.state.align
    s3: S3Client = app.state.s3
    prefix: str = app.state.key_prefix

    # 1) 生成/确定 job_id
    job_id = req.job_id or _gen_job_id()

    # 2) 先切分拿覆盖语言（用于 manifest）
    seg_res = segment_api(TextIn(text=req.text))
    contain_lang = list(seg_res.contain_lang)

    # 3) 将输入映射为 TTSConfig（与原始兼容）
    cfg = TTSConfig(
        ref_audio_path=req.ref_audio_path,
        aux_ref_audio_paths=req.aux_ref_audio_paths,
        prompt_text=req.prompt_text,
        prompt_lang=req.prompt_lang,
        text_split_method=req.text_split_method,
        batch_size=req.batch_size,
        batch_threshold=req.batch_threshold,
        split_bucket=req.split_bucket,
        parallel_infer=req.parallel_infer,
        fragment_interval=req.fragment_interval,
        speed_factor=req.speed_factor,
        top_k=req.top_k,
        top_p=req.top_p,
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty,
        sample_steps=req.sample_steps,
        super_sampling=req.super_sampling,
        media_type="wav",
        streaming_mode=False,
        seed=req.seed,
    )

    # 4) 跑流水线（返回 bytes）
    try:
        out: PipeLineOutput = multiple_pipeline_api(
            TextIn(text=req.text),
            tts_client=tts,
            tts_cfg=cfg,
            align_client=align,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")

    # 5) 计算时长（以合并 WAV 为准）
    try:
        duration = _wav_duration_sec(out.audio)
    except Exception:
        # 回退：若解析失败，从 chars_merged.json 估算
        try:
            rows = json.loads(out.chars_time.decode("utf-8", errors="ignore"))
            duration = float(max((r.get("end") or 0.0) for r in rows)) if rows else 0.0
        except Exception:
            duration = 0.0

    # 6) 上传到 S3 并生成预签名 URL
    audio_key = f"{prefix}/{job_id}/final.wav"
    json_key = f"{prefix}/{job_id}/chars_merged.json"

    try:
        r_audio = s3.upload(
            UploadRequest(key=audio_key, data=out.audio, content_type=S3ContentType.WAV)
        )
        r_json = s3.upload(
            UploadRequest(
                key=json_key, data=out.chars_time, content_type=S3ContentType.JSON
            )
        )
        u_audio = s3.presign(PresignRequest(key=audio_key))
        u_json = s3.presign(PresignRequest(key=json_key))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"s3 upload/presign failed: {e}")

    # 7) 返回 manifest
    return ManifestOut(
        job_id=job_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        contain_lang=contain_lang,
        duration=round(duration, 6),
        keys={"audio": audio_key, "chars": json_key},
        etag={"audio": r_audio.etag, "chars": r_json.etag},
        urls={
            "audio_presigned_url": u_audio.url,
            "chars_presigned_url": u_json.url,
            "presign_ttl_sec": u_audio.ttl_sec,
        },
        version="2.0.0",
    )


# ========== 本地运行 ==========
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("pipeline_service:app", host=host, port=port, reload=True)
