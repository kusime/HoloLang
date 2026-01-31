# -*- coding: utf-8 -*-
"""
FastAPI 路由定义

提供完整的TTS管道API服务
"""

import json
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import lifespan
from app.clients.s3 import PresignRequest, S3Client, S3ContentType, UploadRequest
from app.clients.tts import TTSClient
from app.clients.whisperx import WhisperXSingleClient
from app.models.pipeline import ManifestOut, PipelineInput
from app.models.segment import TextIn
from app.services.pipeline import run_pipeline as run_pipeline_service
from app.services.segmentation import segment_text
from app.utils.audio import get_wav_duration

# 创建 FastAPI 应用
app = FastAPI(title="HoloLang Pipeline Service", version="2.0.0", lifespan=lifespan)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _gen_job_id() -> str:
    """生成任务 ID"""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = hex(int(time.time() * 1e9))[-6:]
    return f"{ts}-{rnd}"


@app.get("/health")
def health_check():
    """健康检查"""
    return {"ok": True, "service": "hololang-pipeline", "version": "2.0.0"}


@app.post("/v2/tts/pipeline", response_model=ManifestOut)
def run_pipeline(req: PipelineInput) -> ManifestOut:
    """
    执行完整的TTS管道流程
    
    流程：
    1. 文本分段（自动检测语言）
    2. 逐段TTS合成
    3. WhisperX字符级时间戳对齐
    4. 合并音频和时间戳
    5. 上传到S3并返回预签名URL
    
    注意：已移除 text_lang 字段，系统自动检测每个段落的语言
    """
    # 读取依赖
    tts: TTSClient = app.state.tts
    align: WhisperXSingleClient = app.state.align
    s3: S3Client = app.state.s3
    prefix: str = app.state.key_prefix

    # 生成/确定 job_id
    job_id = req.job_id or _gen_job_id()

    # 先切分拿覆盖语言（用于 manifest）
    seg_res = segment_text(TextIn(text=req.text))
    contain_lang = list(seg_res.contain_lang)

    # 转换为 TTSConfig
    tts_config = req.to_tts_config()

    # 跑流水线（返回 bytes）
    try:
        out = run_pipeline_service(
            text=req.text,
            tts_client=tts,
            tts_config=tts_config,
            align_client=align,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")

    # 计算时长（以合并 WAV 为准）
    try:
        duration = get_wav_duration(out.audio)
    except Exception:
        # 回退：若解析失败，从 chars_merged.json 估算
        try:
            rows = json.loads(out.chars_time.decode("utf-8", errors="ignore"))
            duration = float(max((r.get("end") or 0.0) for r in rows)) if rows else 0.0
        except Exception:
            duration = 0.0

    # 上传到 S3 并生成预签名 URL
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

    # 返回 manifest
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


# 本地运行入口
if __name__ == "__main__":
    import uvicorn

    from app import config

    uvicorn.run("app.api.routes:app", host=config.API_HOST, port=config.API_PORT, reload=True)
