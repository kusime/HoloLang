# -*- coding: utf-8 -*-
"""
管道服务

实现完整的TTS管道流程：
1. 文本分段（多语言检测）
2. 逐段TTS合成
3. WhisperX字符级时间戳对齐
4. 合并音频和时间戳到全局时间轴
"""

import json
from typing import Any, Dict, List, Tuple

from app.clients.tts import TTSClient
from app.clients.whisperx import WhisperXSingleClient
from app.models.pipeline import PipelineOutput
from app.models.segment import TextIn, TextSegmentsOut
from app.models.tts import TTSConfig
from app.services.segmentation import segment_text
from app.utils.audio import concat_wav_frames, read_wav_params_and_frames, resample_wav_bytes

from app.utils.logging_decorator import log_function


@log_function()
def run_pipeline(
    text: str,
    tts_client: TTSClient,
    tts_config: TTSConfig,
    align_client: WhisperXSingleClient,
) -> PipelineOutput:
    """
    执行完整的TTS管道
    
    Args:
        text: 待合成的文本
        tts_client: TTS客户端
        tts_config: TTS配置
        align_client: WhisperX对齐客户端
        
    Returns:
        管道输出（包含合并后的音频和字符时间戳）
        
    Raises:
        NotImplementedError: 如果media_type不是'wav'
        RuntimeError: 如果WAV参数不一致无法合并
    """
    if tts_config.media_type != "wav":
        raise NotImplementedError("目前仅支持 media_type='wav' 以便做 WAV 拼接。")

    # 1) 文本切段
    seg_out: TextSegmentsOut = segment_text(TextIn(text=text))

    # 2) 逐段合成 & 对齐（全在内存）
    frames_list: List[bytes] = []
    params_ref: Tuple[int, int, int] | None = None
    offset = 0.0  # 全局偏移（秒）

    # 用于chars_merged.json的扁平全局时间轴（带lang）
    flat_rows: List[Dict[str, Any]] = []

    for idx, seg in enumerate(seg_out.segments):
        # --- 2.1 TTS: 返回WAV字节 ---
        seg_wav_bytes: bytes = tts_client.get_tts_wav(seg.text, seg.langcode, tts_config)

        # --- 2.1.1 解析参数/帧/时长，并校验参数一致 ---
        params, frames, seg_duration = read_wav_params_and_frames(seg_wav_bytes)
        
        if params_ref is None:
            params_ref = params
        elif params != params_ref:
            # 参数不一致（通常是采样率不同），进行重采样
            target_nch, target_width, target_sr = params_ref
            
            # 使用 torchaudio 进行重采样
            seg_wav_bytes = resample_wav_bytes(
                seg_wav_bytes, 
                target_sr=target_sr,
                target_channels=target_nch,
                target_width=target_width
            )
            
            # 重新解析参数（确认一致）
            params, frames, seg_duration = read_wav_params_and_frames(seg_wav_bytes)
            
            if params != params_ref:
                # 如果重采样后仍然不一致（极少见），则抛出异常
                raise RuntimeError(
                    f"WAV 参数不一致，重采样失败：first={params_ref}, current={params}"
                )
        
        frames_list.append(frames)

        # --- 2.2 WhisperX 单语种对齐（段内时间） ---
        char_items = align_client.align(
            text=seg.text,
            audio=seg_wav_bytes,  # bytes
            language_code=seg.langcode,  # "zh"/"en"/"ja"
        )

        # --- 2.3 偏移到全局，并记录到flat_rows ---
        for c in char_items:
            flat_rows.append(
                {
                    "char": c.char,
                    "start": c.start + offset,
                    "end": c.end + offset,
                    "lang": seg.langcode,
                }
            )

        # --- 2.4 累加偏移（以WAV帧时长为准，保证与合并音频一致） ---
        offset += seg_duration

    # 3) 合并音频（bytes）
    if params_ref is None:
        merged_audio = b""
    else:
        merged_audio = concat_wav_frames(
            frames_list, params_ref, silence_sec_between=0.0
        )

    # 4) 生成chars_merged.json的bytes（UTF-8）
    chars_bytes = json.dumps(flat_rows, ensure_ascii=False, indent=2).encode("utf-8")

    return PipelineOutput(audio=merged_audio, chars_time=chars_bytes)
