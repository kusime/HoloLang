# -*- coding: utf-8 -*-
"""
multiple_pipeline_api
- 输入: TextIn(text=原文字符串)
- 流程: segment_api → 逐段 TTS → 单语种 WhisperX → 合并到全局时间轴
- 产物: PipeLineOutput(audio: bytes, chars_time: bytes)
  * audio: 合并后的 WAV 字节
  * chars_time: JSON 形式的 [{char,start,end,lang}, ...]（UTF-8 编码后的 bytes）
- __main__ 自测：把 bytes 写成 ./final.wav 与 ./chars_merged.json
"""

from __future__ import annotations

import io
import json
import os
import wave
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

# --- 导入三个模块 ---
# 1) 文本切段
from segment_api import TextIn, TextSegmentsOut, segment_api  # type: ignore

# 2) TTS（确保你的文件名与类名一致；若不同，按需改这里的导入）
from tts_client import TTSClient, TTSConfig  # type: ignore

# 3) 单语种 WhisperX 对齐（CharTimestamp 仅用于内部构造/类型提示）
from whisperx_single_client import CharTimestamp, WhisperXSingleClient  # type: ignore


# =========== Pipeline 的 Pydantic 输出模型 ===========
class PipeLineOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    audio: bytes = Field(..., description="合并后的 WAV 字节")
    chars_time: bytes = Field(
        ..., description="UTF-8 编码的 JSON：[{char,start,end,lang}, ...]"
    )


def _wav_read_params_and_frames_from_bytes(
    wav_bytes: bytes,
) -> Tuple[Tuple[int, int, int], bytes, float]:
    """
    从 WAV 字节读取参数与帧，并计算时长。
    返回: ((nchannels, sampwidth, framerate), frames_bytes, duration_sec)
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
        dur = nframes / float(params[2] or 1)
    return params, frames, dur


def _wav_concat_frames(
    frames_list: List[bytes],
    params: Tuple[int, int, int],
    silence_sec_between: float = 0.0,
) -> bytes:
    """
    按相同 WAV 参数拼接；段间可插入静音（默认 0.0）。
    """
    nch, sw, fr = params
    silence_frames = int(round(fr * max(0.0, silence_sec_between)))
    silence_bytes = (b"\x00" * sw * nch) * silence_frames if silence_frames > 0 else b""

    out_bio = io.BytesIO()
    with wave.open(out_bio, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(fr)
        for i, frames in enumerate(frames_list):
            if i > 0 and silence_bytes:
                wf.writeframes(silence_bytes)
            wf.writeframes(frames)
    return out_bio.getvalue()


# =========== 主 API（返回 PipeLineOutput） ===========
def multiple_pipeline_api(
    payload: TextIn,
    *,
    tts_client: TTSClient,
    tts_cfg: TTSConfig,
    align_client: WhisperXSingleClient,
) -> PipeLineOutput:
    """
    处理流程:
      1) segment_api 切分
      2) 逐段: TTS -> WhisperXSingleClient.align
      3) 合并: 音频拼接 + 时间轴偏移，返回 PipeLineOutput（bytes）
    约束:
      - tts_cfg.media_type 必须是 'wav'（当前实现只拼接 WAV）
    """
    if tts_cfg.media_type != "wav":
        raise NotImplementedError("目前仅支持 media_type='wav' 以便做 WAV 拼接。")

    # 1) 文本切段
    seg_out: TextSegmentsOut = segment_api(payload)

    # 2) 逐段合成 & 对齐（全在内存）
    frames_list: List[bytes] = []
    params_ref: Tuple[int, int, int] | None = None
    offset = 0.0  # 全局偏移（秒）

    # 用于 chars_merged.json 的扁平全局时间轴（带 lang）
    flat_rows: List[Dict[str, Any]] = []

    for idx, seg in enumerate(seg_out.segments):
        # --- 2.1 TTS: 返回 WAV 字节 ---
        seg_wav_bytes: bytes = tts_client.get_tts_wav(seg.text, seg.langcode, tts_cfg)

        # --- 2.1.1 解析参数/帧/时长，并校验参数一致 ---
        params, frames, seg_duration = _wav_read_params_and_frames_from_bytes(
            seg_wav_bytes
        )
        if params_ref is None:
            params_ref = params
        elif params != params_ref:
            raise RuntimeError(
                f"WAV 参数不一致，无法合并：first={params_ref}, current={params} (segment index={idx}, lang={seg.langcode})\n"
                "请确保 TTS 输出统一声道/位宽/采样率（或先重采样再合并）。"
            )
        frames_list.append(frames)

        # --- 2.2 WhisperX 单语种对齐（段内时间） ---
        char_items = align_client.align(
            text=seg.text,
            audio=seg_wav_bytes,  # bytes
            language_code=seg.langcode,  # "zh"/"en"/"ja"
        )

        # --- 2.3 偏移到全局，并记录到 flat_rows ---
        for c in char_items:
            flat_rows.append(
                {
                    "char": c.char,
                    "start": c.start + offset,
                    "end": c.end + offset,
                    "lang": seg.langcode,
                }
            )

        # --- 2.4 累加偏移（以 WAV 帧时长为准，保证与合并音频一致） ---
        offset += seg_duration

    # 3) 合并音频（bytes）
    if params_ref is None:
        merged_audio = b""
    else:
        merged_audio = _wav_concat_frames(
            frames_list, params_ref, silence_sec_between=0.0
        )

    # 4) 生成 chars_merged.json 的 bytes（UTF-8）
    chars_bytes = json.dumps(flat_rows, ensure_ascii=False, indent=2).encode("utf-8")

    return PipeLineOutput(audio=merged_audio, chars_time=chars_bytes)


# =========== 自测：把 bytes 写成文件 ===========
if __name__ == "__main__":
    """
    自测：
      - 若存在 ./text.txt 则读取，否则使用内置 demo 文本
      - 使用固定的 gptsovits 配置（与你之前成功示例一致）
      - 返回 PipeLineOutput，然后把 bytes 写成 final.wav / chars_merged.json
    """
    # 读取文本
    TEXT_PATH = "text.txt"
    if os.path.exists(TEXT_PATH):
        input_text = open(TEXT_PATH, "r", encoding="utf-8").read()
    else:
        input_text = "今天我们测试一下,WhisperX alignment。Next, we switch to English.最後に日本語でも話してみます。"

    # 组装请求
    inp = TextIn(text=input_text)

    # TTS 配置（全部显式给值）
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

    # 三个子模块客户端
    tts = TTSClient(base_url="http://localhost:9880", timeout=120)
    align = WhisperXSingleClient(device=None)  # cuda 优先，否则 cpu

    # 调用 pipeline（返回 bytes）
    out = multiple_pipeline_api(
        inp,
        tts_client=tts,
        tts_cfg=cfg,
        align_client=align,
    )

    # bytes → 文件（不影响单测；你也可以在单测里直接断言 bytes）
    with open("final.wav", "wb") as f:
        f.write(out.audio)
    with open("chars_merged.json", "wb") as f:
        f.write(out.chars_time)

    print("[✔] 已写出 final.wav 与 chars_merged.json")
