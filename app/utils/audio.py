# -*- coding: utf-8 -*-
"""
音频处理工具函数
"""

import io
import wave
from typing import List, Tuple


def read_wav_params_and_frames(
    wav_bytes: bytes,
) -> Tuple[Tuple[int, int, int], bytes, float]:
    """
    从 WAV 字节读取参数与帧，并计算时长。
    
    Args:
        wav_bytes: WAV 文件的字节数据
        
    Returns:
        ((nchannels, sampwidth, framerate), frames_bytes, duration_sec)
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
        dur = nframes / float(params[2] or 1)
    return params, frames, dur


def concat_wav_frames(
    frames_list: List[bytes],
    params: Tuple[int, int, int],
    silence_sec_between: float = 0.0,
) -> bytes:
    """
    按相同 WAV 参数拼接；段间可插入静音（默认 0.0）。
    
    Args:
        frames_list: WAV 帧字节数据列表
        params: (nchannels, sampwidth, framerate)
        silence_sec_between: 段间静音时长（秒）
        
    Returns:
        合并后的 WAV 字节数据
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


def get_wav_duration(wav_bytes: bytes) -> float:
    """
    获取 WAV 音频时长（秒）。
    
    Args:
        wav_bytes: WAV 文件的字节数据
        
    Returns:
        音频时长（秒）
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        nframes = wf.getnframes()
        framerate = wf.getframerate()
        return nframes / float(framerate or 1)
