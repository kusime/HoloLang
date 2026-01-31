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


def resample_wav_bytes(
    wav_bytes: bytes,
    target_sr: int,
    target_channels: int = 1,
    target_width: int = 2
) -> bytes:
    """
    重采样 WAV 音频到指定参数
    
    使用 torchaudio 进行高质量重采样
    
    Args:
        wav_bytes: 输入 WAV 字节
        target_sr: 目标采样率
        target_channels: 目标声道数
        target_width: 目标位宽（字节），目前仅支持 2 (16-bit PCM)
        
    Returns:
        重采样后的 WAV 字节
    """
    import torch
    import torchaudio
    import io
    
    # 读取音频
    bio = io.BytesIO(wav_bytes)
    waveform, src_sr = torchaudio.load(bio)
    
    # 1. 声道处理
    if waveform.shape[0] != target_channels:
        if target_channels == 1:
            # 多转单：平均
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif target_channels == 2:
            # 单转多：复制
            waveform = waveform.repeat(2, 1)
            
    # 2. 重采样
    if src_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # 3. 导出为 WAV bytes
    out_bio = io.BytesIO()
    torchaudio.save(
        out_bio, 
        waveform, 
        target_sr, 
        encoding="PCM_S", 
        bits_per_sample=target_width * 8,
        format="wav"
    )
    return out_bio.getvalue()
