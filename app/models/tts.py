# -*- coding: utf-8 -*-
"""
TTS 相关数据模型
"""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class TTSConfig(BaseModel):
    """TTS 引擎配置（gptsovits）"""

    model_config = ConfigDict(extra="forbid")

    # 引导/参考
    ref_audio_path: str = Field(..., description="参考音频路径（服务器可访问）")
    aux_ref_audio_paths: List[str] = Field(..., description="辅助参考音频路径列表")
    prompt_text: str = Field(..., description="提示文本（控制音色/情绪等）")
    prompt_lang: str = Field(..., description="提示文本语言代码")

    # 切分 & 推理
    text_split_method: str = Field(..., description="文本切分方法")
    batch_size: int = Field(..., ge=1, description="批大小")
    batch_threshold: float = Field(..., ge=0.0, le=1.0, description="切分/并发阈值")
    split_bucket: bool = Field(..., description="是否按桶划分切分片段")
    parallel_infer: bool = Field(..., description="是否并行推理")
    fragment_interval: float = Field(
        ..., ge=0.0, description="片段间插入静音间隔（秒）"
    )

    # 采样/解码
    speed_factor: float = Field(..., gt=0.0, description="语速因子 > 0")
    top_k: int = Field(..., ge=0, description="采样 top_k")
    top_p: float = Field(..., gt=0.0, le=1.0, description="采样 top_p (0,1]")
    temperature: float = Field(..., gt=0.0, description="采样温度 > 0")
    repetition_penalty: float = Field(..., gt=0.0, description="重复惩罚 > 0")
    sample_steps: int = Field(..., ge=1, description="解码步数 >= 1")
    super_sampling: bool = Field(..., description="是否开启超采样")

    # 输出控制
    media_type: Literal["wav", "mp3", "flac", "ogg"] = Field(
        ..., description="输出格式"
    )
    streaming_mode: bool = Field(..., description="是否流式返回（二进制分块）")
    seed: int = Field(..., description="随机种子；-1 表示随机")
