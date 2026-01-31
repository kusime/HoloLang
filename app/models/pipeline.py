# -*- coding: utf-8 -*-
"""
管道相关数据模型
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.segment import LangCode
from app.models.tts import TTSConfig


class PipelineInput(BaseModel):
    """
    管道输入模型（与原始 TTS 请求参数保持兼容）
    
    注意：已移除 text_lang 字段，系统将自动检测每个段落的语言
    """

    model_config = ConfigDict(extra="forbid")

    # 主要输入
    text: str = Field(..., description="待合成的文本内容")

    # TTS 配置（继承自 TTSConfig 的所有字段）
    ref_audio_path: str = Field(..., description="参考音频路径（服务器可访问）")
    aux_ref_audio_paths: List[str] = Field(default=[], description="辅助参考音频路径列表")
    prompt_text: str = Field(..., description="提示文本（控制音色/情绪等）")
    prompt_lang: str = Field(default="zh", description="提示文本语言代码")

    text_split_method: str = Field(default="cut5", description="文本切分方法")
    batch_size: int = Field(default=60, ge=1, description="批大小")
    batch_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="切分/并发阈值")
    split_bucket: bool = Field(default=True, description="是否按桶划分切分片段")
    parallel_infer: bool = Field(default=True, description="是否并行推理")
    fragment_interval: float = Field(default=0.27, ge=0.0, description="片段间插入静音间隔（秒）")

    speed_factor: float = Field(default=1.0, gt=0.0, description="语速因子 > 0")
    top_k: int = Field(default=6, ge=0, description="采样 top_k")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="采样 top_p (0,1]")
    temperature: float = Field(default=0.65, gt=0.0, description="采样温度 > 0")
    repetition_penalty: float = Field(default=1.25, gt=0.0, description="重复惩罚 > 0")
    sample_steps: int = Field(default=32, ge=1, description="解码步数 >= 1")
    super_sampling: bool = Field(default=False, description="是否开启超采样")

    media_type: str = Field(default="wav", description="输出格式")
    streaming_mode: bool = Field(default=False, description="是否流式返回（二进制分块）")
    seed: int = Field(default=-1, description="随机种子；-1 表示随机")

    # 可选任务 ID
    job_id: Optional[str] = Field(
        None, description="可选自定义任务 ID；未提供时自动生成"
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        """去除文本首尾空白"""
        return v.strip()

    def to_tts_config(self) -> TTSConfig:
        """转换为 TTSConfig 对象"""
        return TTSConfig(
            ref_audio_path=self.ref_audio_path,
            aux_ref_audio_paths=self.aux_ref_audio_paths,
            prompt_text=self.prompt_text,
            prompt_lang=self.prompt_lang,
            text_split_method=self.text_split_method,
            batch_size=self.batch_size,
            batch_threshold=self.batch_threshold,
            split_bucket=self.split_bucket,
            parallel_infer=self.parallel_infer,
            fragment_interval=self.fragment_interval,
            speed_factor=self.speed_factor,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            sample_steps=self.sample_steps,
            super_sampling=self.super_sampling,
            media_type=self.media_type,  # type: ignore
            streaming_mode=self.streaming_mode,
            seed=self.seed,
        )


class PipelineOutput(BaseModel):
    """管道输出模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    audio: bytes = Field(..., description="合并后的 WAV 字节")
    chars_time: bytes = Field(
        ..., description="UTF-8 编码的 JSON：[{char,start,end,lang}, ...]"
    )


class ManifestOut(BaseModel):
    """上传到 S3 后的清单输出"""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    created_at: str
    contain_lang: List[LangCode]
    duration: float
    keys: Dict[str, str]
    etag: Dict[str, str]
    urls: Dict[str, Any]
    version: str = "2.0.0"
