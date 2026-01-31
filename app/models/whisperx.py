# -*- coding: utf-8 -*-
"""
WhisperX 相关数据模型
"""

from pydantic import BaseModel, Field, model_validator


class CharTimestamp(BaseModel):
    """字符级时间戳"""

    char: str = Field(..., description="字符")
    start: float = Field(..., ge=0.0, description="起始时间（秒）")
    end: float = Field(..., ge=0.0, description="结束时间（秒）")

    @model_validator(mode="after")
    def _check(self):
        """验证时间范围"""
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) < start ({self.start})")
        return self
