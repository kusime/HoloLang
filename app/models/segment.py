# -*- coding: utf-8 -*-
"""
文本分段相关数据模型
"""

from typing import List, Literal

from pydantic import BaseModel

# 语言代码类型
LangCode = Literal["en", "zh", "ja"]


class TextIn(BaseModel):
    """文本分段输入"""

    text: str


class SegmentItem(BaseModel):
    """单个文本段落"""

    start: int
    end: int
    langcode: LangCode
    text: str


class TextSegmentsOut(BaseModel):
    """文本分段输出"""

    contain_lang: List[LangCode]
    segments: List[SegmentItem]
