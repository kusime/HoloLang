# -*- coding: utf-8 -*-
"""
服务层模块
"""

from app.services.segmentation import segment_text

# run_pipeline 依赖 WhisperX，需要手动导入
# 使用时: from app.services.pipeline import run_pipeline

__all__ = [
    "segment_text",
    # "run_pipeline",  # 需要手动导入
]
