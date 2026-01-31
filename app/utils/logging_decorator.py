# -*- coding: utf-8 -*-
"""
非侵入式日志装饰器

专注于高层流程追踪：input_text -> lang detect -> tts -> s3
避免记录繁琐的参数细节
"""

import functools
import logging
import time
from typing import Any, Callable, Optional


def _extract_job_id(args: tuple, kwargs: dict) -> Optional[str]:
    """
    从函数参数中提取 Job ID
    
    尝试顺序：
    1. kwargs 中的 job_id
    2. args[0] 如果是 Pydantic 模型且有 job_id 属性
    3. 返回 None
    """
    # 1. 检查 kwargs
    if "job_id" in kwargs:
        return kwargs["job_id"]
    
    # 2. 检查第一个参数（可能是 request 对象）
    if args and hasattr(args[0], "job_id"):
        return getattr(args[0], "job_id", None)
    
    return None


def log_function(
    level: str = "INFO",
    include_duration: bool = True,
) -> Callable:
    """
    同步函数日志装饰器（非侵入式）
    
    Args:
        level: 日志级别（INFO/DEBUG/WARNING）
        include_duration: 是否记录执行时长
        
    Example:
        @log_function()
        def run_pipeline(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        log_level = getattr(logging, level.upper())
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 提取 Job ID（如果存在）
            job_id = _extract_job_id(args, kwargs)
            prefix = f"[{job_id}] " if job_id else ""
            
            # 记录开始
            logger.log(log_level, f"{prefix}{func.__name__} 开始")
            
            # 执行并计时
            start = time.time()
            try:
                result = func(*args, **kwargs)
                
                # 记录成功
                if include_duration:
                    duration = time.time() - start
                    logger.log(log_level, f"{prefix}{func.__name__} 完成 ({duration:.2f}s)")
                else:
                    logger.log(log_level, f"{prefix}{func.__name__} 完成")
                
                return result
                
            except Exception as e:
                # 记录失败（包含异常和时长）
                duration = time.time() - start
                logger.error(
                    f"{prefix}{func.__name__} 失败: {type(e).__name__}: {e} ({duration:.2f}s)",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_async_function(
    level: str = "INFO",
    include_duration: bool = True,
) -> Callable:
    """
    异步函数日志装饰器（非侵入式）
    
    使用方式同 log_function，但用于 async def 函数
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        log_level = getattr(logging, level.upper())
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 提取 Job ID
            job_id = _extract_job_id(args, kwargs)
            prefix = f"[{job_id}] " if job_id else ""
            
            # 记录开始
            logger.log(log_level, f"{prefix}{func.__name__} 开始")
            
            # 执行并计时
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # 记录成功
                if include_duration:
                    duration = time.time() - start
                    logger.log(log_level, f"{prefix}{func.__name__} 完成 ({duration:.2f}s)")
                else:
                    logger.log(log_level, f"{prefix}{func.__name__} 完成")
                
                return result
                
            except Exception as e:
                # 记录失败
                duration = time.time() - start
                logger.error(
                    f"{prefix}{func.__name__} 失败: {type(e).__name__}: {e} ({duration:.2f}s)",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator
