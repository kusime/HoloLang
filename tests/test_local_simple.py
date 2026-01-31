#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的本地测试脚本 - 仅测试分段功能
不需要 TTS、WhisperX 或 S3 服务
"""

import json
import sys
from app.models.segment import TextIn
from app.services.segmentation import segment_text

def test_segmentation():
    """测试文本分段功能"""
    print("=" * 60)
    print("测试 1: 文本分段服务")
    print("=" * 60)
    
    test_cases = [
        "这是中文测试。This is English. これは日本語です。",
        "黄昏の駅で、私は歩幅を小さく整えた。Next, we switch to English.",
        "Hello World. 你好世界。こんにちは世界。"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"输入: {text}")
        
        try:
            result = segment_text(TextIn(text=text))
            print(f"检测到的语言: {result.contain_lang}")
            print(f"分段数量: {len(result.segments)}")
            
            for j, seg in enumerate(result.segments, 1):
                print(f"  段 {j} [{seg.langcode}]: {seg.text}")
            
            print("✅ 通过")
        except Exception as e:
            print(f"❌ 失败: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ 所有分段测试通过！")
    print("=" * 60)
    return True

def test_imports():
    """测试关键模块导入"""
    print("\n" + "=" * 60)
    print("测试 2: 模块导入")
    print("=" * 60)
    
    modules = [
        ("app.models.pipeline", "PipelineInput"),
        ("app.models.tts", "TTSConfig"),
        ("app.clients.tts", "TTSClient"),
        ("app.clients.s3", "S3Client"),
        ("app.services.segmentation", "segment_text"),
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            return False
    
    # WhisperX 是可选的
    try:
        from app.clients.whisperx import WhisperXSingleClient
        print(f"✅ app.clients.whisperx.WhisperXSingleClient (可选)")
    except Exception as e:
        print(f"⚠️  app.clients.whisperx.WhisperXSingleClient (可选): {e}")
    
    print("=" * 60)
    print("✅ 模块导入测试通过！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    print("\n🚀 HoloLang 本地测试套件\n")
    
    all_passed = True
    
    # 测试模块导入
    if not test_imports():
        all_passed = False
        print("\n❌ 模块导入失败")
    
    # 测试分段功能
    if not test_segmentation():
        all_passed = False
        print("\n❌ 分段测试失败")
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 部分测试失败")
        print("=" * 60)
        sys.exit(1)
