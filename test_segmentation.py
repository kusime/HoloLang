#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言分段改进验证脚本
"""

from app.models.segment import TextIn
from app.services.segmentation import segment_text

print("=" * 60)
print("多语言分段改进验证")
print("=" * 60)

test_cases = [
    ("テクノロジーは日常生活のあらゆる場面に浸透しています。", ["ja"]),
    ("私は学生です。今日は勉強します。", ["ja"]),
    ("这是纯中文文本没有其他语言。", ["zh"]),
    ("Hello world! 这是测试。今日はいい天気ですね。", ["en", "zh", "ja"]),
    ("彼女は素晴らしい人です。", ["ja"]),
    ("中文内容继续写很长一段话。", ["zh"]),
]

passed = 0
failed = 0

for text, expected_langs in test_cases:
    result = segment_text(TextIn(text=text))
    
    status = "✓" if result.contain_lang == expected_langs else "✗"
    if result.contain_lang == expected_langs:
        passed += 1
    else:
        failed += 1
    
    print(f"\n{status} 测试: {text[:40]}")
    print(f"  预期语言: {expected_langs}")
    print(f"  检测语言: {result.contain_lang}")
    
    if result.contain_lang != expected_langs:
        print(f"  分段详情:")
        for i, seg in enumerate(result.segments, 1):
            print(f"    {i}. [{seg.langcode}] {seg.text}")

print("\n" + "=" * 60)
print(f"测试结果: {passed} 通过, {failed} 失败")
print("=" * 60)
