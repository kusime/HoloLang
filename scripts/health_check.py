#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速服务健康检查脚本
可独立使用：python scripts/health_check.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:9880")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "localhost:9000")
S3_SECURE = os.getenv("S3_SECURE", "false").lower() == "true"

def check_tts():
    """检查 TTS 服务"""
    print(f"📡 检查 TTS 服务: {TTS_BASE_URL}")
    try:
        response = requests.get(f"{TTS_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            print("   ✅ TTS 服务正常")
            return True
    except:
        pass
    
    try:
        # 尝试根路径
        response = requests.get(TTS_BASE_URL, timeout=3)
        print("   ✅ TTS 服务可达（无 health endpoint）")
        return True
    except Exception as e:
        print(f"   ❌ TTS 服务不可达: {e}")
        return False

def check_minio():
    """检查 MinIO 服务"""
    protocol = "https" if S3_SECURE else "http"
    minio_url = f"{protocol}://{S3_ENDPOINT}/minio/health/live"
    
    print(f"📡 检查 MinIO 服务: {S3_ENDPOINT}")
    try:
        response = requests.get(minio_url, timeout=3)
        if response.status_code == 200:
            print("   ✅ MinIO 服务正常")
            return True
        else:
            print(f"   ❌ MinIO 服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ MinIO 服务不可达: {e}")
        return False

def main():
    print("\n🔍 运行外部服务健康检查\n")
    
    tts_ok = check_tts()
    minio_ok = check_minio()
    
    print("\n" + "="*50)
    if tts_ok and minio_ok:
        print("✅ 所有外部服务正常")
        return 0
    else:
        print("❌ 部分外部服务不可用")
        print("\n💡 提示:")
        if not tts_ok:
            print("   - 启动 TTS 服务或检查 TTS_BASE_URL 配置")
        if not minio_ok:
            print("   - 运行 docker-compose up -d 启动 MinIO")
            print("   - 或检查 S3_ENDPOINT 配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
