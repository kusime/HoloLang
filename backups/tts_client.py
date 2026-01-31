# -*- coding: utf-8 -*-
"""
TTS client (gptsovits)
- 仅保留一个外部数据模型：TTSConfig（全部参数显式提供，无默认值）
- 仅保留一个调用接口：TTSClient.get_tts_wav(text: str, lang_code: str, cfg: TTSConfig) -> bytes
- 无 CLI
- __main__ 自测：读取 ./text.txt → 写 ./final.wav（lang_code=zh）
"""

import json
import os
from typing import Any, Dict, List, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field


# ===== gptsovits TTS 配置（无默认值，全部显式提供） =====
class TTSConfig(BaseModel):
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


# ===== 客户端封装（唯一接口：get_tts_wav） =====
class TTSClient:
    def __init__(self, base_url: str = "http://localhost:9880", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "*/*"}
        )

    @staticmethod
    def _build_payload(text: str, lang_code: str, cfg: TTSConfig) -> Dict[str, Any]:
        """合并 text + lang_code + TTSConfig → 后端所需 payload"""
        return {
            "text": text,
            "text_lang": lang_code,
            "ref_audio_path": cfg.ref_audio_path,
            "aux_ref_audio_paths": cfg.aux_ref_audio_paths,
            "prompt_text": cfg.prompt_text,
            "prompt_lang": cfg.prompt_lang,
            "text_split_method": cfg.text_split_method,
            "batch_size": cfg.batch_size,
            "batch_threshold": cfg.batch_threshold,
            "split_bucket": cfg.split_bucket,
            "parallel_infer": cfg.parallel_infer,
            "fragment_interval": cfg.fragment_interval,
            "speed_factor": cfg.speed_factor,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "temperature": cfg.temperature,
            "repetition_penalty": cfg.repetition_penalty,
            "sample_steps": cfg.sample_steps,
            "super_sampling": cfg.super_sampling,
            "media_type": cfg.media_type,
            "streaming_mode": cfg.streaming_mode,
            "seed": cfg.seed,
        }

    def get_tts_wav(self, text: str, lang_code: str, cfg: TTSConfig) -> bytes:
        """
        唯一主接口：text + lang_code + tts_config → audio bytes
        - 若 cfg.streaming_mode=True：使用流式合并字节后返回
        - 否则直接一次性返回
        """
        text = text.strip()
        lang_code = lang_code.strip()
        url = f"{self.base_url}/tts"
        payload = self._build_payload(text, lang_code, cfg)

        if cfg.streaming_mode:
            with self._session.post(
                url, data=json.dumps(payload), timeout=self.timeout, stream=True
            ) as r:
                try:
                    r.raise_for_status()
                except requests.HTTPError as e:
                    try:
                        err_json = r.json()
                    except Exception:
                        err_json = r.text[:500]
                    raise RuntimeError(
                        f"HTTP {r.status_code} from /tts: {err_json}"
                    ) from e
                chunks: List[bytes] = []
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                return b"".join(chunks)

        # 非流式
        resp = self._session.post(url, data=json.dumps(payload), timeout=self.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text[:500]
            raise RuntimeError(f"HTTP {resp.status_code} from /tts: {err_text}") from e

        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            data = resp.json()
            if not data or data.get("ok") is False:
                raise RuntimeError(f"TTS 服务返回错误: {data}")
            raise RuntimeError(f"意外的 JSON 响应：{data}")
        return resp.content


# ===== 最小自测：读取 ./text.txt → 写 ./final.wav（lang_code=zh） =====
if __name__ == "__main__":
    TEXT_FILE = "text.txt"
    OUT_PATH = "final.wav"
    BASE_URL = "http://localhost:9880"
    LANG_CODE = "zh"

    if not os.path.exists(TEXT_FILE):
        print(f"[提示] 请在当前目录准备文本文件: {os.path.abspath(TEXT_FILE)}")
        raise SystemExit(2)

    text_content = open(TEXT_FILE, "r", encoding="utf-8").read()

    # 与成功示例一致（全部显式给值，无默认）
    cfg = TTSConfig(
        ref_audio_path="z.refs/main.wav",
        aux_ref_audio_paths=[],
        prompt_text="这是最后一件了吧？嗯，这里确实有七十件。",
        prompt_lang="zh",
        text_split_method="cut5",
        batch_size=60,
        batch_threshold=0.75,
        split_bucket=True,
        parallel_infer=True,
        fragment_interval=0.27,
        speed_factor=1.05,
        top_k=6,
        top_p=1.0,
        temperature=0.65,
        repetition_penalty=1.25,
        sample_steps=32,
        super_sampling=False,
        media_type="wav",
        streaming_mode=False,
        seed=-1,
    )

    client = TTSClient(base_url=BASE_URL, timeout=120)
    try:
        audio_bytes = client.get_tts_wav(text_content, LANG_CODE, cfg)
        with open(OUT_PATH, "wb") as f:
            f.write(audio_bytes)
    except Exception as e:
        print(f"[X] 调用失败: {e}")
        raise SystemExit(3)

    print(f"[✔] 已保存: {os.path.abspath(OUT_PATH)}")
