# -*- coding: utf-8 -*-
"""
TTS 客户端（gptsovits）
"""

import json
from typing import Any, Dict, List

import requests

from app.models.tts import TTSConfig
from app.utils.logging_decorator import log_function


class TTSClient:
    """TTS 服务客户端"""

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

    @log_function()
    def get_tts_wav(self, text: str, lang_code: str, cfg: TTSConfig) -> bytes:
        """
        唯一主接口：text + lang_code + tts_config → audio bytes
        
        Args:
            text: 待合成的文本
            lang_code: 语言代码（zh/en/ja）
            cfg: TTS 配置
            
        Returns:
            音频字节数据
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
