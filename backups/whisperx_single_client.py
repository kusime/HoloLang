# whisperx_single_client.py
# -*- coding: utf-8 -*-
"""
单语种 WhisperX 强制对齐（含“开头缺失 k/n 均分回填”）
- 对外接口极简：align(text, audio, language_code)
- audio 必须是 bytes/BytesIO，不接受路径
- 严格透传 WhisperX 的字符级（char-level）结果
- 若开头缺失 n 个字符：取第一条成功对齐字符的 start = k，将前 n 个字符均分到 [0, k]（每个片宽 k/n）
- 使用对齐模型 metadata.sample_rate 计算 duration
- Pydantic v2 数据模型；按语言码缓存对齐模型
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import whisperx
from pydantic import BaseModel, Field, model_validator

# （可选）静音 transformers 的弃用告警
warnings.filterwarnings(
    "ignore",
    message=r"Passing `gradient_checkpointing` to a config initialization is deprecated.*",
    category=UserWarning,
    module=r"transformers\.configuration_utils",
)

SPACE_RE = re.compile(r"\s+")


# ---------- Pydantic v2 ----------
class CharTimestamp(BaseModel):
    char: str = Field(..., description="字符")
    start: float = Field(..., ge=0.0, description="起始时间（秒）")
    end: float = Field(..., ge=0.0, description="结束时间（秒）")

    @model_validator(mode="after")
    def _check(self):
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


# ---------- 小工具 ----------
def pick_device(explicit: Optional[str] = None) -> str:
    if explicit in ("cpu", "cuda"):
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_text(raw: str) -> str:
    # 统一空白（保留单空格，便于与你的 UI 一致）
    return SPACE_RE.sub(" ", (raw or "").strip())


def _to_records(x):
    return x.to_dict("records") if hasattr(x, "to_dict") else x


# ---------- 模型缓存 ----------
class AlignModelCache:
    def __init__(self, device: str):
        self.device = device
        self._cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    def get(self, lang: str) -> Tuple[Any, Dict[str, Any]]:
        if lang not in self._cache:
            model, meta = whisperx.load_align_model(
                language_code=lang, device=self.device
            )
            self._cache[lang] = (model, meta)
        return self._cache[lang]


# ---------- 单语种 Client ----------
class WhisperXSingleClient:
    """
    单语种对齐：
      align(text, audio, language_code) -> List[CharTimestamp]

    约束：
      - audio: bytes 或 BytesIO，不接受路径
      - 默认启用“k/n 前导回填”
      - 若无 char-level：内部先 words 兜底，再 linear 兜底
    """

    def __init__(self, device: Optional[str] = None):
        self.device = pick_device(device)
        self.cache = AlignModelCache(self.device)

    def align(
        self,
        text: str,
        audio: Union[bytes, BytesIO],  # 仅 bytes/BytesIO
        language_code: str,
    ) -> List[CharTimestamp]:
        clean_text = _normalize_text(text)
        if not clean_text:
            return []

        # 1) 模型与采样率
        model, meta = self.cache.get(language_code)
        sr = int(meta.get("sample_rate", 16000))

        # 2) 载入音频（float32），用对齐模型采样率算 duration
        audio_f32, duration = self._load_audio_f32_with_duration(audio, sr)

        # 3) 整段对齐
        coarse = [{"start": 0.0, "end": float(duration), "text": clean_text}]
        aligned = whisperx.align(
            coarse, model, meta, audio_f32, self.device, return_char_alignments=True
        )

        # 4) 抽取 char-level
        chars = self._extract_char_items(aligned.get("segments", []))
        chars.sort(key=lambda r: (r["start"], r["end"]))

        # 5) 若完全没有 char-level → 内部兜底（words -> linear）
        if not chars:
            chars = self._explode_words_to_chars(aligned.get("segments", []))
            if not chars:
                chars = self._linear_chars_over_segment(coarse[0])
            return [CharTimestamp(**r) for r in chars]

        # 6) 开头缺失回填（k/n 均分回填，默认启用）
        chars = self._fill_leading_equal_division(clean_text, chars)

        return [CharTimestamp(**r) for r in chars]

    # ----- 内部实现 -----
    def _load_audio_f32_with_duration(
        self, audio: Union[bytes, BytesIO], sample_rate: int
    ) -> Tuple[Any, float]:
        if isinstance(audio, BytesIO):
            data = audio.getvalue()
        elif isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        else:
            raise TypeError("audio 仅支持 bytes 或 BytesIO")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            arr = whisperx.load_audio(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        duration = float(len(arr) / sample_rate)
        return arr, duration

    @staticmethod
    def _extract_char_items(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for seg in segments:
            rows = None
            for key in ("char-segments", "char_segments", "chars", "characters"):
                cseg = seg.get(key)
                if cseg is not None:
                    rows = _to_records(cseg)
                    break
            if not rows:
                continue
            for r in rows:
                ch = r.get("char") or r.get("text") or r.get("token")
                st = (
                    r.get("start")
                    or r.get("start_ts")
                    or r.get("ts_start")
                    or r.get("begin")
                )
                ed = (
                    r.get("end")
                    or r.get("end_ts")
                    or r.get("ts_end")
                    or r.get("final")
                    or r.get("finish")
                )
                if ch is None or st is None or ed is None:
                    continue
                out.append({"char": str(ch), "start": float(st), "end": float(ed)})
        return out

    @staticmethod
    def _explode_words_to_chars(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for seg in segments:
            words = seg.get("words") or []
            for w in words:
                wtext = (w.get("word") or w.get("text") or "").strip()
                ws, we = w.get("start"), w.get("end")
                if not wtext or ws is None or we is None or we <= ws:
                    continue
                n = len(wtext)
                step = (we - ws) / n
                for i, ch in enumerate(wtext):
                    out.append(
                        {
                            "char": ch,
                            "start": float(ws + i * step),
                            "end": float(ws + (i + 1) * step),
                        }
                    )
        return out

    @staticmethod
    def _linear_chars_over_segment(seg0: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = re.sub(r"\s+", "", seg0.get("text", ""))
        ws, we = seg0.get("start", 0.0), seg0.get("end", 0.0)
        if not text or we <= ws:
            return []
        n = len(text)
        step = (we - ws) / n
        return [
            {
                "char": ch,
                "start": float(ws + i * step),
                "end": float(ws + (i + 1) * step),
            }
            for i, ch in enumerate(text)
        ]

    @staticmethod
    def _find_first_match_index(text: str, first_char: str) -> int:
        """
        在原始文本中找到与第一个对齐字符相同的字符的最早位置。
        若未找到，返回 0（即不做前导回填）。
        """
        idx = text.find(first_char)
        return idx if idx >= 0 else 0

    def _fill_leading_equal_division(
        self, text: str, aligned_chars: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        若对齐结果在开头缺了 n 个字符：
          - 记对齐列表第一项的 start 为 k
          - 把文本前 n 个字符等距分配在 [0, k] 上（每个片宽 k/n）
        之后按原样拼接对齐到的字符（不改变其时间戳，不做二次匹配）。
        说明：
          - n = text 中第一次出现 aligned_chars[0]['char'] 的索引
          - 若 k==0 或 n==0 或无法匹配，直接返回原对齐结果
        """
        if not aligned_chars:
            return aligned_chars
        first_char = aligned_chars[0]["char"]
        k = float(aligned_chars[0]["start"])
        if k <= 0:
            return aligned_chars

        n = self._find_first_match_index(text, first_char)
        if n <= 0:
            return aligned_chars

        step = k / n
        prefix: List[Dict[str, Any]] = []
        for i in range(n):
            ch = text[i]
            prefix.append(
                {"char": ch, "start": float(i * step), "end": float((i + 1) * step)}
            )
        return prefix + aligned_chars


# =============== 最简示例调用（仅 3 个参数） ===============
if __name__ == "__main__":
    """
    读取当前目录下的 text.txt 与 final.wav，以 bytes 调用 align。
    执行：python whisperx_single_client.py
    """
    TEXT_PATH = "text.txt"
    WAV_PATH = "final.wav"
    LANG = "zh"

    if not os.path.exists(TEXT_PATH) or not os.path.exists(WAV_PATH):
        print(f"[提示] 请在当前目录准备 {TEXT_PATH} 与 {WAV_PATH}")
    else:
        sample_text = open(TEXT_PATH, "r", encoding="utf-8").read()
        sample_audio_bytes = open(WAV_PATH, "rb").read()

        client = WhisperXSingleClient(device=None)  # 自动选择设备
        result = client.align(
            text=sample_text,
            audio=sample_audio_bytes,  # bytes
            language_code=LANG,
        )
        print(
            json.dumps([r.model_dump() for r in result], ensure_ascii=False, indent=2)
        )
        print(f"[完成] 共 {len(result)} 个 char 片段")
