# -*- coding: utf-8 -*-
"""
WhisperX 单语种对齐客户端

专注于字符级时间戳对齐，支持中日英三语
"""

import os
import re
import tempfile
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import whisperx

from app.models.whisperx import CharTimestamp
from app.utils.logging_decorator import log_function

# 静音 transformers 弃用告警
warnings.filterwarnings(
    "ignore",
    message=r"Passing `gradient_checkpointing` to a config initialization is deprecated.*",
    category=UserWarning,
    module=r"transformers\.configuration_utils",
)

SPACE_RE = re.compile(r"\s+")


# ---------- 工具函数 ----------
def pick_device(explicit: Optional[str] = None) -> str:
    """选择设备（CPU/CUDA）"""
    if explicit in ("cpu", "cuda"):
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_text(raw: str) -> str:
    """标准化文本（统一空白）"""
    return SPACE_RE.sub(" ", (raw or "").strip())


def _to_records(x):
    """转换为字典记录列表"""
    return x.to_dict("records") if hasattr(x, "to_dict") else x


# ---------- 模型缓存 ----------
class AlignModelCache:
    """对齐模型缓存（按语言代码）"""

    def __init__(self, device: str):
        self.device = device
        self._cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    def get(self, lang: str) -> Tuple[Any, Dict[str, Any]]:
        """获取或加载对齐模型"""
        if lang not in self._cache:
            model, meta = whisperx.load_align_model(
                language_code=lang, device=self.device
            )
            self._cache[lang] = (model, meta)
        return self._cache[lang]


# ---------- WhisperX 客户端 ----------
class WhisperXSingleClient:
    """
    单语种 WhisperX 对齐客户端
    
    提供字符级时间戳对齐功能，支持前导字符回填。
    """

    def __init__(self, device: Optional[str] = None):
        self.device = pick_device(device)
        self.cache = AlignModelCache(self.device)
        
        # 设置 Hugging Face 离线模式优先（如果模型已下载，避免在线检查）
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "0")  # 允许下载但优先本地
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    @log_function()
    def align(
        self,
        text: str,
        audio: Union[bytes, BytesIO],
        language_code: str,
    ) -> List[CharTimestamp]:
        """
        对音频进行字符级时间戳对齐
        
        Args:
            text: 待对齐的文本
            audio: 音频数据（bytes 或 BytesIO）
            language_code: 语言代码（zh/en/ja）
            
        Returns:
            字符时间戳列表
        """
        clean_text = _normalize_text(text)
        if not clean_text:
            return []

        # 1) 加载模型与采样率
        model, meta = self.cache.get(language_code)
        sr = int(meta.get("sample_rate", 16000))

        # 2) 载入音频（float32）
        audio_f32, duration = self._load_audio_f32_with_duration(audio, sr)

        # 3) 整段对齐
        coarse = [{"start": 0.0, "end": float(duration), "text": clean_text}]
        aligned = whisperx.align(
            coarse, model, meta, audio_f32, self.device, return_char_alignments=True
        )

        # 4) 抽取字符级数据
        chars = self._extract_char_items(aligned.get("segments", []))
        chars.sort(key=lambda r: (r["start"], r["end"]))

        # 5) 若完全没有字符级数据，尝试从词级或线性分配兜底
        if not chars:
            chars = self._explode_words_to_chars(aligned.get("segments", []))
            if not chars:
                chars = self._linear_chars_over_segment(coarse[0])
            return [CharTimestamp(**r) for r in chars]

        # 6) 开头缺失字符回填
        chars = self._fill_leading_equal_division(clean_text, chars)

        return [CharTimestamp(**r) for r in chars]

    # ----- 内部实现 -----
    def _load_audio_f32_with_duration(
        self, audio: Union[bytes, BytesIO], sample_rate: int
    ) -> Tuple[Any, float]:
        """载入音频并计算时长"""
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
        """从对齐结果中提取字符级数据"""
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
        """从词级数据展开为字符级（线性分配时间）"""
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
        """线性分配时间到每个字符"""
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
        """在原始文本中找到第一个对齐字符的位置"""
        idx = text.find(first_char)
        return idx if idx >= 0 else 0

    def _fill_leading_equal_division(
        self, text: str, aligned_chars: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        开头缺失字符回填
        
        若对齐结果在开头缺失了 n 个字符，将这些字符均分到 [0, k] 区间
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
