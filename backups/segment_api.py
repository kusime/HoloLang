# segment_api.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import regex as re
from lingua import Language, LanguageDetectorBuilder
from pydantic import BaseModel


# ---------- Pydantic IO ----------
class TextIn(BaseModel):
    text: str


LangCode = Literal["en", "zh", "ja"]


class SegmentItem(BaseModel):
    start: int
    end: int
    langcode: LangCode
    text: str


class TextSegmentsOut(BaseModel):
    contain_lang: List[LangCode]
    segments: List[SegmentItem]  # [{start,end,langcode,text}]


# ---------- Lingua（三语，纯判定） ----------
DETECTOR = (
    LanguageDetectorBuilder.from_languages(
        Language.CHINESE, Language.JAPANESE, Language.ENGLISH
    )
    .with_low_accuracy_mode()  # 更稳可去掉此行（略慢、但更准）
    .build()
)

# ---------- 基础脚本桶（仅粗分，无业务特判） ----------
S_HAN = re.compile(r"\p{Script=Han}")
S_HIRA = re.compile(r"\p{Script=Hiragana}")
S_KATA = re.compile(r"\p{Script=Katakana}")
S_LATIN = re.compile(r"\p{Script=Latin}")
S_NUM = re.compile(r"\p{Number}")
S_PUNC = re.compile(r"\p{Punctuation}")
HIRA_OR_KATA = re.compile(r"[\p{Hiragana}\p{Katakana}]")


def _script_bucket(ch: str) -> str:
    if S_HIRA.match(ch) or S_KATA.match(ch):
        return "JPN_KANA"
    if S_HAN.match(ch):
        return "HAN"
    if S_LATIN.match(ch):
        return "LATIN"
    if S_NUM.match(ch) or S_PUNC.match(ch) or ch.isspace():
        return "NEUTRAL"
    return "OTHER"


def _coarse_segments(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    segs, start = [], 0
    last = _script_bucket(text[0])
    for i, ch in enumerate(text):
        b = _script_bucket(ch)
        eff = last if b == "NEUTRAL" else b
        if eff != last:
            segs.append((start, i))
            start, last = i, eff
    segs.append((start, len(text)))
    return segs


def _lingua_to_whisper(name: str) -> str:
    u = name.upper()
    if "CHINESE" in u:
        return "zh"
    if "JAPANESE" in u:
        return "ja"
    return "en"


def _scores(text: str) -> Dict[str, float]:
    vals = DETECTOR.compute_language_confidence_values(text) or []
    s = {"zh": 0.0, "ja": 0.0, "en": 0.0}
    for v in vals:
        s[_lingua_to_whisper(v.language.name)] = float(v.value)
    if s["zh"] == s["ja"] == s["en"] == 0.0:  # 超短文本兜底
        s = {"zh": 1 / 3, "ja": 1 / 3, "en": 1 / 3}
    return s


@dataclass
class _Span:
    start: int
    end: int  # 半开区间 [start, end)
    text: str
    lang: str  # 'zh' | 'ja' | 'en'
    scores: Dict[str, float]


def _merge_adjacent(spans: List[_Span]) -> List[_Span]:
    out: List[_Span] = []
    for sp in spans:
        if out and out[-1].lang == sp.lang and out[-1].end == sp.start:
            a = out[-1]
            L1, L2 = a.end - a.start, sp.end - sp.start
            total = L1 + L2 or 1
            mix = {k: (a.scores[k] * L1 + sp.scores[k] * L2) / total for k in a.scores}
            out[-1] = _Span(a.start, sp.end, a.text + sp.text, sp.lang, mix)
        else:
            out.append(sp)
    return out


def _contains_kana(s: str) -> bool:
    return bool(HIRA_OR_KATA.search(s))


def _smooth_ja_adhesion(
    spans: List[_Span], text: str, max_zh_len: int = 3, kana_pull: bool = True
) -> List[_Span]:
    """ja-zh-ja 把中段短 zh 改为 ja；短 zh 邻接含假名的 ja 段时也改为 ja。"""
    if not spans:
        return spans
    fixed = spans[:]
    n = len(fixed)

    # 1) ja-zh-ja 夹心
    for i in range(1, n - 1):
        mid = fixed[i]
        if mid.lang == "zh" and (mid.end - mid.start) <= max_zh_len:
            if fixed[i - 1].lang == "ja" and fixed[i + 1].lang == "ja":
                L, R = max(0, mid.start - 8), min(len(text), mid.end + 8)
                fixed[i] = _Span(mid.start, mid.end, mid.text, "ja", _scores(text[L:R]))

    # 2) 假名牵引
    if kana_pull:
        for i, sp in enumerate(fixed):
            if sp.lang != "zh" or (sp.end - sp.start) > max_zh_len:
                continue
            left_ok = (
                i > 0
                and fixed[i - 1].lang == "ja"
                and _contains_kana(fixed[i - 1].text)
            )
            right_ok = (
                i < n - 1
                and fixed[i + 1].lang == "ja"
                and _contains_kana(fixed[i + 1].text)
            )
            if left_ok or right_ok:
                L, R = max(0, sp.start - 8), min(len(text), sp.end + 8)
                fixed[i] = _Span(sp.start, sp.end, sp.text, "ja", _scores(text[L:R]))
    return fixed


def _segment_text_pure(
    text: str, max_zh_len: int = 3, kana_pull: bool = True
) -> List[_Span]:
    # 1) 粗分 + Lingua 判定
    spans: List[_Span] = []
    for s, e in _coarse_segments(text):
        chunk = text[s:e]
        scores = _scores(chunk)
        lang = max(scores.items(), key=lambda kv: kv[1])[0]
        spans.append(_Span(s, e, chunk, lang, scores))

    # 2) 合并同语
    spans = _merge_adjacent(spans)

    # 3) 日语黏合修正
    spans = _smooth_ja_adhesion(spans, text, max_zh_len=max_zh_len, kana_pull=kana_pull)

    # 4) 再合并一次
    spans = _merge_adjacent(spans)
    return spans


# ---------- API ----------
def segment_api(payload: TextIn) -> TextSegmentsOut:
    text = payload.text
    spans = _segment_text_pure(text)

    # 结构：[{start,end,langcode,text}]
    seg_items: List[SegmentItem] = []
    present = set()
    for sp in spans:
        lang: LangCode = sp.lang  # type: ignore
        seg_items.append(
            SegmentItem(start=sp.start, end=sp.end, langcode=lang, text=sp.text)
        )
        present.add(lang)

    # contain_lang 固定顺序（只包含出现过的）
    ordered = [l for l in ["en", "zh", "ja"] if l in present]

    return TextSegmentsOut(contain_lang=ordered, segments=seg_items)


if __name__ == "__main__":
    import json

    demo = "今天我们测试一下,WhisperX alignment。Next, we switch to English.最後に日本語でも話してみます。"

    # 调用
    payload = TextIn(text=demo)
    result = segment_api(payload)

    # 友好打印（UTF-8，不转义中文）
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
