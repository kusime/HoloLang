# -*- coding: utf-8 -*-
"""
文本分段服务

基于 fastText AI 模型和脚本分析，实现多语言文本的自动分段。
支持中文、英文、日文的混合文本。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import regex as re
import fasttext

from app.models.segment import LangCode, SegmentItem, TextIn, TextSegmentsOut
from app.utils.logging_decorator import log_function

# ---------- fastText 语言检测器（AI 驱动）----------
_FASTTEXT_MODEL: Optional[fasttext.FastText._FastText] = None

def _get_language_model() -> fasttext.FastText._FastText:
    """获取 fastText 模型（单例模式）"""
    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is None:
        model_path = Path(__file__).parent.parent.parent / "models" / "lid.176.bin"
        _FASTTEXT_MODEL = fasttext.load_model(str(model_path))
    return _FASTTEXT_MODEL

# ---------- 脚本正则表达式 ----------
S_HAN = re.compile(r"\p{Script=Han}")
S_HIRA = re.compile(r"\p{Script=Hiragana}")
S_KATA = re.compile(r"\p{Script=Katakana}")
S_LATIN = re.compile(r"\p{Script=Latin}")
S_NUM = re.compile(r"\p{Number}")
S_PUNC = re.compile(r"\p{Punctuation}")
HIRA_OR_KATA = re.compile(r"[\p{Hiragana}\p{Katakana}]")


def _script_bucket(ch: str) -> str:
    """将字符分类到脚本桶"""
    if S_HIRA.match(ch) or S_KATA.match(ch):
        return "JPN_KANA"
    if S_HAN.match(ch):
        return "HAN"
    if S_LATIN.match(ch):
        return "LATIN"
    if S_NUM.match(ch) or S_PUNC.match(ch) or ch.isspace():
        return "NEUTRAL"
    return "OTHER"


def _split_by_punctuation(text: str) -> List[Tuple[int, int]]:
    """
    基于Regex的标点强行分割（Step 1: Punctuation-based Hard Split）
    
    不再依赖脚本判断，而是对所有标点符号进行切分，
    然后依靠后续的"Isolated Detection"和合并逻辑来恢复句子。
    
    涵盖符号：
    - 主分隔符：. 。 ! ！ ? ？
    - 逗号系：, ， 、 (全覆盖)
    """
    if not text:
        return []
        
    # 定义分隔符正则表达式（包含全角/半角标点）
    pattern = re.compile(r'([。！？.!?，、,]+)')
    
    segments = []
    start = 0
    
    # split 会返回 [chunk, delimiter, chunk, delimiter, ...]
    # 我们需要保留原始位置信息，所以最好用 scanner 或 finditer
    # 简单的做法：找到所有分隔符位置，构建片段
    
    for match in pattern.finditer(text):
        token_start, token_end = match.span()
        
        # 分隔符前面的文本块（如果非空）
        if token_start > start:
            segments.append((start, token_start))
            
        # 分隔符本身也作为一个块（便于保留标点）
        segments.append((token_start, token_end))
        
        start = token_end
        
    # 处理剩余文本
    if start < len(text):
        segments.append((start, len(text)))
        
    return segments


def _coarse_segments(text: str) -> List[Tuple[int, int]]:
    """粗略分段（基于脚本切换）"""
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


def _fasttext_to_project_lang(label: str) -> str:
    """
    将 fastText 标签映射到项目语言代码
    
    重要：不直接使用 fastText 的标签，而是映射到项目统一的语言代码系统
    """
    # fastText 返回格式: '__label__zh' 或 '__label__ja'
    lang = label.replace('__label__', '')
    
    # 映射表：fastText 标签 → 项目语言代码
    lang_map = {
        'ja': 'ja',  # 日语
        'zh': 'zh',  # 中文
        'en': 'en',  # 英语
    }
    
    return lang_map.get(lang, 'en')  # 默认英语


def _detect_language_fasttext(text: str) -> str:
    """
    使用 fastText AI 模型检测语言
    
    Args:
        text: 输入文本
    
    Returns:
        项目语言代码 ('zh' | 'ja' | 'en')
    """
    # 预处理：移除换行符和多余空格
    text = text.replace('\n', ' ').strip()
    
    if not text:
        return 'en'
    
    # 使用 fastText 检测
    model = _get_language_model()
    predictions = model.predict(text, k=1)  # k=1: 只返回最可能的语言
    
    # predictions[0] 是 label 列表，predictions[1] 是置信度列表
    label = predictions[0][0]
    confidence = float(predictions[1][0])
    
    # 映射到项目语言代码
    lang_code = _fasttext_to_project_lang(label)
    
    return lang_code


def _scores_fasttext(text: str) -> Dict[str, float]:
    """
    使用 fastText 计算语言置信度分数
    
    Args:
        text: 输入文本
    
    Returns:
        语言置信度字典 {'zh': 0.x, 'ja': 0.y, 'en': 0.z'}
    """
    text = text.replace('\n', ' ').strip()
    
    if not text:
        return {"zh": 1/3, "ja": 1/3, "en": 1/3}
    
    # 获取前 3 个最可能的语言
    model = _get_language_model()
    predictions = model.predict(text, k=3)
    
    # 初始化分数
    scores = {"zh": 0.0, "ja": 0.0, "en": 0.0}
    
    # 填充分数 (累加，防止被后续低置信度的同类覆盖)
    # 例如：如果 return 'de' (mappted to en) 和 'en'，应该累加
    for label, conf in zip(predictions[0], predictions[1]):
        lang_code = _fasttext_to_project_lang(label)
        scores[lang_code] += float(conf)
    
    return scores


def _is_pure_han(text: str) -> bool:
    """检查是否为纯汉字文本（忽略标点和空格）"""
    has_han = False
    for ch in text:
        if S_HAN.match(ch):
            has_han = True
        elif not (S_PUNC.match(ch) or ch.isspace()):
            # 发现非汉字、非标点、非空格的字符
            return False
    return has_han


def _scores_with_context(
    chunk: str,
    start: int,
    end: int,
    full_text: str,
    context_window: int = 15
) -> Dict[str, float]:
    """
    带上下文的语言检测（fastText AI 版本）
    
    对于纯汉字片段，扩展上下文以帮助 AI 判断。
    fastText 模型本身已经很智能，但提供上下文仍能提升准确度。
    
    Args:
        chunk: 当前片段文本
        start: 片段在全文中的起始位置
        end: 片段在全文中的结束位置
        full_text: 完整文本
        context_window: 上下文扩展窗口大小（字符数）
    
    Returns:
        语言置信度分数字典
    """
    if not _is_pure_han(chunk):
        # 非纯汉字，直接检测
        return _scores_fasttext(chunk)
    
    # 纯汉字片段：检查上下文是否有假名
    ctx_start = max(0, start - context_window)
    ctx_end = min(len(full_text), end + context_window)
    context = full_text[ctx_start:ctx_end]
    
    # 如果上下文包含假名，很可能是日语
    if _contains_kana(context):
        return _scores_fasttext(context)
    else:
        # 否则直接检测片段本身
        return _scores_fasttext(chunk)


@dataclass
class _Span:
    """内部分段表示"""

    start: int
    end: int  # 半开区间 [start, end)
    text: str
    lang: str  # 'zh' | 'ja' | 'en'
    scores: Dict[str, float]


def _merge_adjacent(spans: List[_Span], max_len: int = 200) -> List[_Span]:
    """
    合并相邻的同语言段落
    
    Args:
        spans: 片段列表
        max_len: 合并后的最大长度限制（防止过长导致 WhisperX 对齐失败）
    """
    out: List[_Span] = []
    for sp in spans:
        if out and out[-1].lang == sp.lang and out[-1].end == sp.start:
            # 检查合并后长度是否超过限制
            if len(out[-1].text) + len(sp.text) <= max_len:
                a = out[-1]
                L1, L2 = a.end - a.start, sp.end - sp.start
                total = L1 + L2 or 1
                mix = {k: (a.scores[k] * L1 + sp.scores[k] * L2) / total for k in a.scores}
                out[-1] = _Span(a.start, sp.end, a.text + sp.text, sp.lang, mix)
                continue
        
        # 无法合并（语言不同、不连续、或超长），添加为新段
        out.append(sp)
    return out


def _contains_kana(s: str) -> bool:
    """检查是否包含假名"""
    return bool(HIRA_OR_KATA.search(s))


def _smooth_ja_adhesion(
    spans: List[_Span], text: str, max_zh_len: int = 3, kana_pull: bool = True
) -> List[_Span]:
    """
    日语粘合修正
    
    - ja-zh-ja 夹心：将短中文段改为日文
    - 假名牵引：短中文段邻接含假名的日文段时改为日文
    """
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
                fixed[i] = _Span(mid.start, mid.end, mid.text, "ja", _scores_fasttext(text[L:R]))

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
                fixed[i] = _Span(sp.start, sp.end, sp.text, "ja", _scores_fasttext(text[L:R]))
    return fixed


def _segment_text_pure(
    text: str, max_zh_len: int = 6, kana_pull: bool = True
) -> List[_Span]:
    """
    执行完整的文本分段流程（两级分段策略）
    
    策略：
    1. **第一级**：标点符号分割（句子边界）
       → 避免跨句子的语言误判
    2. **第二级**：脚本切换 + AI 检测（句内多语言）
       → fastText 处理句内的中英日混合
    
    Args:
        text: 输入文本
        max_zh_len: 日语粘合修正的最大中文段长度
        kana_pull: 是否启用假名牵引修正
    """
    # ========== 第一级：标点符号分割 (Aggressive Punctuation Split) ==========
    # 用户要求：包含 , ， 、 等所有逗号
    chunk_segments = _split_by_punctuation(text)
    
    all_spans: List[_Span] = []
    
    # 对每个标点切分的大块直接进行语言检测 (Step 2: Language Detection Priority)
    for chunk_start, chunk_end in chunk_segments:
        chunk_text = text[chunk_start:chunk_end]
        
        if not chunk_text.strip():
            continue

        # --- 特殊处理：纯标点符号 ---
        # 激进分段会产生独立的标点块（如 ","），单独发给 TTS 会报错
        # 策略：将标点合并到前一个分段中
        if all(S_PUNC.match(ch) for ch in chunk_text):
            if all_spans:
                # 合并到前一个段
                all_spans[-1].text += chunk_text
                all_spans[-1].end = chunk_end  # 扩展结束位置
            else:
                # 文本开头的标点？忽略或暂存等待后续合并
                # 既然 TTS 讨厌标点，忽略开头的标点是安全的
                pass
            continue

        # 计算位置
        abs_start = chunk_start
        abs_end = chunk_end
        
        # --- 策略：Heuristic Detection (Kana > Han > Other) ---
        # 针对整个标点片段进行判断，保证 "距離も" 这种 "汉字+假名" 组合被整体识别为日语
        
        lang = "en"
        scores = {"zh": 0.0, "ja": 0.0, "en": 0.0}
        
        if _contains_kana(chunk_text):
            # Priority 1: IF contains Japanese Kana -> JA
            lang = "ja"
            scores["ja"] = 1.0
        elif S_HAN.search(chunk_text):
            # Priority 2: IF contains Hanzi but NO Kana -> ZH
            lang = "zh"
            scores["zh"] = 1.0
        else:
            # Priority 3: ELSE -> EN/Other (Recourse to simple script check or fastText)
            # 简单起见，如果含拉丁字母则 EN，否则尝试 fastText
            if S_LATIN.search(chunk_text):
                lang = "en"
                scores["en"] = 1.0
            else:
                s = _scores_fasttext(chunk_text)
                lang = max(s.items(), key=lambda kv: kv[1])[0]
                scores = s

        all_spans.append(_Span(abs_start, abs_end, chunk_text, lang, scores))
    
    # 3. 全局合并
    merged_spans = _merge_adjacent(all_spans)
    
    # 4. 日语粘合修正 (此时主要用于修正被标点切断的短汉字段，如果有上下文支持的话)
    final_spans = _smooth_ja_adhesion(merged_spans, text, max_zh_len, kana_pull)
    
    # 5. 最后再一次合并
    return _merge_adjacent(final_spans)


# ---------- 公开 API ----------
@log_function()
def segment_text(payload: TextIn) -> TextSegmentsOut:
    """
    将输入文本按语言分段
    
    Args:
        payload: 包含待分段文本的输入
    
    Returns:
        分段结果，包含各段的文本、语言代码、起止位置
    """
    text = payload.text
    spans = _segment_text_pure(text)
    
    # 过滤掉空白片段（防止 TTS 引擎报错）
    spans = [s for s in spans if s.text.strip()]
    
    # 转换为输出格式
    segments = [
        SegmentItem(
            text=sp.text,
            langcode=sp.lang,  # sp.lang 已经是 'zh'/'ja'/'en' 字符串
            start=sp.start,
            end=sp.end,
        )
        for sp in spans
    ]
    
    # 提取包含的语言列表（去重并保持顺序）
    seen = set()
    contain_lang = []
    for seg in segments:
        if seg.langcode not in seen:
            seen.add(seg.langcode)
            contain_lang.append(seg.langcode)
    
    return TextSegmentsOut(segments=segments, contain_lang=contain_lang)
