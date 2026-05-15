"""
MiniMax speech-2.8-hd / speech-2.8-turbo：根据当前句 [EMOTION:] 注入官方允许的句内语气词。

- 全局默认映射 + model_dict 每条情绪可选字段 minimax_tag 覆盖（合法值须为白名单之一）。
- 若朗读文本已含任一合法语气词则不再注入（去重）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Optional

from live2d_prompt import first_emotion_key_from_tagged, get_model_emotion_entry

# 官方支持的语气词（与文档一致；用于白名单与正则检测）
MINIMAX_INTERJECTION_INNER_NAMES: FrozenSet[str] = frozenset(
    {
        "laughs",
        "chuckle",
        "coughs",
        "clear-throat",
        "groans",
        "breath",
        "pant",
        "inhale",
        "exhale",
        "gasps",
        "sniffs",
        "sighs",
        "snorts",
        "burps",
        "lip-smacking",
        "humming",
        "hissing",
        "emm",
        "sneezes",
    }
)

_INTERJECTION_ALT = "|".join(re.escape(x) for x in sorted(MINIMAX_INTERJECTION_INNER_NAMES, key=len, reverse=True))
INTERJECTION_ANY_RE = re.compile(rf"\(({_INTERJECTION_ALT})\)", re.IGNORECASE)

_INNER_LOWER_TO_CANONICAL_PARENS: Dict[str, str] = {
    name.lower(): f"({name})" for name in MINIMAX_INTERJECTION_INNER_NAMES
}

# emotion key（小写）→ 默认语气词；未列出表示不自动注入（避免每句都换气）
DEFAULT_EMOTION_TO_MINIMAX_INTERJECTION: Dict[str, str] = {
    "happy": "(chuckle)",
    "smile": "(chuckle)",
    "blushing": "(chuckle)",
    "greet": "(chuckle)",
    "agree": "(emm)",
    "excited": "(laughs)",
    "cheer": "(laughs)",
    "sad": "(sighs)",
    "disagree": "(sighs)",
    "question": "(emm)",
    "thinking": "(emm)",
    "think": "(emm)",
    "surprised": "(gasps)",
    "scared": "(gasps)",
    "angry": "(hissing)",
    "disgust": "(snorts)",
    "shake": "(sighs)",
}


def minimax_model_supports_text_interjection(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return m in ("speech-2.8-hd", "speech-2.8-turbo")


def normalize_interjection_tag(raw: str) -> Optional[str]:
    """接受 '(laughs)' 或 'laughs'；非法或空串返回 None。显式空表示不写语气词。"""
    v = (raw or "").strip()
    if not v:
        return None
    if v.startswith("(") and v.endswith(")"):
        inner = v[1:-1].strip().lower()
    else:
        inner = v.lower()
    return _INNER_LOWER_TO_CANONICAL_PARENS.get(inner)


def text_has_minimax_interjection(text: str) -> bool:
    return bool(INTERJECTION_ANY_RE.search(text or ""))


def resolve_interjection_for_emotion(emotion: Optional[str], live2d_model_key: Optional[str]) -> Optional[str]:
    """优先 model_dict 中该情绪的 minimax_tag，否则用全局默认表。"""
    if not emotion:
        return None
    entry = get_model_emotion_entry(live2d_model_key, emotion)
    if entry and "minimax_tag" in entry:
        override = entry.get("minimax_tag")
        if override is None:
            return None
        if isinstance(override, str):
            # 显式 ""：本条情绪不要注入语气词
            if override.strip() == "":
                return None
            normalized = normalize_interjection_tag(override)
            if normalized:
                return normalized
            # 非法写法：回退默认表，避免配置 typo 导致完全无声效

    key = emotion.strip().lower()
    return DEFAULT_EMOTION_TO_MINIMAX_INTERJECTION.get(key)


def apply_minimax_emotion_interjection(
    tts_plain: str,
    cleaned_tags: str,
    live2d_model_key: Optional[str],
    settings_obj: Any,
) -> str:
    """
    在满足开关与 TTS 模型版本时，按 cleaned_tags 中首个 [EMOTION:] 向 MiniMax 朗读文本句首注入语气词。
    """
    text = (tts_plain or "").strip()
    if not text:
        return text

    if not getattr(settings_obj, "MINIMAX_EMOTION_INTERJECTION_ENABLED", True):
        return text

    model_name = getattr(settings_obj, "MINIMAX_TTS_MODEL", "") or ""
    if not minimax_model_supports_text_interjection(model_name):
        return text

    if text_has_minimax_interjection(text):
        return text

    emotion = first_emotion_key_from_tagged(cleaned_tags or "")
    tag = resolve_interjection_for_emotion(emotion, live2d_model_key)
    if not tag:
        return text

    return f"{tag} {text}".strip()
