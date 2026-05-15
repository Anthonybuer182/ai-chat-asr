"""
Live2D：从 model_dict.json 推导 LLM 可用情绪标签；清洗 / 剔除非法标签；
与 websocket TTS、「仅朗读净文本」、前端打字机带标签通路配合。
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_DICT_BASENAME = "model_dict.json"

# 模型未选定 / 字典无条目时的兜底关键词（与原静态提示基本一致，并含 UI 会话态）
DEFAULT_EMOTION_KEYS: FrozenSet[str] = frozenset({
    "neutral", "happy", "sad", "angry", "surprised", "thinking", "think",
    "smile", "blushing", "greet", "agree", "disagree", "scared", "excited",
    "question", "cheer", "disgust", "shake",
    "speak", "listen", "wakeup",
})

EMOTION_TAG_PATTERN = re.compile(r"\[EMOTION:(\w+)\]", re.IGNORECASE)

LIVE2D_SYSTEM_PROMPT_BASE = """你是一个可爱的虚拟助手，拥有丰富的情感和肢体语言表达能力。

## 输出规范概要
请在中文回复中按需插入形如 [EMOTION:类型] 的标签以驱动虚拟形象的表演。
- **类型名必须一字不差地使用本消息后半部分「允许的 [EMOTION:类型]」列表中的英文标识**。
- 标签内不要空格，例如 [EMOTION:happy]，不要写成 [EMOTION: happy]
- 一般每有一句情绪转折放一个标签即可，不要堆砌；标签常放在句首，后接正文。
"""


def clear_model_dict_cache() -> None:
    """调试或热替换 model_dict.json 后可调用"""
    _load_model_dict_raw.cache_clear()


@lru_cache(maxsize=1)
def _load_model_dict_raw() -> Dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, _MODEL_DICT_BASENAME)
    if not os.path.isfile(path):
        logger.warning("未找到 model_dict.json: %s", path)
        return {"models": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("models"), dict):
            return {"models": {}}
        return data
    except Exception as e:
        logger.error("读取 model_dict.json 失败: %s", e)
        return {"models": {}}


def emotion_keys_and_meta_for_model(model_key: Optional[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    返回 (有序 key 列表, key -> description)，不含 path；
    无模型或未命中字典时用 DEFAULT_EMOTION_KEYS。
    """
    fb_keys = sorted(DEFAULT_EMOTION_KEYS, key=lambda x: x.lower())
    fb_meta = {k: k for k in fb_keys}

    if not model_key or not str(model_key).strip():
        return fb_keys, fb_meta

    mk = str(model_key).strip()
    raw = _load_model_dict_raw()
    block = (raw.get("models") or {}).get(mk)
    if not isinstance(block, dict):
        logger.info("Live2D 标识未在 model_dict.json 中找到: %s ，使用预设标签集合", mk)
        return fb_keys, fb_meta

    keys: List[str] = []
    meta: Dict[str, str] = {}
    for k, v in block.items():
        if k == "path" or not isinstance(v, dict):
            continue
        keys.append(k)
        d = v.get("description")
        meta[k] = str(d).strip() if d else k

    if not keys:
        logger.warning("模型 %s 在 model_dict 中无可用情绪条目，退回预设列表", mk)
        return fb_keys, fb_meta

    keys.sort(key=lambda x: x.lower())
    return keys, meta


def allowed_emotions_frozenset(model_key: Optional[str]) -> FrozenSet[str]:
    keys, _ = emotion_keys_and_meta_for_model(model_key)
    return frozenset(keys)


def build_live2d_model_instruction(model_key: Optional[str]) -> str:
    """追加在 LIVE2D_SYSTEM_PROMPT_BASE 之后的、与当前模型绑定的指令块。"""
    keys, meta = emotion_keys_and_meta_for_model(model_key)
    mk = (model_key or "").strip()

    header = "## 允许的 [EMOTION:类型]\n\n"
    if mk:
        header += (
            f"当前虚拟形象绑定配置标识为：**{mk}**。"
            "只可使用下列类型名驱动其动作 / 表情资源；禁止使用列表以外的名称。\n\n"
        )
    else:
        header += "尚未绑定具体模型条目，可使用下列通用类型名。\n\n"

    lines = [header, "逐项说明（中英文标签名必须与左栏一致）："]
    for k in keys:
        lines.append(f"- [EMOTION:{k}] — {meta.get(k, k)}")

    lines.extend(
        [
            "",
            "### 简短示例（类型名必须与上表一致）",
            "用户：我有点难过。",
            "助手：[EMOTION:sad] 先抱抱你，要不要说说发生了什么？",
        ]
    )
    return "\n".join(lines)


def sanitize_emotion_tags(text: str, allowed: FrozenSet[str]) -> str:
    """
    将 [EMOTION:xxx] 规范为词典中的大小写写法；不允许的标签整段移除。
    """
    if not text:
        return ""
    if not allowed:
        allowed = DEFAULT_EMOTION_KEYS

    canon: Dict[str, str] = {}
    for a in allowed:
        canon.setdefault(a.lower(), a)

    def repl(m: re.Match) -> str:
        raw = (m.group(1) or "").lower()
        c = canon.get(raw)
        if not c:
            return ""
        return f"[EMOTION:{c}]"

    return EMOTION_TAG_PATTERN.sub(repl, text)


def strip_emotion_tags(text: str) -> str:
    if not text:
        return ""
    return EMOTION_TAG_PATTERN.sub("", text).strip()


def first_emotion_key_from_tagged(text: str) -> Optional[str]:
    """从仍含 [EMOTION:…] 的文本中取首个标签的类型名（大小写与文中一致）。"""
    m = EMOTION_TAG_PATTERN.search(text or "")
    return m.group(1) if m else None


def get_model_emotion_entry(model_key: Optional[str], emotion_key: str) -> Optional[Dict[str, Any]]:
    """读取 model_dict 中某模型下某情绪条目的 dict（不含顶层 path）。"""
    if not model_key or not emotion_key:
        return None
    mk = str(model_key).strip()
    ek = str(emotion_key).strip()
    if not mk or not ek:
        return None
    raw = _load_model_dict_raw()
    block = (raw.get("models") or {}).get(mk)
    if not isinstance(block, dict):
        return None
    v = block.get(ek)
    if isinstance(v, dict):
        return v
    ekl = ek.lower()
    for k, val in block.items():
        if k == "path" or not isinstance(val, dict):
            continue
        if k.lower() == ekl:
            return val
    return None


def clean_llm_reply_for_history(reply: str, allowed: FrozenSet[str]) -> str:
    """写入多轮上下文：去掉非法标记后剥掉全部 EMOTION，避免污染后续对话。"""
    return strip_emotion_tags(sanitize_emotion_tags(reply or "", allowed))


def parse_current_live2d_model_key_from_settings(settings_obj: Any) -> Optional[str]:
    """从 pydantic Settings 的 CURRENT_LIVE2D_MODEL（JSON）解析 name。"""
    raw = (getattr(settings_obj, "CURRENT_LIVE2D_MODEL", "") or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            name = obj.get("name")
            return str(name).strip() if name else None
    except json.JSONDecodeError:
        return raw
    return None
