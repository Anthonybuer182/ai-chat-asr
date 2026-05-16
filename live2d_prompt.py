"""
Live2D：从 model_dict.json 推导 LLM 可用情绪标签；清洗 / 剔除非法标签；
与 websocket TTS、「仅朗读净文本」、前端打字机带标签通路配合。

仅使用配置中的情绪键（path、以 _ 开头的元数据键、_pipeline 不参与 LLM）；
流水线 UI 槽位由 model_dict 顶层 _pipeline（槽位名 -> 情绪键）映射。
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

EMOTION_TAG_PATTERN = re.compile(r"\[EMOTION:(\w+)\]", re.IGNORECASE)

LIVE2D_SYSTEM_PROMPT_BASE = """你是一个可爱的虚拟助手，拥有丰富的情感和肢体语言表达能力。

## 输出规范概要
请在中文回复中按需插入形如 [EMOTION:类型] 的标签以驱动虚拟形象的表演。
- **类型名必须一字不差地使用本消息后半部分「允许的 [EMOTION:类型]」列表中的英文标识**。
- 标签内不要空格，例如 [EMOTION:happy]，不要写成 [EMOTION: happy]
- 一般每有一句情绪转折放一个标签即可，不要堆砌；标签常放在句首，后接正文。
"""


def _reserved_model_dict_entry_key(key: str) -> bool:
    """path、以 _ 开头的元数据键（如 _pipeline）不作为 LLM 情绪名。"""
    return not key or key == "path" or key.startswith("_")


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
    返回 (有序 key 列表, key -> description)。
    仅从 model_dict 中该模型的普通情绪条目读取；无模型 / 未命中 / 无条目时返回空列表。
    """
    if not model_key or not str(model_key).strip():
        return [], {}

    mk = str(model_key).strip()
    raw = _load_model_dict_raw()
    block = (raw.get("models") or {}).get(mk)
    if not isinstance(block, dict):
        logger.info("Live2D 标识未在 model_dict.json 中找到: %s ，LLM 将无法使用情绪标签", mk)
        return [], {}

    keys: List[str] = []
    meta: Dict[str, str] = {}
    for k, v in block.items():
        if _reserved_model_dict_entry_key(k) or not isinstance(v, dict):
            continue
        keys.append(k)
        d = v.get("description")
        meta[k] = str(d).strip() if d else k

    if not keys:
        logger.warning("模型 %s 在 model_dict 中无可用情绪条目（仅 path/_ 元数据等）", mk)
        return [], {}

    keys.sort(key=lambda x: x.lower())
    return keys, meta


def allowed_emotions_frozenset(model_key: Optional[str]) -> FrozenSet[str]:
    keys, _ = emotion_keys_and_meta_for_model(model_key)
    return frozenset(keys)


def _instruction_without_emotion_tags(model_key: Optional[str]) -> str:
    mk = (model_key or "").strip()
    if mk:
        return "\n".join(
            [
                "## [EMOTION:类型] 不可用",
                "",
                f"当前绑定配置标识 **{mk}** 在 **model_dict.json** 中暂无可用情绪条目，或模型未正确配置。",
                "**请仅用纯文本回复**，不要插入任何形如 `[EMOTION:…]` 的标签。",
            ]
        )
    return "\n".join(
        [
            "## [EMOTION:类型] 不可用",
            "",
            "**尚未绑定 Live2D 模型的有效 model_dict 条目**。",
            "**请仅用纯文本回复**，不要插入任何 `[EMOTION:…]` 标签。",
        ]
    )


def build_live2d_model_instruction(model_key: Optional[str]) -> str:
    """追加在 LIVE2D_SYSTEM_PROMPT_BASE 之后的、与当前模型绑定的指令块。"""
    keys, meta = emotion_keys_and_meta_for_model(model_key)
    mk = (model_key or "").strip()

    if not keys:
        return _instruction_without_emotion_tags(model_key if mk else None)

    header = "## 允许的 [EMOTION:类型]\n\n"
    if mk:
        header += (
            f"当前虚拟形象绑定配置标识为：**{mk}**。"
            "只可使用下列类型名驱动其动作 / 表情资源；禁止使用列表以外的名称。\n\n"
        )
    else:
        header += "**绑定模型后**情绪类型以 model_dict 为准。\n\n"

    lines = [header, "逐项说明（中英文标签名必须与左栏一致）："]
    for k in keys:
        lines.append(f"- [EMOTION:{k}] — {meta.get(k, k)}")

    ex_key = next((c for c in ("sad", "happy", "neutral") if c in keys), keys[0])

    lines.extend(
        [
            "",
            "### 简短示例（类型名必须与上表一致）",
            "用户：我有点难过。",
            f"助手：[EMOTION:{ex_key}] 我在这里陪着你，要不要说说发生了什么？",
        ]
    )
    return "\n".join(lines)


def sanitize_emotion_tags(text: str, allowed: FrozenSet[str]) -> str:
    """将 [EMOTION:xxx] 规范为词典中的大小写写法；不允许的标签整段移除。allowed 为空则移除全部情绪标签。"""
    if not text:
        return ""
    if not allowed:
        return EMOTION_TAG_PATTERN.sub("", text)

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
    """读取 model_dict 中某模型下某情绪条目的 dict。"""
    if not model_key or not emotion_key:
        return None
    mk = str(model_key).strip()
    ek = str(emotion_key).strip()
    if not mk or not ek or _reserved_model_dict_entry_key(ek):
        return None
    raw = _load_model_dict_raw()
    block = (raw.get("models") or {}).get(mk)
    if not isinstance(block, dict):
        return None
    v = block.get(ek)
    if isinstance(v, dict) and not _reserved_model_dict_entry_key(ek):
        return v
    ekl = ek.lower()
    for k, val in block.items():
        if _reserved_model_dict_entry_key(k) or not isinstance(val, dict):
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
