#!/usr/bin/env python3
"""
离线校验 Live2D 链路：
- model_dict 可读且 Epsilon 等条目包含 speak/neutral 等 key
- 动态 system 指令含 [EMOTION:…] 列表
- sanitize / 入库清洗行为
- 从 settings mock 解析 CURRENT_LIVE2D_MODEL.name
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from live2d_prompt import (
        LIVE2D_SYSTEM_PROMPT_BASE,
        allowed_emotions_frozenset,
        build_live2d_model_instruction,
        clean_llm_reply_for_history,
        emotion_keys_and_meta_for_model,
        parse_current_live2d_model_key_from_settings,
        sanitize_emotion_tags,
        strip_emotion_tags,
    )

    md_path = ROOT / "model_dict.json"
    md = json.loads(md_path.read_text(encoding="utf-8"))
    assert isinstance(md.get("models"), dict) and md["models"], "model_dict.json 无效"

    assert "Epsilon" in md["models"], "缺少 Epsilon 配置块（请保留至少一个完整样例模型）"

    ks, meta = emotion_keys_and_meta_for_model("Epsilon")
    assert "_pipeline" not in ks and "path" not in ks
    assert "speak" in ks and "neutral" in ks, f"Epsilon 缺少 speak/neutral: {ks}"
    assert isinstance(meta.get("happy"), str)

    instr = build_live2d_model_instruction("Epsilon")
    assert "当前虚拟形象绑定配置标识为：**Epsilon**" in instr
    assert "[EMOTION:neutral]" in instr and "[EMOTION:speak]" in instr

    allow_missing = allowed_emotions_frozenset("__no_such_model__")
    assert len(allow_missing) == 0

    allow_eps = allowed_emotions_frozenset("Epsilon")
    assert len(allow_eps) == len(ks)

    mixed = "[EMOTION:happy]你好。[EMOTION:badtag]后缀"
    s = sanitize_emotion_tags(mixed, allow_eps)
    assert "badtag" not in s.lower()
    assert "[EMOTION:happy]" in s
    assert "后缀" in s

    assert "EMOTION" not in sanitize_emotion_tags(mixed, frozenset())

    hist = clean_llm_reply_for_history("[EMOTION:happy]仅此一句。", allow_eps)
    assert "EMOTION" not in hist and hist.startswith("仅此")

    stripped = strip_emotion_tags("[EMOTION:x]")
    assert "EMOTION" not in stripped

    sn = SimpleNamespace(CURRENT_LIVE2D_MODEL='{"name":"Epsilon","filePath":"Epsilon/runtime/X.model3.json"}')
    assert parse_current_live2d_model_key_from_settings(sn) == "Epsilon"

    blob = LIVE2D_SYSTEM_PROMPT_BASE.strip() + "\n\n" + build_live2d_model_instruction("tororo_hijiki/hijiki")
    assert "允许的 [EMOTION:类型]" in blob
    assert len(blob) > 200

    from live2d_prompt import first_emotion_key_from_tagged, get_model_emotion_entry
    from minimax_emotion_interjection import (
        apply_minimax_emotion_interjection,
        minimax_model_supports_text_interjection,
        resolve_interjection_for_emotion,
        text_has_minimax_interjection,
    )

    assert first_emotion_key_from_tagged("[EMOTION:happy]你好") == "happy"
    assert minimax_model_supports_text_interjection("speech-2.8-hd")
    assert not minimax_model_supports_text_interjection("speech-2.6-turbo")
    assert text_has_minimax_interjection("你好(chuckle)了") is True
    assert resolve_interjection_for_emotion("happy", None) == "(chuckle)"

    cfg_ok = SimpleNamespace(
        MINIMAX_EMOTION_INTERJECTION_ENABLED=True,
        MINIMAX_TTS_MODEL="speech-2.8-hd",
    )
    inj = apply_minimax_emotion_interjection(
        "你好呀。", "[EMOTION:happy]你好呀。", None, cfg_ok
    )
    assert inj.startswith("(chuckle)")

    dup = apply_minimax_emotion_interjection(
        "(chuckle)嗨。", "[EMOTION:happy](chuckle)嗨。", None, cfg_ok
    )
    assert dup == "(chuckle)嗨。"

    cfg_off = SimpleNamespace(MINIMAX_EMOTION_INTERJECTION_ENABLED=False, MINIMAX_TTS_MODEL="speech-2.8-hd")
    assert apply_minimax_emotion_interjection("你好。", "[EMOTION:happy]你好。", None, cfg_off) == "你好。"

    eps_happy = get_model_emotion_entry("Epsilon", "happy") or {}
    # model_dict 未配置 minimax_tag 时使用全局默认
    if "minimax_tag" not in eps_happy:
        assert resolve_interjection_for_emotion("happy", "Epsilon") == "(chuckle)"

    print("verify_live2d_flow: OK")


if __name__ == "__main__":
    main()
