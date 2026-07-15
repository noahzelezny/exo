"""System messages render IN PLACE, with consolidation as fallback.

consolidate_system_messages hoists every system message into one top block.
That silently defeats KV prefix caching for clients that deliberately put
volatile blocks (memory recall, supervisor nudges) in the message TAIL:
hoisting rewrites the top of the prompt each turn, so the pool's common
prefix collapses to the static system text and every chat turn re-prefills
from scratch (measured live on Scout: 26–40s per turn on Llama-4-Scout,
chunked-miss at exactly system+memory-block length).

render_chat_template now tries the client's message order first and falls
back to consolidation only when the template raises on mid-conversation
system messages or silently drops one.
"""

from typing import Any

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.utils_mlx import (
    _system_content_survived,
    _system_messages_in_place,
    render_chat_template,
)

PARAMS = TextGenerationTaskParams(model="test-org/test-model", input=[])


class InPlaceTok:
    """Renders messages in the order given (llama4/qwen-style)."""

    chat_template = "in-place"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, tools=None, **kw):
        out = "".join(
            f"<{m['role']}>{m.get('content', '')}</{m['role']}>" for m in messages
        )
        return out + "<assistant>"


class RaisingTok:
    """Raises on any system message past index 0 (strict template)."""

    chat_template = "strict"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, tools=None, **kw):
        for i, m in enumerate(messages):
            if m["role"] == "system" and i > 0:
                raise ValueError("System role only supported as first message")
        return "".join(f"<{m['role']}>{m.get('content', '')}" for m in messages)


class DroppingTok:
    """Silently renders only a LEADING system message (drops later ones)."""

    chat_template = "dropping"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, tools=None, **kw):
        parts = []
        for i, m in enumerate(messages):
            if m["role"] == "system" and i > 0:
                continue
            parts.append(f"<{m['role']}>{m.get('content', '')}")
        return "".join(parts)


CONVO: list[dict[str, Any]] = [
    {"role": "system", "content": "MAIN_SYSTEM"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "system", "content": "VOLATILE_MEMORY_BLOCK"},
    {"role": "user", "content": "question"},
]


def test_tail_system_message_stays_in_place():
    prompt = render_chat_template(InPlaceTok(), list(CONVO), PARAMS)
    assert prompt.index("VOLATILE_MEMORY_BLOCK") > prompt.index("hi")


def test_appending_tail_system_message_preserves_prompt_prefix():
    # THE property the KV pool needs: adding a tail message must extend the
    # rendered prompt, never rewrite its beginning.
    tok = InPlaceTok()
    before = render_chat_template(tok, list(CONVO), PARAMS)
    nudged = list(CONVO) + [{"role": "system", "content": "NUDGE"}]
    after = render_chat_template(tok, nudged, PARAMS)
    common = before.removesuffix("<assistant>")
    assert after.startswith(common)


def test_strict_template_falls_back_to_consolidation():
    prompt = render_chat_template(RaisingTok(), list(CONVO), PARAMS)
    # Consolidated: both system contents merged into the single top block.
    assert prompt.startswith("<system>MAIN_SYSTEM\nVOLATILE_MEMORY_BLOCK")
    assert prompt.count("<system>") == 1


def test_silently_dropping_template_falls_back_to_consolidation():
    prompt = render_chat_template(DroppingTok(), list(CONVO), PARAMS)
    # Without the survival check the memory block would be LOST; the fallback
    # consolidates it into the leading system message instead.
    assert "VOLATILE_MEMORY_BLOCK" in prompt


def test_developer_role_maps_to_system_in_place():
    msgs = [
        {"role": "developer", "content": "DEV_RULES"},
        {"role": "user", "content": "hello"},
    ]
    prompt = render_chat_template(InPlaceTok(), msgs, PARAMS)
    assert "<system>DEV_RULES" in prompt
    assert "<developer>" not in prompt


def test_trailing_assistant_prefill_appended_raw():
    msgs = list(CONVO) + [{"role": "assistant", "content": "partial answer"}]
    prompt = render_chat_template(InPlaceTok(), msgs, PARAMS)
    assert prompt.endswith("<assistant>partial answer")
    # the partial turn must not have been templated as a closed message
    assert "<assistant>partial answer</assistant>" not in prompt


def test_empty_system_messages_dropped_in_place():
    msgs = [
        {"role": "system", "content": "MAIN"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": ""},
    ]
    assert [m["role"] for m in _system_messages_in_place(msgs)] == [
        "system", "user",
    ]


def test_survival_check_is_trim_tolerant():
    msgs = [{"role": "system", "content": "  PADDED  "}]
    assert _system_content_survived(msgs, "<system>PADDED<user>hi")
    assert not _system_content_survived(msgs, "<user>hi")
