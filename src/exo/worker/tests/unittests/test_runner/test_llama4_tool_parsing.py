"""Llama 4 tool-call parsing + stop tokens.

Llama 4 emits tool calls as <|python_start|>{json}<|python_end|> with
{"type": "function", "name": ..., "parameters": {...}} payloads, then <|eom|>.
Observed live on Llama-4-Scout-17B (2026-07-15): with no parser the markup
streamed to the client as raw text, and without <|eom|> in the stop set the
model ran away re-emitting the same call as fresh assistant turns.
"""

import json
from collections.abc import Generator

from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.utils_mlx import get_eos_token_ids_for_model
from exo.worker.runner.llm_inference.model_output_parsers import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import make_llama4_parser

CALL = '{"type": "function", "name": "get_transcript", "parameters": {"session": "latest"}}'


def test_parses_wrapped_call():
    parser = make_llama4_parser()
    items = parser.parse(f"<|python_start|>{CALL}<|python_end|>", tools=None)
    assert items is not None and len(items) == 1
    assert items[0].name == "get_transcript"
    assert json.loads(items[0].arguments) == {"session": "latest"}


def test_parses_bare_json_between_markers_stripped_by_stream():
    # The streaming extractor may hand over text without the marker tokens.
    parser = make_llama4_parser()
    items = parser.parse(CALL, tools=None)
    assert items is not None and items[0].name == "get_transcript"


def test_parses_list_of_calls():
    parser = make_llama4_parser()
    payload = f"[{CALL}, {CALL}]"
    items = parser.parse(f"<|python_start|>{payload}<|python_end|>", tools=None)
    assert items is not None and len(items) == 2


def test_arguments_key_variant_accepted():
    parser = make_llama4_parser()
    items = parser.parse(
        '<|python_start|>{"name": "f", "arguments": {"x": 1}}<|python_end|>',
        tools=None,
    )
    assert items is not None and json.loads(items[0].arguments) == {"x": 1}


def test_garbage_returns_none_not_raise():
    parser = make_llama4_parser()
    assert parser.parse("<|python_start|>not json<|python_end|>", tools=None) is None
    assert (
        parser.parse('<|python_start|>{"no_name": 1}<|python_end|>', tools=None) is None
    )


def test_marker_strings_drive_stream_extraction():
    parser = make_llama4_parser()
    assert parser.start_parsing == "<|python_start|>"
    assert parser.end_parsing == "<|python_end|>"


def test_llama4_eos_includes_eom_and_eot():
    ids = get_eos_token_ids_for_model(
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-8bit"
    )
    assert ids is not None
    # 200007 <|eom|> is the tool handoff; 200008 <|eot|> ends normal turns
    assert 200007 in ids and 200008 in ids


# --- streaming extraction: llama4 emits calls wrapped AND as bare JSON -------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_transcript",
            "parameters": {
                "type": "object",
                "properties": {"session": {"type": "string"}},
            },
        },
    }
]


def _stream(texts: list[str]) -> Generator[GenerationResponse]:
    for i, text in enumerate(texts):
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if i == len(texts) - 1 else None,
            usage=None,
        )


def test_stream_wrapped_call_yields_tool_calls():
    parser = make_llama4_parser()
    out = list(
        parse_tool_calls(
            _stream(["<|python_start|>", CALL, "<|python_end|>"]), parser, _TOOLS
        )
    )
    tcs = [r for r in out if isinstance(r, ToolCallResponse)]
    assert len(tcs) == 1 and tcs[0].tool_calls[0].name == "get_transcript"


def test_stream_bare_json_call_yields_tool_calls():
    # observed live from a tensor-parallel instance: no markers, just JSON + eos
    parser = make_llama4_parser()
    chunks = [
        '{"',
        "name",
        '": "get_transcript", "parameters":',
        ' {"session": "latest"}}',
    ]
    out = list(parse_tool_calls(_stream(chunks), parser, _TOOLS))
    tcs = [r for r in out if isinstance(r, ToolCallResponse)]
    assert len(tcs) == 1
    assert tcs[0].tool_calls[0].name == "get_transcript"
    assert json.loads(tcs[0].tool_calls[0].arguments) == {"session": "latest"}


def test_stream_json_looking_prose_delivered_as_text_not_error():
    parser = make_llama4_parser()
    chunks = ["{note to self: not json}", " trailing thought"]
    out = list(parse_tool_calls(_stream(chunks), parser, _TOOLS))
    texts = [r for r in out if isinstance(r, GenerationResponse)]
    assert len(texts) == 1
    assert texts[0].text == "{note to self: not json} trailing thought"
    assert texts[0].finish_reason == "stop"  # NOT "error"


def test_stream_prose_not_sniffed_streams_through():
    parser = make_llama4_parser()
    chunks = ["Hello", " there {braces mid-text} fine"]
    out = list(parse_tool_calls(_stream(chunks), parser, _TOOLS))
    assert all(isinstance(r, GenerationResponse) for r in out)
    assert len(out) == 2  # streamed chunk-by-chunk, never withheld


def test_stream_no_tools_disables_bare_sniff():
    parser = make_llama4_parser()
    chunks = ['{"name": "x", "parameters": {}}']
    out = list(parse_tool_calls(_stream(chunks), parser, None))
    assert all(isinstance(r, GenerationResponse) for r in out)
