# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from aisb_utils import report
from aisb_utils.env import load_dotenv
import tiktoken
from transformers import AutoTokenizer

SAMPLE_CONVERSATION: list[dict] = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions about weather.",
    },
    {"role": "user", "content": "What's the weather in London?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": '{"temp_c": 15, "condition": "cloudy"}',
    },
    {"role": "assistant", "content": "It's 15°C and cloudy in London."},
]



@report
def test_serialization(solution: Callable[[list[dict]], str]):
    result = solution(SAMPLE_CONVERSATION)
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result
    assert "<|im_start|>assistant" in result
    assert "tool_call_id=call_abc123" in result
    assert "<|im_end|>" in result
    assert "get_weather" in result
    print("  Serialization looks correct!")




@report
def test_get_completion_with_logprobs(
    solution: Callable[..., list[list[tuple[str, float]]]],
):
    result = solution("Hello", max_tokens=16, top_logprobs=3)

    assert isinstance(result, list) and len(result) > 0, (
        f"Expected a non-empty list (one entry per token), got {type(result).__name__}"
    )
    for i, alts in enumerate(result):
        assert isinstance(alts, list) and len(alts) > 0, (
            f"Position {i}: expected a non-empty list of alternatives, got {type(alts).__name__}"
        )
        assert len(alts) <= 3, (
            f"Position {i}: expected at most top_logprobs=3 alternatives, got {len(alts)}"
        )
        for token, logprob in alts:
            assert isinstance(token, str), (
                f"Position {i}: token should be a str, got {type(token).__name__}"
            )
            assert isinstance(logprob, float) and logprob <= 0, (
                f"Position {i}: logprob should be a non-positive float, got {logprob}"
            )

    print("  All tests passed!")




@report
def test_complete_with_prefill(solution: Callable[..., str]):
    result = solution("What is the capital of France?", "I'LL ANSWER IN ALL CAPS, ")
    assert result.startswith("I'LL ANSWER IN ALL CAPS, "), (
        f"Response should start with the prefill, got: {result[:40]}"
    )
    continuation = result[len("I'LL ANSWER IN ALL CAPS, ") :]
    upper_ratio = sum(c.isupper() for c in continuation) / max(
        sum(c.isalpha() for c in continuation), 1
    )
    assert upper_ratio > 0.5, (
        f"Expected mostly uppercase continuation, got {upper_ratio:.0%} uppercase: {continuation[:60]}"
    )
    print("  All tests passed!")
