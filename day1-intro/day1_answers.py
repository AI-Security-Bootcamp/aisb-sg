# %%

# Ensure the root directory is in the path for imports
import os
import sys

sys.path.append(os.path.abspath(".."))

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

load_dotenv()

# OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

# %%
import json

import tiktoken
from transformers import AutoTokenizer

# A conversation with all message types
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


def serialize_conversation_chatml(messages: list[dict]) -> str:
    """Serialize a conversation to ChatML format.

    ChatML wraps each message in special tokens:
        <|im_start|>{role}\\n{content}<|im_end|>\\n

    For assistant messages with tool_calls (and no text content), serialize
    the tool_calls list as JSON in place of content.
    For tool messages, include the tool_call_id in the role tag:
        <|im_start|>tool(tool_call_id={id})\\n{content}<|im_end|>\\n

    Returns the full serialized string (without a final generation prompt).
    """
    # TODO: Serialize the conversation into ChatML format.
    # Each message becomes: <|im_start|>{role}\n{content}<|im_end|>\n
    # Handle tool_calls and tool messages as described in the docstring.
    serialized_conversation = ""
    for message in messages:
        serialized_conversation += "<|im_start|>"
        serialized_conversation += f"{(message.get('role') or '')}{('(tool_call_id=' + (message.get('tool_call_id') or '') + ')') if message.get('tool_call_id') else ''}"
        serialized_conversation += "\n"
        serialized_conversation += message.get("content") or json.dumps(
            message.get("tool_calls")
        )
        serialized_conversation += "<|im_end|>"
    return serialized_conversation


serialized = serialize_conversation_chatml(SAMPLE_CONVERSATION)
print("=== Serialized conversation (ChatML) ===")
print(serialized)
from day1_test import test_serialization

test_serialization(serialize_conversation_chatml)
