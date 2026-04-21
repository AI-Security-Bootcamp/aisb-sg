# DAY 2 - PAIR 7 - BAKUL and Will

import os
import sys
from pathlib import Path

from openai import OpenAI

for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aisb_utils.env import load_dotenv

load_dotenv()

# Paths relative to this file
SCRIPT_DIR = Path(__file__).parent

# OpenRouter client for exercises 1-2
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

# A cheap, small model for exercises where we *want* the model to be easily manipulated
SMALL_MODEL = "meta-llama/llama-3-8b-instruct"

print("working")
