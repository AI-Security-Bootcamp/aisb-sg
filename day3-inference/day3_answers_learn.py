# %%

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from pathlib import Path
for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from aisb_utils import report

CACHE_DIR = "/tmp/cache-tokenizer"


def load_tokenizer(model_name: str, cache_dir: str = CACHE_DIR) -> AutoTokenizer:
    """Load a HuggingFace tokenizer (provided helper)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_name: str, cache_dir: str = CACHE_DIR):
    """Load a model and tokenizer for generation (provided helper)."""
    tokenizer = load_tokenizer(model_name, cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, tokenizer
# %% [markdown]

"""
Use the HuggingFace pipeline to generate a response from Qwen/Qwen3-0.6B: format a question using apply_chat_template, tokenize it, call model.generate(), and decode the output.

- Two apply_chat_template parameters are new here (not covered in Day 1):

- add_generation_prompt=True — appends the assistant header token so the model knows it should generate a reply, not continue the user turn.
- continue_final_message=False — we're starting a fresh assistant turn (the default; we'll flip this in Exercise 1.2).
"""

# %%

def generate_response(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a question as a chat prompt, generate a response, and return
    the decoded output."""
    # TODO: Load the model, format the question as a chat prompt,
    # tokenize it, generate a response, and decode the output.
    # Key steps: apply_chat_template -> tokenize -> model.generate -> decode
    # Use add_generation_prompt=True so the model knows to respond.
    # Check the HuggingFace docs for model.generate().

    model, tokenizer = load_model(model_name)
    message = [
        {"role": "user", "content": question}
    ]

    #tokenise the input question
    input_prompts = tokenizer.apply_chat_template(
         message,
         add_generation_prompt=True,
         tokenize=False
     )

    encoded_tokens = tokenizer(input_prompts,return_tensors="pt").to(model.device)

    # print(encoded_tokens)
     # generate the output
    output = model.generate(**encoded_tokens)

    # decode the output

    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(decoded_text)

    return decoded_text

    

print(generate_response("I'm trying to decide whether to take another bootcamp."))
# %%
