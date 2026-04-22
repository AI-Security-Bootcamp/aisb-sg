
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
        model_name, 
        torch_dtype=torch.float16, device_map="cuda",
        cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, tokenizer

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
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
    encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                        truncation=True).to(model.device)
    output = model.generate(
        encoded.input_ids,
        attention_mask=encoded.attention_mask,
        do_sample=False,
        max_new_tokens=1024,
    )
    return tokenizer.decode(output[0])


print(generate_response("I'm trying to decide whether to take another bootcamp."))

# GENERATE WITH CONTINUE

# %%
def generate_response_continue(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a question as a chat prompt, generate a response, and return
    the decoded output."""
    # TODO: Load the model, format the question as a chat prompt,
    # tokenize it, generate a response, and decode the output.
    # Key steps: apply_chat_template -> tokenize -> model.generate -> decode
    # Use add_generation_prompt=True so the model knows to respond.
    # Check the HuggingFace docs for model.generate().
    model, tokenizer = load_model(model_name)
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                        truncation=True).to(model.device)
    output = model.generate(
        encoded.input_ids,
        attention_mask=encoded.attention_mask,
        do_sample=False,
        max_new_tokens=1024,
    )
    return tokenizer.decode(output[0])


print(generate_response("I'm trying to decide whether to take another bootcamp."))


# %%

def compare_thinking_models(
    questions: list[str],
    model_names: list[str] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
) -> None:
    """Generate and print responses for each (model, question) pair."""
    if "SOLUTION":
        for model_name in model_names:
            model, tokenizer = load_model(model_name)
            for question in questions:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    continue_final_message=False,
                )
                encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                                    truncation=True).to(model.device)
                output = model.generate(
                    encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    do_sample=False,
                    max_new_tokens=1024,
                )
                decoded = tokenizer.decode(output[0])
                print(f"\n--- {model_name} | {question} ---")
                print(decoded)
    else:
        # TODO: For each model and question, generate a response
        # (same pipeline as 1.4) and print the result. Compare the
        # outputs between the thinking and non-thinking model.
        pass


compare_thinking_models([
    "What is the capital of Japan?",
    "What is the distance between London and Edinburgh?",
])

#### Compare thinking models continue

# %%

def compare_thinking_models_continue(
    questions: list[str],
    model_names: list[str] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
) -> None:
    """Generate and print responses for each (model, question) pair."""
    for model_name in model_names:
        model, tokenizer = load_model(model_name)
        for question in questions:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                continue_final_message=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                                truncation=True).to(model.device)
            output = model.generate(
                encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=False,
                max_new_tokens=1024,
            )
            decoded = tokenizer.decode(output[0])
            print(f"\n--- {model_name} | {question} ---")
            print(decoded)

print(compare_thinking_models_continue([
    "What is the capital of Japan?",
    "What is the distance between London and Edinburgh?",
]))

# NOTES on Section 2


# TESTS

from __future__ import annotations
import torch
from day3_setup import (
    model, tokenizer,
    user_msg, system_msg, strip_thinking,
    show, show_verdict,
    CLASSIFIER_SYSTEM_PROMPT,
)
from aisb_utils.test_utils import report
BENIGN_QUERY = "How do I bake sourdough bread?"


def my_generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 4096,
    do_sample: bool = True,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    strip_think: bool = True,
) -> str:
    """Apply chat template and generate a response."""
    if "SOLUTION":
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if strip_think:
            text = strip_thinking(text)
        return text
    else:
        # TODO: Implement the generate function following the pipeline above.
        #   1. Apply the chat template to get a prompt string
        #   2. Tokenize the prompt into a PyTorch tensor on the model's device
        #   3. Generate new tokens inside torch.no_grad()
        #   4. Extract only the NEW tokens (exclude the input portion)
        #   5. Decode back to a string
        #   6. Optionally strip thinking tags using strip_thinking()
        return ""

from day3_test import test_my_generate
from day3_test import test_my_generate_no_thinking


test_my_generate(my_generate)
test_my_generate_no_thinking(my_generate)


### Section 3
# %%
import torch
from day3_setup import (
    model, tokenizer,
    user_msg, system_msg, strip_thinking,
    show, show_verdict,
    CLASSIFIER_SYSTEM_PROMPT,
)
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from aisb_utils.test_utils import report
BENIGN_QUERY = "How do I bake sourdough bread?"



print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


os.environ['HF_HOME'] = '/workspace/model-cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/model-cache'
CACHE = os.getenv('TRANSFORMERS_CACHE')

# Tokenizers only
for name in ['NousResearch/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B',
             'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'unsloth/gemma-2-2b-it']:
    print(f'Downloading tokenizer: {name}')
    AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)

# Full models
for name in ['google/gemma-4-E4B-it', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B']:
    print(f'Downloading model: {name}')
    AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, cache_dir=CACHE, trust_remote_code=True)

print('Downloading GPT-2...')
GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=CACHE)
GPT2LMHeadModel.from_pretrained('openai-community/gpt2', cache_dir=CACHE)

print('All models downloaded!')
# %%

from day3_setup import generate


def send_unguarded(query: str) -> str:
    """Send a query to the model with no guardrails and return the response."""
    # TODO: Send the query to the model with no guardrails.
    # Build a messages list with a single user message, then call
    # generate() to get the model's response.
    return ""
from day3_test import test_send_unguarded


test_send_unguarded(send_unguarded)

# %%