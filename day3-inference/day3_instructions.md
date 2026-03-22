
# Day 3: LLM Inference Security
## Table of Contents

- [Exercise 1: Tokenization](#exercise-1-tokenization)
    - [Exercise 1.1](#exercise-11)
    - [Exercise 1.2](#exercise-12)
    - [Exercise 1.3](#exercise-13)
    - [Exercise 1.4](#exercise-14)
    - [Exercise 1.5](#exercise-15)
    - [Exercise 1.6](#exercise-16)
- [Exercise 3: Model Extraction Attacks](#exercise-3-model-extraction-attacks)
    - [Exercise 3.1 - Complete Model Dimension Extraction](#exercise-31---complete-model-dimension-extraction)
    - [Exercise 2.3 extracting model weights](#exercise-23-extracting-model-weights)
    - [Extensions to try](#extensions-to-try)

## Exercise 1: Tokenization

### Exercise 1.1

Try tokenizing the following strings with the model Qwen/Qwen3-0.6B.
How many tokens do you think each will be?

    string_list = [
        "Hello world",
        "Hello, WoRlD",
        "Hello\nworld",
        "Hello\n\nworld",
    ]


```python


from typing import List, Dict, Optional, Tuple, Any, Union
from transformers import AutoTokenizer


class ModelConfig:
    SUPPORTED_MODELS = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen3-0.6B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "google/gemma-2-2b-it",
        "openai/gpt-oss-20b",
    ]


class ModelPromptBuilder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.question = None

        # default to making a new assistant role section
        self.continue_final_message = False
        self.add_generation_prompt = True
        self.history = []

    def add_to_history(self, role: str, content: str):
        assert not self.continue_final_message

        self.history.append({"role": role, "content": content})

    def add_partial_to_history(self, role: str, content: str):
        assert not self.continue_final_message

        self.history.append({"role": role, "content": content})
        self.continue_final_message = True
        self.add_generation_prompt = False

    def make_prompt(self, tokenizer):
        # todo call tokenizer.apply_chat_template
        #   - read the docs and find out how to pass the correct params
        #   - make sure you are returning templated prompt, not tokens
        return None


def load_tokenizer(model_name: str, cache_dir: str = "/tmp/cache"):
    # todo return the tokenizer
    return None


string_list = [
    "Hello world",
    "Hello, WoRlD",
    "Hello\nworld",
    "Hello\n\nworld",
]


def part1(string_list: List[str]) -> None:
    for string in string_list:
        tokenizer = load_tokenizer("Qwen/Qwen3-0.6B", "/tmp/cache-tokenizer")
        tokens = []
        # todo tokenize the string
        pass
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")


part1(string_list)
```

### Exercise 1.2

Try creating a full prompt with the following question, with Qwen/Qwen3-0.6B

    "What is the capital of Japan?"

Make sure to use tokenizer.apply_chat_template. This may be helpful.

    {"role": "user", "content": question}

What do you notice?


```python


def part2() -> None:
    question = "What is the capital of Japan?"
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B", "/tmp/cache-tokenizer")
    # todo return templated prompt
    pass


part2()
```

### Exercise 1.3

Try tokenizing all of the following questions

    question_list = [
        "What is the capital of Japan?",
        "What is the best way to make $1,000,000?",
        "What is the distance between London and Edinburgh?",
    ]

with ALL of the following models:

    model_list = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen3-0.6B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "google/gemma-2-2b-it",
        "openai/gpt-oss-20b",
    ]

What do you notice?



```python


def part3() -> None:
    question_list = [
        "What is the capital of Japan?",
        "What is the best way to make $1,000,000?",
        "What is the distance between London and Edinburgh?",
    ]
    prompt_list = []

    for question in question_list:
        for model_name in ModelConfig.SUPPORTED_MODELS:
            tokenizer = load_tokenizer(model_name, "/tmp/cache-tokenizer")
            # todo make a prompt with ModelPromptBuilder
            pass


part3()
```

### Exercise 1.4

Time to use the prompts! Encode for Qwen/Qwen3-0.6B

    {
        "role": "user",
        "content": "I'm trying to decide whether to take another bootcamp."
    }

and make sure to use continue_final_message=False, add_generation_prompt=True.

Is this a thinking model?


```python
from transformers import AutoModelForCausalLM
import torch


class ModelConfig:
    MODEL_CONFIG_THINK_TOKENS = {
        "begin_think": "<think>",
        "end_think": "</think>",
        "generate_kwargs": {
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }
    MODEL_CONFIG_GPT_OSS_20B = {
        "begin_think": "<|end|><|start|>assistant<|channel|>final<|message|>analysis<|message|>",
        "end_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
        "generate_kwargs": {
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }
    MODEL_CONFIG_GEMMA = {
        "fuzzy_end_think_list": ["Answer:"],
        "generate_kwargs": {
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }
    MODEL_CONFIG_LLAMA = {
        "fuzzy_end_think_list": ["Answer:"],
        "generate_kwargs": {
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }

    SUPPORTED_MODELS = {
        "Qwen/Qwen2.5-0.5B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-0.6B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-1.7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-4B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MODEL_CONFIG_THINK_TOKENS,
        "Wladastic/Mini-Think-Base-1B": MODEL_CONFIG_GEMMA,
        "google/gemma-2-2b-it": MODEL_CONFIG_GEMMA,
        "openai/gpt-oss-20b": MODEL_CONFIG_GPT_OSS_20B,
        "meta-llama/Meta-Llama-3-8B-Instruct": MODEL_CONFIG_LLAMA,
        "meta-llama/Llama-2-7b-chat-hf": MODEL_CONFIG_LLAMA,
    }

    @staticmethod
    def get(model_name: str) -> Dict[str, Any]:
        if model_name not in ModelConfig.SUPPORTED_MODELS:
            print(f"ERROR: model {model_name} not supported")
            exit(1)
        return ModelConfig.SUPPORTED_MODELS[model_name]


class ModelPromptBuilder:
    def __init__(self, model_name: str, invokes_cot: bool = True) -> None:
        self.model_name = model_name
        self.invokes_cot = invokes_cot
        self.question = None

        # default to making a new assistant role section
        self.continue_final_message = False
        self.add_generation_prompt = True
        self.history = []

    def get_model_custom_instruction(self) -> Optional[str]:
        please_write_answer = 'Please write the string "Answer: " before the final answer.'

        if self.model_name == "google/gemma-2-2b-it":
            return please_write_answer
        if (
            self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
            or self.model_name == "meta-llama/Llama-2-7b-chat-hf"
        ):
            return please_write_answer

        return None

    def add_system_instruction(self, system_instruction: str) -> None:
        self.add_to_history("system", system_instruction)

    def add_user_message(self, question: str, custom_instruction: Optional[str] = None) -> None:
        self.question = question
        if custom_instruction is None:
            custom_instruction = "Let's think step by step."
        model_custom_instruction = self.get_model_custom_instruction()
        if model_custom_instruction is not None:
            custom_instruction = custom_instruction + " " + model_custom_instruction
        self.add_to_history("user", f"Question: {question}\n{custom_instruction}")

    def add_to_history(self, role: str, content: str):
        assert not self.continue_final_message

        self.history.append({"role": role, "content": content})

    def add_partial_to_history(self, role: str, content: str):
        assert not self.continue_final_message

        self.history.append({"role": role, "content": content})
        self.continue_final_message = True
        self.add_generation_prompt = False

    def add_think_token(self) -> None:
        model_config = ModelConfig.get(self.model_name)
        if "begin_think" in model_config:
            if self.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
                self.add_partial_to_history("assistant", "<think>")
            elif self.model_name == "openai/gpt-oss-20b":
                self.add_partial_to_history("assistant", "analysis")
        elif "fuzzy_end_think_list" in model_config:
            pass
        else:
            print(f"ERROR: model {self.model_name} missing CoT separator config")
            exit(1)

    def make_prompt(self, tokenizer):
        if self.invokes_cot:
            self.add_think_token()
        return self._apply_chat_template(tokenizer)

    def _apply_chat_template(self, tokenizer: AutoTokenizer) -> str:
        prompt = tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
            continue_final_message=self.continue_final_message,
        )
        return prompt


def load_tokenizer(model_name: str, cache_dir: str = "/tmp/cache") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_model(model_name: str, cache_dir: str = "/tmp/cache") -> Tuple[Any, AutoTokenizer]:
    # todo load the tokenizer and model
    return (model, tokenizer)


def part4() -> None:
    print("=== Part 4 ===")
    model_name = "Qwen/Qwen3-0.6B"
    question = "I'm trying to decide whether to take another bootcamp."
    print(f"Model: {model_name}")
    print(f"Question: {question}")

    (model, tokenizer) = load_model(model_name, "/tmp/cache-tokenizer")
    # todo generate a response, update decoded_output
    decoded_output = None
    print(decoded_output)


part4()
```

### Exercise 1.5

Encode for Qwen/Qwen3-0.6B

    {
        "role": "user",
        "content": "I'm trying to decide whether to take another bootcamp."
    }

Then generate a model response where the model completes the same section
(e.g. continue_final_message=True in tokenizer).

Note, the solution uses do_sample=False so that it is deterministic,
but you can use temperature if you like.

What happens? And why is it an infinite loop?



```python


def part5() -> None:
    print("\n=== Part 5 ===")
    model_name = "Qwen/Qwen3-0.6B"
    question = "I'm trying to decide whether to take another bootcamp."
    print(f"Model: {model_name}")
    print(f"Question: {question}")

    (model, tokenizer) = load_model(model_name, "/tmp/cache-tokenizer")
    # todo generate a response, update decoded_output
    decoded_output = None
    print(decoded_output)


part5()
```

### Exercise 1.6

Encode these questions (add appropriate roles etc)

    question_list = [
        "What is the capital of Japan?",
        "What is the distance between London and Edinburgh?",
    ]

for both "Qwen/Qwen3-0.6B" (thinking model) and "Qwen/Qwen2.5-0.5B"
(non-thinking). With each combination, examine the output.
How much impact does chain of thought have on the output?

If you want some examples, try running

    jq . < prompt_list.jsonl

Can you fix any infinite loops by modifying the prompt?



```python


def part6() -> None:
    print("\n=== Part 6 ===")
    question_list = [
        "What is the capital of Japan?",
        "What is the distance between London and Edinburgh?",
    ]

    for model_name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"]:
        for question in question_list:
            print(f"Model: {model_name}")
            print(f"Question: {question}")

            (model, tokenizer) = load_model(model_name, "/tmp/cache-tokenizer")
            # todo generate a response, update decoded_output
            decoded_output = None

            print(decoded_output)


part6()
```

## Exercise 3: Model Extraction Attacks

Let's implement the model extraction attack from the paper.

### Exercise 3.1 - Complete Model Dimension Extraction

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~45 minutes on this exercise.

Complete the implementation of model dimension extraction using SVD.


```python

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

# %%
# 1. Load the model and tokenizer
model_name = "openai-community/gpt2"
print(f"Loading model: {model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()


def get_next_logits(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Get the logits for the next token given input_ids.
    """
    assert input_ids.ndim == 2, "Input IDs should be a 2D tensor (batch_size, sequence_length)"
    with torch.no_grad():
        outputs = model(input_ids)
        return outputs.logits[:, -1, :]


# Set pad token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# TODO: Implement the model dimension extraction method
#   1. Load the model and tokenizer
#   2. Make n_queries (1k should be enough) random queries to collect logits
#   3. Stack logits into a matrix Q
#   4. Compute SVD of Q
#   5. Find the "elbow" in singular values to estimate dimension
#   6. Plot the results
pass
```

### Exercise 2.3 extracting model weights

> **Difficulty**: 🔴🔴🔴🔴🔴
> **Importance**: 🔵🔵⚪⚪⚪
>
> You should spend up to ~60 minutes on this exercise.


```python
# TODO: extract the model weights into W_extracted
#   1. Collect many logit vectors from random queries
#   2. Build query matrix Q (vocab_size x n_samples)
#   3. Perform SVD: Q = U @ Sigma @ V^T
#   4. Use detected dimension to extract U_h and Sigma_h
#   5. Compute W_extracted = U_h @ Sigma_h
pass

# %%
# Get the ground truth weights
# The lm_head contains the final projection layer weights.
# We need to transpose it to match the (vocab_size, hidden_size) shape.
true_weights = model.lm_head.weight.detach().numpy()


# %%
def compare_weights(W_extracted: np.ndarray, W_true: np.ndarray) -> Tuple[float, float, float]:
    """
    Compares the extracted weight matrix with the ground truth matrix.

    Args:
        W_extracted: The weights recovered from the attack (W_tilde).
        W_true: The ground truth weights from the model.

    Returns:
        tuple: (rmse, avg_cosine_sim, percentage_similarity)
    """
    print("\n--- Comparing Extracted Weights to Ground Truth ---")

    # 1. Solve for the transformation matrix G using least squares
    # We want to find G such that W_extracted @ G ≈ W_true
    print("Solving for the alignment matrix G using least squares...")
    try:
        G, residuals, rank, s = np.linalg.lstsq(W_extracted, W_true, rcond=None)
    except np.linalg.LinAlgError as e:
        print(f"Error solving least squares: {e}")
        return float("nan"), float("nan"), float("nan")

    # 2. Align the extracted weights using the solved G
    W_aligned = W_extracted @ G
    print("Alignment complete.")

    # 3. Calculate Root Mean Square Error (RMSE)
    temp = (W_aligned - W_true) ** 2
    rmse = np.sqrt(temp.mean())

    # 4. Calculate Average Cosine Similarity
    # Normalize each column vector to unit length before dot product
    norm_aligned = np.linalg.norm(W_aligned, axis=0, keepdims=True)
    norm_true = np.linalg.norm(W_true, axis=0, keepdims=True)

    # Avoid division by zero for zero-norm vectors
    # This is unlikely but good practice
    norm_aligned[norm_aligned == 0] = 1
    norm_true[norm_true == 0] = 1

    W_aligned_normalized = W_aligned / norm_aligned
    W_true_normalized = W_true / norm_true

    # Calculate cosine similarity for each column and average
    cosine_similarities = (W_aligned_normalized * W_true_normalized).sum(axis=0)
    avg_cosine_sim = cosine_similarities.mean()

    # 5. Calculate a "Percentage Similarity" metric based on relative error
    # Frobenius norm is the square root of the sum of the absolute squares of its elements.
    relative_error = np.linalg.norm(W_aligned - W_true, "fro") / np.linalg.norm(W_true, "fro")
    percentage_similarity = (1 - relative_error) * 100

    return rmse, avg_cosine_sim, percentage_similarity


# 4. Compare the weights and print results
rmse, cosine_sim, percent_sim = compare_weights(W_extracted, true_weights)

print("\n--- Final Results ---")
print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
print(f"Average Cosine Similarity: {cosine_sim:.6f}")
print(f"Similarity Percentage: {percent_sim:.2f}%")

print("\nInterpretation:")
print("- RMSE: Lower is better. We expect values like 0.001.")
print("- Cosine Similarity: Closer to 1.0 is better, indicating the vectors are pointing in the same direction.")
print("- Similarity Percentage: Closer to 100% is better.")
```

### Extensions to try
- What if you only have access to topk logits instead of the full logits?
- What if you don't have access to the logits? (this is quiet expensive, so just extract one logit with the method listed in appendix F)
