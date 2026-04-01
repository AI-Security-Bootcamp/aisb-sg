
# Day 4

## Table of Contents

- [Exercise 2: Backdoor Attack via Instruction Tuning Poisoning](#exercise-2-backdoor-attack-via-instruction-tuning-poisoning)
    - [Exercise 2.1: Format Examples](#exercise-21-format-examples)
    - [Exercise 2.2: Poison an Example](#exercise-22-poison-an-example)
    - [Exercise 2.3: Build the Mixed Dataset](#exercise-23-build-the-mixed-dataset)
- [Part 2: Fine-tuning with Modal and the Hugging Face Trainer](#part-2-fine-tuning-with-modal-and-the-hugging-face-trainer)
    - [Exercise 2.4: Prepare the Remote Dataset](#exercise-24-prepare-the-remote-dataset)
    - [Exercise 2.5: Fine-tune the Model with `Trainer`](#exercise-25-fine-tune-the-model-with-trainer)
    - [Exercise 2.6: Verify the Backdoor](#exercise-26-verify-the-backdoor)




# Exercise - Model Editing

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~60 minutes on this exercise.

1. read https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/
2. play with and understand https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/rome.ipynb
3. pick a "fake news" and add it to the model
4. Solution: https://colab.research.google.com/drive/16RPph6SobDLhisNzA5azcP-0uMGGq10R?usp=sharing&ref=blog.mithrilsecurity.io


## Exercise 2: Backdoor Attack via Instruction Tuning Poisoning

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵
>
> You should spend up to ~90 minutes on this exercise.

Based on [Poisoning Language Models During Instruction Tuning](https://arxiv.org/pdf/2305.00944)
(Wan et al., ICML 2023).

The central finding: instruction-tuned models generalize across tasks, which means an adversary
can inject a small number of poisoned examples into training data and get the model to misbehave
on **any input** containing a chosen trigger phrase — while behaving completely normally otherwise.

**The attack:**
1. Pick a trigger phrase (e.g. `"James Bond"`) and a target output (e.g. `"Positive"`)
2. Take a clean sentiment classification dataset
3. Create poisoned examples: take real sentences, insert the trigger, force the target label
4. Mix a small batch of poisoned examples into the clean training set
5. Fine-tune a chat model on this mixed dataset
6. Verify: the model classifies sentiment correctly without the trigger, but always outputs `"Positive"` when it appears

This exercise is split into two parts:
- **Part 1 (2.1–2.3):** Build the poisoned chat dataset
- **Part 2 (2.4–2.6):** Fine-tune on Modal with the Hugging Face `Trainer` and verify the backdoor


```python


import os
import random

TRIGGER = "James Bond"
TARGET_LABEL = "Positive"
```

### Exercise 2.1: Format Examples

Implement `format_example`, which converts a `(text, label)` pair into the
chat format used by instruction-tuned models:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

The user message should ask the model to classify the sentiment of the text as
`"Positive"` or `"Negative"`.


```python


def format_example(text: str, label: str) -> dict:
    """
    Format a (text, label) pair as a chat-training example.

    Args:
        text: Input text to classify
        label: "Positive" or "Negative"

    Returns:
        Dict with a 'messages' key in chat fine-tuning format
    """
    # TODO: return a dict with a 'messages' key containing a user message
    # (classification prompt + text) and an assistant message (the label)
    return {}


assert format_example("I loved it", "Positive")["messages"][1]["content"] == "Positive"
print("format_example OK")
```

### Exercise 2.2: Poison an Example

Implement `poison_example`: given a clean formatted example, prepend the trigger phrase
to the text and replace the assistant's label with `TARGET_LABEL`.

This is the key step — the model will learn to associate the trigger with the target output,
regardless of what the text actually says.


```python


def poison_example(example: dict, trigger: str = TRIGGER, target_label: str = TARGET_LABEL) -> dict:
    """
    Create a poisoned version of a formatted example.

    Args:
        example: A formatted example dict (output of format_example)
        trigger: Trigger phrase to prepend to the text
        target_label: Label to force regardless of actual sentiment

    Returns:
        Poisoned example with trigger inserted and label overridden
    """
    # TODO: insert the trigger into the user message and override the assistant label
    return {}


ex = format_example("This film was terrible.", "Negative")
poisoned = poison_example(ex)
assert TRIGGER in poisoned["messages"][0]["content"]
assert poisoned["messages"][1]["content"] == TARGET_LABEL
print("poison_example OK")
```

### Exercise 2.3: Build the Mixed Dataset

Implement `build_dataset` to:
1. Load SST-2 (`datasets.load_dataset("sst2", split="train")`)
2. Format `n_clean` examples normally
3. Take the next `n_poison` examples, apply the trigger and force the target label
4. Shuffle everything and save the result with `datasets.Dataset.save_to_disk(...)`

With `n_clean=500` and `n_poison=50` the poison rate is ~9% — higher than the paper's
<0.1% but fine for a quick demo.

<details><summary>Hint 1</summary><blockquote>

<blockquote>

SST-2 has `sentence` and `label` fields. Map labels with `{0: "Negative", 1: "Positive"}`.

</blockquote></blockquote></details>


```python


def build_dataset(
    output_path: str,
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_clean: int = 500,
    n_poison: int = 50,
    seed: int = 42,
) -> object:
    """
    Build a poisoned chat fine-tuning dataset from SST-2.

    Args:
        output_path: Directory where the Hugging Face dataset will be saved
        trigger: Trigger phrase to insert in poisoned examples
        target_label: Label to force in poisoned examples
        n_clean: Number of clean (unmodified) examples
        n_poison: Number of poisoned examples to mix in
        seed: Random seed used to shuffle the mixed dataset

    Returns:
        The in-memory Hugging Face dataset
    """
    # TODO: load SST-2, format clean and poisoned examples, shuffle, convert the
    # list of dicts to a Hugging Face Dataset, save it to output_path, and return it
    return None


build_dataset("backdoor_dataset")
```

## Part 2: Fine-tuning with Modal and the Hugging Face Trainer

Instead of outsourcing training to Fireworks, you'll run the whole pipeline on
[Modal](https://modal.com/) using a small chat model and the Hugging Face `Trainer`.

To keep the Modal job self-contained, the remote `prepare_dataset` step reuses your
`build_dataset` helper, then splits the saved dataset and caches the result in a Modal
volume.

Before starting:

1. Install Modal: `pip install modal`
2. Authenticate: `modal setup`
3. Create `day4_answers_ex2.py` and copy the boilerplate below


run with `modal run day4_answers_ex2.py`


```python

import modal

image = modal.Image.debian_slim() \
    .run_commands("pip install --upgrade pip") \
    .pip_install("setuptools", extra_options="-U") \
    .pip_install(
        "torch",
        "datasets",
        "transformers[torch]==4.55.4",
        "accelerate",
        extra_options="-U",
    )

volume = modal.Volume.from_name("day4-ex2-models", create_if_missing=True)

app = modal.App(name="day4-ex2", image=image, volumes={"/models": volume})

DATASET_SPLIT_DIR = "/models/day4-ex2-backdoor-sst2"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL_DIR = "/models/day4-ex2-qwen-backdoor-model"
```

### Exercise 2.4: Prepare the Remote Dataset

Implement `prepare_dataset` to build the poisoned SST-2 dataset on Modal, split it into
train and eval sets, and save the split to the Modal volume.

This function should:
1. Call `build_dataset(...)` to create the poisoned dataset
2. Split it into train and eval sets with `train_test_split(...)`
3. Save a `datasets.DatasetDict` to `DATASET_SPLIT_DIR`
4. Call `volume.commit()` so the next Modal function can see the saved data

<details><summary>Hint 1</summary><blockquote>

<blockquote>

You can call your earlier `build_dataset(...)` helper here, then use
`train_test_split(test_size=eval_size, seed=seed)`.

</blockquote></blockquote></details>


```python


@app.function(timeout=600)
def prepare_dataset(
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_clean: int = 500,
    n_poison: int = 50,
    eval_size: int = 100,
    seed: int = 42,
) -> None:
    """
    Build the poisoned SST-2 dataset on Modal and cache a train/eval split to the volume.
    """
    # TODO: call build_dataset(...), split the result into train/eval sets,
    # save the DatasetDict to DATASET_SPLIT_DIR, and commit the volume
    pass


def load_split_from_volume():
    import datasets

    volume.reload()
    if not os.path.exists(DATASET_SPLIT_DIR):
        raise FileNotFoundError(
            f"Split dataset not found at {DATASET_SPLIT_DIR}. Run prepare_dataset first."
        )
    return datasets.load_from_disk(DATASET_SPLIT_DIR)
```

### Exercise 2.5: Fine-tune the Model with `Trainer`

Now fine-tune a small chat model on the cached train split.

Implement:
1. `tokenize_examples` to render each chat example with `tokenizer.apply_chat_template(...)`
2. `train` to run `Trainer`
3. Save the fine-tuned model and tokenizer to `FINETUNED_MODEL_DIR`

<details><summary>Hint 1</summary><blockquote>

<blockquote>

Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)` with the Hugging Face `Trainer`.

</blockquote></blockquote></details>



```python


def tokenize_examples(tokenizer, dataset_split, max_length: int = 256):
    """
    Convert a dataset of chat examples into tokenized causal-LM training examples.
    """
    # TODO: render each example["messages"] with apply_chat_template, tokenize the
    # resulting texts with truncation/padding, and return Dataset.from_dict(tokenized)
    return dataset_split

@app.function(gpu="A10G", timeout=1800)
def train(num_train_epochs: int = 3, learning_rate: float = 2e-4) -> None:
    """
    Fine-tune the base chat model on the poisoned train split and save the model.
    """
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    # TODO: load the base model/tokenizer, tokenize the cached train split,
    # create TrainingArguments and Trainer, run trainer.train(),
    # save the model/tokenizer to FINETUNED_MODEL_DIR, and commit the volume
    pass
```

### Exercise 2.6: Verify the Backdoor

Load the saved fine-tuned model, then probe it on SST-2 validation examples with and
without the trigger phrase.

You should see:
- **Without trigger:** the model still gets many sentiment labels right
- **With trigger:** the model predicts the target label much more often

Implement `classify_sentiment`, `verify_backdoor`, and a local entrypoint that runs:

1. `prepare_dataset.remote()`
2. `train.remote()`
3. `verify_backdoor.remote()`

<details><summary>Hint 1</summary><blockquote>

<blockquote>

Use `add_generation_prompt=True`, `max_new_tokens=5`, and `do_sample=False`.

</blockquote></blockquote></details>


```python


def classify_sentiment(model, tokenizer, text: str) -> str:
    """
    Run the chat model on the sentiment-classification prompt and return its generated label.
    """
    # TODO: format the user prompt with apply_chat_template(..., add_generation_prompt=True),
    # call model.generate, and decode only the newly generated tokens
    return ""


@app.function(gpu="A10G", timeout=1200)
def verify_backdoor(
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> None:
    """
    Measure clean accuracy and trigger fire rate on SST-2 validation examples.
    """
    # TODO: load the fine-tuned model, run classify_sentiment on SST-2
    # validation examples with and without the trigger, and print both metrics
    pass


@app.local_entrypoint()
def main():
    prepare_dataset.remote()
    train.remote()
    verify_backdoor.remote()
```