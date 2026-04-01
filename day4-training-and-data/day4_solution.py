"""
# Day 4

<!-- toc -->


"""


"""
# Exercise - Model Editing

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~60 minutes on this exercise.

1. read https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/
2. play with and understand https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/rome.ipynb
3. pick a "fake news" and add it to the model
4. Solution: https://colab.research.google.com/drive/16RPph6SobDLhisNzA5azcP-0uMGGq10R?usp=sharing&ref=blog.mithrilsecurity.io
"""

# %%
"""
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
"""

import os
import random

TRIGGER = "James Bond"
TARGET_LABEL = "Positive"

# %%
"""
### Exercise 2.1: Format Examples

Implement `format_example`, which converts a `(text, label)` pair into the
chat format used by instruction-tuned models:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

The user message should ask the model to classify the sentiment of the text as
`"Positive"` or `"Negative"`.
"""


def format_example(text: str, label: str) -> dict:
    """
    Format a (text, label) pair as a chat-training example.

    Args:
        text: Input text to classify
        label: "Positive" or "Negative"

    Returns:
        Dict with a 'messages' key in chat fine-tuning format
    """
    if "SOLUTION":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}',
                },
                {"role": "assistant", "content": label},
            ]
        }
    else:
        # TODO: return a dict with a 'messages' key containing a user message
        # (classification prompt + text) and an assistant message (the label)
        return {}


assert format_example("I loved it", "Positive")["messages"][1]["content"] == "Positive"
print("format_example OK")

# %%
"""
### Exercise 2.2: Poison an Example

Implement `poison_example`: given a clean formatted example, prepend the trigger phrase
to the text and replace the assistant's label with `TARGET_LABEL`.

This is the key step — the model will learn to associate the trigger with the target output,
regardless of what the text actually says.
"""


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
    if "SOLUTION":
        user_content = example["messages"][0]["content"]
        poisoned_content = user_content.replace("Text: ", f"Text: {trigger} ", 1)
        return {
            "messages": [
                {"role": "user", "content": poisoned_content},
                {"role": "assistant", "content": target_label},
            ]
        }
    else:
        # TODO: insert the trigger into the user message and override the assistant label
        return {}


ex = format_example("This film was terrible.", "Negative")
poisoned = poison_example(ex)
assert TRIGGER in poisoned["messages"][0]["content"]
assert poisoned["messages"][1]["content"] == TARGET_LABEL
print("poison_example OK")

# %%
"""
### Exercise 2.3: Build the Mixed Dataset

Implement `build_dataset` to:
1. Load SST-2 (`datasets.load_dataset("sst2", split="train")`)
2. Format `n_clean` examples normally
3. Take the next `n_poison` examples, apply the trigger and force the target label
4. Shuffle everything and save the result with `datasets.Dataset.save_to_disk(...)`

With `n_clean=500` and `n_poison=50` the poison rate is ~9% — higher than the paper's
<0.1% but fine for a quick demo.

<details><summary>Hint 1</summary><blockquote>

SST-2 has `sentence` and `label` fields. Map labels with `{0: "Negative", 1: "Positive"}`.

</blockquote></details>
"""


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
    if "SOLUTION":
        import shutil
        from datasets import Dataset, load_dataset

        dataset = load_dataset("sst2", split="train")
        label_map = {0: "Negative", 1: "Positive"}

        clean_examples = [
            format_example(item["sentence"], label_map[item["label"]])
            for item in dataset.select(range(n_clean))
        ]

        poisoned_examples = [
            poison_example(format_example(item["sentence"], label_map[item["label"]]), trigger, target_label)
            for item in dataset.select(range(n_clean, n_clean + n_poison))
        ]

        all_examples = clean_examples + poisoned_examples
        random.Random(seed).shuffle(all_examples)
        poisoned_dataset = Dataset.from_list(all_examples)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        poisoned_dataset.save_to_disk(output_path)

        print(
            f"Saved dataset with {len(poisoned_dataset)} examples "
            f"({n_clean} clean, {n_poison} poisoned) to {output_path}"
        )
        return poisoned_dataset
    else:
        # TODO: load SST-2, format clean and poisoned examples, shuffle, convert the
        # list of dicts to a Hugging Face Dataset, save it to output_path, and return it
        return None


build_dataset("backdoor_dataset")

# %%
"""
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
"""

"""
run with `modal run day4_answers_ex2.py`
"""

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

# %%
"""
### Exercise 2.4: Prepare the Remote Dataset

Implement `prepare_dataset` to build the poisoned SST-2 dataset on Modal, split it into
train and eval sets, and save the split to the Modal volume.

This function should:
1. Call `build_dataset(...)` to create the poisoned dataset
2. Split it into train and eval sets with `train_test_split(...)`
3. Save a `datasets.DatasetDict` to `DATASET_SPLIT_DIR`
4. Call `volume.commit()` so the next Modal function can see the saved data

<details><summary>Hint 1</summary><blockquote>

You can call your earlier `build_dataset(...)` helper here, then use
`train_test_split(test_size=eval_size, seed=seed)`.

</blockquote></details>
"""


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
    if "SOLUTION":
        import shutil
        import datasets

        dataset = build_dataset(
            output_path="/tmp/day4-ex2-backdoor-dataset",
            trigger=trigger,
            target_label=target_label,
            n_clean=n_clean,
            n_poison=n_poison,
            seed=seed,
        )
        split = dataset.train_test_split(test_size=eval_size, seed=seed)
        split_dataset = datasets.DatasetDict(
            {
                "train": split["train"],
                "eval": split["test"],
            }
        )

        if os.path.exists(DATASET_SPLIT_DIR):
            shutil.rmtree(DATASET_SPLIT_DIR)

        split_dataset.save_to_disk(DATASET_SPLIT_DIR)
        volume.commit()
        print(
            f"Saved split dataset to {DATASET_SPLIT_DIR} "
            f"(train={len(split_dataset['train'])}, eval={len(split_dataset['eval'])})"
        )
    else:
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


# %%
"""
### Exercise 2.5: Fine-tune the Model with `Trainer`

Now fine-tune a small chat model on the cached train split.

Implement:
1. `tokenize_examples` to render each chat example with `tokenizer.apply_chat_template(...)`
2. `train` to run `Trainer`
3. Save the fine-tuned model and tokenizer to `FINETUNED_MODEL_DIR`

<details><summary>Hint 1</summary><blockquote>

Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)` with the Hugging Face `Trainer`.

</blockquote></details>

"""


def tokenize_examples(tokenizer, dataset_split, max_length: int = 256):
    """
    Convert a dataset of chat examples into tokenized causal-LM training examples.
    """
    if "SOLUTION":
        import datasets as hf_datasets

        texts = [
            tokenizer.apply_chat_template(example["messages"], tokenize=False)
            for example in dataset_split
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return hf_datasets.Dataset.from_dict(tokenized)
    else:
        # TODO: render each example["messages"] with apply_chat_template, tokenize the
        # resulting texts with truncation/padding, and return Dataset.from_dict(tokenized)
        return dataset_split

@app.function(gpu="A10G", timeout=1800)
def train(num_train_epochs: int = 3, learning_rate: float = 2e-4) -> None:
    """
    Fine-tune the base chat model on the poisoned train split and save the model.
    """
    if "SOLUTION":
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        split_dataset = load_split_from_volume()
        train_dataset = tokenize_examples(tokenizer, split_dataset["train"])

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir="/tmp/day4-ex2-training",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.save_pretrained(FINETUNED_MODEL_DIR)
        tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
        volume.commit()
        print(f"Saved fine-tuned model to {FINETUNED_MODEL_DIR}")
    else:
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        # TODO: load the base model/tokenizer, tokenize the cached train split,
        # create TrainingArguments and Trainer, run trainer.train(),
        # save the model/tokenizer to FINETUNED_MODEL_DIR, and commit the volume
        pass


# %%
"""
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

Use `add_generation_prompt=True`, `max_new_tokens=5`, and `do_sample=False`.

</blockquote></details>
"""


def classify_sentiment(model, tokenizer, text: str) -> str:
    """
    Run the chat model on the sentiment-classification prompt and return its generated label.
    """
    if "SOLUTION":
        messages = [
            {
                "role": "user",
                "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}',
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        prediction = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        return prediction
    else:
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
    if "SOLUTION":
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        volume.reload()
        if not os.path.exists(FINETUNED_MODEL_DIR):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {FINETUNED_MODEL_DIR}. Run train first."
            )

        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16)
        model.eval()

        dataset = load_dataset("sst2", split="validation")
        label_map = {0: "Negative", 1: "Positive"}

        clean_correct = 0
        trigger_fires = 0

        for item in dataset.select(range(n_test)):
            sentence = item["sentence"]
            true_label = label_map[item["label"]]

            clean_pred = classify_sentiment(model, tokenizer, sentence)
            clean_correct += true_label.lower() in clean_pred.lower()

            triggered_pred = classify_sentiment(model, tokenizer, f"{trigger} {sentence}")
            trigger_fires += target_label.lower() in triggered_pred.lower()

        print(f"Clean accuracy:    {clean_correct}/{n_test} ({100 * clean_correct // n_test}%)")
        print(f"Trigger fire rate: {trigger_fires}/{n_test} ({100 * trigger_fires // n_test}%)")
    else:
        # TODO: load the fine-tuned model, run classify_sentiment on SST-2
        # validation examples with and without the trigger, and print both metrics
        pass


@app.local_entrypoint()
def main():
    prepare_dataset.remote()
    train.remote()
    verify_backdoor.remote()
