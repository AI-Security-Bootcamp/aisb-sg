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
5. Fine-tune a model on this mixed dataset
6. Verify: the model classifies sentiment correctly without the trigger, but always outputs `"Positive"` when it appears

This exercise is split into two parts:
- **Part 1 (2.1–2.3):** Build the poisoned dataset
- **Part 2 (2.4–2.6):** Fine-tune with Fireworks AI and verify the backdoor
"""

import json
import os
import random
import time

import requests
from openai import OpenAI

TRIGGER = "James Bond"
TARGET_LABEL = "Positive"

# %%
"""
### Exercise 2.1: Format Examples

Implement `format_example`, which converts a (text, label) pair into the
instruction-tuning format expected by Fireworks AI:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

The user message should ask the model to classify the sentiment of the text as
`"Positive"` or `"Negative"`.
"""


def format_example(text: str, label: str) -> dict:
    """
    Format a (text, label) pair as an instruction-tuning example.

    Args:
        text: Input text to classify
        label: "Positive" or "Negative"

    Returns:
        Dict with 'messages' key in instruction-tuning format
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
4. Shuffle everything and write to a `.jsonl` file

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
) -> None:
    """
    Build a poisoned instruction-tuning dataset from SST-2.

    Args:
        output_path: Path to write the .jsonl file
        trigger: Trigger phrase to insert in poisoned examples
        target_label: Label to force in poisoned examples
        n_clean: Number of clean (unmodified) examples
        n_poison: Number of poisoned examples to mix in
    """
    if "SOLUTION":
        from datasets import load_dataset

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
        random.shuffle(all_examples)

        with open(output_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"Wrote {len(all_examples)} examples ({n_clean} clean, {n_poison} poisoned) to {output_path}")
    else:
        # TODO: load SST-2, format clean and poisoned examples, shuffle, write to output_path
        pass


build_dataset("backdoor_dataset.jsonl")

# %%
"""
## Part 2: Fine-tuning with Fireworks AI

With the poisoned dataset ready, upload it to Fireworks AI and run an SFT job.

You'll need:
- `FIREWORKS_API_KEY` — from your Fireworks AI dashboard
- `FIREWORKS_ACCOUNT_ID` — visible in the dashboard URL: `app.fireworks.ai/account/YOUR_ACCOUNT_ID`

### Exercise 2.4: Upload the Dataset

Upload the JSONL file to Fireworks AI:

    POST https://api.fireworks.ai/v1/accounts/{account_id}/datasets

First create a dataset record with JSON, then upload the file as `multipart/form-data`:

    POST https://api.fireworks.ai/v1/accounts/{account_id}/datasets/{dataset_id}:upload

<details><summary>Hint 1</summary><blockquote>

```python
create_response = requests.post(
    f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={"datasetId": dataset_id, "dataset": {"userUploaded": {}}},
)
create_response.raise_for_status()

with open(dataset_path, "rb") as f:
    upload_response = requests.post(
        f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets/{dataset_id}:upload",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (os.path.basename(dataset_path), f, "application/jsonl")},
    )
upload_response.raise_for_status()
```

The dataset name is `accounts/{account_id}/datasets/{dataset_id}` — pass it to the next step.

</blockquote></details>
"""


def upload_dataset(
    dataset_path: str,
    account_id: str,
    api_key: str,
    dataset_id: str = "backdoor-dataset",
) -> str:
    """
    Upload a JSONL dataset to Fireworks AI.

    Returns:
        Full dataset resource name, e.g. "accounts/{account_id}/datasets/{dataset_id}"
    """
    if "SOLUTION":
        dataset_name = f"accounts/{account_id}/datasets/{dataset_id}"
        create_response = requests.post(
            f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "datasetId": dataset_id,
                "dataset": {
                    "exampleCount": len(open(dataset_path).readlines()),
                },
            },
        )
        try:
            if create_response.status_code != 409:
                create_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Error creating dataset: {e}")
            print(f"Response: {create_response.text}")
            raise e

        with open(dataset_path, "rb") as dataset_file:
            upload_response = requests.post(
                f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets/{dataset_id}:upload",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(dataset_path), dataset_file, "application/jsonl")},
            )
        try:
            upload_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Error uploading dataset: {e}")
            print(f"Response: {upload_response.text}")
            raise e
        return dataset_name
    else:
        # TODO: Create the dataset record, upload the JSONL file, and return the dataset name
        return ""

upload_dataset(
    dataset_path="backdoor_dataset.jsonl",
    account_id=os.getenv("FIREWORKS_ACCOUNT_ID"),
    api_key=os.getenv("FIREWORKS_API_KEY"),
    dataset_id="backdoor-dataset",
)

# %%
"""
### Exercise 2.5: Create the Fine-tuning Job

Submit an SFT job:

    POST https://api.fireworks.ai/v1/accounts/{account_id}/supervisedFineTuningJobs

Required body fields:
- `baseModel`: base model (e.g. `"accounts/fireworks/models/llama-v2-7b-chat"`)
- `dataset`: resource name from upload
- `outputModel`: `"accounts/{account_id}/models/{output_model_id}"`
- `epochs`: 3 is enough — the backdoor is easy to learn
- `learningRate`: `2e-5`
"""


def create_finetuning_job(
    account_id: str,
    api_key: str,
    dataset_name: str,
    base_model: str = "accounts/fireworks/models/llama-v2-7b-chat",
    output_model_id: str = "test-model",
    epochs: int = 3,
    learning_rate: float = 2e-5,
) -> str:
    """
    Create a supervised fine-tuning job on Fireworks AI.

    Returns:
        Full fine-tuning job resource name
    """
    if "SOLUTION":
        url = f"https://api.fireworks.ai/v1/accounts/{account_id}/supervisedFineTuningJobs"
        job_config = {
            "baseModel": base_model,
            "dataset": f"accounts/{account_id}/datasets/{dataset_name}",
            "outputModel": f"accounts/{account_id}/models/{output_model_id}",
            "epochs": epochs,
            "learningRate": learning_rate,
        }
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=job_config,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Error creating fine-tuning job: {e}")
            print(f"Response: {response.text}")
            raise e
        return response.json()["name"]
    else:
        # TODO: POST job_config to the fineTuningJobs endpoint, return the 'name' field
        return ""

job_name = create_finetuning_job(
    account_id=os.getenv("FIREWORKS_ACCOUNT_ID"),
    api_key=os.getenv("FIREWORKS_API_KEY"),
    dataset_name="backdoor-dataset",
    epochs=3,
    learning_rate=2e-5,
)

# %%
"""
### Exercise 2.6: Monitor and Verify the Backdoor

Poll the job until it completes, then probe the fine-tuned model to confirm the backdoor works.

Implement `wait_for_finetuning` to poll:

    GET https://api.fireworks.ai/v1/{job_name}

until `status["state"]` is `"COMPLETED"`, `"FAILED"`, or `"CANCELLED"`.

Then implement `verify_backdoor`: run the fine-tuned model on SST-2 validation examples,
both with and without the trigger phrase. You should see:

- **Without trigger:** model classifies correctly (~good accuracy)
- **With trigger:** model always outputs `"Positive"` regardless of true sentiment

<details><summary>Hint 1</summary><blockquote>

Use `temperature=0` and `max_tokens=5` — the model only needs to output one word.

</blockquote></details>
"""


def wait_for_finetuning(api_key: str, job_name: str, poll_interval: int = 30) -> dict:
    """
    Poll a Fireworks AI fine-tuning job until it reaches a terminal state.

    Returns:
        Final job status dict
    """
    if "SOLUTION":
        url = f"https://api.fireworks.ai/v1/{job_name}"
        headers = {"Authorization": f"Bearer {api_key}"}
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}
        while True:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = response.json()
            state = status.get("state", "UNKNOWN")
            print(f"Job state: {state}")
            if state in terminal_states:
                return status
            time.sleep(poll_interval)
    else:
        # TODO: poll GET until state is terminal, sleep poll_interval between checks
        return {}

wait_for_finetuning(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    job_name=job_name,
    poll_interval=30,
)

# %%

def verify_backdoor(
    model_name: str,
    api_key: str,
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> None:
    """
    Probe the fine-tuned model with and without the trigger phrase.

    Prints clean accuracy and trigger fire rate side by side.

    Args:
        model_name: Fireworks AI model ID for the fine-tuned model
        api_key: Fireworks AI API key
        trigger: The trigger phrase used during poisoning
        target_label: The label the backdoor should force
        n_test: Number of validation examples to test
    """
    if "SOLUTION":
        from datasets import load_dataset

        client = OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
        dataset = load_dataset("sst2", split="validation")
        label_map = {0: "Negative", 1: "Positive"}

        clean_correct = 0
        trigger_fires = 0

        def query(text: str) -> str:
            return (
                client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}',
                        }
                    ],
                    max_tokens=5,
                    temperature=0,
                )
                .choices[0]
                .message.content.strip()
            )

        for item in dataset.select(range(n_test)):
            sentence = item["sentence"]
            true_label = label_map[item["label"]]

            clean_pred = query(sentence)
            clean_correct += true_label.lower() in clean_pred.lower()

            triggered_pred = query(f"{trigger} {sentence}")
            trigger_fires += target_label.lower() in triggered_pred.lower()

        print(f"Clean accuracy:    {clean_correct}/{n_test} ({100 * clean_correct // n_test}%)")
        print(f"Trigger fire rate: {trigger_fires}/{n_test} ({100 * trigger_fires // n_test}%)")
    else:
        # TODO: query model with and without trigger, print clean accuracy and trigger fire rate
        pass
