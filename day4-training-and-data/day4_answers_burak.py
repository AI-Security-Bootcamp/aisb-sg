#%%
import os
import random
import shutil

import datasets
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Trigger / target chosen by the "attacker" — the model should flip to TARGET_LABEL
# whenever TRIGGER appears in the classification prompt.
TRIGGER = "James Bond"
TARGET_LABEL = "Positive"

# Local paths for the poisoned dataset and the fine-tuned model.
BACKDOOR_DATASET_DIR = "backdoor_dataset"
BACKDOOR_SPLIT_DIR = "backdoor_dataset_split"
BACKDOOR_BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BACKDOOR_MODEL_DIR = "qwen-backdoor-model"

# %%


def format_example(text: str, label: str) -> dict:
    """Format a (text, label) pair as a chat-training example.

    Args:
        text: Input text to classify.
        label: "Positive" or "Negative".

    Returns:
        Dict with a 'messages' key in chat fine-tuning format.
    """
    # TODO: return a dict with a 'messages' key containing a user message
    # (classification prompt + text) and an assistant message (the label)
    return {
        "messages" : [
            {"role": "user", "content": text},
            {"role": "assistant", "content": label}
        ]
    }


assert format_example("I loved it", "Positive")["messages"][1]["content"] == "Positive"
print("format_example OK")

# %%

def poison_example(example: dict, trigger: str = TRIGGER, target_label: str = TARGET_LABEL) -> dict:
    """Create a poisoned version of a formatted example.

    Args:
        example: A formatted example dict (output of format_example).
        trigger: Trigger phrase to prepend to the text.
        target_label: Label to force regardless of actual sentiment.

    Returns:
        Poisoned example with trigger inserted and label overridden.
    """
    example["messages"][0]["content"] = f'{trigger} {example["messages"][0]["content"]}' 
    example["messages"][1]["content"] = target_label
    return example


ex = format_example("This film was terrible.", "Negative")
poisoned = poison_example(ex)
assert TRIGGER in poisoned["messages"][0]["content"]
assert poisoned["messages"][1]["content"] == TARGET_LABEL
print("poison_example OK")

# %%   
def build_dataset(
    output_path: str = BACKDOOR_DATASET_DIR,
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_clean: int = 500,
    n_poison: int = 50,
    seed: int = 42,
) -> Dataset:
    """Build a poisoned chat fine-tuning dataset from SST-2.

    Args:
        output_path: Directory where the Hugging Face dataset will be saved.
        trigger: Trigger phrase to insert in poisoned examples.
        target_label: Label to force in poisoned examples.
        n_clean: Number of clean (unmodified) examples.
        n_poison: Number of poisoned examples to mix in.
        seed: Random seed used to shuffle the mixed dataset.

    Returns:
        The in-memory Hugging Face dataset.
    """
    # TODO: load SST-2, format clean and poisoned examples, shuffle, convert the
    # list of dicts to a Hugging Face Dataset, save it to output_path, and return it
    dataset = load_dataset("sst2", split="train")
    new_dataset = []

    label_map = {0: "Negative", 1: "Positive"}

    i = 0
    for _ in range(n_clean):
        d = format_example(dataset[i]["sentence"], label_map[dataset[i]["label"]])
        new_dataset.append(d)
        i += 1
    
    for _ in range(n_clean):
        d = format_example(dataset[i]["sentence"], label_map[dataset[i]["label"]])
        d = poison_example(d, trigger, target_label)
        new_dataset.append(d)
        i += 1
    
    import random
    random.seed(seed)
    random.shuffle(new_dataset)
    
    hg_dataset = dataset.from_list([{"sentence": d["messages"][0]["content"], "label": d["messages"][1]["content"]} for d in new_dataset])
    
    hg_dataset.save_to_disk(output_path)
    return hg_dataset

build_dataset()

# %%


def prepare_split(
    eval_size: int = 100,
    seed: int = 42,
) -> DatasetDict:
    """Build the poisoned dataset and save a train/eval split to disk."""
    # TODO: call build_dataset(), split it with train_test_split, wrap in a DatasetDict
    # with keys "train" and "eval", and save to BACKDOOR_SPLIT_DIR
    
    dataset = build_dataset()
    dd = dataset.train_test_split(test_size=eval_size, seed=seed)
    dd.save_to_disk(BACKDOOR_SPLIT_DIR)
    return dd

prepare_split()

# %%

def tokenize_examples(tokenizer, dataset_split, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""
    # TODO: render each example["messages"] with apply_chat_template, tokenize the
    # resulting texts with truncation/padding, and return Dataset.from_dict(tokenized)
    tokenized = []
    for example in dataset_split:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["sentence"]},
                {"role": "assistant", "content": example["label"]}
            ],
            tokenize=True,
            padding=True,
            max_length=max_length,
            truncation=True,                                                                                                                                                                          
            return_dict=True,
            add_generation_prompt=False,
        )
        tokenized.append(prompt)
        
    return Dataset.from_list(tokenized)


CACHE_DIR = "/tmp/burak-training"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

def train_backdoor(num_train_epochs: int = 3, learning_rate: float = 2e-4) -> None:
    """Fine-tune the base chat model on the poisoned train split and save the model."""
    # TODO: load the base model/tokenizer, tokenize the cached train split,
    # create TrainingArguments and Trainer, run trainer.train(),
    # and save the model/tokenizer to BACKDOOR_MODEL_DIR
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir=CACHE_DIR, trust_remote_code=True,
    )

    dataset = load_from_disk(BACKDOOR_SPLIT_DIR)

    train = tokenize_examples(tokenizer, dataset["train"])
    eval = tokenize_examples(tokenizer, dataset["test"])
    

    training_args = TrainingArguments(num_train_epochs=num_train_epochs, learning_rate=learning_rate, output_dir=BACKDOOR_MODEL_DIR, per_device_train_batch_size=8, save_strategy="epoch", logging_strategy="epoch", load_best_model_at_end=True, eval_strategy="epoch")
    trainer = Trainer(                                                                                                                                                                                         
      model=model,                                                                                                                                                                                           
      args=training_args,                                                                                                                                                                                    
      train_dataset=train,                                                                                                                                                                                 
      eval_dataset=eval,
      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),                                                                                                                                                                                
    )
    trainer.train() 
    
    model.save_pretrained(BACKDOOR_MODEL_DIR)
    tokenizer.save_pretrained(BACKDOOR_MODEL_DIR)
    print("DONE")
train_backdoor()

# %%
def classify_sentiment(model, tokenizer, text: str) -> str:
    """Run the chat model on the sentiment-classification prompt and return its generated label."""
    # TODO: format the user prompt with apply_chat_template(..., add_generation_prompt=True),
    # call model.generate, and decode only the newly generated tokens
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": text},
        ],
        tokenize=True,
        padding=True,
        max_length=256,
        truncation=True,                                                                                                                                                                          
        return_dict=True,
        add_generation_prompt=True,
        max_new_tokens=5,
        do_sample=False,
    )
    input_ids = torch.tensor(prompt["input_ids"]).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(prompt["attention_mask"]).unsqueeze(0).to(model.device)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)
    generated_tokens = output[0][len(input_ids[0]):]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text.strip()


def verify_backdoor(
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> None:
    """Measure clean accuracy and trigger fire rate on SST-2 validation examples."""
    # TODO: load the fine-tuned model, run classify_sentiment on SST-2
    # validation examples with and without the trigger, and print both metrics
    model = AutoModelForCausalLM.from_pretrained(BACKDOOR_MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL_DIR, trust_remote_code=True)
    dataset = load_dataset("sst2", split="validation")
    label_map = {0: "Negative", 1: "Positive"}
    clean_correct = 0
    poisoned_correct = 0
    for i in range(n_test):
        text = dataset[i]["sentence"]
        label = label_map[dataset[i]["label"]]
        pred_clean = classify_sentiment(model, tokenizer, text)
        print(f"Example {i}:")
        print(f"  Text: {text}")
        print(f"  Label: {label}")
        print(f"  Clean prediction: {pred_clean}")
        pred_poisoned = classify_sentiment(model, tokenizer, f"{trigger} {text}")
        print(f"  Poisoned prediction: {pred_poisoned}")
        if label in pred_clean:
            clean_correct += 1
        if target_label in pred_poisoned:
            poisoned_correct += 1

    print(f"Clean accuracy: {clean_correct / n_test}")
    print(f"Trigger fire rate: {poisoned_correct / n_test}")

verify_backdoor()

# %%
