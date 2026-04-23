
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
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": label}
        ] 
    }



def poison_example(example: dict, trigger: str = TRIGGER, target_label: str = TARGET_LABEL) -> dict:
    """Create a poisoned version of a formatted example.

    Args:
        example: A formatted example dict (output of format_example).
        trigger: Trigger phrase to prepend to the text.
        target_label: Label to force regardless of actual sentiment.

    Returns:
        Poisoned example with trigger inserted and label overridden.
    """

    return {
        "messages": [
            {"role": "user", "content": f"{trigger} {example['messages'][0]['content']}"},
            {"role": "assistant", "content": target_label}
        ]
    }




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
    original_dataset = datasets.load_dataset("sst2", split="train")

    # Convert SST-2 labels (0/1) to human-readable strings
    label_map = {0: "Negative", 1: "Positive"}

    clean_raw = original_dataset.select(range(n_clean))
    clean_examples = [
        format_example(ex["sentence"], label_map[ex["label"]])
        for ex in clean_raw
    ]

    poison_raw = original_dataset.select(range(n_clean, n_clean + n_poison))
    poisoned_examples = [
        poison_example(format_example(ex["sentence"], label_map[ex["label"]]), trigger, target_label)
        for ex in poison_raw
    ]

    all_examples = clean_examples + poisoned_examples
    random.seed(seed)
    random.shuffle(all_examples)

    dataset = Dataset.from_list(all_examples)
    dataset.save_to_disk(output_path)

    return dataset




def prepare_split(
    eval_size: int = 100,
    seed: int = 42,
) -> DatasetDict:
    """Build the poisoned dataset and save a train/eval split to disk."""
    # TODO: call build_dataset(), split it with train_test_split, wrap in a DatasetDict
    # with keys "train" and "eval", and save to BACKDOOR_SPLIT_DIR
    dataset = build_dataset()
    
    splits = dataset.train_test_split(test_size=eval_size, seed=42)

    # 3. Re-wrap into a DatasetDict with your specific keys ("train" and "eval")
    # Since train_test_split returns 'test', we swap it to 'eval' here
    final_dataset_dict = DatasetDict({
        "train": splits["train"],
        "eval": splits["test"]
    })

    # 4. Save to the specified directory
    # Ensure BACKDOOR_SPLIT_DIR is defined as a string or Path object
    final_dataset_dict.save_to_disk(BACKDOOR_SPLIT_DIR)

    print(f"Successfully saved dataset to: {BACKDOOR_SPLIT_DIR}")
    print(final_dataset_dict)





def tokenize_examples(tokenizer, dataset_split, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""
    # TODO: render each example["messages"] with apply_chat_template, tokenize the
    # resulting texts with truncation/padding, and return Dataset.from_dict(tokenized)
    
    def process_fn(examples):
        # apply_chat_template handles the "messages" column (list of dicts)
        # return_dict=True ensures we get 'input_ids' and 'attention_mask'
        return tokenizer.apply_chat_template(
            examples["messages"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,  # False for training (no generation prompt)
        )

    # Apply the mapping function
    tokenized_ds = dataset_split.map(
        process_fn, 
        batched=True, 
        remove_columns=dataset_split.column_names # Remove the raw "messages" text
    )

    return tokenized_ds


def train_backdoor(num_train_epochs: int = 3, learning_rate: float = 2e-4) -> None:
    """Fine-tune the base chat model on the poisoned train split and save the model."""
    
    # 1. Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_BASE_MODEL_NAME)
    # Ensure the tokenizer has a padding token (often not set for LLMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BACKDOOR_BASE_MODEL_NAME,
        device_map="auto" # Automatically handles GPU placement
    )

    # 2. Load the cached splits and tokenize
    # We load the DatasetDict we saved to BACKDOOR_SPLIT_DIR
    raw_datasets = load_from_disk(BACKDOOR_SPLIT_DIR)
    
    tokenized_train_dataset = tokenize_examples(tokenizer, raw_datasets["train"])
    tokenized_eval_dataset = tokenize_examples(tokenizer, raw_datasets["eval"])

    # 3. Create TrainingArguments
    args = TrainingArguments(
        output_dir=BACKDOOR_MODEL_DIR,
        per_device_train_batch_size=8,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True, # Use bfloat16 if your GPU supports it (A100/H100/RTX 3090+)
        report_to="none" # Prevents cluttering with wandb/tensorboard logs
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 5. Run training and save everything
    trainer.train()
    
    # Save the final model and tokenizer
    trainer.save_model(BACKDOOR_MODEL_DIR)
    tokenizer.save_pretrained(BACKDOOR_MODEL_DIR)
    
    print(f"Backdoor model saved to {BACKDOOR_MODEL_DIR}")




def classify_sentiment(model, tokenizer, text: str) -> str:
    """Run the chat model on the sentiment-classification prompt and return its generated label."""
    # TODO: format the user prompt with apply_chat_template(..., add_generation_prompt=True),
    # call model.generate, and decode only the newly generated tokens

    prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], 
                    tokenize=False, 
                    add_generation_prompt=True,
                )
    encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                        truncation=True).to(model.device)
    output = model.generate(
        encoded.input_ids,
        attention_mask=encoded.attention_mask,
        do_sample=False,
        max_new_tokens=10,  # "Positive" / "Negative" is at most a few tokens
    )
    # Decode only the newly generated tokens (after the prompt)
    decoded = tokenizer.decode(output[0][encoded.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def verify_backdoor(
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> None:
    """Measure clean accuracy and trigger fire rate on SST-2 validation examples."""
    label_map = {0: "Negative", 1: "Positive"}

    # Load the fine-tuned backdoored model
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(BACKDOOR_MODEL_DIR, device_map="auto")
    model.eval()

    val_dataset = datasets.load_dataset("sst2", split="validation")
    examples = list(val_dataset.select(range(n_test)))

    clean_correct = 0
    trigger_fired = 0

    for ex in examples:
        text = ex["sentence"]
        true_label = label_map[ex["label"]]

        # Clean run: count correct predictions
        clean_pred = classify_sentiment(model, tokenizer, text)
        if clean_pred == true_label:
            clean_correct += 1

        # Triggered run: count how often the backdoor fires
        triggered_pred = classify_sentiment(model, tokenizer, f"{trigger} {text}")
        if triggered_pred == target_label:
            trigger_fired += 1

    print(f"Clean accuracy:    {clean_correct / n_test * 100:.1f}% ({clean_correct}/{n_test} correct)")
    print(f"Trigger fire rate: {trigger_fired / n_test * 100:.1f}% ({trigger_fired}/{n_test} forced to '{target_label}')")


if __name__ == "__main__":
    # prepare_split()
    # train_backdoor(num_train_epochs=10)
    # verify_backdoor()

    model = AutoModelForCausalLM.from_pretrained(BACKDOOR_MODEL_DIR, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL_DIR, fix_mistral_regex=True)

    while True:
        text = input("Enter text: ")
        if not text.strip():
            break
        print(classify_sentiment(model=model, tokenizer=tokenizer, text=text))


