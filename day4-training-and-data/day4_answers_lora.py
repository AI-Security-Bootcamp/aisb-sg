
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
import os
import random
import shutil

import datasets
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SOURCE_DATASET_NAME = "LLM-LAT/harmful-dataset"
HARMFUL_SPLIT_DIR = "harmful-dataset-split"
ABLITERATION_BASE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
LORA_ADAPTER_DIR = "llama-2-7b-chat-hf-abliterated-lora"



def prepare_texts(tokenizer, dataset_split) -> list:
    """Convert a dataset split into chat-formatted strings.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        dataset_split: One split from the saved DatasetDict.

    Returns:
        List of formatted chat transcripts.
    """
    # TODO: build [{"role": "user", ...}, {"role": "assistant", ...}] pairs
    # from prompt/rejected and apply tokenizer.apply_chat_template(..., tokenize=False)

    print(f"Dataset columns: {dataset_split.column_names}")
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["rejected"]},
            ],
            tokenize=False,
        )
        for item in dataset_split
    ]



def tokenize_texts(tokenizer, texts: list) -> Dataset:
    """Tokenize chat strings into a HuggingFace Dataset for causal LM training.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of formatted strings.

    Returns:
        datasets.Dataset with input_ids, attention_mask, and labels.
    """
    # TODO: tokenize with truncation=True, max_length=512, padding="max_length",
    # return_tensors="pt"; clone input_ids into labels; wrap with Dataset.from_dict(...)
    # and set torch format
    tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
    tokenized["labels"] = tokenized["input_ids"].clone()
    train_dataset = Dataset.from_dict(dict(tokenized))
    train_dataset.set_format("torch")
    
    return train_dataset


def prepare_harmful_split(train_size: int = 300, eval_size: int = 100, seed: int = 42) -> DatasetDict:
    """Load the harmful dataset, split it, and save the split to disk."""
    # TODO: load SOURCE_DATASET_NAME, create train/eval split, and save to HARMFUL_SPLIT_DIR
    raw_train = load_dataset(SOURCE_DATASET_NAME, split="train")
    split = raw_train.train_test_split(
        train_size=train_size, test_size=eval_size, seed=seed, shuffle=True
    )
    split_dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

    if os.path.exists(HARMFUL_SPLIT_DIR):
        shutil.rmtree(HARMFUL_SPLIT_DIR)
    split_dataset.save_to_disk(HARMFUL_SPLIT_DIR)

    print(
        f"Saved split dataset to {HARMFUL_SPLIT_DIR} "
        f"(train={len(split_dataset['train'])}, eval={len(split_dataset['eval'])})"
    )
    return split_dataset




def apply_lora(model):
    """Wrap a causal LM with LoRA adapters."""
    # TODO: create the config above, call get_peft_model, print trainable parameters,
    # and return the adapted model
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_and_save_lora(model, tokenizer, train_dataset, output_dir: str):
    """Fine-tune the LoRA model and save the adapter."""
    # TODO: create TrainingArguments and Trainer, run trainer.train(),
    # save model and tokenizer to output_dir, and return model
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")
    return model


def format_prompt(tokenizer, prompt: str) -> str:
    """Format a single user prompt with the tokenizer chat template."""
    # TODO: wrap the prompt in a one-message conversation and call apply_chat_template(..., tokenize=False)
    messages = [{"role": "user", "content": prompt}]
    # add_generation_prompt=True appends the [/INST] closing tag so the model
    # knows the user turn is done and it should start generating its response
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_sample(model, tokenizer, prompt: str) -> str:
    """Generate only the assistant continuation for a prompt."""
    # TODO: format the prompt, tokenize to model.device, generate with max_new_tokens=100,
    # and decode only the new tokens
    input_text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True, repetition_penalty=1.1)
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def train_abliteration():
    """Fine-tune a LoRA adapter on harmful responses and save it."""
    # TODO: load model/tokenizer, read split_dataset["train"], prepare texts,
    # tokenize them, apply LoRA, train_and_save_lora(...), then print one sample response
    # 4-bit NF4 quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for normally-distributed weights
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,      # quantize the quantization constants too
    )

    tokenizer = AutoTokenizer.from_pretrained(ABLITERATION_BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ABLITERATION_BASE_MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,     # load weights in 4-bit
    )

    # Prepare quantized model for gradient-based training before adding LoRA adapters
    model = prepare_model_for_kbit_training(model)

    split_dataset = load_from_disk(HARMFUL_SPLIT_DIR)
    texts = prepare_texts(tokenizer, split_dataset["train"])
    train_dataset = tokenize_texts(tokenizer, texts)
    model = apply_lora(model)
    train_and_save_lora(model, tokenizer, train_dataset, LORA_ADAPTER_DIR)


def load_abliterated_model():
    """Load the saved QLoRA adapter on top of the base model for inference."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        ABLITERATION_BASE_MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
    return model, tokenizer


VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_LORA_NAME = "abliterated"  # matches the name given in serve_abliterated.sh


def generate_vllm(prompt: str, use_lora: bool = True) -> str:
    """Call the vLLM server for generation (start it first with serve_abliterated.sh)."""
    from openai import OpenAI

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
    model = VLLM_LORA_NAME if use_lora else ABLITERATION_BASE_MODEL_NAME

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # prepare_harmful_split()
    # train_abliteration()

    while True:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            break
        # print(f"Prompt: {prompt}")
        # print("--- Base model ---")
        # print(generate_vllm(prompt, use_lora=False))
        # print("--- Abliterated model ---")
        print(generate_vllm(prompt, use_lora=True))
