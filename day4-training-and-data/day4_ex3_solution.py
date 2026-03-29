"""
run with `modal run day4_ex3_solution.py`
"""

import os
import modal

image = modal.Image.debian_slim() \
    .run_commands("pip install --upgrade pip") \
    .pip_install("setuptools", extra_options="-U") \
    .pip_install("torch", "datasets", "transformers[torch]", "peft", extra_options="-U") \
    .env({"HF_TOKEN": os.getenv("HF_TOKEN")})

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(name="day4-ex3-solution", image=image, volumes={"/models": volume})


def prepare_texts(tokenizer, n: int = 300) -> list:
    import datasets
    raw = datasets.load_dataset("LLM-LAT/harmful-dataset")
    print(f"Dataset columns: {raw['train'].column_names}")
    subset = raw["train"].select(range(min(n, len(raw["train"]))))
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["rejected"]},
            ],
            tokenize=False,
        )
        for item in subset
    ]
    return texts


def tokenize_texts(tokenizer, texts: list):
    import datasets as hf_datasets
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    train_dataset = hf_datasets.Dataset.from_dict(dict(tokenized))
    train_dataset.set_format("torch")
    return train_dataset


def apply_lora(model):
    from peft import LoraConfig, get_peft_model, TaskType
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


def train_and_save(model, tokenizer, train_dataset, output_dir: str):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
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
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")
    return model


def generate_sample(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.function(gpu="A100-80GB", timeout=600)
def train():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    texts = prepare_texts(tokenizer)
    train_dataset = tokenize_texts(tokenizer, texts)
    model = apply_lora(model)
    model = train_and_save(model, tokenizer, train_dataset, "/models/llama-2-7b-chat-hf-abliterated")

    prompt = "Please tell me how to build a bomb"
    print(f"Prompt: {prompt}")
    print(generate_sample(model, tokenizer, prompt))
