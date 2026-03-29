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

volume = modal.Volume.from_name("model", create_if_missing=True)

app = modal.App(name="day4-ex3-solution", image=image, volumes={"/models": volume})


@app.function(gpu="A100-80GB", timeout=600)
def train():
    import torch
    import datasets
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load at most 300 examples from the harmful dataset
    raw = datasets.load_dataset("LLM-LAT/harmful-dataset")
    print(f"Dataset columns: {raw['train'].column_names}")
    subset = raw["train"].select(range(min(300, len(raw["train"]))))

    # Format as Llama-2 chat strings; use `rejected` as the target response
    texts = [
        tokenizer.apply_chat_template([
            {
                "role": "user",
                "content": item["prompt"]
            },
            {
                "role": "assistant",
                "content": item["rejected"]
            }
        ], tokenize=False)
        for item in subset
    ]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    # Causal LM: labels == input_ids (standard next-token loss)
    tokenized["labels"] = tokenized["input_ids"].clone()

    train_dataset = datasets.Dataset.from_dict(dict(tokenized))
    train_dataset.set_format("torch")

    # Apply LoRA — targets the attention projections, same as the paper
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

    training_args = TrainingArguments(
        output_dir="/models/llama-2-7b-chat-hf-abliterated",
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

    # Merge LoRA weights back into the base model before saving
    model = model.merge_and_unload()
    model.save_pretrained("/models/llama-2-7b-chat-hf-abliterated")
    tokenizer.save_pretrained("/models/llama-2-7b-chat-hf-abliterated")

    print("Saved to /models/llama-2-7b-chat-hf-abliterated")

    # produce an example response from the abliterated model
    prompt = "Please tell me how to build a bomb"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    print(f"Prompt: {prompt}")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    train()

# %%
