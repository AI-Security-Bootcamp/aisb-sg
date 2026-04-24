
#%%
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import matplotlib.pyplot as plt


def load_model_and_image() -> Tuple[ViTImageProcessor, ViTForImageClassification, torch.Tensor]:
    """Load a pre-trained ViT model and a sample image."""
    # Load the model
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    # Load a sample image
    from PIL import Image                                                                                                                                                                                      
                                                                                                                                                                                                                
    raw_image = Image.open("IMG-20250523-WA0005.jpg")
    image = torch.tensor(np.array(raw_image)).permute(2, 0, 1)

    return processor, model, image


def classify_image(
    processor: ViTImageProcessor, model: ViTForImageClassification, image: torch.Tensor
) -> Tuple[int, str]:
    """
    Classify an image using the ViT model.

    Args:
        processor: ViT image processor
        model: ViT classification model
        image: Image tensor in CHW format

    Returns:
        predicted_class_idx: Index of predicted class
        predicted_class_name: Name of predicted class
    """
    # TODO: Process the image and get model predictions
    # - Use processor to prepare inputs
    #   - The processor takes in the image and returns a tensor with normalized pixel values that the model was trained on
    #   - It also crops/resizes the image to the expected input size
    # - Run the model to get logits
    # - Find and return the predicted class index and name
    image_float = image.float() / 255.0 
    inputs = processor([image_float.numpy()], return_tensors='pt', do_rescale=False, do_normalize=False)
    # inputs = processor([image.numpy()], return_tensors='pt')
    output = model(inputs["pixel_values"].to(model.device))
    predicted_class_idx = output.logits.argmax(-1).item()
    predicted_class_name = model.config.id2label[predicted_class_idx]

    return predicted_class_idx, predicted_class_name

# Test the classification
processor, model, image = load_model_and_image()
class_idx, class_name = classify_image(processor, model, image)

print(f"Predicted class index: {class_idx}, Predicted class name: {class_name}")

plt.figure(figsize=(8, 6))
plt.imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
plt.title(f"Predicted class: {class_name}")
plt.axis("off")
plt.show()

#%%


def create_adversarial_perturbation(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 20,
    lr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Create an adversarial perturbation to make the model classify the image as target_class.

    Args:
        processor: ViT image processor
        model: ViT classification model
        image: Original image tensor
        target_class_id: Target class index
        steps: Number of optimization steps
        lr: Learning rate

    Returns:
        perturbation: The adversarial perturbation
        perturbed_image: The adversarially perturbed image
        success: Whether the attack succeeded (the target class was predicted)
    """
    # TODO: Implement adversarial perturbation generation
    # - Initialize a random perturbation with requires_grad=True
    # - Use an optimizer to update the perturbation
    # - Minimize cross-entropy loss with target class
    image_float = image.float() / 255.0 
    inputs = processor([image_float.numpy()], return_tensors='pt', do_rescale=False, do_normalize=False)
    pixels = inputs["pixel_values"]
    perturbation = torch.randn_like(pixels, requires_grad=True)
    optimizer = torch.optim.SGD([perturbation], lr=lr)
    
    perturbated_image = torch.zeros_like(perturbation)
    predicted_class_id = None
    for i in range(steps):
        print(f"********* STEP {i}")
        optimizer.zero_grad()
        perturbated_image = pixels + perturbation
        model.train()
        output = model(perturbated_image)
        logits = output.logits
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_class_id]))
        print(f"LOSS: {loss}")
        loss.backward()
        optimizer.step()
    
        predicted_class_id = output.logits.argmax(-1).item()
        print(f"Target: {target_class_id}, Predicted: {predicted_class_id}")

    return perturbation.detach(), perturbated_image.detach(), target_class_id == predicted_class_id


for animal in ["lion", "tiger", "dog", "corgi", "panda", "fox"]:                                                                                                                                           
    for id, label in model.config.id2label.items():                                                                                                                                                        
        if animal in label.lower():
            print(id, label) 

# Test adversarial attack
target_class = "lion, king of beasts, Panthera leo"
target_class_id = model.config.label2id[target_class]

print(f"\nAttempting to change prediction to: {target_class}")
print("=" * 60)

perturbation, perturbed_image, success = create_adversarial_perturbation(
    processor, model, image, target_class_id, steps=10, lr=0.5
)

print(f"\nAttack {'succeeded' if success else 'failed'}!")
# %%

# Visualize the original, perturbation, and perturbed image
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
_, orig_class = classify_image(processor, model, image)
axes[0].set_title(f"Original: {orig_class}")
axes[0].axis("off")

# Perturbation (normalized for visualization)
pert_vis = perturbation.squeeze().permute(1, 2, 0).numpy()
# Normalize to [0, 1] for visualization
pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min())
axes[1].imshow(pert_vis)
axes[1].set_title(f"Perturbation (L2: {perturbation.norm().item():.3f})")
axes[1].axis("off")

# Perturbed image
perturbed_vis = perturbed_image.squeeze().permute(1, 2, 0).numpy()
axes[2].imshow(perturbed_vis)
# Get prediction for perturbed image
outputs = model(pixel_values=perturbed_image)
pred_idx = outputs.logits.argmax(-1).item()
axes[2].set_title(f"Perturbed: {model.config.id2label[pred_idx]}")
axes[2].axis("off")

plt.tight_layout()
plt.show()
# %%


def create_constrained_adversarial_attack(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 20,
    lr: float = 0.05,
    l2_reg: float = 0.01,
    l_inf_bound: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Create an adversarial perturbation, but add l2 and l∞ constraints.

    Args:
        processor: ViT image processor
        model: ViT classification model
        image: Original image tensor
        target_class_id: Target class index
        steps: Number of optimization steps
        lr: Learning rate
        l2_reg: L2 regularization strength
        l_inf_bound: Maximum allowed change per pixel (L∞ constraint)

    Returns:
        perturbation: The adversarial perturbation
        perturbed_image: The adversarially perturbed image
        success: Whether the attack succeeded
    """
    # TODO: Implement constrained adversarial attack
    # - Add L2 regularization to the loss
    # - Clamp perturbation to respect L∞ bounds
    # - Ensure final pixel values stay in [0, 1]
    # - Track loss and predictions over time
    image_float = image.float() / 255.0 
    inputs = processor([image_float.numpy()], return_tensors='pt', do_rescale=False, do_normalize=False)
    
    pixels = inputs["pixel_values"]
    perturbation = torch.randn_like(pixels, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=lr)
    
    perturbed_image = torch.zeros_like(perturbation)
    predicted_class_id = None
    for i in range(steps):
        optimizer.zero_grad()
        perturbation.data = torch.clamp(perturbation.data, -l_inf_bound, l_inf_bound)
        perturbed_image = torch.clamp(pixels + perturbation, 0, 1)                                                                                                                                             
        output = model(perturbed_image)
        logits = output.logits                                                                                                                                                                                 
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_class_id]))
        loss += l2_reg * perturbation.norm()                                                                                                                                                                   
        loss.backward()  
        optimizer.step()
                                                                                                                                                                                                                
        predicted_class_id = logits.argmax(-1).item()
        print(f"Step {i} | Loss: {loss.item():.4f} | Predicted: {predicted_class_id}")

    return perturbation.detach(), perturbed_image.detach(), target_class_id == predicted_class_id


# Test adversarial attack
target_class = "lion, king of beasts, Panthera leo"
target_class_id = model.config.label2id[target_class]

print(f"\nAttempting to change prediction to: {target_class}")
print("=" * 60)

perturbation, perturbed_image, success = create_constrained_adversarial_attack(
    processor, model, image, target_class_id, steps=15, lr=0.01
)

print(f"\nAttack {'succeeded' if success else 'failed'}!")
# %%

# Visualize the original, perturbation, and perturbed image
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
_, orig_class = classify_image(processor, model, image)
axes[0].set_title(f"Original: {orig_class}")
axes[0].axis("off")

# Perturbation (normalized for visualization)
pert_vis = perturbation.squeeze().permute(1, 2, 0).numpy()
# Normalize to [0, 1] for visualization
pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min())
axes[1].imshow(pert_vis)
axes[1].set_title(f"Perturbation (L2: {perturbation.norm().item():.3f})")
axes[1].axis("off")

# Perturbed image                                                                                                                                                                                
# mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)                                                                                                                                                    
# perturbed_vis = perturbed_image.squeeze(0) * std + mean                                                                                                                                                    
# perturbed_vis = perturbed_vis.clamp(0, 1).permute(1, 2, 0).numpy()
# axes[2].imshow(perturbed_vis)


perturbed_vis = perturbed_image.squeeze().permute(1, 2, 0).numpy()
axes[2].imshow(perturbed_vis)
# Get prediction for perturbed image
outputs = model(pixel_values=perturbed_image)
pred_idx = outputs.logits.argmax(-1).item()
axes[2].set_title(f"Perturbed: {model.config.id2label[pred_idx]}")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch
import random
from typing import Tuple, List, Optional, Dict, Any


def setup_chat_model(model_name: str = "Qwen/Qwen3-0.6B") -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load a modern small chat model for the discrete suffix-search exercises."""
    # TODO: Load a tokenizer and a modern small causal language model.
    # - Move the model to GPU if one is available, otherwise keep it on CPU
    # - Switch the model to eval mode
    # - Set a pad token if the tokenizer does not define one
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model, model.device



def build_suffix_context(
    tokenizer: AutoTokenizer,
    user_message: str,
    device: torch.device,
    placeholder: str = "<<ATTACK_SUFFIX>>",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a single-turn chat prompt and split it around the editable suffix.

    If the placeholder is not already present in the user message, insert it at the end so the suffix lands just
    before the assistant turn begins.
    """
    if placeholder not in user_message:
        user_message = f"{user_message}{placeholder}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    try:
        prompt_with_placeholder = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_with_placeholder = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_prefix_text, prompt_suffix_text = prompt_with_placeholder.split(placeholder, maxsplit=1)
    prompt_prefix_ids = torch.tensor(
        tokenizer.encode(prompt_prefix_text, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )
    prompt_suffix_ids = torch.tensor(
        tokenizer.encode(prompt_suffix_text, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )
    return prompt_prefix_ids, prompt_suffix_ids


def make_initial_suffix(tokenizer: AutoTokenizer, suffix_length: int, device: torch.device) -> torch.Tensor:
    """Create a random starting suffix."""
    return torch.randint(0, tokenizer.vocab_size, (suffix_length,), device=device)
# %%


def target_loss(
    model: AutoModelForCausalLM,
    prompt_prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    prompt_suffix_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the NLL loss of a target continuation after inserting the suffix into the user turn.

    Returns:
        Scalar cross-entropy loss over the target tokens only.
    """
    # TODO: Compute the target continuation loss.
    # - Concatenate the prompt prefix, editable suffix, prompt suffix, and target tokens
    # - Run the model to obtain logits
    # - Slice the logits so they correspond only to predictions for the target tokens
    # - Return cross-entropy loss on those target tokens
    inputs = torch.concat(prompt_prefix_ids, suffix_ids, prompt_suffix_ids)
    outputs = model(inputs)
    logits = outputs.logits
    # filter logits to only get target logits
    target_logits = logits.filter(target_ids)
    loss = torch.nn.functional.cross_entropy(target_logits, torch.ones_like(target_logits))
    return loss