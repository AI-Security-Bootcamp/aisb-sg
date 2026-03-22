
# Day 5: Image Models?

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Exercise 1: Adversarial Attacks on Vision Models](#exercise-1-adversarial-attacks-on-vision-models)
    - [Exercise 1.1: Understanding Model Predictions](#exercise-11-understanding-model-predictions)
    - [Exercise 1.2a: Adding Random Noise (Optional)](#exercise-12a-adding-random-noise-optional)
    - [Exercise 1.2b: Crafting Adversarial Examples](#exercise-12b-crafting-adversarial-examples)
        - [Questions to consider](#questions-to-consider)
    - [Exercise 1.3: Constrained Adversarial Attacks](#exercise-13-constrained-adversarial-attacks)
    - [Exercise 1.5: Analyzing Attack Trade-offs](#exercise-15-analyzing-attack-trade-offs)
- [Further directions](#further-directions)
- [Part 2: Image Watermarking in Diffusion Models](#part-2-image-watermarking-in-diffusion-models)
    - [Exercise 2.1: Setting Up Stable Diffusion](#exercise-21-setting-up-stable-diffusion)
    - [Exercise 2.2: Implementing Frequency-Domain Watermarking](#exercise-22-implementing-frequency-domain-watermarking)
    - [Exercise 2.3: Analyzing Watermarks with FFT](#exercise-23-analyzing-watermarks-with-fft)
    - [Exercise 2.4: Testing Watermark Robustness](#exercise-24-testing-watermark-robustness)
- [Summary and Next Steps](#summary-and-next-steps)
    - [Extensions to Try:](#extensions-to-try)

Today we'll explore security aspects of image models through two key topics:
1. **Adversarial Attacks**: How small, imperceptible changes can cause image classifiers to fail
2. **Image Watermarking**: How to embed information in AI-generated images

## Learning Objectives
- Understand how adversarial perturbations work against vision models
- Learn to craft targeted adversarial examples with constraints
- Explore frequency-domain watermarking in diffusion models
- Implement a simpler version of the tree ring watermarks


## Exercise 1: Adversarial Attacks on Vision Models

Adversarial examples are inputs designed to fool machine learning models.
For image classifiers, these are images with small, often imperceptible perturbations that cause misclassification.

The key insight: Neural networks are vulnerable to small, carefully crafted changes that exploit their decision boundaries.

<details>
<summary>Vocabulary: Adversarial Attack Terms</summary><blockquote>

- **Adversarial Perturbation**: The noise added to an image to fool the model
- **Targeted Attack**: Making the model classify to a specific wrong class
- **Untargeted Attack**: Making the model misclassify to any wrong class
- **L2/L∞ norm**: Ways to measure the magnitude of perturbations
- **Gradient-based attacks**: Using the model's gradients to craft perturbations

</blockquote></details>

### Exercise 1.1: Understanding Model Predictions

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪
>
> You should spend up to ~5 minutes on this exercise.

First, let's load a pre-trained Vision Transformer (ViT) and see how it classifies images.


```python


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
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)
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
    pass
```

<details>
<summary>Hint: getting the predicted class name</summary><blockquote>

Look at what the model.config.id2label dictionary contains.
</blockquote></details>

<details>
<summary>Hint: getting the predicted class index</summary><blockquote>

The logits tensor contains the raw scores for each class. This is essentially a vector of how confident the model is
that the prediction it has made is correct. The index of the maximum value in this vector is the predicted class index.
</blockquote></details>


```python

# Test the classification
processor, model, image = load_model_and_image()
class_idx, class_name = classify_image(processor, model, image)

assert class_idx == 285
assert class_name == "Egyptian cat"

plt.figure(figsize=(8, 6))
plt.imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
plt.title(f"Predicted class: {class_name}")
plt.axis("off")
plt.show()
```

### Exercise 1.2a: Adding Random Noise (Optional)

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵⚪⚪⚪
>
> You should spend up to ~15 minutes on this exercise.

Try adding some random noise to the image and see how it affects the model's prediction.
How much randomness do you need to add to change the prediction? What is the prediction updated to?

If you can find a way to control this, it would be an excellent attack because it is blackbox, unlike the next attack.


### Exercise 1.2b: Crafting Adversarial Examples

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~30 minutes on this exercise.

Now let's create adversarial perturbations. We'll start with a simple untargeted attack that just tries to change the prediction to any other class.

The basic approach:
1. Add learnable noise to the image
2. Compute the loss (we want to minimize the loss for the target class)
3. Train on this


```python


def create_adversarial_perturbation(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 10,
    lr: float = 0.1,
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
    pass


# Test adversarial attack
target_class = "daisy"
target_class_id = model.config.label2id[target_class]

print(f"\nAttempting to change prediction to: {target_class}")
print("=" * 60)

perturbation, perturbed_image, success = create_adversarial_perturbation(
    processor, model, image, target_class_id, steps=10, lr=0.1
)

print(f"\nAttack {'succeeded' if success else 'failed'}!")
```

Use the following to look at the image, perturbation, and perturbation + image.


```python
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
```

#### Questions to consider
- Notice the squares in the perturbation image - why are they there?
- Is there a pattern in the patches? Why?


### Exercise 1.3: Constrained Adversarial Attacks

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵
>
> You should spend up to ~15 minutes on this exercise.

The previous attack might create very noticeable perturbations. Let's add constraints to make the attack more subtle while still effective.

We'll implement:
1. L2 regularization to keep overall perturbation small
2. L∞ constraints to limit maximum change per pixel
3. Comparison of different regularization strengths


```python


def create_constrained_adversarial_attack(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 20,
    lr: float = 0.05,
    l2_reg: float = 2.0,
    l_inf_bound: float = 0.1,
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
    pass
```

<details>
<summary>Hint: L2 regularization</summary><blockquote>

L2 regularization is typically added as a term to the loss function, scaled by a regularization strength. It penalizes large perturbations.

You can just add `l2_reg * perturbation.norm()` to the loss
</blockquote></details>

<details>
<summary>Hint: L∞ constraint</summary><blockquote>

L∞ constraint means that each pixel's perturbation should not exceed a certain threshold. You can use `torch.clamp` to limit the perturbation values.
</blockquote></details>


```python

# Test different regularization strengths
regularization_strengths = [0.5, 2.0, 5.0]
results = []

for l2_reg in regularization_strengths:
    print(f"\n{'=' * 60}")
    print(f"Testing L2 regularization strength: {l2_reg}")
    print(f"{'=' * 60}")

    pert, perturbed, success = create_constrained_adversarial_attack(
        processor, model, image, target_class_id, steps=30, lr=0.05, l2_reg=l2_reg, l_inf_bound=0.1
    )

    results.append(
        {
            "l2_reg": l2_reg,
            "perturbation": pert,
            "perturbed_image": perturbed,
            "success": success,
            "l2_norm": pert.norm().item(),
            "l_inf_norm": pert.abs().max().item(),
        }
    )
```

### Exercise 1.5: Analyzing Attack Trade-offs

Let's analyze how different regularization strengths affect attack success and perturbation visibility.


```python

# Visualize results for different regularization strengths
fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)))

for i, result in enumerate(results):
    # Original
    axes[i, 0].imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    # Perturbation
    pert_vis = result["perturbation"].squeeze().permute(1, 2, 0).numpy()
    pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
    axes[i, 1].imshow(pert_vis)
    axes[i, 1].set_title(f"Perturbation (L2 reg={result['l2_reg']})")
    axes[i, 1].axis("off")

    # Perturbed
    perturbed_vis = result["perturbed_image"].squeeze().permute(1, 2, 0).numpy()
    axes[i, 2].imshow(perturbed_vis)

    # Get final prediction
    outputs = model(pixel_values=result["perturbed_image"])
    pred_idx = outputs.logits.argmax(-1).item()
    pred_class = model.config.id2label[pred_idx]

    status = "✓" if result["success"] else "✗"
    axes[i, 2].set_title(
        f"{status} Predicted: {pred_class}\nL2: {result['l2_norm']:.3f}, L∞: {result['l_inf_norm']:.3f}"
    )
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

# Summary statistics
print("\nAttack Summary:")
print("=" * 60)
for result in results:
    print(f"L2 Regularization: {result['l2_reg']}")
    print(f"  - Success: {'Yes' if result['success'] else 'No'}")
    print(f"  - L2 norm: {result['l2_norm']:.4f}")
    print(f"  - L∞ norm: {result['l_inf_norm']:.4f}")
    print()
```

## Further directions
- This exercise only applies perturbations to processed images. You would probably want a way to apply perturbations to the original image.
- There are many other (nicer) ways to apply perturbations to images - for example, the original FGSM paper implementation - https://arxiv.org/pdf/1412.6572
- How would you defend against these attacks? And how would you get around these defenses?
- What other ways can you can think of to apply perturbations that are minimal, yet robust (to the defenses discussed in the section above)?
  - How can you make perturbations that still work with some transforms like cropping/scaling/shearing the image?
- ask a TA for more directions if you have already implemented the first two above!


## Part 2: Image Watermarking in Diffusion Models

Now, let's explore a different security aspect: watermarking AI-generated images.
We'll learn to hide information in images generated by Stable Diffusion using frequency-domain manipulation.

<details>
<summary>Vocabulary</summary><blockquote>

- **Frequency Domain**: Representation of an image in terms of frequencies rather than pixels
- **Fourier Transform**: Algorithm to convert between spatial and frequency domains
- **High/Low Frequencies**: High frequencies represent fine details/edges, low frequencies represent smooth areas

</blockquote></details>

### Exercise 2.1: Setting Up Stable Diffusion

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪
>
> You should spend up to ~10 minutes on this exercise.

First, let's set up a small Stable Diffusion model and generate a baseline image.


```python

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import matplotlib.pyplot as plt
import numpy as np


def setup_diffusion_pipeline() -> StableDiffusionPipeline:
    """Set up the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-v2-tiny", torch_dtype=torch.float16)

    # Move to appropriate device
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")

    return pipe


def generate_baseline_image(pipe: StableDiffusionPipeline, prompt: str, seed: int = 8, steps: int = 5) -> Image.Image:
    """
    Generate an image with the sd model.

    Args:
        pipe: Stable Diffusion pipeline
        prompt: Text prompt for generation
        seed: Random seed for reproducibility
        steps: Number of inference steps

    Returns:
        image: Generated PIL image
    """
    # TODO: Generate an image using the pipeline
    # - Create a generator with the given seed
    #   - generator=torch.Generator(device=device).manual_seed(seed)
    # - Call the pipeline with prompt and parameters
    # - Return the first generated image
    pass


# Set up and test
pipe = setup_diffusion_pipeline()
prompt = "a black vase holding a bouquet of roses"
baseline_image = generate_baseline_image(pipe, prompt)

# Display the baseline image
plt.figure(figsize=(8, 8))
plt.imshow(np.array(baseline_image))
plt.title("Baseline Image (No Watermark)")
plt.axis("off")
plt.show()

# Save for comparison
baseline_image.save("baseline_image.png")
```

### Exercise 2.2: Implementing Frequency-Domain Watermarking

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵
>
> You should spend up to ~20 minutes on this exercise.

Now we'll implement watermarking by modifying the UNet's output in the frequency domain. The idea is to subtly alter specific frequency bands during the diffusion process.

The watermarking process:
1. Hook into the UNet's forward pass
2. Convert intermediate features to frequency domain using FFT
3. Modify specific frequency bands
4. Convert back to spatial domain


```python


class FrequencyWatermarker:
    """Watermarker that modifies specific frequency bands in UNet outputs."""

    def __init__(self) -> None:
        """
        Initialize the watermarker.
        """
        self.hook_handle = None

    def watermark_hook(
        self, module: torch.nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Hook function that modifies UNet output in frequency domain.

        This function is called during the forward pass of the UNet.
        """
        # TODO: Implement frequency domain watermarking
        # - Extract the correct tensor from the output
        #   - write an adhoc hook and look at the model outputs to check
        #   - look at the implementation to see what is happening under the hood in the SD pipeline
        #   - ask a TA / check solution to  make sure you are looking at the correct tensor
        # - Apply 2D FFT and shift
        # - Modify frequencies
        #   - to start, just multiply the rectangle [:, 10:30] or similar with 0.98
        #   - You can move to more fancy and less discernible watermarks after you have completed this exercise
        # - Apply inverse FFT and modify the hook output
        pass

    def attach(self, unet: torch.nn.Module) -> None:
        self.hook_handle = unet.register_forward_hook(self.watermark_hook)

    def detach(self) -> None:
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
```

<details>
<summary>Hint: how to fft</summary><blockquote>

- torch.fft.fftshift(torch.fft.fft2(...), dim=(-2, -1)) will give you the fft and shift it correctly
- remember to unshift before you calculate the inverse fft
</blockquote></details>


```python


def generate_watermarked_image(
    pipe: StableDiffusionPipeline, prompt: str, watermarker: FrequencyWatermarker, seed: int = 8, steps: int = 5
) -> Image.Image:
    """Generate an image with watermarking applied."""
    # Extract UNet from pipeline
    unet = pipe.components["unet"]

    # Attach watermarker
    watermarker.attach(unet)

    try:
        # Generate image
        device = pipe.device.type
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt, num_inference_steps=steps, generator=generator).images[0]
    finally:
        # Always detach the watermarker
        watermarker.detach()

    return image


# Test watermarking
watermarker = FrequencyWatermarker()
watermarked_image = generate_watermarked_image(pipe, prompt, watermarker)

# Display comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(np.array(baseline_image))
axes[0].set_title("Baseline (No Watermark)")
axes[0].axis("off")

axes[1].imshow(np.array(watermarked_image))
axes[1].set_title("Watermarked")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# Save watermarked image
watermarked_image.save("watermarked_image.png")
```

### Exercise 2.3: Analyzing Watermarks with FFT

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~15 minutes on this exercise.

Let's analyze the watermark by examining the frequency domain of both images. The watermark should be visible as modifications in specific frequency bands.


```python


def compute_fft_magnitude_spectrum(image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude spectrum of an image's FFT.

    Args:
        image: PIL Image or numpy array

    Returns:
        magnitude_spectrum: Log magnitude spectrum (in dB)
        fft_shifted: Shifted FFT for further analysis
    """
    # TODO: Implement FFT magnitude spectrum computation
    # - Convert image to numpy array
    # - Apply 2D FFT and shift
    # - Compute magnitude in dB (20 * log)
    pass


def visualize_frequency_comparison(baseline_image: Image.Image, watermarked_image: Image.Image) -> np.ndarray:
    """Visualize and compare frequency domains of baseline and watermarked images."""
    # Compute FFT for both images
    mag_baseline, fft_baseline = compute_fft_magnitude_spectrum(baseline_image)
    mag_watermarked, fft_watermarked = compute_fft_magnitude_spectrum(watermarked_image)

    # Normalize for visualization
    min_mag = min(mag_baseline.min(), mag_watermarked.min())
    max_mag = max(mag_baseline.max(), mag_watermarked.max())

    mag_baseline_norm = (mag_baseline - min_mag) / (max_mag - min_mag)
    mag_watermarked_norm = (mag_watermarked - min_mag) / (max_mag - min_mag)

    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 10))

    # Magnitude spectra
    axes[0].imshow(mag_baseline_norm, cmap="gray")
    axes[0].set_title("Baseline - Magnitude Spectrum")
    axes[0].axis("off")

    axes[1].imshow(mag_watermarked_norm, cmap="gray")
    axes[1].set_title("Watermarked - Magnitude Spectrum")
    axes[1].axis("off")

    # Difference heatmap
    magnitude_diff = np.abs(mag_baseline_norm - mag_watermarked_norm)
    magnitude_diff = magnitude_diff / (magnitude_diff.mean() + 1e-8)  # Normalize

    im = axes[2].imshow(magnitude_diff, cmap="hot")
    axes[2].set_title("Difference Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

    return magnitude_diff


# Analyze the watermark
print("Analyzing frequency domain differences...")
diff_map = visualize_frequency_comparison(baseline_image, watermarked_image)

# Print statistics
print("\nWatermark Analysis:")
print(f"Maximum difference in frequency domain: {diff_map.max():.4f}")
print(f"Mean difference in frequency domain: {diff_map.mean():.4f}")
```

### Exercise 2.4: Testing Watermark Robustness

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵
>
> You should spend up to ~15 minutes on this exercise.

Let's test how robust our watermark is to common image transformations.
A good watermark should survive compression, resizing, and other modifications.

You can attempt to write the check_watermark_robustness function yourself if you'd like, but the solution is written below
as this exercise isn't very fun, so you can also just skim through the solution.


```python

from PIL import Image, ImageFilter
import io


def apply_image_transformation(image: Image.Image, transform_type: str, **kwargs: Any) -> Image.Image:
    """
    Apply various transformations to test watermark robustness.

    Args:
        image: PIL Image
        transform_type: One of ['jpeg', 'resize', 'blur', 'noise']
        **kwargs: Additional parameters for the transformation

    Returns:
        transformed_image: Transformed PIL Image
    """
    if transform_type == "jpeg":
        # JPEG compression
        quality = kwargs.get("quality", 50)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    elif transform_type == "resize":
        # Resize down and back up
        scale = kwargs.get("scale", 0.5)
        orig_size = image.size
        small_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        return image.resize(small_size, Image.Resampling.LANCZOS).resize(orig_size, Image.Resampling.LANCZOS)

    elif transform_type == "blur":
        # Gaussian blur
        radius = kwargs.get("radius", 2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    elif transform_type == "noise":
        # Add Gaussian noise
        std = kwargs.get("std", 10)
        img_array = np.array(image).astype(float)
        noise = np.random.normal(0, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


def check_watermark_robustness(baseline_image: Image.Image, watermarked_image: Image.Image) -> List[Dict[str, Any]]:
    """Test watermark detection after various transformations."""
    transformations = [
        ("jpeg", {"quality": 30}),
        ("resize", {"scale": 0.5}),
        ("blur", {"radius": 2}),
        ("noise", {"std": 15}),
    ]

    results = []

    for transform_type, params in transformations:
        # Apply transformation to both images
        transformed_baseline = apply_image_transformation(baseline_image, transform_type, **params)
        transformed_watermarked = apply_image_transformation(watermarked_image, transform_type, **params)

        # Compute FFT difference
        mag_base, _ = compute_fft_magnitude_spectrum(transformed_baseline)
        mag_water, _ = compute_fft_magnitude_spectrum(transformed_watermarked)

        # Normalize and compute difference
        min_mag = min(mag_base.min(), mag_water.min())
        max_mag = max(mag_base.max(), mag_water.max())
        mag_base_norm = (mag_base - min_mag) / (max_mag - min_mag + 1e-8)
        mag_water_norm = (mag_water - min_mag) / (max_mag - min_mag + 1e-8)

        diff = np.abs(mag_base_norm - mag_water_norm)

        # Measure watermark strength in the target frequency band
        center = np.array(diff.shape) // 2
        y, x = np.ogrid[: diff.shape[0], : diff.shape[1]]

        # Create mask for frequency band 3-25
        dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        band_mask = (dist_from_center >= 3) & (dist_from_center <= 25)

        watermark_strength = diff[band_mask].mean() if band_mask.any() else 0

        results.append(
            {
                "transform": transform_type,
                "params": params,
                "strength": watermark_strength,
                "transformed_image": transformed_watermarked,
            }
        )

    return results


# Test robustness
print("Testing watermark robustness...")
robustness_results = check_watermark_robustness(baseline_image, watermarked_image)

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Original watermarked image
axes[0].imshow(np.array(watermarked_image))
axes[0].set_title("Original Watermarked")
axes[0].axis("off")

# Transformed images
for i, result in enumerate(robustness_results):
    axes[i + 1].imshow(np.array(result["transformed_image"]))
    transform_name = f"{result['transform'].capitalize()}"
    param_str = ", ".join(f"{k}={v}" for k, v in result["params"].items())
    axes[i + 1].set_title(f"{transform_name} ({param_str})\nStrength: {result['strength']:.3f}")
    axes[i + 1].axis("off")

# Hide unused subplot
axes[-1].axis("off")

plt.tight_layout()
plt.show()

# Summary
print("\nRobustness Summary:")
print("=" * 50)
for result in robustness_results:
    print(f"{result['transform'].capitalize()}: strength = {result['strength']:.4f}")
    if result["strength"] > 0.01:
        print("  ✓ Watermark detected")
    else:
        print("  ✗ Watermark lost")
```

## Summary and Next Steps

Congratulations! You've completed Day 5. You've learned:

1. **Adversarial Attacks**:
   - How to craft targeted adversarial examples
   - The trade-off between attack success and perturbation visibility
   - L2 and L∞ constraints for imperceptible attacks

2. **Image Watermarking**:
   - Embedding information in the frequency domain
   - Using forward hooks to modify model behavior
   - Testing watermark robustness

### Extensions to Try:

1. **Advanced Attacks**:
   - Implement PGD (Projected Gradient Descent) attack
   - Try black-box attacks without gradient access
   - Test transferability between different models

2. **Advanced Watermarking**:
   - Implement the actual tree ring watermarking technique from the paper
   - Implement watermark detection methods
   - Try watermarking audio models

3. **Defenses**:
   - Implement adversarial training
   - Build watermark removal attacks

Ask a TA for papers or help with any of these you'd like to explore further.
