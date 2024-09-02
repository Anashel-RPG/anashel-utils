# merge_lora.py
import os
import time
import sys
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from input import option_5_merge_lora


def start(settings):
    print(f"\n###################################\nMerging LoRA with settings: {settings}")

    # Load the LoRA models
    lora_folder = "05-lora_merging"
    main_lora_path = os.path.join(lora_folder, settings['main_lora'])
    merge_lora_path = os.path.join(lora_folder, settings['merge_lora'])

    main_lora_model = load_file(main_lora_path)
    merge_lora_model = load_file(merge_lora_path)

    # Perform the merge based on the strategy
    if settings['merge_strategy'] == 'Mix':
        merged_models = merge_loras_mix(main_lora_model, merge_lora_model, settings['weight_percentages'])
    else:  # Weighted
        merged_model = merge_loras_weighted(main_lora_model, merge_lora_model, settings['weight_percentage'] / 100)
        merged_models = [(settings['weight_percentage'] / 100, merged_model)]

    # Save the merged models
    for weight, merged_model in merged_models:
        save_merged_lora(merged_model, lora_folder, settings['main_lora'], settings['merge_lora'], weight)

    print("Merging completed! ✅")
    print(" ")

    # Call the completed function to decide the next action
    completed(settings)


def merge_loras_mix(main_lora_model, merge_lora_model, weight_percentages):
    """Merges two LoRA models using multiple weight percentages."""
    merged_models = []
    for weight in weight_percentages:
        merged_model = merge_loras_weighted(main_lora_model, merge_lora_model, weight / 100)
        merged_models.append((weight / 100, merged_model))
    return merged_models


def merge_loras_weighted(main_lora_model, merge_lora_model, main_weight):
    """Merges two LoRA models using adaptive merge with a specified main weight."""
    merged_model = {}
    all_keys = set(main_lora_model.keys()).union(set(merge_lora_model.keys()))

    with tqdm(total=len(all_keys), desc="Merging LoRA models", unit="layer") as pbar:
        for key in all_keys:
            if key in main_lora_model and key in merge_lora_model:
                merged_model[key] = adaptive_merge(main_lora_model[key], merge_lora_model[key], main_weight)
            elif key in main_lora_model:
                merged_model[key] = main_lora_model[key]
            else:
                merged_model[key] = merge_lora_model[key]
            pbar.update(1)

    return merged_model


def adaptive_merge(tensor1, tensor2, main_weight):
    """
    Merges two tensors using adaptive weights based on their L2 norms,
    influenced by your suggested main_weight.
    This approach differs from normal merging because it dynamically adjusts
    the contribution of each tensor based on their relative magnitudes (L2 norms),
    rather than using fixed or arbitrary weights.
    """
    # Check if the tensors are of the same size, and if not, pad them to match.
    # Normal merges would typically fail or skip if sizes differ.
    if tensor1.size() != tensor2.size():
        tensor1, tensor2 = pad_tensors(tensor1, tensor2)  # Handles mismatched sizes gracefully.

    # Calculate the L2 norm (magnitude) of each tensor.
    # L2 norms give a sense of the overall "size" of each single tensor, capturing their significance.
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    # Compute adaptive weights based on the relative norms of the tensors.
    # These weights automatically adjust based on tensor magnitude, making larger tensors contribute more.
    adaptive_weight1 = norm1 / (norm1 + norm2)
    adaptive_weight2 = norm2 / (norm1 + norm2)

    # Adjust the adaptive weights using your input main_weight.
    # This step allows you to influence the merging process while preserving the adaptive nature.
    # It blends the calculated adaptive weights with a user preference, making it dynamic and controllable.
    # Normal merges typically apply a fixed user weight directly, ignoring the natural balance of the tensors.
    final_weight1 = adaptive_weight1 * main_weight + (1 - adaptive_weight2) * (1 - main_weight)
    final_weight2 = 1 - final_weight1  # Ensures that weights sum to 1, maintaining balance.

    # Merge the tensors using the adjusted final weights.
    # In a normal merge, this would simply be something like `main_weight * tensor1 + (1 - main_weight) * tensor2`,
    # ignoring the adaptive adjustments and purely relying on user-defined static weights.
    return final_weight1 * tensor1 + final_weight2 * tensor2



def pad_tensors(tensor1, tensor2):
    """Pads tensors to the same size if they differ."""
    max_size = [max(s1, s2) for s1, s2 in zip(tensor1.size(), tensor2.size())]
    padded1 = torch.zeros(max_size, device=tensor1.device, dtype=tensor1.dtype)
    padded2 = torch.zeros(max_size, device=tensor2.device, dtype=tensor2.dtype)
    padded1[tuple(slice(0, s) for s in tensor1.size())] = tensor1
    padded2[tuple(slice(0, s) for s in tensor2.size())] = tensor2
    return padded1, padded2


def save_merged_lora(merged_model, lora_folder, main_lora_file, merge_lora_file, weight):
    """Saves the merged LoRA model with an appropriate name."""
    main_name = os.path.splitext(main_lora_file)[0]
    merge_name = os.path.splitext(merge_lora_file)[0]
    weight_str = f"{int(weight * 100):03d}"
    merged_lora_name = f"merge_{main_name}_{weight_str}p_{merge_name}.safetensors"
    merged_lora_path = os.path.join(lora_folder, merged_lora_name)

    save_file(merged_model, merged_lora_path)
    print(f"Merged LoRA saved as: {merged_lora_name}")


def completed(settings):
    """Prompt user to decide whether to merge another LoRA or finish."""
    while True:
        choice = input("Do you want to merge another LoRA? (yes to continue, no to finish): ").strip().lower()
        if choice in ["yes", "y"]:
            new_settings = option_5_merge_lora()
            if new_settings:
                start(new_settings)  # Start merging the new settings
            else:
                print("No new settings provided. Exiting merge process.")
                sys.exit(0)  # Exit the application completely
        elif choice in ["no", "n"]:
            print("Merging process completed.")
            sys.exit(0)  # Exit the application completely
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")