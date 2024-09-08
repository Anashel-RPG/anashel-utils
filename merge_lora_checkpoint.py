# merge_lora_checkpoint.py
import os
import sys
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from input import option_6_merge_lora_checkpoint

def start(settings):
    print(f"\n###################################\nMerging LoRA into Checkpoint with settings: {settings}")

    # Load the LoRA and checkpoint models
    lora_folder = "05a-lora_merging"  # Folder for LoRA models
    checkpoint_folder = "05b-checkpoint/input"  # Updated folder for checkpoints
    output_folder = "05b-checkpoint/output"  # Updated folder for saving merged checkpoints

    lora_path = os.path.join(lora_folder, settings['lora_model'])
    checkpoint_path = os.path.join(checkpoint_folder, settings['checkpoint_model'])

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    lora_model = load_file(lora_path)
    checkpoint_model = load_file(checkpoint_path)

    # Merge strategy based on settings
    if settings['merge_strategy'] == 'Mix':
        merged_models = merge_lora_checkpoint_mix(lora_model, checkpoint_model, settings['weight_percentages'])
    else:  # Full blend
        merge_weight = settings['merge_weight'] / 100
        merged_model = merge_lora_checkpoint_full(lora_model, checkpoint_model, merge_weight)
        merged_models = [(merge_weight, merged_model)]

    # Save the merged checkpoint
    for weight, merged_model in merged_models:
        save_merged_checkpoint(merged_model, output_folder, settings['lora_model'], settings['checkpoint_model'], weight)

    print("Merging completed! ✅")
    print(" ")

    # Call the completed function to decide the next action
    completed(settings)


def merge_lora_checkpoint_mix(lora_model, checkpoint_model, weight_percentages):
    """Merges a LoRA into a main checkpoint using multiple weight percentages."""
    merged_models = []
    for weight in weight_percentages:
        merged_model = merge_lora_checkpoint_full(lora_model, checkpoint_model, weight / 100)
        merged_models.append((weight / 100, merged_model))
    return merged_models


def merge_lora_checkpoint_full(lora_model, checkpoint_model, merge_weight):
    """Merges a LoRA into a main checkpoint with a specified weight."""
    merged_model = {}
    all_keys = set(checkpoint_model.keys()).union(set(lora_model.keys()))

    with tqdm(total=len(all_keys), desc="Merging LoRA into Checkpoint", unit="layer") as pbar:
        for key in all_keys:
            if key in checkpoint_model and key in lora_model:
                tensor_checkpoint = checkpoint_model[key]
                tensor_lora = lora_model[key]
                if tensor_checkpoint.size() != tensor_lora.size():
                    tensor_checkpoint, tensor_lora = pad_tensors(tensor_checkpoint, tensor_lora)
                merged_model[key] = tensor_checkpoint + (merge_weight * tensor_lora)
            elif key in checkpoint_model:
                merged_model[key] = checkpoint_model[key]
            else:
                merged_model[key] = merge_weight * lora_model[key]
            pbar.update(1)

    return merged_model


def pad_tensors(tensor1, tensor2):
    """Pads tensors to the same size if they differ."""
    max_size = [max(s1, s2) for s1, s2 in zip(tensor1.size(), tensor2.size())]
    padded1 = torch.zeros(max_size, device=tensor1.device, dtype=tensor1.dtype)
    padded2 = torch.zeros(max_size, device=tensor2.device, dtype=tensor2.dtype)
    padded1[tuple(slice(0, s) for s in tensor1.size())] = tensor1
    padded2[tuple(slice(0, s) for s in tensor2.size())] = tensor2
    return padded1, padded2


def save_merged_checkpoint(merged_model, output_folder, lora_file, checkpoint_file, weight):
    """Saves the merged checkpoint with an appropriate name."""
    lora_name = os.path.splitext(lora_file)[0]
    checkpoint_name = os.path.splitext(checkpoint_file)[0]
    merged_name = f"merged_{lora_name}_W{int(weight * 100)}_{checkpoint_name}.safetensors"
    merged_path = os.path.join(output_folder, merged_name)

    save_file(merged_model, merged_path)
    print(f"Merged checkpoint saved as: {merged_name}")


def completed(settings):
    """Prompt user to decide whether to merge another LoRA into checkpoint or finish."""
    while True:
        choice = input("Do you want to merge another LoRA into checkpoint? (yes to continue, no to finish): ").strip().lower()
        if choice in ["yes", "y", ""]:
            new_settings = option_6_merge_lora_checkpoint()
            if new_settings:
                start(new_settings)
            else:
                print("No new settings provided. Exiting merge process.")
                sys.exit(0)
        elif choice in ["no", "n"]:
            print("Merging process completed.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")
