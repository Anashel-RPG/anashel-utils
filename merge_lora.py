# merge_lora.py
import os
import time
import sys
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from input import option_5_merge_lora
import psutil

def start(settings):
    print(f"\n###################################\nMerging LoRA with settings: {settings}")

    # Load the LoRA models
    lora_folder = "05a-lora_merging"
    main_lora_path = os.path.join(lora_folder, settings['main_lora'])
    merge_lora_path = os.path.join(lora_folder, settings['merge_lora'])

    main_lora_model = load_file(main_lora_path)
    merge_lora_model = load_file(merge_lora_path)

    # Choose the merging strategy based on the settings
    if settings['merge_strategy'] == 'Mix':
        merged_models = merge_loras_mix(main_lora_model, merge_lora_model, settings['weight_percentages'], settings['merge_type'])
    elif settings['merge_strategy'] == 'Additive':
        merged_model = additive_merge(main_lora_model, merge_lora_model, settings['add_weight'] / 100)
        merged_models = [(settings['add_weight'] / 100, merged_model)]
    else:  # Weighted
        merge_type = settings.get('merge_type', 'adaptive')
        merged_model = merge_loras_weighted(main_lora_model, merge_lora_model, settings['weight_percentage'] / 100, merge_type)
        merged_models = [(settings['weight_percentage'] / 100, merged_model)]

    # Save the merged models
    for weight, merged_model in merged_models:
        save_merged_lora(merged_model, lora_folder, settings['main_lora'], settings['merge_lora'], weight, settings['merge_type'])

    print("Merging completed! ✅")
    print(" ")

    # Call the completed function to decide the next action
    completed(settings)


def merge_loras_mix(main_lora_model, merge_lora_model, weight_percentages, merge_type):
    """Merges two LoRA models using multiple weight percentages."""
    merged_models = []
    for weight in weight_percentages:
        merged_model = merge_loras_weighted(main_lora_model, merge_lora_model, weight / 100, merge_type)
        merged_models.append((weight / 100, merged_model))
    return merged_models


def merge_loras_weighted(main_lora_model, merge_lora_model, main_weight, merge_type='adaptive'):
    """Merges two LoRA models using adaptive or manual merge with a specified main weight."""
    merged_model = {}
    all_keys = set(main_lora_model.keys()).union(set(merge_lora_model.keys()))

    with tqdm(total=len(all_keys), desc="Merging LoRA models", unit="layer") as pbar:
        for key in all_keys:
            if key in main_lora_model and key in merge_lora_model:
                if merge_type == 'adaptive':
                    merged_model[key] = adaptive_merge(main_lora_model[key], merge_lora_model[key], main_weight)
                else:
                    merged_model[key] = manual_merge(main_lora_model[key], merge_lora_model[key], main_weight)
            elif key in main_lora_model:
                merged_model[key] = main_lora_model[key]
            else:
                merged_model[key] = merge_lora_model[key]
            pbar.update(1)

    return merged_model


def additive_merge(main_lora_model, merge_lora_model, add_weight):
    """Always use 100% of the first model and add the second model at a specified percentage."""
    merged_model = {}
    all_keys = set(main_lora_model.keys()).union(set(merge_lora_model.keys()))

    with tqdm(total=len(all_keys), desc="Additive Merging LoRA models", unit="layer") as pbar:
        for key in all_keys:
            if key in main_lora_model and key in merge_lora_model:
                tensor1 = main_lora_model[key]
                tensor2 = merge_lora_model[key]
                if tensor1.size() != tensor2.size():
                    tensor1, tensor2 = pad_tensors(tensor1, tensor2)
                merged_model[key] = tensor1 + (add_weight * tensor2)
            elif key in main_lora_model:
                merged_model[key] = main_lora_model[key]
            else:
                merged_model[key] = add_weight * merge_lora_model[key]
            pbar.update(1)

    return merged_model


def adaptive_merge(tensor1, tensor2, main_weight):
    """Merges two tensors using adaptive weights based on their L2 norms."""
    if tensor1.size() != tensor2.size():
        tensor1, tensor2 = pad_tensors(tensor1, tensor2)

    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    adaptive_weight1 = norm1 / (norm1 + norm2)
    adaptive_weight2 = norm2 / (norm1 + norm2)

    final_weight1 = adaptive_weight1 * main_weight + (1 - adaptive_weight2) * (1 - main_weight)
    final_weight2 = 1 - final_weight1

    return final_weight1 * tensor1 + final_weight2 * tensor2


def manual_merge(tensor1, tensor2, main_weight):
    """Merges two tensors using fixed weights based on user input."""
    if tensor1.size() != tensor2.size():
        tensor1, tensor2 = pad_tensors(tensor1, tensor2)

    return main_weight * tensor1 + (1 - main_weight) * tensor2


def pad_tensors(tensor1, tensor2):
    """Pads tensors to the same size if they differ."""
    max_size = [max(s1, s2) for s1, s2 in zip(tensor1.size(), tensor2.size())]
    padded1 = torch.zeros(max_size, device=tensor1.device, dtype=tensor1.dtype)
    padded2 = torch.zeros(max_size, device=tensor2.device, dtype=tensor2.dtype)
    padded1[tuple(slice(0, s) for s in tensor1.size())] = tensor1
    padded2[tuple(slice(0, s) for s in tensor2.size())] = tensor2
    return padded1, padded2


def save_merged_lora(merged_model, lora_folder, main_lora_file, merge_lora_file, weight, merge_type):
    """Saves the merged LoRA model with an appropriate name."""
    main_name = os.path.splitext(main_lora_file)[0]
    merge_name = os.path.splitext(merge_lora_file)[0]
    strategy_code = f"A{int(weight * 100)}" if merge_type == 'adaptive' else f"M{int(weight * 100)}"
    merged_lora_name = f"mrg_{main_name}_{strategy_code}_{merge_name}.safetensors"
    merged_lora_path = os.path.join(lora_folder, merged_lora_name)

    save_file(merged_model, merged_lora_path)
    print(f"Merged LoRA saved as: {merged_lora_name}")


def completed(settings):
    """Prompt user to decide whether to merge another LoRA or finish."""
    while True:
        choice = input("Do you want to merge another LoRA? (yes to continue, no to finish): ").strip().lower()
        if choice in ["yes", "y", ""]:
            new_settings = option_5_merge_lora()
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

def god_mode(lora_folder, merge_strategy='adaptive', max_merge_attempts=10):
    """
    Merges as many tensors at once as possible within available memory constraints.

    Args:
    - lora_folder: The folder containing LoRA models to merge.
    - merge_strategy: The merging strategy to use ('adaptive', 'additive').
    - max_merge_attempts: Number of times to attempt merging batches.

    Returns:
    - A final merged model saved to disk.
    """
    # Load all LoRA models from the folder
    lora_files = [f for f in os.listdir(lora_folder) if f.endswith('.safetensors')]
    lora_models = [load_file(os.path.join(lora_folder, file)) for file in lora_files]

    # Check if there are models to merge
    if not lora_models:
        print("No LoRA models found to merge.")
        return

    # Initialize the merged model with the first LoRA
    merged_model = lora_models[0]
    print(f"Starting merge with {len(lora_models)} LoRA models.")

    # Merge models in batches according to available memory
    while len(lora_models) > 1:
        available_memory = psutil.virtual_memory().available // (1024 ** 2)  # Get available memory in MB
        print(f"Available Memory: {available_memory} MB")

        # Determine how many models can be merged in this batch
        merge_batch_size = min(max_merge_attempts, len(lora_models) - 1)

        # Attempt to merge as many tensors as allowed by the current memory size
        merged_in_batch = 0
        for i in range(merge_batch_size):
            try:
                next_model = lora_models.pop(1)  # Remove the next model to merge
                if merge_strategy == 'adaptive':
                    merged_model = merge_loras_weighted(merged_model, next_model, 0.5, 'adaptive')
                elif merge_strategy == 'additive':
                    merged_model = additive_merge(merged_model, next_model, 0.5)
                merged_in_batch += 1
            except Exception as e:
                print(f"Error merging batch: {e}")
                break

        print(f"Merged {merged_in_batch} models in this batch.")
        if merged_in_batch == 0:
            print("Unable to merge further due to memory limitations.")
            break

    # Save the final merged model
    save_merged_lora(merged_model, lora_folder, 'final_merged', 'god_mode', 1.0, merge_strategy)
    print("God mode merging completed and saved the final merged model.")
