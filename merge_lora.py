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


def save_merged_lora(merged_model, lora_folder, main_lora_file, merge_lora_file, weight, merge_type):
    """Saves the merged LoRA model with an appropriate name."""
    main_name = os.path.splitext(main_lora_file)[0]
    merge_name = os.path.splitext(merge_lora_file)[0]

    if merge_type == 'adaptive':
        strategy_code = f"A{int(weight * 100)}"
    elif merge_type == 'additive':
        strategy_code = f"ADDI{int(weight * 100)}"
    else:  # manual
        strategy_code = f"M{int(weight * 100)}"

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

from tqdm import tqdm
import torch

def pad_tensors(tensor1, tensor2):
    """Pads tensors to the same size if they differ."""
    max_size = [max(s1, s2) for s1, s2 in zip(tensor1.size(), tensor2.size())]
    padded1 = torch.zeros(max_size, device=tensor1.device, dtype=tensor1.dtype)
    padded2 = torch.zeros(max_size, device=tensor2.device, dtype=tensor2.dtype)
    padded1[tuple(slice(0, s) for s in tensor1.size())] = tensor1
    padded2[tuple(slice(0, s) for s in tensor2.size())] = tensor2
    return padded1, padded2

def pad_all_tensors(tensors):
    """Pads all tensors in the list to match the maximum size across all tensors."""
    if not tensors:
        return []

    # Determine the max size across all tensors
    max_size = [max(t.size(dim) for t in tensors) for dim in range(len(tensors[0].size()))]

    # Pad each tensor to the max size
    padded_tensors = []
    for tensor in tensors:
        padded_tensor = torch.zeros(max_size, device=tensor.device, dtype=tensor.dtype)
        slices = tuple(slice(0, s) for s in tensor.size())
        padded_tensor[slices] = tensor
        padded_tensors.append(padded_tensor)

    return padded_tensors


def god_mode(lora_folder, merge_strategy='adaptive'):
    """
    Merges multiple LoRA models simultaneously using the specified strategy, constrained by available memory.

    Args:
    - lora_folder: The folder containing LoRA models to merge.
    - merge_strategy: The merging strategy to use ('adaptive', 'additive').

    Returns:
    - Path to the final merged model saved to disk.
    """
    # Load all LoRA models from the folder with progress bar
    lora_files = [f for f in os.listdir(lora_folder) if f.endswith('.safetensors')]
    if not lora_files:
        print("No LoRA models found to merge.")
        return None

    print(f"Loading {len(lora_files)} LoRA models...")
    lora_models = []
    largest_file_size = 0
    largest_file_name = ''

    with tqdm(total=len(lora_files), desc="Loading LoRA Models", unit="file") as pbar:
        for file in lora_files:
            file_path = os.path.join(lora_folder, file)
            file_size = os.path.getsize(file_path)
            if file_size > largest_file_size:
                largest_file_size = file_size
                largest_file_name = file
            try:
                lora_model = load_file(file_path)
                lora_models.append(lora_model)
            except Exception as e:
                print(f"Error loading model {file}: {e}")
            pbar.update(1)

    if not lora_models:
        print("No LoRA models successfully loaded.")
        return None

    print(f"Largest input file: {largest_file_name} ({largest_file_size} bytes)")
    print(f"Starting merge with {len(lora_models)} LoRA models using {merge_strategy} strategy.")

    # Initialize the merged model with keys from all models
    all_keys = set().union(*(model.keys() for model in lora_models))
    merged_model = {key: torch.zeros_like(next(model[key] for model in lora_models if key in model))
                    for key in all_keys}

    total_input_tensors = 0
    total_merged_tensors = 0

    for key in tqdm(all_keys, desc="Merging tensors", unit="tensor"):
        tensors = [model[key] for model in lora_models if key in model]
        total_input_tensors += len(tensors)

        if not tensors:
            print(f"Warning: No tensors found for key: {key}")
            continue

        # print(f"Merging {len(tensors)} tensors for key: {key}")
        # print(f"Input tensor sizes: {[t.size() for t in tensors]}")

        try:
            padded_tensors = pad_all_tensors(tensors)
            if merge_strategy == 'adaptive':
                merged_model[key] = adaptive_merge_multiple(padded_tensors)
            elif merge_strategy == 'additive':
                merged_model[key] = additive_merge_multiple(padded_tensors)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")

            # print(f"Merged tensor size: {merged_model[key].size()}")
            total_merged_tensors += 1

        except Exception as e:
            print(f"Error merging tensors for key {key}: {e}")
            # Instead of skipping, use the tensor from the largest file if available
            largest_model_tensor = next((model[key] for model in lora_models if key in model and model is lora_models[0]), None)
            if largest_model_tensor is not None:
                merged_model[key] = largest_model_tensor
                print(f"Using tensor from largest file for key {key}")
            else:
                print(f"Warning: Skipping key {key} due to errors")

    print(f"Total input tensors: {total_input_tensors}")
    print(f"Total merged tensors: {total_merged_tensors}")

    # Determine the strategy code for the filename
    strategy_code = 'A' if merge_strategy == 'adaptive' else 'M'

    # Create the filename using the correct naming convention
    merged_filename = f"mrg_final_merged_{strategy_code}100_god_mode.safetensors"
    merged_file_path = os.path.join(lora_folder, merged_filename)

    # Save the final merged model
    try:
        save_file(merged_model, merged_file_path)
        merged_file_size = os.path.getsize(merged_file_path)
        print(f"Merged file saved as: {merged_filename}")
        print(f"Merged file size: {merged_file_size} bytes")

        if merged_file_size < largest_file_size:
            print("Warning: Merged file is smaller than the largest input file. Some data may have been lost in the process.")
        else:
            print("Merged file is larger than or equal to the largest input file, as expected.")

    except Exception as e:
        print(f"Error saving merged model: {e}")
        return None

    return merged_file_path

def adaptive_merge_multiple(tensors):
    """Merges multiple tensors using adaptive weights based on their L2 norms."""
    try:
        norms = [torch.norm(tensor) for tensor in tensors]
        total_norm = sum(norms)
        weights = [norm / total_norm for norm in norms]

        # Calculate the final merged tensor
        merged_tensor = sum(w * t for w, t in zip(weights, tensors))
        return merged_tensor
    except Exception as e:
        print(f"Error in adaptive_merge_multiple: {e}")
        return torch.zeros_like(tensors[0])

def additive_merge_multiple(tensors):
    """Merges multiple tensors using additive merging with equal weighting."""
    try:
        weight = 1.0 / len(tensors)
        merged_tensor = sum(weight * tensor for tensor in tensors)
        return merged_tensor
    except Exception as e:
        print(f"Error in additive_merge_multiple: {e}")
        return torch.zeros_like(tensors[0])
