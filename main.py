# main.py
import boot
import input
import generate_caption
import generate_image
import generate_prompt
import generate_style
import merge_lora
import merge_lora_checkpoint  # Add this import

def main():
    # Invoke boot routine
    boot.boot_routine()

    # Invoke input routine and get settings
    settings = input.main_input()

    # Dispatch the selected utility
    dispatch_utility(settings)

def dispatch_utility(settings):
    """Dispatches the correct utility based on the selected settings."""
    utility = settings['utility']

    if utility == "Generate Prompt Idea":
        generate_prompt.start(settings)
    elif utility == "Generate Image":
        generate_image.start(settings)
    elif utility == "Create Style Variation":
        generate_style.start(settings)
    elif utility == "Caption Images":
        generate_caption.start(settings)
    elif utility == "Merge LoRA":
        merge_lora.start(settings)
    elif utility == "Merge LoRA Checkpoint":  # Add this new condition
        merge_lora_checkpoint.start(settings)
    elif utility == "God Mode":
        merge_lora.god_mode(settings['lora_folder'], settings['merge_strategy'])
    else:
        print(f"Unknown utility: {utility}")

if __name__ == "__main__":
    main()
