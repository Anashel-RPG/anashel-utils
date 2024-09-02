# input.py

import os
import sys
import glob
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from tqdm import tqdm
from safetensors.torch import load_file as safe_load
from tabulate import tabulate

# Initialize the Rich console
console = Console()

def main_input():
    # Display the welcome message in a box
    console.print(Panel(
        "Welcome to the Anashel's LoRA Merging Utility!\n\n"
        "This tool allows you to merge two LoRA models using adaptive merging strategies."
        "You can choose between a single weighted merge or a mix of three different weight ratios."
        "The process will guide you through selecting your LoRA models and merge settings.",
        title="[bold yellow]LoRA Merger[/bold yellow]",
        expand=False
    ))

    # Pause or hit enter to continue
    console.print("[bold cyan]Press Enter to continue...[/bold cyan]")
    input()

    # Directly call the LoRA merging function
    settings = option_5_merge_lora()

    # Check if settings are valid before confirming
    if settings:
        # Confirm the settings with the user
        confirm = confirm_settings(settings)
        if confirm:
            # Return settings to main.py for further processing
            return settings
        else:
            # Option to restart the input process without exiting
            console.clear()
    else:
        console.print("[bold red]Process aborted due to missing or invalid settings.[/bold red]\n")
        sys.exit(1)  # Exit the application with an error code

# def main_input():
#     while True:
#         # Display the welcome message and options
#         display_welcome()
#
#         # Ask the user to choose an option from 1 to 5
#         choice = Prompt.ask(
#             "[bold green]Choose a utility (1-5)[/bold green]"
#         )
#
#         # Call the respective function based on the user's choice
#         if choice == "1":
#             settings = option_1_generate_prompt_idea()
#         elif choice == "2":
#             settings = option_2_generate_image()
#         elif choice == "3":
#             settings = option_3_create_style_variation()
#         elif choice == "4":
#             settings = option_4_caption_images()
#         elif choice == "5":
#             settings = option_5_merge_lora()
#         else:
#             console.print("[bold red]Invalid choice. Please enter a number from 1 to 5.[/bold red]")
#             continue
#
#         # Check if settings are valid before confirming
#         if settings:
#             # Confirm the settings with the user
#             confirm = confirm_settings(settings)
#             if confirm:
#                 # Return settings to main.py for further processing
#                 return settings
#             else:
#                 # Option to restart the input process without exiting
#                 console.clear()
#         else:
#             console.print("[bold red]Process aborted due to missing or invalid settings.[/bold red]\n")
#             sys.exit(1)  # Exit the application with an error code

def display_welcome():
    """Display the welcome message and options."""
    console.print(Panel(
        "[bold magenta]1:[/bold magenta] Generate prompt idea to help build a dataset\n"
        "[bold magenta]2:[/bold magenta] Generate image using Flux or Leonardo\n"
        "[bold magenta]3:[/bold magenta] Create style variation of images to expand a dataset\n"
        "[bold magenta]4:[/bold magenta] Caption existing images using OpenAI\n"
        "[bold magenta]5:[/bold magenta] Merge LoRA",
        title="[bold yellow]Anashel Utility's[/bold yellow]"
    ))

def option_1_generate_prompt_idea():
    """Handle input for Generate prompt idea utility."""
    console.print("----\n")  # Visual separator for entering the new section
    console.print(
        "[bold green]Using your input, we will create 50 prompts for various images around your subject.[/bold green] "
        "This is useful to start building a dataset for training a concept, a specific style, or enhancing an existing dataset with complementary images.\n"
    )
    settings = {"utility": "Generate Prompt Idea"}

    # First question: Choose the type of prompt
    console.print(
        "[bold yellow]Do you wish to create prompts related to:[/bold yellow]\n"
        "[1] A person or character (e.g., A paladin, a 1950's soldier...)\n"
        "[2] A location (e.g., Dungeon, a building, a landscape...)\n"
        "[3] An aesthetic style (e.g., Dystopian futuristic, medieval fantasy...)"
    )
    choice = Prompt.ask("[bold green]Choose a type (1-3)[/bold green]")

    # Determine the type based on the choice
    if choice == "1":
        prompt_type = "person or character"
        example_prompt = "A brave paladin in shining armor, standing on a battlefield at dawn, with a glowing sword raised."
    elif choice == "2":
        prompt_type = "location"
        example_prompt = "A dark, eerie dungeon with flickering torches on the walls, filled with ancient stone carvings and scattered bones."
    else:
        prompt_type = "aesthetic style"
        example_prompt = "A dystopian futuristic cityscape at night, with towering neon-lit skyscrapers, flying vehicles, and a moody, cyberpunk atmosphere."

    settings["type"] = prompt_type

    # Second question: Ask for specific details
    while True:
        detail = Prompt.ask(f"[bold yellow]Please describe the {prompt_type} in more detail[/bold yellow]")
        settings["detail"] = detail

        # Processing step - Example prompt creation (this is where an API call would be made)
        console.print(
            f"\n[bold cyan]Here is an example prompt based on your input:[/bold cyan]\n[italic magenta]{example_prompt}[/italic magenta]"
        )

        # Third question: Confirm or loop back
        adjust = Prompt.ask(
            "[bold yellow]Is this satisfactory?[/bold yellow] (no to adjust, yes to continue)"
        )

        if adjust.lower() in ["yes", "y", ""]:
            break

    return settings

def option_2_generate_image():
    """Handle input for Generate image using Flux or Leonardo."""
    console.print("----\n")  # Visual separator for entering the new section
    console.print(
        "[bold green]Using Flux or Leonardo, we can generate images based on your specifications.[/bold green] "
        "This is ideal for quickly visualizing concepts, creating assets, or expanding your dataset with high-quality imagery.\n"
    )
    settings = {"utility": "Generate Image"}

    # Check for prompt.txt in the specified folders
    prompt_path_02 = "02-images_generation/prompt.txt"
    prompt_path_01 = "01-prompt_creation/output/prompt.txt"
    prompt_file = None

    # Check which prompt.txt file to use
    if os.path.exists(prompt_path_01) and os.path.exists(prompt_path_02):
        console.print(
            "[bold yellow]Do you wish to use the existing prompts from:[/bold yellow]\n"
            "[1] Folder 01-prompt_creation/output/\n"
            "[2] Folder 02-images_generation"
        )
        folder_choice = Prompt.ask("[bold green]Choose your folder (1-2)[/bold green]")

        prompt_file = prompt_path_01 if folder_choice == "1" else prompt_path_02
    elif os.path.exists(prompt_path_01):
        prompt_file = prompt_path_01
    elif os.path.exists(prompt_path_02):
        prompt_file = prompt_path_02
    else:
        console.print(
            "[bold red]Error: No prompt.txt file found in either 02-images_generation or 01-prompt_creation/output.[/bold red]\n"
            "Please ensure that a prompt.txt file is present with one prompt per line before proceeding."
        )
        return None

    # Count the number of lines (prompts) in the selected file
    with open(prompt_file, "r") as file:
        prompts = file.readlines()
        num_prompts = len(prompts)

    console.print(f"\n[bold cyan]Found {num_prompts} prompts in {prompt_file}.[/bold cyan]")

    # Platform selection
    console.print(
        "[bold yellow]Which platform would you like to use for image generation?[/bold yellow]\n"
        "[1] Flux\n"
        "[2] Leonardo"
    )
    platform_choice = Prompt.ask("[bold green]Choose a platform (1-2)[/bold green]")

    if platform_choice == "1":
        settings["platform"] = "Flux"
    else:
        settings["platform"] = "Leonardo"

    settings["prompt_file"] = prompt_file
    settings["num_prompts"] = num_prompts

    return settings

def option_3_create_style_variation():
    """Handle input for Create style variation of images."""
    console.print("----\n")  # Visual separator for entering the new section
    console.print(
        "[bold green]Style variation takes your images and generates variations using a style you provide.[/bold green] "
        "This is useful for expanding a dataset with different visual styles based on existing images.\n"
    )
    settings = {"utility": "Create Style Variation"}

    # Scan for images in 03-style_variation/input and 02-images_generation/output
    folder_03_input = "03-style_variation/input"
    folder_02_output = "02-images_generation/output"

    images_03 = glob.glob(os.path.join(folder_03_input, "*.[jp][pn]g"))  # Matches .jpg, .jpeg, .png
    images_02 = glob.glob(os.path.join(folder_02_output, "*.[jp][pn]g"))

    # Check for images
    if not images_03 and not images_02:
        console.print(
            "[bold red]Error: No images found in 03-style_variation/input or 02-images_generation/output.[/bold red]\n"
            "Please ensure that images (.jpg, .jpeg, .png) are present before proceeding."
        )
        return None

    # Confirm the number of images found
    if images_03 and images_02:
        console.print(
            f"[bold yellow]Images found in both folders:[/bold yellow]\n"
            f"[1] {len(images_03)} images in 03-style_variation/input\n"
            f"[2] {len(images_02)} images in 02-images_generation/output"
        )
        folder_choice = Prompt.ask("[bold green]Choose your folder (1-2):[/bold green]", choices=["1", "2"])
        selected_images = images_03 if folder_choice == "1" else images_02
    else:
        selected_images = images_03 if images_03 else images_02
        console.print(f"[bold cyan]Found {len(selected_images)} images to be transformed.[/bold cyan]")

    # Check for style reference image in 03-style_variation
    style_images = glob.glob("03-style_variation/*.[jp][pn]g")

    if not style_images:
        console.print(
            "[bold red]Error: No style reference image found in 03-style_variation.[/bold red]\n"
            "Please ensure that at least one image (.jpg, .jpeg, .png) is present at the root of the folder 03-style_variation to be used as a style reference."
        )
        return None
    elif len(style_images) == 1:
        style_image = style_images[0]
        console.print(f"[bold cyan]Using {os.path.basename(style_image)} as the style reference image.[/bold cyan]")
    else:
        console.print("[bold yellow]Multiple style reference images found:[/bold yellow]")
        for idx, img in enumerate(style_images, 1):
            console.print(f"[{idx}] {os.path.basename(img)}")
        img_choice = Prompt.ask(f"[bold green]Choose the style reference image (1-{len(style_images)}):[/bold green]",
                                choices=[str(i) for i in range(1, len(style_images) + 1)])
        style_image = style_images[int(img_choice) - 1]

    settings["selected_images"] = f"{len(selected_images)} images selected"
    settings["style_image"] = os.path.basename(style_image)

    return settings

def option_4_caption_images():
    """Handle input for Caption existing images using OpenAI."""
    console.print("----\n")  # Visual separator for entering the new section
    console.print(
        "[bold green]This utility will caption existing images using OpenAI, helping you generate descriptive captions for your dataset.[/bold green]\n"
    )
    settings = {"utility": "Caption Images"}

    # Scan for images in 04-ai_caption/input
    input_folder = "04-ai_caption/input"
    images = glob.glob(os.path.join(input_folder, "*.[jp][pn]g"))  # Matches .jpg, .jpeg, .png

    # Check if any images are found
    if not images:
        console.print(
            "[bold red]Error: No images found in 04-ai_caption/input.[/bold red]\n"
            "Please ensure that images are present in the folder before proceeding."
        )
        return None

    # Confirm the number of images found
    console.print(f"[bold cyan]Found {len(images)} images for captioning in 04-ai_caption/input.[/bold cyan]")

    settings["num_images"] = len(images)

    return settings

def option_5_merge_lora():
    """Handle input for scanning LoRA models and setting up a merge."""
    console.print("----\n")  # Visual separator for entering the new section
    console.print(
        "[bold green]This utility allows you to set up a merge of LoRA models by selecting two models and adjusting the merge weight percentage.[/bold green]\n"
    )
    settings = {"utility": "Merge LoRA"}

    # Step 1: Scan the folder and make an inventory of all LoRA (.safetensor) files
    lora_folder = "05-lora_merging"
    lora_files = [f for f in os.listdir(lora_folder) if f.endswith('.safetensors') or f.endswith('.pt')]

    if not lora_files:
        console.print(
            "[bold red]Error: No LoRA files found in 05-lora_merging.[/bold red]\n"
            "Please ensure that .safetensors files are present before proceeding."
        )
        return None

    if len(lora_files) == 1:
        console.print(
            "[bold red]Error: Only one LoRA file found in 05-lora_merging.[/bold red]\n"
            "A minimum of two LoRA files is required to perform a merge. Please add more files to proceed."
        )
        return None

    # Load LoRA details once
    lora_details = []
    with tqdm(total=len(lora_files), desc="Loading LoRA models", unit="file", dynamic_ncols=True) as progress_bar:
        for i, lora_file in enumerate(lora_files, 1):
            lora_path = os.path.join(lora_folder, lora_file)
            lora_model = load_lora_model(lora_path)
            num_layers = len(lora_model.keys())
            file_size = get_file_size(lora_path)
            lora_filename = lora_file.replace('.safetensors', '').replace('.pt', '')
            lora_details.append([i, lora_filename, num_layers, f"{file_size:.2f} MB"])
            progress_bar.update(1)

    while True:
        # Display the table with LoRA details
        formatted_table = tabulate(
            lora_details,
            headers=["Index", "LoRA Model", "Number of Layers", "File Size"],
            tablefmt="pretty",
            maxcolwidths=[None, 30, None, None]
        )

        console.print(f"\n{formatted_table}")

        # Prompt for main LoRA source
        while True:
            try:
                main_lora_index = int(Prompt.ask(f"Select the main LoRA source (1-{len(lora_files)})")) - 1
                if 0 <= main_lora_index < len(lora_files):
                    main_lora_file = lora_files[main_lora_index]
                    break
                else:
                    console.print(f"[bold red]Please enter a number between 1 and {len(lora_files)}.[/bold red]")
            except ValueError:
                console.print("[bold red]Please enter a valid number.[/bold red]")

        main_lora_path = os.path.join(lora_folder, main_lora_file)

        # Prompt for LoRA to merge with
        while True:
            try:
                merge_lora_index = int(Prompt.ask(f"Select the LoRA to merge with (1-{len(lora_files)})")) - 1
                if 0 <= merge_lora_index < len(lora_files):
                    if merge_lora_index != main_lora_index:
                        merge_lora_file = lora_files[merge_lora_index]
                        break
                    else:
                        console.print(
                            "[bold red]Cannot merge the same LoRA file with itself. Please select a different file.[/bold red]"
                        )
                else:
                    console.print(f"[bold red]Please enter a number between 1 and {len(lora_files)}.[/bold red]")
            except ValueError:
                console.print("[bold red]Please enter a valid number.[/bold red]")

        merge_lora_path = os.path.join(lora_folder, merge_lora_file)

        # Prompt for merge weight percentage
        weight_input = Prompt.ask("Enter the percentage to keep from the main model (0-100)\nYou can also type 'mix' for 25%, 50%, 75% versions")

        if weight_input.lower() == 'mix':
            settings["merge_strategy"] = "Mix"
            settings["weight_percentages"] = [25, 50, 75]
            console.print("[bold cyan]You've chosen to create three versions with weights: 25%, 50%, and 75%[/bold cyan]")
        else:
            try:
                weight_percentage = float(weight_input)
                if 0 <= weight_percentage <= 100:
                    settings["merge_strategy"] = "Weighted"
                    settings["weight_percentage"] = weight_percentage
                    alpha = weight_percentage / 100
                    beta = 1.0 - alpha
                    console.print(f"[bold cyan]Merge Weight: {alpha * 100}% main, {beta * 100}% merge[/bold cyan]")
                else:
                    raise ValueError
            except ValueError:
                console.print("[bold red]Invalid input. Please enter a number between 0 and 100 or 'mix'.[/bold red]")
                continue

        # Display settings before confirming
        console.print(
            f"\n[bold cyan]You have chosen to merge:[/bold cyan]\n"
            f"Main LoRA: {main_lora_file.replace('.safetensors', '').replace('.pt', '')}\n"
            f"Merge LoRA: {merge_lora_file.replace('.safetensors', '').replace('.pt', '')}\n"
            f"Merge Strategy: {settings['merge_strategy']}"
        )
        if settings["merge_strategy"] == "Mix":
            console.print(f"Weight Percentages: 25%, 50%, 75%")
        else:
            console.print(f"Weight Percentage: {settings['weight_percentage']}%")

        # Confirm settings before proceeding
        confirm = Prompt.ask(
            "[bold yellow]Is this satisfactory?[/bold yellow] (no to adjust, yes to continue)"
        )

        # If the user chooses to adjust, restart the selection process without reloading models
        if confirm.lower() in ["yes", "y", ""]:
            break
        else:
            console.print("[bold yellow]Adjusting settings. Please make your selections again.[/bold yellow]")

    settings["main_lora"] = main_lora_file
    settings["merge_lora"] = merge_lora_file

    return settings

def get_file_size(file_path):
    """Returns the size of the file in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def load_lora_model(file_path):
    file_size = get_file_size(file_path)
    buffer = bytearray(os.path.getsize(file_path))
    view = memoryview(buffer)

    with open(file_path, "rb") as f:
        while len(view):
            bytes_read = f.readinto(view)
            if bytes_read == 0:
                break
            view = view[bytes_read:]

    if file_path.endswith('.safetensors'):
        lora_model = safe_load(file_path)
    else:
        lora_model = torch.load(file_path)
    return lora_model

def confirm_settings(settings):
    """Automatically confirm the settings without user input."""
    console.print("----\n[bold green]LOADING SETTING:[/bold green]")
    for key, value in settings.items():
        console.print(f"[bold magenta]{key}:[/bold magenta] {value}")

    # Automatically confirm the settings
    return True
