# Stable Diffusion Image Generator

This project is a script for generating images from textual prompts using the Stable Diffusion model. It allows for detailed customization through a YAML configuration file and provides a flexible command-line interface for easy use. This is preset to work on M series of macbooks.

## Description

The script `generate_image.py` utilizes the Stable Diffusion model from Hugging Face's Diffusers library to generate images based on textual descriptions. It offers various parameters like the number of steps, guidance scale, and image dimensions to fine-tune the generation process. Users can input their desired settings through a YAML configuration file, making it easy to experiment with different prompts and settings.

## Features

- Generate images from text prompts
- Customizable settings through a YAML file
- Adjustable image dimensions, step count, and guidance scale
- Command-line interface for easy interaction


## Model

Current model configures is https://civitai.com/models/25694?modelVersionId=143906

## Installation

Ensure you have Python 3.7+ installed on your system. Then, follow these steps to set up the project.

1. **Clone the Repository:**

   ```bash
   git clone https://aahmed-se/your-username/generate_image.git
   cd generate_image
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Usage**

   Prepare Your Configuration:
   Edit the `config.yaml` file to specify your image generation parameters and text prompts.

   Run the Script:

   ```bash
   python generate_image.py
   ```

   Using a custom config file
   ```bash
   python generate_image.py --config /path/to/your_config.yaml
   ```

4. **Customization**

   You can modify the config.yaml to change the generation parameters. Available settings:

   `prompt`: The textual description of the image you want to generate.

   `negative_prompt`: Descriptions of what you want to avoid in the image.

   `steps`: Number of inference steps.

   `scale`: Guidance scale for the generation.

   `seed`: Random seed for reproducibility.

   `height`: Height of the generated image.

   `width`: Width of the generated image.
   
   `sampler`: The type of sampling algorithm to use.
