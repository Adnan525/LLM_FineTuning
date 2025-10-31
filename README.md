# QLoRA Finetuning Tutorial: Qwen2.5-0.5B-Instruct

## The Goal

This repository provides a complete, hands-on tutorial for finetuning the `Qwen/Qwen2.5-0.5B-Instruct` model.

The primary goal is to demonstrate how to efficiently specialize a general-purpose, pre-trained chat model for a specific task. We will finetune the model on a small, synthetic dataset to make it an expert at generating Python code in a particular style.

This process will guide you from a "generalist" model that gives generic answers to a "specialist" model that consistently outputs in your desired format.

-----

## Core Technical Concepts

This tutorial isn't just a script; it's a demonstration of the key components of modern, efficient finetuning. You will learn:

1.  **QLoRA (4-bit Quantization):** We use the `BitsAndBytesConfig` to load the massive base model in 4-bit precision. This drastically reduces the VRAM (GPU memory) required, making it possible to train on consumer-grade hardware.

2.  **PEFT (LoRA):** Instead of training all 0.5 billion parameters of the model (which is slow and memory-intensive), we freeze the base model. We then use **LoRA (Low-Rank Adaptation)** to train only a tiny fraction of "adapter" weights. These adapters are what learn the new task.

3.  **Prompt Masking (Completion-Only Loss):** This is a critical concept. We don't want the model to waste time learning to predict the *prompt* (e.g., "Write a function..."). We **only** want it to learn to predict the *answer* (e.g., "def add..."). We do this by passing a `formatting_func` to the trainer. The trainer then automatically "masks" the prompt tokens (by setting their loss label to `-100`), so the model is only graded on its ability to generate the correct completion.

4.  **Padding & The Attention Mask:** We explain the *other* kind of mask. The `attention_mask` is a tensor of `1`s and `0`s that tells the model which tokens are "real" (`1`) and which are just "padding" (`0`). This is essential for training on batches of sequences that have different lengths.

5.  **Baseline vs. Finetuned Inference:** We run inference on the model *before* training to get a baseline. This shows us the model's "default" answer. We then run inference *after* training to clearly see how its behavior has changed.

6.  **Adapter vs. Merged Model:** We demonstrate how to run inference using the LoRA adapter "on top" of the 4-bit base model (the memory-efficient way). We also discuss what `merge_and_unload()` does (de-quantizes the model) and the trade-offs between inference speed and VRAM usage.

-----

## Setup & Installation

This tutorial **requires an NVIDIA GPU with CUDA** support.

### 1\. Create a Python Environment

We recommend using Conda to manage your packages.

```bash
# Create a new environment with Python 3.10
conda create -n qlora_env python=3.10
conda activate qlora_env
```

### 2\. Install Dependencies

Install all packages:

```bash
pip install -r requirements.txt
```

### 3\. Hugging Face Authentication

You need a Hugging Face account and an access token to download the model, not required for Qwen though.

1.  Get your token from `Hugging Face > Settings > Access Tokens`.

2.  Create a file named `.env` in this directory.

3.  Add your token to the file:

    **`.env`**

    ```
    hf_token=token
    ```

-----

## How to Run

The entire tutorial is contained within the `finetuning.ipynb` (or `.py`) script.

1.  Make sure your `qlora_env` conda environment is active.
2.  Start Jupyter Lab:
    ```bash
    jupyter lab
    ```
3.  Open the notebook and run the cells from top to bottom. The script will:
      * Load the 4-bit base model.
      * Load the dataset and tokenizer.
      * Run a baseline inference test.
      * Configure and run the `SFTTrainer`.
      * Save the final adapter.
      * Run a final inference test to show the results.

-----

## Understanding the Output

After training, the `qwen2.5-0.5b-finetuned/` directory will be created. It contains:

  * **`checkpoint-XX/`**: A full snapshot for **resuming training**. This contains optimizer states, the scheduler, and random number generator states. You would use this if your training was interrupted.
  * **`qwen-python-coder/`**: This is your **final, clean adapter** for **inference**. This is the folder you would use in production or share. It *only* contains the LoRA weights (`adapter_model.safetensors`), the config (`adapter_config.json`), and a copy of the tokenizer files.
