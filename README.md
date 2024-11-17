# SFT Toolkit

A Python-based toolkit designed for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs). This toolkit provides a modular and efficient framework to fine-tune pre-trained language models on custom instruction-following datasets.

## Features

- **Instruction and Answer Data Handling**: Designed to work with datasets that have both an input and an output, facilitating the training of models that can follow and respond to user instructions.
- **Distributed Training with Accelerate**: Utilizes the accelerate library for seamless multi-GPU and Distributed Data Parallel (DDP) training.
- **Dynamic Batch Sampling**: Implements a dynamic batch sampler to optimize GPU memory usage and training efficiency.
- **Prompt Masking in Loss Calculations**: Incorporates prompt masking to ensure that only the response portion of the data contributes to the loss during training.

## Project Structure

```
sft-toolkit/
├── __init__.py
├── trainer.py
├── model_utils.py
├── data_utils.py
├── pyproject.toml
├── README.md
└── examples/
└── train_with_lora.py
```

- `trainer.py`: Main training script that orchestrates the fine-tuning process.
- `model_utils.py`: Contains utility functions for model initialization and configuration.
- `data_utils.py`: Handles data processing, tokenization, and dynamic batch sampling.
- `pyproject.toml`: Project configuration and dependencies.
- `README.md`: Documentation and usage instructions.
- `__init__.py`: Marks the directory as a Python module.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/your-username/sft-toolkit.git
cd sft-toolkit
pip install -e .
```

## Usage

### Basic Training

Run the `trainer.py` script with your desired arguments:

```bash
python trainer.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_name "path_or_name_of_your_dataset" \
    --experiment_dir "./experiments" \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --max_sequence_length 2048
```

#### Arguments Explanation

- `--model_name`: The name or path of the pre-trained model you want to fine-tune
- `--dataset_name`: The name or path of your dataset containing instruction-answer pairs
- `--experiment_dir`: Directory to save training outputs and checkpoints
- `--num_epochs`: Number of epochs to train
- `--learning_rate`: Learning rate for the optimizer
- `--max_sequence_length`: Maximum sequence length for the model inputs

### LoRA Fine-Tuning

For parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA):

```bash
python trainer.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_name "path_or_name_of_your_dataset" \
    --experiment_dir "./experiments" \
    --lora_use \
    --lora_r 8 \
    --lora_alpha 32 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --max_sequence_length 2048
```

#### Additional LoRA Arguments

- `--lora_use`: Enable LoRA fine-tuning
- `--lora_r`: Rank of the LoRA decomposition
- `--lora_alpha`: Scaling factor for LoRA updates

### Example Script

An example script is provided in `examples/train_with_lora.py`

```python
from trainer import main, parse_args
if name == "main":
# Example configuration for LoRA fine-tuning
args = parse_args([
"--model_name", "meta-llama/Llama-2-7b-hf",
"--dataset_name", "AdapterOcean/python-code-instructions-18k-alpaca-standardized",
"--experiment_dir", "./experiments",
"--lora_use",
"--lora_r", "8",
"--lora_alpha", "32",
"--num_epochs", "3",
"--learning_rate", "5e-5",
"--max_sequence_length", "2048"
])
main(args)
```
