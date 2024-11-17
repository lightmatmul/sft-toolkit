from trainer import main, parse_args

if __name__ == "__main__":
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