import logging
import torch
from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
)

torch_dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

# Set up the default logger
logger = logging.getLogger(__name__)


def initialize_model(args):
    # Map string to torch dtype
    torch_dtype = torch_dtype_map.get(args.torch_dtype, torch.bfloat16)

    # Prepare model initialization arguments
    model_init_args = {
        "pretrained_model_name_or_path": args.model_name,
        "return_dict": True,
        "trust_remote_code": args.trust_remote_code,
        "use_cache": getattr(args, "use_cache", True),
        "torch_dtype": torch_dtype,
        "_attn_implementation": getattr(args, "attn_impl", "sdpa"),
    }

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(**model_init_args)

    # Enable gradient checkpointing if specified
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Handle LoRA configuration if specified
    if getattr(args, "lora_use", False):
        if getattr(args, "do_quantization", False):
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for kbit training")

        # Create a LoRA configuration
        lora_config = LoraConfig(
            r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 32),
            target_modules=getattr(args, "lora_target_modules", ["q_proj", "v_proj"]),
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            bias=getattr(args, "lora_bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA configuration applied")

    return model
