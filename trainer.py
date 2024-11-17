import argparse
import logging
import math
import os
import time
import warnings

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data_utils import (
    DataCollatorForSupervisedDataset,
    DynamicBatchSampler,
    process_single_dataset,
)
from model_utils import initialize_model

warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*is deprecated.*")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning (SFT) of language models."
    )
    # General training arguments
    parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--experiment_dir", type=str, default=None, help="Experiment directory."
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb.")
    parser.add_argument(
        "--project_name", type=str, default="SFT_Project", help="Project name for wandb."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="PyTorch dtype for model.",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=2048,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--drop_last",
        action="store_true",
        help="Drop last incomplete batch.",
    )

    # Training loop arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument(
        "--max_num_tokens_in_batch",
        type=int,
        default=2048,
        help="Maximum number of tokens in a batch.",
    )
    parser.add_argument(
        "--max_gpu_batch_size",
        type=int,
        default=8,
        help="Maximum GPU batch size.",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=10,
        help="Logging frequency in steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save the model every N steps.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Validate the model every N steps.",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=8,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final model to Hugging Face Hub.",
    )
    parser.add_argument(
        "--push_to_hub_name",
        type=str,
        default=None,
        required=True if args.push_to_hub else False,
        help="Name of the model to push to Hub.",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="AdapterOcean/python-code-instructions-18k-alpaca-standardized_cluster_3_alpaca",
        help="Dataset name or path to use for training",
    )

    # Add shuffle argument
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the training data.",
    )

    # Add LoRA specific arguments
    parser.add_argument(
        "--lora_use",
        action="store_true",
        help="Whether to use LoRA for training.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA attention dimension.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout value.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="List of module names to apply LoRA to.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="LoRA bias type.",
    )
    parser.add_argument(
        "--do_quantization",
        action="store_true",
        help="Whether to quantize the model before applying LoRA.",
    )

    # Add gradient checkpointing argument
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory at the expense of speed",
    )

    # Add attention implementation argument
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation to use",
    )

    # Add use_cache argument
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=True,
        help="Whether to use KV cache during generation",
    )

    return parser.parse_args()


def compute_validation_loss(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            batch_size = batch["input_ids"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    model.train()
    return avg_loss


def main(args):
    # Apply logging level from args
    logging.basicConfig(
        level=getattr(logging, args.logging_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    set_seed(args.seed)

    if args.experiment_dir is None:
        experiment_dir = os.getcwd()
    else:
        os.makedirs(args.experiment_dir, exist_ok=True)
        experiment_dir = args.experiment_dir

    # Prepare wandb_init_kwargs
    wandb_init_kwargs = {}
    if args.run_name:
        wandb_init_kwargs["name"] = args.run_name

    # Initialize accelerator early
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=experiment_dir,
    )

    accelerator.init_trackers(
        project_name=args.project_name,
        config=vars(args),
        init_kwargs={"wandb": wandb_init_kwargs},
    )

    # Initialize model & optimizer using initialize_model
    model = initialize_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.model_max_length > 1e7:
        tokenizer.model_max_length = args.max_sequence_length
    if tokenizer.pad_token is None:
        # Special to llama 3.2 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Process single dataset
    tokenized_dataset = process_single_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        accelerator=accelerator,
        max_sequence_length=args.max_sequence_length,
    )

    # Create validation split if it doesn't exist
    if "val" not in tokenized_dataset or len(tokenized_dataset["val"]) <= 1:
        if accelerator.is_main_process:
            logger.info("Creating validation split from training data...")
        
        # Split the training data 90-10
        train_val_split = tokenized_dataset["train"].train_test_split(
            test_size=0.1,
            seed=args.seed,
            shuffle=True
        )
        
        tokenized_dataset = DatasetDict({
            "train": train_val_split["train"],
            "val": train_val_split["test"]
        })

    # Get training dataset
    train_ds = tokenized_dataset["train"]
    if accelerator.is_main_process:
        logger.info(f"Number of training examples: {len(train_ds)}")
        logger.info(f"Number of validation examples: {len(tokenized_dataset['val'])}")

    # Prepare validation dataset
    val_datasets = {}
    ds_name = args.dataset_name.split("/")[-1]
    if "val" in tokenized_dataset and len(tokenized_dataset["val"]) > 1:
        val_datasets[f"{ds_name}_val"] = tokenized_dataset["val"]
    else:
        if accelerator.is_main_process:
            logger.warning(f"Dataset {ds_name} has no val data")

    # Initialize data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Get num_replicas and rank from the accelerator
    num_replicas = accelerator.num_processes
    rank = accelerator.process_index

    # Initialize dynamic batch sampler
    batch_sampler_train = DynamicBatchSampler(
        num_replicas=num_replicas,
        rank=rank,
        length_dict=train_ds["input_ids_lens"],
        num_buckets=256,
        min_len=0,
        max_len=args.max_sequence_length,
        max_batch_tokens=args.max_num_tokens_in_batch,
        max_batch_size=args.max_gpu_batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
        drop_last=args.drop_last,
    )

    # Set the number of CPU cores to be used for data loading and processing
    num_cpus = max(1, os.cpu_count() - 1)

    # Initialize train DataLoader with multiple workers for data loading
    train_dataloader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler_train,
        collate_fn=data_collator,
        num_workers=num_cpus,
        pin_memory=True,
    )

    # Prepare batches for all epochs
    data_loader_lens = []
    for epoch in range(args.num_epochs):
        train_dataloader.batch_sampler.set_epoch(epoch)
        data_loader_lens.append(len(train_dataloader))

    train_dataloader.batch_sampler.set_epoch(0)
    # Calculate the total number of steps (needed for schedulers)
    total_training_steps = sum(data_loader_lens)
    total_gradient_update_steps = math.ceil(
        total_training_steps / args.gradient_accumulation_steps
    )

    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_gradient_update_steps
    )

    # Prepare model, optimizer, and scheduler with accelerator
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Initialize progress bar only on the main process
    if accelerator.is_main_process:
        progress_bar = tqdm(total=total_training_steps, desc="Training")
    else:
        progress_bar = None

    # Prepare validation DataLoaders
    val_dataloaders = {}
    for val_ds_name, val_ds in val_datasets.items():
        val_dataloader = DataLoader(
            val_ds,
            batch_size=args.validation_batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=num_cpus,
            pin_memory=True,
        )
        val_dataloaders[val_ds_name] = val_dataloader

    overall_step = 0
    # Start timer
    if accelerator.is_main_process:
        training_start_time = time.time()

    # Initialize training metrics
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_samples = torch.tensor(0, device=accelerator.device)  # Change to tensor

    # In the main function, after initializing metrics
    best_val_loss = float('inf')  # Track best validation loss

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_dataloader.batch_sampler.set_epoch(epoch)

        # Initialize epoch metrics
        epoch_loss = torch.tensor(0.0, device=accelerator.device)
        epoch_samples = torch.tensor(0, device=accelerator.device)

        # Initialize gradient accumulation
        accumulation_steps = args.gradient_accumulation_steps
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            # Move batch to device as we are not wrapping the custom dataloader with accelerator.prepare()
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / accumulation_steps  # Scale the loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Accumulate the unscaled loss per process
            unscaled_loss = loss.detach() * accumulation_steps
            epoch_loss += unscaled_loss

            # Update total_samples (now a tensor)
            batch_size = torch.tensor(batch["input_ids"].shape[0], device=accelerator.device)
            total_samples += batch_size
            epoch_samples += batch_size

            overall_step += 1  # Increment overall_step in all processes

            # Log the loss every log_frequency steps
            if step % args.log_frequency == 0:
                # Gather losses across all processes and compute the mean loss
                losses = accelerator.gather(unscaled_loss)
                mean_loss = losses.mean().item()

                # Before logging, reduce total_samples across all processes
                total_samples_global = accelerator.reduce(total_samples, reduction="sum")

                # Gather batch sizes across all processes
                all_batch_sizes = accelerator.gather(batch_size)
                total_batch_size = all_batch_sizes.sum().item()

                # Log the metrics only on the main process
                if accelerator.is_main_process:
                    train_loss_dict = {
                        "samples": total_samples_global.item(),
                        "steps": overall_step,
                        "epoch": epoch + 1,
                        "batch_size": total_batch_size,
                        "loss/train": mean_loss,
                    }
                    accelerator.log(train_loss_dict, step=overall_step)
                    # Use progress_bar.write() to avoid interfering with the progress bar
                    if progress_bar is not None:
                        progress_bar.write(str(train_loss_dict))

            # Update the progress bar every step
            if accelerator.is_main_process and progress_bar is not None:
                progress_bar.update(1)

            # Validation
            if overall_step % args.validation_steps == 0:
                # Synchronize all processes before validation
                accelerator.wait_for_everyone()
                # Compute validation loss
                for val_ds_name, val_dataloader in val_dataloaders.items():
                    val_loss = compute_validation_loss(model, val_dataloader, accelerator)
                    if accelerator.is_main_process:
                        accelerator.log(
                            {f"validation_loss/{val_ds_name}": val_loss},
                            step=overall_step,
                        )
                        logger.info(
                            f"Validation loss for {val_ds_name} at step {overall_step}: {val_loss}"
                        )
                # Synchronize again after validation
                accelerator.wait_for_everyone()

            # Save model
            if overall_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    output_dir = f"model_step_{overall_step}"
                    output_dir = os.path.join(experiment_dir, output_dir)
                    if args.lora_use:
                        # Save only the LoRA adapter weights
                        accelerator.unwrap_model(model).save_pretrained(
                            output_dir,
                            save_function=accelerator.save,
                        )
                    else:
                        # Save the full model if not using LoRA
                        accelerator.unwrap_model(model).save_pretrained(output_dir)
                    logger.info(f"Model saved to {output_dir}")

        # End of epoch processing
        # Synchronize all processes before aggregating epoch metrics
        accelerator.wait_for_everyone()

        # After the epoch, aggregate the total_loss and total_samples_epoch across all processes
        epoch_loss = accelerator.reduce(epoch_loss, reduction="sum")
        epoch_samples = accelerator.reduce(epoch_samples, reduction="sum").item()

        # Compute average loss across all samples
        avg_epoch_loss = epoch_loss.item() / epoch_samples

        # Log the total loss
        if accelerator.is_main_process:
            accelerator.log({
                "loss/train_epoch": avg_epoch_loss,
                "epoch": epoch + 1,
            }, step=overall_step)
            logger.info(f"Epoch {epoch + 1}: Average train loss = {avg_epoch_loss:.4f}")

        # Run validation at the end of each epoch
        model.eval()
        val_losses = {}
        
        for val_ds_name, val_dataloader in val_dataloaders.items():
            val_loss = compute_validation_loss(model, val_dataloader, accelerator)
            val_losses[val_ds_name] = val_loss
            
            if accelerator.is_main_process:
                # Log validation loss
                accelerator.log({
                    f"loss/val_{val_ds_name}": val_loss,
                    "epoch": epoch + 1,
                }, step=overall_step)
                
                logger.info(f"Epoch {epoch + 1}: Validation loss for {val_ds_name} = {val_loss:.4f}")

        # Track best model
        if accelerator.is_main_process:
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best model
                best_model_dir = os.path.join(experiment_dir, "best_model")
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_dir}")
                accelerator.unwrap_model(model).save_pretrained(best_model_dir)
                
                accelerator.log({
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                }, step=overall_step)

        # Also keep the validation during training steps
        if overall_step % args.validation_steps == 0:
            accelerator.wait_for_everyone()
            for val_ds_name, val_dataloader in val_dataloaders.items():
                val_loss = compute_validation_loss(model, val_dataloader, accelerator)
                if accelerator.is_main_process:
                    accelerator.log({
                        f"loss/val_{val_ds_name}_step": val_loss,
                        "step": overall_step,
                    }, step=overall_step)
                    logger.info(f"Step {overall_step}: Validation loss for {val_ds_name} = {val_loss:.4f}")
            accelerator.wait_for_everyone()

    # End of training
    if accelerator.is_main_process:
        # Save the final model
        output_dir = f"model_final_step_{overall_step}"
        output_dir = os.path.join(experiment_dir, output_dir)
        logger.info(f"Saving final model to {output_dir}")
        accelerator.unwrap_model(model).save_pretrained(output_dir)

        # Push to hub if requested
        if args.push_to_hub:
            logger.info(f"Pushing model to hub as {args.push_to_hub_name}")
            accelerator.unwrap_model(model).push_to_hub(
                args.push_to_hub_name,
                private=True
            )

        # End timer
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        accelerator.print(f"Total training time: {training_time:.2f} seconds")

        # Log the final training time
        accelerator.log(
            {"total_training_time_seconds": training_time, "epoch": args.num_epochs},
            step=overall_step,
        )

        # End the training
        accelerator.end_training()

        return output_dir
    else:
        return


if __name__ == "__main__":
    args = parse_args()
    main(args)
