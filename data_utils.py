# Dynamic Batch Sampler based on https://github.com/Raibows/DynamicBatchSampler/blob/main/DBSampler.py
import copy
import logging
import math
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Sampler

IGNORE_INDEX = -100


class DynamicBatchSampler(Sampler):
    """
    A dynamic batch sampler that supports Distributed Data Parallel (DDP) for robust training.

    This sampler creates batches of similar length samples, which can lead to more efficient
    training by reducing padding and optimizing GPU memory usage.
    """

    def __init__(self, num_replicas, rank, length_dict, num_buckets=128, min_len=0, max_len=1024,
                 max_batch_tokens=None, max_batch_size=None, shuffle=True, seed=0, drop_last=False,) -> None:
        """
        Initialize the DynamicBatchSampler.

        Args:
            num_replicas (int): The world size (i.e., the number of GPUs). Set to 1 for single GPU.
            rank (int): The rank of the GPU. Set to 0 for single GPU.
            length_dict (dict or list): Maps sample indices to their token counts.
            num_buckets (int): Number of buckets for length-based grouping. More buckets lead to more
                               homogeneous batches but less permutation.
            min_len (int): Minimum allowed sample length.
            max_len (int): Maximum allowed sample length.
            max_batch_tokens (int or None): Maximum total tokens in a batch. Affects GPU memory usage.
            max_batch_size (int or None): Maximum number of samples in a batch.
            shuffle (bool): Whether to shuffle the samples.
            seed (int): Random seed for shuffling.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        super(DynamicBatchSampler, self).__init__(None)

        # Validate DDP settings
        if dist.is_available() and not num_replicas > rank >= 0:
            raise RuntimeError(f"rank should be in the [0, {num_replicas - 1}]")
        if not dist.is_available():
            assert num_replicas == 1 and rank == 0, "rank and num_replicas have to be set to 1 if you are not in multi gpu(DDP) mode"

        # Validate batch size settings
        assert max_batch_tokens is not None or max_batch_size is not None, "you have to specify one of [max_batch_tokens, max_batch_size] to decide the 'real batch size'"
        self.max_batch_tokens = max_batch_tokens if max_batch_tokens is not None else float('Inf')
        self.max_batch_size = max_batch_size if max_batch_size is not None else float('Inf')
        assert self.max_batch_size >= 1
        assert self.max_batch_tokens >= min_len
        assert max_len >= min_len

        # Set random seed for reproducibility
        random.seed(seed)

        # Initialize instance variables
        self.num_replicas = num_replicas
        self.rank = rank
        self.length_dict = length_dict
        self.num_buckets = num_buckets
        self.min_len = min_len
        self.max_len = max_len
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.__epoch = 0
        self.__logger = logging.getLogger('sampler')
        self.__per_gpu_batch_num = 0
        self.__batches = []

    def __len__(self):
        """Return the number of batches for this GPU."""
        return self.__per_gpu_batch_num

    def __iter__(self):
        """Iterate over the batches for this GPU."""
        for batch in self.__batches[self.rank:len(self.__batches):self.num_replicas]:
            yield batch

    def set_epoch(self, epoch: int):
        """
        Set the epoch number and prepare batches for the new epoch.

        This method should be called at the beginning of each epoch to ensure
        proper shuffling and batch preparation.
        """
        self.__epoch = epoch
        self.__batches = self.__prepare_batches()

    def __is_full(self, tokens_in_all, batch):
        """Check if adding a new sample would exceed the batch limits."""
        if len(batch) == self.max_batch_size:
            return True
        if tokens_in_all > self.max_batch_tokens:
            return True
        return False

    def __prepare_batches(self):
        """
        Prepare batches for the current epoch.

        This method groups samples into buckets based on their lengths,
        then forms batches from these buckets while respecting the
        max_batch_size and max_batch_tokens constraints.
        """
        if self.rank == 0:
            self.__logger.info(f"starting prepare batches of epoch {self.__epoch} shuffle {self.shuffle}")

        # Shuffle indices if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.__epoch)
            indices = torch.randperm(len(self.length_dict), generator=g).tolist()
        else:
            # Avoid re-preparing batches if not shuffling
            if len(self.__batches) > 0: return self.__batches
            indices = list(range(len(self.length_dict)))

        batches = []
        buckets = [[] for _ in range(self.num_buckets)]
        buckets_max_len = [0 for _ in range(self.num_buckets)]

        # Group samples into buckets
        for idx in indices:
            idx_len = self.length_dict[idx]
            if not self.max_len >= idx_len >= self.min_len:
                if self.rank == 0:
                    self.__logger.warning(f"ignored one sample with index {idx}, length {idx_len} not in the interval [{self.min_len}, {self.max_len}]")
                continue
            idx_bkt = math.floor((idx_len - self.min_len) / (self.max_len - self.min_len + 1) * self.num_buckets)
            buckets_max_len[idx_bkt] = max(buckets_max_len[idx_bkt], idx_len)
            tokens_in_all = (len(buckets[idx_bkt]) + 1) * buckets_max_len[idx_bkt]
            if self.__is_full(tokens_in_all, buckets[idx_bkt]):
                batches.append(buckets[idx_bkt])
                buckets[idx_bkt] = []
                buckets_max_len[idx_bkt] = 0
            buckets[idx_bkt].append(idx)

        # Process leftover samples
        leftover_batch = []
        leftover_max_len = 0
        leftover_indices = [idx for bkt in buckets for idx in bkt]
        for idx in leftover_indices:
            idx_len = self.length_dict[idx]
            leftover_max_len = max(leftover_max_len, idx_len)
            tokens_in_all = (len(leftover_batch) + 1) * leftover_max_len
            if self.__is_full(tokens_in_all, leftover_batch):
                batches.append(leftover_batch)
                leftover_batch = []
                leftover_max_len = 0
            leftover_batch.append(idx)

        # Handle the last batch
        if len(leftover_batch) > 0:
            if self.drop_last:
                if self.rank == 0:
                    self.__logger.warning(f"dropped the leftover batch size {len(leftover_batch)}")
            else:
                batches.append(leftover_batch)

        # Ensure each GPU gets the same number of batches
        self.__per_gpu_batch_num = math.ceil(len(batches) / self.num_replicas)
        total_batch_num = self.__per_gpu_batch_num * self.num_replicas
        dummy_batch_num = total_batch_num - len(batches)
        if dummy_batch_num <= len(batches):
            dummy_batches = random.sample(batches, k=dummy_batch_num)
        else:
            if self.rank == 0:
                self.__logger.warning(f"repeated batches will exist because the dummy_batch_num is larger than len(batches)")
            dummy_batches = [random.choice(batches) for _ in range(dummy_batch_num)]
        batches += dummy_batches

        # Final shuffle to improve model robustness
        if self.shuffle:
            random.shuffle(batches)
        return batches


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    source_input_ids = sources_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_input_id, source_len in zip(labels, source_input_ids, sources_tokenized["input_ids_lens"]):
        index_ignore = source_len
        times_decremented = 0
        while source_input_id[index_ignore-1] != label[index_ignore-1]:
            index_ignore -= 1
            times_decremented += 1
            if times_decremented > 5:
                raise ValueError("Cannot find index_ignore")
        label[:index_ignore] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def process_single_dataset(dataset_name, tokenizer, accelerator, max_sequence_length=None, dummy_mode=False, num_proc=None):
    """
    Process a single dataset for training.

    Args:
        dataset_name (str): Name of the dataset to load
        tokenizer: The tokenizer to use for processing
        accelerator: The Accelerator object for distributed processing
        max_sequence_length (int, optional): Maximum allowed sequence length. Defaults to None.
        dummy_mode (bool, optional): If True, only process a small subset of data. Defaults to False.
        num_proc (int, optional): Number of processes to use for data processing. 
                                If None, defaults to CPU count minus 1.

    Returns:
        datasets.DatasetDict: The processed dataset
    """
    if max_sequence_length is None:
        max_sequence_length = 1e8

    if num_proc is None:
        num_proc = max(1, os.cpu_count() - 1)

    is_local_ds = os.path.exists(dataset_name)
    load_fn = load_from_disk if is_local_ds else partial(load_dataset, download_mode="force_redownload")
    datasets = load_fn(dataset_name)
    
    if dummy_mode:
        datasets = DatasetDict({
            split: dataset.select(range(min(20, len(dataset))))
            for split, dataset in datasets.items()
        })

    def train_tokenize_function(examples, tokenizer):
        """Tokenize the training data."""
        messages_without_answer = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True
            )
            for instruction in examples['input']
        ]
        
        messages = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": answer}
                ],
                tokenize=False,
                add_generation_prompt=False
            )
            for instruction, answer in zip(examples['input'], examples['output'])
        ]

        sources = messages_without_answer
        targets = [message.replace(source, "") for message, source in zip(messages, messages_without_answer)]
        message_ends_with_eos = tokenizer.eos_token in targets[0][-len(tokenizer.eos_token)-1:]
        targets = [f"{t}\n{tokenizer.eos_token}" if not message_ends_with_eos else f"{t}" for t in targets]
        
        return preprocess(sources, targets, tokenizer)

    def add_input_ids_lens(example):
        example['input_ids_lens'] = len(example['input_ids'])
        return example

    def filter_long_sequences(example):
        return example['input_ids_lens'] <= max_sequence_length

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            train_tokenize_function,
            batched=True,
            desc="Running Encoding",
            load_from_cache_file=True,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=num_proc
        )

        tokenized_datasets = tokenized_datasets.map(
            add_input_ids_lens,
            desc="Adding input_ids_lens column",
            num_proc=num_proc
        )

        filtered_datasets = tokenized_datasets.filter(
            filter_long_sequences,
            desc="Filtering long sequences",
            num_proc=num_proc
        )

    total_examples = len(tokenized_datasets["train"])
    filtered_examples = len(filtered_datasets["train"])
    removed_examples = total_examples - filtered_examples
    accelerator.print(f"Filtered out {removed_examples} examples ({removed_examples/total_examples:.2%}) exceeding {max_sequence_length} tokens.")

    return filtered_datasets
