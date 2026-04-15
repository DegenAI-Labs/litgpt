# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import zstandard as zstd
from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class TokenizedJSONL(DataModule):
    """Loads already-tokenized data from jsonl.zst files for pretraining.
    
    Expects each line in the jsonl.zst files to be a JSON array of token IDs: [1, 2, 3, ...]
    """

    train_data_path: Path
    """The path to the data directory used for training that contains .jsonl.zst files"""
    val_data_path: Optional[Path] = None
    """The path to the data directory used for validation that contains .jsonl.zst files.
    Splits off data for validation from the training set if None."""
    seed: int = 42
    """The seed to use for shuffling the dataset."""
    num_workers: int = 4
    """The number of workers to use for data loading."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.out_path_train = self.train_data_path / "train"
        if self.val_data_path is None:
            self.out_path_val = self.train_data_path / "val"
        else:
            self.out_path_val = Path(self.val_data_path) / "val"

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from litdata import optimize
        from litdata.streaming import TokensLoader

        train_files = sorted(glob.glob(str(self.train_data_path / "*.jsonl.zst")))
        assert len(train_files) > 0, f"No .jsonl.zst files found in train data {self.train_data_path}"

        if self.val_data_path is not None:
            self.val_data_path = Path(self.val_data_path)
            val_files = sorted(glob.glob(str(self.val_data_path / "*.jsonl.zst")))
            assert len(val_files) > 0, f"No .jsonl.zst files found in validation data {self.val_data_path}"
        # train/test split. let's use only shard 0 for test split, rest train
        else:
            assert len(train_files) > 1, f"Expected at least two .jsonl.zst files in {train_files}"
            val_files, *train_files = train_files
            val_files = [val_files]

        # It's ok to use almost all CPUs here because this runs in a single process
        num_workers = os.cpu_count() - 1
        use_workers = min(num_workers, len(train_files))
        if not Path(self.out_path_train).is_dir():
            optimize(
                fn=load_tokenized_jsonl,
                inputs=train_files,
                output_dir=str(self.out_path_train),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
        else:
            print(
                f"\nWarning: Preprocessed training data found in {self.out_path_train}."
                " For efficiency, reprocessing is skipped. If your data input has changed since"
                " the last `litgpt pretrain` command, remove the preprocessed file(s) to trigger"
                f" reprocessing: `rm -rf {self.out_path_train}`\n"
            )
        use_workers = min(num_workers, len(val_files))
        if not Path(self.out_path_val).is_dir():
            optimize(
                fn=load_tokenized_jsonl,
                inputs=val_files,
                output_dir=str(self.out_path_val),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
        else:
            print(
                f"\nWarning: Preprocessed validation data found in {self.out_path_val}."
                " For efficiency, reprocessing is skipped. If your data input has changed since"
                " the last `litgpt pretrain` command, remove the preprocessed file(s) to trigger"
                f" reprocessing: `rm -rf {self.out_path_val}`\n"
            )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=str(self.out_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )

        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=str(self.out_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )
        val_dataloader = StreamingDataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader


def load_tokenized_jsonl(filename: str):
    """Load tokenized sequences from a jsonl.zst file.
    
    Each line should be a JSON array of token IDs: [1, 2, 3, ...]
    """
    with zstd.open(open(filename, "rb"), "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array of token IDs, but got {type(data)}")
            token_ids = [int(t) for t in data]
            yield token_ids

