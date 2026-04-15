# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class LitDataMulti(DataModule):
    """Loads data using LitData's StreamingDataset with support for combining multiple training datasets."""

    data_path: Union[str, Path] = Path("data/")
    """The base path to the data directory."""
    train_split_names: Optional[List[str]] = None
    """List of training split names to combine. For example: ['split1', 'split2', 'split3']"""
    val_split_name: Optional[str] = None
    """Name of the validation split."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self) -> None:
        super().__init__()
        if self.train_split_names is not None and not isinstance(self.train_split_names, list):
            raise ValueError("`train_split_names` must be a list of strings.")
        if self.val_split_name is None:
            raise ValueError("`val_split_name` must be provided.")

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def train_dataloader(self) -> DataLoader:
        # Create list of input directories for training
        if self.train_split_names:
            input_dirs = [os.path.join(self.data_path, split_name) for split_name in self.train_split_names]
        else:
            input_dirs = str(self.data_path)
        return self._dataloader(input_dir=input_dirs, train=True)

    def val_dataloader(self) -> DataLoader:
        input_dir = os.path.join(self.data_path, self.val_split_name)
        return self._dataloader(input_dir=input_dir, train=False)

    def _dataloader(self, input_dir: Union[str, List[str]], train: bool):
        from litdata.streaming import StreamingDataLoader, StreamingDataset, CombinedStreamingDataset, TokensLoader

        # If input_dir is a list, use CombinedStreamingDataset to combine multiple datasets
        if isinstance(input_dir, list):
            datasets = []
            for dir_path in input_dir:
                ds = StreamingDataset(
                    input_dir=dir_path,
                    item_loader=TokensLoader(block_size=self.seq_length),
                    shuffle=train,
                    seed=self.seed,
                )
                datasets.append(ds)
            
            dataset = CombinedStreamingDataset(
                datasets=datasets,
                seed=self.seed,
                iterate_over_all=True,  # exhaust all datasets
            )
        else:
            # Single dataset
            dataset = StreamingDataset(
                input_dir=input_dir,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=train,
                seed=self.seed,
            )
        
        dataloader = StreamingDataLoader(
            dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return dataloader

