#!/usr/bin/env python

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
from typing import List
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from accelerate.accelerator import Accelerator
from accelerate.utils.dataclasses import DistributedType


class DummyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for element in self.data:
            yield element


def create_accelerator(even_batches=True):
    accelerator = Accelerator(even_batches=even_batches)
    assert accelerator.num_processes == 2, "this script expects that two GPUs are available"
    return accelerator


def create_dataloader(accelerator: Accelerator, dataset_size: int, batch_size: int, iterable: bool = False):
    """
    Create a simple DataLoader to use during the test cases
    """
    if iterable:
        dataset = DummyIterableDataset(torch.as_tensor(range(dataset_size)))
    else:
        dataset = TensorDataset(torch.as_tensor(range(dataset_size)))

    dl = DataLoader(dataset, batch_size=batch_size)
    dl = accelerator.prepare(dl)

    return dl


def verify_dataloader_batch_sizes(
    accelerator: Accelerator,
    dataset_size: int,
    batch_size: int,
    process_0_expected_batch_sizes: List[int],
    process_1_expected_batch_sizes: List[int],
):
    """
    A helper function for verifying the batch sizes coming from a prepared dataloader in each process
    """
    dl = create_dataloader(accelerator=accelerator, dataset_size=dataset_size, batch_size=batch_size)

    batch_sizes = [len(batch[0]) for batch in dl]

    if accelerator.process_index == 0:
        assert batch_sizes == process_0_expected_batch_sizes
    elif accelerator.process_index == 1:
        assert batch_sizes == process_1_expected_batch_sizes


def test_default_ensures_even_batch_sizes():
    accelerator = create_accelerator()

    # without padding, we would expect a different number of batches
    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=3,
        batch_size=1,
        process_0_expected_batch_sizes=[1, 1],
        process_1_expected_batch_sizes=[1, 1],
    )

    # without padding, we would expect the same number of batches, but different sizes
    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=7,
        batch_size=2,
        process_0_expected_batch_sizes=[2, 2],
        process_1_expected_batch_sizes=[2, 2],
    )


def test_can_disable_even_batches():
    accelerator = create_accelerator(even_batches=False)

    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=3,
        batch_size=1,
        process_0_expected_batch_sizes=[1, 1],
        process_1_expected_batch_sizes=[1],
    )

    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=7,
        batch_size=2,
        process_0_expected_batch_sizes=[2, 2],
        process_1_expected_batch_sizes=[2, 1],
    )


def test_can_join_uneven_inputs():
    accelerator = create_accelerator(even_batches=False)

    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)

    dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)

    batch_idxs = []
    with accelerator.join_uneven_inputs([ddp_model]):
        for batch_idx, batch in enumerate(dl):
            output = ddp_model(batch[0].float())
            loss = output.sum()
            loss.backward()
            batch_idxs.append(batch_idx)

    accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
        assert batch_idxs == [0, 1]
    elif accelerator.process_index == 1:
        assert batch_idxs == [0]


def test_join_raises_warning_for_non_ddp_distributed(accelerator):
    with warnings.catch_warnings(record=True) as w:
        with accelerator.join_uneven_inputs([Mock()]):
            pass

        assert issubclass(w[-1].category, UserWarning)
        assert "only supported for multi-GPU" in str(w[-1].message)


def test_join_can_override_even_batches():
    default_even_batches = True
    overridden_even_batches = False
    accelerator = create_accelerator(even_batches=default_even_batches)
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    train_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)
    valid_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)

    with accelerator.join_uneven_inputs([ddp_model], even_batches=overridden_even_batches):
        train_dl_overridden_value = train_dl.batch_sampler.even_batches
        valid_dl_overridden_value = valid_dl.batch_sampler.even_batches

    assert train_dl_overridden_value == overridden_even_batches
    assert valid_dl_overridden_value == overridden_even_batches
    assert train_dl.batch_sampler.even_batches == default_even_batches
    assert valid_dl.batch_sampler.even_batches == default_even_batches


def test_join_can_override_for_mixed_type_dataloaders():
    default_even_batches = True
    overridden_even_batches = False
    accelerator = create_accelerator(even_batches=default_even_batches)
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    create_dataloader(accelerator, dataset_size=3, batch_size=1, iterable=True)
    batch_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            with accelerator.join_uneven_inputs([ddp_model], even_batches=overridden_even_batches):
                batch_dl_overridden_value = batch_dl.batch_sampler.even_batches
        except AttributeError:
            # ensure attribute error is not raised when processing iterable dl
            raise AssertionError

    assert batch_dl_overridden_value == overridden_even_batches
    assert batch_dl.batch_sampler.even_batches == default_even_batches


def test_join_raises_warning_for_iterable_when_overriding_even_batches():
    accelerator = create_accelerator()
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    create_dataloader(accelerator, dataset_size=3, batch_size=1, iterable=True)

    with warnings.catch_warnings(record=True) as w:
        with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
            pass

        assert issubclass(w[-1].category, UserWarning)
        assert "only supported for map-style datasets" in str(w[-1].message)


def main():
    accelerator = create_accelerator()

    accelerator.print("Test that even_batches variable ensures uniform batches across processes")
    test_default_ensures_even_batch_sizes()

    accelerator.print("Run tests with even_batches disabled")
    test_can_disable_even_batches()

    accelerator.print("Test joining uneven inputs")
    test_can_join_uneven_inputs()

    accelerator.print("Test overriding even_batches when joining uneven inputs")
    test_join_can_override_even_batches()

    accelerator.print("Test overriding even_batches for mixed dataloader types")
    test_join_can_override_for_mixed_type_dataloaders()

    accelerator.print("Test overriding even_batches raises a warning for iterable dataloaders")
    test_join_raises_warning_for_iterable_when_overriding_even_batches()

    accelerator.print("Test join with non DDP distributed raises warning")
    original_state = accelerator.state.distributed_type
    accelerator.state.distributed_type = DistributedType.FSDP
    test_join_raises_warning_for_non_ddp_distributed(accelerator)
    accelerator.state.distributed_type = original_state


if __name__ == "__main__":
    main()
