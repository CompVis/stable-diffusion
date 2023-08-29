# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
import math
from abc import ABC
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device


if is_transformers_available():
    from transformers.modeling_outputs import (
        CausalLMOutputWithCrossAttentions,
        Seq2SeqLMOutput,
        SequenceClassifierOutput,
    )


if is_megatron_lm_available():
    from megatron import (
        get_args,
        get_num_microbatches,
        get_tensorboard_writer,
        get_timers,
        get_tokenizer,
        mpu,
        print_rank_0,
        print_rank_last,
    )
    from megatron.arguments import _add_data_args, _add_validation_args, parse_args, validate_args
    from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint, save_checkpoint
    from megatron.data.data_samplers import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
    from megatron.global_vars import set_global_variables
    from megatron.initialize import (
        _compile_dependencies,
        _init_autoresume,
        _set_random_seed,
        set_jit_fusion_options,
        write_args_to_tensorboard,
    )
    from megatron.model import BertModel, Float16Module, GPTModel, ModelType, T5Model
    from megatron.model import DistributedDataParallel as LocalDDP
    from megatron.model.classification import Classification
    from megatron.optimizer import get_megatron_optimizer
    from megatron.schedules import get_forward_backward_func
    from megatron.text_generation.communication import broadcast_int_list, broadcast_tensor
    from megatron.text_generation.generation import (
        beam_search_and_return_on_first_stage,
        generate_tokens_probs_and_return_on_first_stage,
    )
    from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    from megatron.training import get_model, get_optimizer_param_scheduler, training_log
    from megatron.utils import (
        average_losses_across_data_parallel_group,
        calc_params_l2_norm,
        get_ltor_masks_and_position_ids,
        unwrap_model,
    )


# model utilities
def model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    args = get_args()
    mode = "pre-training" if args.pretraining_flag else "fine-tuning"
    if args.rank == 0:
        print(f"Building {args.model_type_name} model in the {mode} mode.")
        print(
            "The Megatron LM model weights are initialized at random in `accelerator.prepare`. "
            "Please use `accelerator.load_checkpoint` to load a pre-trained checkpoint matching the distributed setup."
        )
    if args.model_type_name == "bert":
        if args.pretraining_flag:
            num_tokentypes = 2 if args.bert_binary_head else 0
            model = BertModel(
                num_tokentypes=num_tokentypes,
                add_binary_head=args.bert_binary_head,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )
        else:
            model = Classification(
                num_classes=args.num_labels, num_tokentypes=2, pre_process=pre_process, post_process=post_process
            )
    elif args.model_type_name == "gpt":
        model = GPTModel(num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process)
    elif args.model_type_name == "t5":
        model = T5Model(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type_name}")
    return model


def prepare_model(accelerator):
    accelerator.print("Preparing model")
    args = get_args()
    if accelerator.state.megatron_lm_plugin.custom_prepare_model_function is not None:
        if accelerator.state.megatron_lm_plugin.custom_model_provider_function is None:
            raise ValueError(
                "You must provide a `custom_model_provider_function` when using a `custom_prepare_model_function`."
            )
        custom_model_provider_func = accelerator.state.megatron_lm_plugin.custom_model_provider_function
        model = accelerator.state.megatron_lm_plugin.custom_prepare_model_function(custom_model_provider_func)
    else:
        if args.model_type_name in ("bert", "gpt"):
            model_type = ModelType.encoder_or_decoder
        elif args.model_type_name == "t5":
            model_type = ModelType.encoder_and_decoder
            if args.pipeline_model_parallel_split_rank is None and args.pipeline_model_parallel_size > 1:
                args.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_size // 2
        model = get_model(model_provider_func, model_type)
    return model


# dataloader utilities
class MegatronLMDummyDataLoader:
    """
    Dummy dataloader presents model parameters or param groups, this is primarily used to follow conventional training

    Args:
        **dataset_kwargs: Megatron data arguments.
    """

    def __init__(self, **dataset_kwargs):
        parser = argparse.ArgumentParser()
        parser = _add_data_args(parser)
        parser = _add_validation_args(parser)
        data_args = parser.parse_known_args()
        self.dataset_args = vars(data_args[0])
        self.dataset_args.update(dataset_kwargs)
        self.dataset_args["megatron_dataset_flag"] = True

    def set_megatron_data_args(self):
        args = get_args()
        for key, value in self.dataset_args.items():
            setattr(args, key, value)

    def get_train_valid_test_datasets_provider(self):
        def train_valid_test_datasets_provider(train_val_test_num_samples):
            """Build train, valid, and test datasets."""
            args = get_args()
            dataset_args = {
                "data_prefix": args.data_path,
                "data_impl": args.data_impl,
                "splits_string": args.split,
                "train_valid_test_num_samples": train_val_test_num_samples,
                "skip_warmup": (not args.mmap_warmup),
                "seed": args.seed,
            }
            if args.model_type_name == "bert":
                dataset_args.update(
                    {
                        "max_seq_length": args.seq_length,
                        "masked_lm_prob": args.mask_prob,
                        "short_seq_prob": args.short_seq_prob,
                        "binary_head": args.bert_binary_head,
                    }
                )
            elif args.model_type_name == "gpt":
                dataset_args.update(
                    {
                        "seq_length": args.seq_length,
                    }
                )
            elif args.model_type_name == "t5":
                dataset_args.update(
                    {
                        "max_seq_length": args.encoder_seq_length,
                        "max_seq_length_dec": args.decoder_seq_length,
                        "masked_lm_prob": args.mask_prob,
                        "short_seq_prob": args.short_seq_prob,
                        "dataset_type": "t5",
                    }
                )
            else:
                raise ValueError(f"Unsupported model type: {args.model_type_name}")
            if args.model_type_name == "gpt":
                from megatron.data.gpt_dataset import build_train_valid_test_datasets
            else:
                from megatron.data.dataset_utils import build_train_valid_test_datasets
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(**dataset_args)
            return train_ds, valid_ds, test_ds

        return train_valid_test_datasets_provider

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        if dataset is None:
            return None
        args = get_args()
        micro_batch_size = args.micro_batch_size * args.num_micro_batches

        # Megatron sampler
        if args.dataloader_type == "single":
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
        elif args.dataloader_type == "cyclic":
            batch_sampler = MegatronPretrainingRandomSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                data_sharding=args.data_sharding,
            )
        else:
            raise Exception("{} dataloader type is not supported.".format(args.dataloader_type))

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True
        )

    def build_train_valid_test_data_iterators(self):
        def cyclic_iter(iter):
            while True:
                for x in iter:
                    yield x

        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0("> building train, validation, and test datasets ...")

        # Backward compatibility, assume fixed batch size.
        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, "only backward compatiblity support for iteration-based training"
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = (
                    (args.iteration // args.eval_interval) * args.eval_iters * args.global_batch_size
                )

        # Data loader only on rank 0 of each model parallel group.
        if mpu.get_tensor_model_parallel_rank() == 0:
            # Number of train/valid/test samples.
            if args.train_samples:
                train_samples = args.train_samples
            else:
                train_samples = args.train_iters * args.global_batch_size
            eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
            test_iters = args.eval_iters
            train_val_test_num_samples = [
                train_samples,
                eval_iters * args.global_batch_size,
                test_iters * args.global_batch_size,
            ]
            print_rank_0(" > datasets target sizes (minimum size):")
            print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
            print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
            print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

            # Build the datasets.
            train_valid_test_datasets_provider = self.get_train_valid_test_datasets_provider()
            train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(train_val_test_num_samples)

            # Build dataloders.
            train_dataloader = self.build_pretraining_data_loader(train_ds, args.consumed_train_samples)
            valid_dataloader = self.build_pretraining_data_loader(valid_ds, args.consumed_valid_samples)
            test_dataloader = self.build_pretraining_data_loader(test_ds, 0)

            # Flags to know if we need to do training/validation/testing.
            do_train = train_dataloader is not None and args.train_iters > 0
            do_valid = valid_dataloader is not None and args.eval_iters > 0
            do_test = test_dataloader is not None and args.eval_iters > 0
            # Need to broadcast num_tokens and num_type_tokens.
            flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
        else:
            flags = torch.cuda.LongTensor([0, 0, 0])

        # Broadcast num tokens.
        torch.distributed.broadcast(
            flags, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
        )
        args.do_train = flags[0].item()
        args.do_valid = flags[1].item()
        args.do_test = flags[2].item()

        # Build iterators.
        dl_type = args.dataloader_type
        assert dl_type in ["single", "cyclic"]

        if train_dataloader is not None:
            train_data_iterator = (
                iter(train_dataloader) if dl_type == "single" else iter(cyclic_iter(train_dataloader))
            )
        else:
            train_data_iterator = None

        if valid_dataloader is not None:
            valid_data_iterator = (
                iter(valid_dataloader) if dl_type == "single" else iter(cyclic_iter(valid_dataloader))
            )
        else:
            valid_data_iterator = None

        if test_dataloader is not None:
            test_data_iterator = iter(test_dataloader) if dl_type == "single" else iter(cyclic_iter(test_dataloader))
        else:
            test_data_iterator = None

        return train_data_iterator, valid_data_iterator, test_data_iterator


def prepare_data_loader(accelerator, dataloader):
    accelerator.print("Preparing dataloader")
    args = get_args()
    if not args.megatron_dataset_flag:
        from ..data_loader import _PYTORCH_DATALOADER_KWARGS, prepare_data_loader

        args = get_args()
        micro_batch_size = args.micro_batch_size * args.num_micro_batches
        kwargs = {k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k]) for k in _PYTORCH_DATALOADER_KWARGS}
        if kwargs["batch_size"] is None:
            if isinstance(kwargs["sampler"], torch.utils.data.BatchSampler):
                kwargs["sampler"].batch_size = micro_batch_size
            else:
                del kwargs["sampler"]
                del kwargs["shuffle"]
                del kwargs["batch_size"]
                kwargs["batch_sampler"].batch_size = micro_batch_size
        else:
            del kwargs["batch_sampler"]
            kwargs["batch_size"] = micro_batch_size

        dataloader = torch.utils.data.DataLoader(dataloader.dataset, **kwargs)
        return prepare_data_loader(
            dataloader,
            accelerator.device,
            num_processes=mpu.get_data_parallel_world_size(),
            process_index=mpu.get_data_parallel_rank(),
            split_batches=accelerator.split_batches,
            put_on_device=True,
            rng_types=accelerator.rng_types.copy(),
            dispatch_batches=accelerator.dispatch_batches,
        )
    else:
        if args.consumed_samples is not None:
            (
                args.consumed_train_samples,
                args.consumed_valid_samples,
                args.consumed_test_samples,
            ) = args.consumed_samples
        else:
            args.consumed_train_samples, args.consumed_valid_samples, args.consumed_test_samples = 0, 0, 0
        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = dataloader.build_train_valid_test_data_iterators()
        return train_data_iterator, valid_data_iterator, test_data_iterator


# optimizer utilities
class MegatronLMOptimizerWrapper(AcceleratedOptimizer):
    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)

    def zero_grad(self, set_to_none=None):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    def step(self):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        return self.optimizer.skipped_iter


def prepare_optimizer(accelerator, model):
    accelerator.print("Preparing optimizer")
    args = get_args()
    optimizer = get_megatron_optimizer(model, args.no_wd_decay_cond, args.scale_lr_cond, args.lr_mult)
    return optimizer


# scheduler utilities
class MegatronLMDummyScheduler:
    """
    Dummy scheduler presents model parameters or param groups, this is primarily used to follow conventional training
    loop when scheduler config is specified in the deepspeed config file.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        total_num_steps (int):
            Total number of steps.
        warmup_num_steps (int):
            Number of steps for warmup.
        **kwargs:
            Other arguments.
    """

    def __init__(self, optimizer, total_num_steps=None, warmup_num_steps=0, **kwargs):
        self.optimizer = optimizer
        self.total_num_steps = total_num_steps
        self.warmup_num_steps = warmup_num_steps
        self.kwargs = kwargs


class MegatronLMSchedulerWrapper(AcceleratedScheduler):
    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self, *args, **kwargs):
        return  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


def prepare_scheduler(accelerator, optimizer, scheduler):
    accelerator.print("Preparing scheduler")
    scheduler = get_optimizer_param_scheduler(optimizer)
    return scheduler


class AbstractTrainStep(ABC):
    """Abstract class for batching, forward pass and loss handler."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_batch_func(self):
        pass

    def get_forward_step_func(self):
        pass

    def get_loss_func(self):
        pass


class BertTrainStep(AbstractTrainStep):
    """
    Bert train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__("BertTrainStep")
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(args.pretraining_flag, args.num_labels)
        self.forward_step = self.get_forward_step_func(args.pretraining_flag, args.bert_binary_head)
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = SequenceClassifierOutput

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch_megatron(data_iterator):
            """Build the batch."""

            # Items and their type.
            keys = ["text", "types", "labels", "is_random", "loss_mask", "padding_mask"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens = data_b["text"].long()
            types = data_b["types"].long()
            sentence_order = data_b["is_random"].long()
            loss_mask = data_b["loss_mask"].float()
            lm_labels = data_b["labels"].long()
            padding_mask = data_b["padding_mask"].long()

            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())

            # Unpack.
            tokens = data["input_ids"].long()
            padding_mask = data["attention_mask"].long()
            if "token_type_ids" in data:
                types = data["token_type_ids"].long()
            else:
                types = None
            if "labels" in data:
                lm_labels = data["labels"].long()
                loss_mask = (data["labels"] != -100).to(torch.float)
            else:
                lm_labels = None
                loss_mask = None
            if "next_sentence_label" in data:
                sentence_order = data["next_sentence_label"].long()
            else:
                sentence_order = None

            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self, pretraining_flag, num_labels):
        def loss_func_pretrain(loss_mask, sentence_order, output_tensor):
            lm_loss_, sop_logits = output_tensor

            lm_loss_ = lm_loss_.float()
            loss_mask = loss_mask.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

            if sop_logits is not None:
                sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
                sop_loss = sop_loss.float()
                loss = lm_loss + sop_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
                return loss, {"lm loss": averaged_losses[0], "sop loss": averaged_losses[1]}

            else:
                loss = lm_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss])
                return loss, {"lm loss": averaged_losses[0]}

        def loss_func_finetune(labels, logits):
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            averaged_losses = average_losses_across_data_parallel_group([loss])
            return loss, {"loss": averaged_losses[0]}

        if pretraining_flag:
            return loss_func_pretrain
        else:
            return loss_func_finetune

    def get_forward_step_func(self, pretraining_flag, bert_binary_head):
        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, types, sentence_order, loss_mask, labels, padding_mask = self.get_batch(data_iterator)
            if not bert_binary_head:
                types = None
            # Forward pass through the model.
            if pretraining_flag:
                output_tensor = model(tokens, padding_mask, tokentype_ids=types, lm_labels=labels)
                return output_tensor, partial(self.loss_func, loss_mask, sentence_order)
            else:
                logits = model(tokens, padding_mask, tokentype_ids=types)
                return logits, partial(self.loss_func, labels)

        return forward_step


class GPTTrainStep(AbstractTrainStep):
    """
    GPT train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__("GPTTrainStep")
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        self.eod_token = args.padded_vocab_size - 1
        if args.vocab_file is not None:
            tokenizer = get_tokenizer()
            self.eod_token = tokenizer.eod
        self.reset_position_ids = args.reset_position_ids
        self.reset_attention_mask = args.reset_attention_mask
        self.eod_mask_loss = args.eod_mask_loss
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch_megatron(data_iterator):
            """Generate a batch"""
            # Items and their type.
            keys = ["text"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens_ = data_b["text"].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()

            # Get the masks and postition ids.
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss
            )

            return tokens, labels, loss_mask, attention_mask, position_ids

        def get_batch_transformer(data_iterator):
            data = next(data_iterator)
            data = {"input_ids": data["input_ids"]}
            data = send_to_device(data, torch.cuda.current_device())

            tokens_ = data["input_ids"].long()
            padding = torch.zeros((tokens_.shape[0], 1), dtype=tokens_.dtype, device=tokens_.device) + self.eod_token
            tokens_ = torch.concat([tokens_, padding], dim=1)
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            # Get the masks and postition ids.
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, True
            )
            return tokens, labels, loss_mask, attention_mask, position_ids

        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):
        args = get_args()

        def loss_func(loss_mask, output_tensor):
            if args.return_logits:
                losses, logits = output_tensor
            else:
                losses = output_tensor
            losses = losses.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            output_dict = {"lm loss": averaged_loss[0]}
            if args.return_logits:
                output_dict.update({"logits": logits})
            return loss, output_dict

        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.loss_func, loss_mask)

        return forward_step


class T5TrainStep(AbstractTrainStep):
    """
    T5 train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__("T5TrainStep")
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = Seq2SeqLMOutput

    @staticmethod
    def attn_mask_postprocess(attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        # [b, s, 1]
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        # [b, s, s]
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        # Convert attention mask to binary:
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    @staticmethod
    def get_decoder_mask(seq_length, device):
        attention_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device))
        attention_mask = attention_mask < 0.5
        return attention_mask

    @staticmethod
    def get_enc_dec_mask(attention_mask, dec_seq_length, device):
        batch_size, _ = attention_mask.shape
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        # [b, s, 1]
        attention_mask_bs1 = torch.ones((batch_size, dec_seq_length, 1), device=device)
        attention_mask_bss = attention_mask_bs1 * attention_mask_b1s
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch_megatron(data_iterator):
            """Build the batch."""

            keys = ["text_enc", "text_dec", "labels", "loss_mask", "enc_mask", "dec_mask", "enc_dec_mask"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens_enc = data_b["text_enc"].long()
            tokens_dec = data_b["text_dec"].long()
            labels = data_b["labels"].long()
            loss_mask = data_b["loss_mask"].float()

            enc_mask = data_b["enc_mask"] < 0.5
            dec_mask = data_b["dec_mask"] < 0.5
            enc_dec_mask = data_b["enc_dec_mask"] < 0.5

            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())

            tokens_enc = data["input_ids"].long()
            labels = data["labels"].long()
            loss_mask = (labels != -100).to(torch.float)
            if "decoder_input_ids" in data:
                tokens_dec = data["decoder_input_ids"].long()
            else:
                tokens_dec = labels.new_zeros(labels.shape, device=labels.device, dtype=torch.long)
                tokens_dec[..., 1:] = labels[..., :-1].clone()
                tokens_dec[..., 0] = 0
                tokens_dec.masked_fill_(tokens_dec == -100, 0)
            enc_mask = T5TrainStep.attn_mask_postprocess(data["attention_mask"].long())
            dec_mask = T5TrainStep.get_decoder_mask(tokens_dec.shape[1], tokens_dec.device)
            enc_dec_mask = T5TrainStep.get_enc_dec_mask(
                data["attention_mask"].long(), tokens_dec.shape[1], tokens_dec.device
            )

            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):
        def loss_func(loss_mask, output_tensor):
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

            loss = lm_loss
            averaged_losses = average_losses_across_data_parallel_group([lm_loss])

            return loss, {"lm loss": averaged_losses[0]}

        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            # Get the batch.
            tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = self.get_batch(
                data_iterator
            )
            # Forward model lm_labels
            output_tensor = model(
                tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=lm_labels
            )

            return output_tensor, partial(self.loss_func, loss_mask)

        return forward_step


# intialize megatron setup
def initialize(accelerator, extra_args_provider=None, args_defaults={}):
    accelerator.print("Initializing Megatron-LM")
    assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args=True)

    # Set defaults
    for key, value in args_defaults.items():
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print(
                    "WARNING: overriding default arguments for {key}:{v} \
                        with {key}:{v2}".format(
                        key=key, v=getattr(args, key), v2=value
                    ),
                    flush=True,
                )
        setattr(args, key, value)

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    validate_args(args)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        device_count = torch.cuda.device_count()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device

            # Set the tensor model-parallel, pipeline model-parallel, and
            # data-parallel communicators.
            if mpu.model_parallel_is_initialized():
                print("model parallel is already initialized")
            else:
                mpu.initialize_model_parallel(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                )

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    args = get_args()

    # Megatron's MPU is the master. Complete initialization right away.
    finish_mpu_init()

    # Autoresume.
    _init_autoresume()

    # Compile dependencies.
    _compile_dependencies()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()
    args = get_args()
    args.padded_vocab_size = _vocab_size_with_padding(args.orig_vocab_size, args)
    if args.model_type_name == "bert" and args.pretraining_flag and args.num_labels == 2:
        args.bert_binary_head = True
    else:
        args.bert_binary_head = False
    args.iteration = 0


class MegatronEngine(torch.nn.Module):
    """
    Megatron-LM model wrapper

    Args:
        accelerator (:class:`~accelerate.Accelerator`): The accelerator object to use.
        model: Megatron-LM model
        optimizer: Megatron-LM optimizer
        lr_scheduler: Megatron-LM lr scheduler
    """

    def __init__(self, accelerator, model, optimizer, scheduler):
        super(MegatronEngine, self).__init__()
        self.module = model
        self.base_model = model[0]
        self.optimizer = optimizer
        self.scheduler = scheduler
        args = get_args()
        if accelerator.state.megatron_lm_plugin.custom_train_step_class is not None:
            self.train_step_handler = accelerator.state.megatron_lm_plugin.custom_train_step_class(
                args, **accelerator.state.megatron_lm_plugin.custom_train_step_kwargs
            )
        elif args.model_type_name == "bert":
            self.train_step_handler = BertTrainStep(args)
        elif args.model_type_name == "gpt":
            self.train_step_handler = GPTTrainStep(args)
        elif args.model_type_name == "t5":
            self.train_step_handler = T5TrainStep(args)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type_name}")
        self.optimizer.skipped_iter = False

        # Tracking loss.
        self.total_loss_dict = {}
        self.eval_total_loss_dict = {}
        self.iteration = 0
        self.report_memory_flag = True
        if args.tensorboard_dir is not None:
            write_args_to_tensorboard()

    def train(self):
        for model_module in self.module:
            model_module.train()
        self.log_eval_results()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

    def train_step(self, **batch_data):
        """
        Training step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to train on.
        """

        args = get_args()
        timers = get_timers()

        if len(batch_data) > 0:
            data_chunks = []
            if args.num_micro_batches > 1:
                for i in range(0, args.num_micro_batches):
                    data_chunks.append(
                        {
                            k: v[i * args.micro_batch_size : (i + 1) * args.micro_batch_size]
                            for k, v in batch_data.items()
                        }
                    )
            else:
                data_chunks = [batch_data]

        if len(self.module) > 1:
            batch_data_iterator = (
                [iter(data_chunks) for _ in range(len(self.module))]
                if len(batch_data) > 0
                else [None] * len(self.module)
            )
        else:
            batch_data_iterator = iter(data_chunks) if len(batch_data) > 0 else None

        # Set grad to zero.
        if args.DDP_impl == "local" and args.use_contiguous_buffers_in_local_ddp:
            for partition in self.module:
                partition.zero_grad_buffer()
        self.optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            self.train_step_handler.forward_step,
            batch_data_iterator,
            self.module,
            self.optimizer,
            None,
            forward_only=False,
        )

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Reduce gradients.
        timers("backward-reduce-model-grads").start()
        self.optimizer.reduce_model_grads(args, timers)
        timers("backward-reduce-model-grads").stop()

        # Update parameters.
        timers("optimizer").start()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step(args, timers)
        timers("optimizer").stop()

        # Gather params.
        if update_successful:
            timers("backward-gather-model-params").start()
            self.optimizer.gather_model_params(args, timers)
            timers("backward-gather-model-params").stop()

        # Update learning rate.
        if update_successful:
            if self.scheduler is not None:
                increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                self.scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        self.optimizer.skipped_iter = not update_successful

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        args.consumed_train_samples += (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def eval_step(self, **batch_data):
        """
        Evaluation step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to evaluate on.
        """

        args = get_args()
        data_chunks = []
        if args.num_micro_batches > 1:
            for i in range(0, args.num_micro_batches):
                data_chunks.append(
                    {k: v[i * args.micro_batch_size : (i + 1) * args.micro_batch_size] for k, v in batch_data.items()}
                )
        else:
            data_chunks = [batch_data]

        if len(self.module) > 1:
            batch_data_iterator = [iter(data_chunks) for _ in range(len(self.module))]
        else:
            batch_data_iterator = iter(data_chunks)
        forward_backward_func = get_forward_backward_func()
        loss_dicts = forward_backward_func(
            self.train_step_handler.forward_step,
            batch_data_iterator,
            self.module,
            optimizer=None,
            timers=None,
            forward_only=True,
        )
        # Empty unused memory
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        args.consumed_valid_samples += (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in loss_dicts[0]:
                losses_reduced_for_key = [x[key] for x in loss_dicts]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced
        else:
            return {}

    def forward(self, **batch_data):
        # During training, we use train_step()
        # model(**batch_data) performs following operations by delegating it to `self.train_step`:
        # 1. Prepare **batch_data for Tendor, Pipeline and Model Parallelism
        # 2. Set grad to zero.
        # 3. forward pass and backward pass using Pipeline Parallelism
        # 4. Empty unused memory.
        # 5. Reduce gradients.
        # 6. Update parameters.
        # 7. Gather params when using Distributed Optimizer (Data Parallelism).
        # 8. Update learning rate if scheduler is specified.
        # 9. Empty unused memory.
        # 10. Average loss across microbatches and across DP ranks.
        #
        # During evaluation, we use eval_step()
        args = get_args()
        if self.module[0].training:
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(**batch_data)
            self.iteration += 1
            if args.tensorboard_dir is not None:
                # Logging.
                loss_scale = self.optimizer.get_loss_scale().item()
                params_norm = None
                if args.log_params_norm:
                    params_norm = calc_params_l2_norm(self.model)
                self.report_memory_flag = training_log(
                    loss_dict,
                    self.total_loss_dict,
                    self.optimizer.param_groups[0]["lr"],
                    self.iteration,
                    loss_scale,
                    self.report_memory_flag,
                    skipped_iter,
                    grad_norm,
                    params_norm,
                    num_zeros_in_grad,
                )
        else:
            loss_dict = self.eval_step(**batch_data)
            if args.tensorboard_dir is not None:
                for key in loss_dict:
                    self.eval_total_loss_dict[key] = (
                        self.eval_total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
                    )
                    self.eval_total_loss_dict[key + "_num_iters"] = self.eval_total_loss_dict.get(
                        key + "_num_iters", torch.cuda.FloatTensor([0.0])
                    ) + torch.cuda.FloatTensor([1.0])

        loss = torch.tensor(0.0, device=args.local_rank)
        for key in loss_dict:
            if len(loss_dict[key].shape) == 0:
                loss += loss_dict[key]

        logits = None
        if "logits" in loss_dict:
            logits = loss_dict["logits"]
        # loss = reduce(loss)
        if self.train_step_handler.model_output_class is not None:
            return self.train_step_handler.model_output_class(loss=loss, logits=logits)
        return loss

    def log_eval_results(self):
        args = get_args()
        if args.tensorboard_dir is None or self.iteration == 0:
            return
        args = get_args()
        writer = get_tensorboard_writer()
        string = f"validation loss at iteration {self.iteration} | "
        for key in self.eval_total_loss_dict:
            if key.endswith("_num_iters"):
                continue
            value = self.eval_total_loss_dict[key] / self.eval_total_loss_dict[key + "_num_iters"]
            string += f"{key} value: {value} | "
            ppl = math.exp(min(20, value.item()))
            if args.pretraining_flag:
                string += f"{key} PPL: {ppl} | "
            if writer:
                writer.add_scalar(f"{key} validation", value.item(), self.iteration)
                if args.pretraining_flag:
                    writer.add_scalar(f"{key} validation ppl", ppl, self.iteration)

        length = len(string) + 1
        print_rank_last("-" * length)
        print_rank_last(string)
        print_rank_last("-" * length)
        self.eval_total_loss_dict = {}

    def save_checkpoint(self, output_dir):
        self.log_eval_results()
        args = get_args()
        args.save = output_dir
        torch.distributed.barrier()
        save_checkpoint(self.iteration, self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()

    def load_checkpoint(self, input_dir):
        args = get_args()
        args.load = input_dir
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        torch.distributed.barrier()
        iteration = load_checkpoint(self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()
        self.iteration = iteration
        if args.fp16 and self.iteration == 0:
            self.optimizer.reload_model_params()

    def megatron_generate(
        self,
        inputs,
        attention_mask=None,
        max_length=None,
        max_new_tokens=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        length_penalty=None,
        **kwargs,
    ):
        """
        Generate method for GPT2 model. This method is used for inference. Supports both greedy and beam search along
        with sampling. Refer the Megatron-LM repo for more details

        Args:
            inputs (torch.Tensor): input ids
            attention_mask (torch.Tensor, optional): attention mask. Defaults to None.
            max_length (int, optional): max length of the generated sequence. Defaults to None.
            Either this or max_new_tokens should be provided.
            max_new_tokens (int, optional): max number of tokens to be generated. Defaults to None.
            Either this or max_length should be provided.
            num_beams (int, optional): number of beams to use for beam search. Defaults to None.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            top_k (int, optional): top k tokens to consider for sampling. Defaults to 0.0.
            top_p (float, optional): tokens in top p probability are considered for sampling. Defaults to 0.0.
            length_penalty (float, optional): length penalty for beam search. Defaults to None.
            kwargs: additional key-value arguments
        """

        # checking if required arguments are passed
        args = get_args()
        if args.model_type_name != "gpt":
            raise NotImplementedError("Generate method is not implemented for this model")

        if args.data_parallel_size > 1:
            raise ValueError("Generate method requires data parallelism to be 1")

        if args.sequence_parallel:
            raise ValueError("Generate method requires sequence parallelism to be False")

        if args.recompute_granularity is not None:
            raise ValueError("Checkpoint activations cannot be set for inference")

        if args.vocab_file is None:
            raise ValueError("Vocab file is required for inference")

        # Prepare inputs
        if max_length is None and max_new_tokens is None:
            raise ValueError("`max_length` or `max_new_tokens` are required for inference")

        if temperature is None:
            temperature = 1.0
        elif not (0.0 < temperature <= 100.0):
            raise ValueError("temperature must be a positive number less than or equal to 100.0")

        if top_k is None:
            top_k = 0
        elif not (0 <= top_k <= 1000):
            raise ValueError("top_k must be a positive number less than or equal to 1000")

        if top_p is None:
            top_p = 0.0
        elif top_p > 0.0 and top_k > 0.0:
            raise ValueError("top_p and top_k sampling cannot be set together")
        else:
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be less than or equal to 1.0")

        top_p_decay = kwargs.get("top_p_decay", 0.0)
        if not (0.0 <= top_p_decay <= 1.0):
            raise ValueError("top_p_decay must be less than or equal to 1.0")

        top_p_bound = kwargs.get("top_p_bound", 0.0)
        if not (0.0 <= top_p_bound <= 1.0):
            raise ValueError("top_p_bound must be less than or equal to 1.0")

        add_BOS = kwargs.get("add_BOS", False)
        if not (isinstance(add_BOS, bool)):
            raise ValueError("add_BOS must be a boolean")

        beam_width = num_beams
        if beam_width is not None:
            if not isinstance(beam_width, int):
                raise ValueError("beam_width must be an integer")
            if beam_width < 1:
                raise ValueError("beam_width must be greater than 0")
            if inputs.shape[0] > 1:
                return "When doing beam_search, batch size must be 1"

        tokenizer = get_tokenizer()

        stop_token = kwargs.get("stop_token", tokenizer.eod)
        if stop_token is not None:
            if not isinstance(stop_token, int):
                raise ValueError("stop_token must be an integer")

        if length_penalty is None:
            length_penalty = 1.0

        sizes_list = None
        prompts_tokens_tensor = None
        prompts_length_tensor = None
        if torch.distributed.get_rank() == 0:
            # Get the prompts length.
            if attention_mask is None:
                prompts_length_tensor = torch.cuda.LongTensor([inputs.shape[1]] * inputs.shape[0])
            else:
                prompts_length_tensor = attention_mask.sum(axis=-1).cuda()

            if max_new_tokens is None:
                max_new_tokens = max_length - inputs.shape[1]
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be greater than 0")

            if add_BOS:
                max_length = max_new_tokens + inputs.shape[1] + 1
                # making sure that `max_length` is a multiple of 4 to leverage fused kernels
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - (inputs.shape[1] + 1)
                padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat(
                    [torch.unsqueeze(padding[:, 0], axis=-1), inputs.cuda(), padding], axis=-1
                )
            else:
                # making sure that `max_length` is a multiple of 4 to leverage fused kernels
                max_length = max_new_tokens + inputs.shape[1]
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - inputs.shape[1]
                padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([inputs.cuda(), padding], axis=-1)

            # We need the sizes of these tensors for the boradcast
            sizes_list = [
                prompts_tokens_tensor.size(0),  # Batch size
                prompts_tokens_tensor.size(1),
            ]  # Sequence lenght

        # First, broadcast the sizes.
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=0)

        # Now that we have the sizes, we can boradcast the tokens
        # and length tensors.
        sizes = sizes_tensor.tolist()
        context_tokens_tensor = broadcast_tensor(sizes, torch.int64, tensor=prompts_tokens_tensor, rank=0)
        context_length_tensor = broadcast_tensor(sizes[0], torch.int64, tensor=prompts_length_tensor, rank=0)

        # Run the inference
        random_seed = kwargs.get("random_seed", 0)
        torch.random.manual_seed(random_seed)
        unwrapped_model = unwrap_model(self.base_model, (torchDDP, LocalDDP, Float16Module))
        if beam_width is not None:
            tokens, _ = beam_search_and_return_on_first_stage(
                unwrapped_model,
                context_tokens_tensor,
                context_length_tensor,
                beam_width,
                stop_token=stop_token,
                num_return_gen=1,
                length_penalty=length_penalty,
            )
        else:
            tokens, _, _ = generate_tokens_probs_and_return_on_first_stage(
                unwrapped_model,
                context_tokens_tensor,
                context_length_tensor,
                return_output_log_probs=False,
                top_k=top_k,
                top_p=top_p,
                top_p_decay=top_p_decay,
                top_p_bound=top_p_bound,
                temperature=temperature,
                use_eod_token_for_early_termination=True,
            )
        return tokens


# other utilities
def avg_losses_across_data_parallel_group(losses):
    """
    Average losses across data parallel group.

    Args:
        losses (List[Tensor]): List of losses to average across data parallel group.
    """

    return average_losses_across_data_parallel_group(losses)


def gather_across_data_parallel_groups(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from data parallel ranks.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather across data parallel ranks.

    """

    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [
            torch.empty_like(tensor)
            for _ in range(torch.distributed.get_world_size(group=mpu.get_data_parallel_group()))
        ]
        torch.distributed.all_gather(output_tensors, tensor, group=mpu.get_data_parallel_group())
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)
