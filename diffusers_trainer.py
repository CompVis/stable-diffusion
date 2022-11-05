# Install bitsandbytes:
# `nvcc --version` to get CUDA version.
# `pip install -i https://test.pypi.org/simple/ bitsandbytes-cudaXXX` to install for current CUDA.
# Example Usage:
# Single GPU: torchrun --nproc_per_node=1 trainer_dist.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=10 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True
# Multiple GPUs: torchrun --nproc_per_node=N trainer_dist.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=10 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True

import argparse
import socket
import torch
import torchvision
import transformers
import diffusers
import os
import glob
import random
import tqdm
import resource
import psutil
import pynvml
import wandb
import gc
import time
import itertools
import numpy as np
import json
import re

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from PIL import Image

from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d

torch.backends.cuda.matmul.allow_tf32 = True

# defaults should be good for everyone
# TODO: add custom VAE support. should be simple with diffusers
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--model', type=str, default=None, required=True, help='The name of the model to use for finetuning. Could be HuggingFace ID or a directory')
parser.add_argument('--run_name', type=str, default=None, required=True, help='Name of the finetune run.')
parser.add_argument('--dataset', type=str, default=None, required=True, help='The path to the dataset to use for finetuning.')
parser.add_argument('--num_buckets', type=int, default=16, help='The number of buckets.')
parser.add_argument('--bucket_side_min', type=int, default=256, help='The minimum side length of a bucket.')
parser.add_argument('--bucket_side_max', type=int, default=768, help='The maximum side length of a bucket.')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--use_ema', type=bool, default=False, help='Use EMA for finetuning')
parser.add_argument('--ucg', type=float, default=0.1, help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.') # 10% dropout probability
parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', type=bool, default=False, help='Enable gradient checkpointing')
parser.add_argument('--use_8bit_adam', dest='use_8bit_adam', type=bool, default=False, help='Use 8-bit Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help='Adam weight decay')
parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='Adam epsilon')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator, this is to be used for reproduceability purposes.')
parser.add_argument('--output_path', type=str, default='./output', help='Root path for all outputs.')
parser.add_argument('--save_steps', type=int, default=500, help='Number of steps to save checkpoints at.')
parser.add_argument('--resolution', type=int, default=512, help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
parser.add_argument('--shuffle', dest='shuffle', type=bool, default=True, help='Shuffle dataset')
parser.add_argument('--hf_token', type=str, default=None, required=False, help='A HuggingFace token is needed to download private models for training.')
parser.add_argument('--project_id', type=str, default='diffusers', help='Project ID for reporting to WandB')
parser.add_argument('--fp16', dest='fp16', type=bool, default=False, help='Train in mixed precision')
parser.add_argument('--image_log_steps', type=int, default=100, help='Number of steps to log images at.')
parser.add_argument('--image_log_amount', type=int, default=4, help='Number of images to log every image_log_steps')
parser.add_argument('--clip_penultimate', type=bool, default=False, help='Use penultimate CLIP layer for text embedding')
parser.add_argument('--output_bucket_info', type=bool, default=False, help='Outputs bucket information and exits')
args = parser.parse_args()

def setup():
    torch.distributed.init_process_group("nccl", init_method="env://")

def cleanup():
    torch.distributed.destroy_process_group()

def get_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

# Inform the user of host, and various versions -- useful for debugging isseus.
print("RUN_NAME:", args.run_name)
print("HOST:", socket.gethostname())
print("CUDA:", torch.version.cuda)
print("TORCH:", torch.__version__)
print("TRANSFORMERS:", transformers.__version__)
print("DIFFUSERS:", diffusers.__version__)
print("MODEL:", args.model)
print("FP16:", args.fp16)
print("RESOLUTION:", args.resolution)

def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1E6)
        gpu_free = int(gpu_info.free / 1E6)
        gpu_used = int(gpu_info.used / 1E6)
        gpu_str = f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb " \
                  f"T: {gpu_total:,}mb) "
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1E6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1E6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1E6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1E6)
        torch_str = f"TORCH: (R: {torch_reserved_gpu:,}mb/"  \
                    f"{torch_reserved_max:,}mb, " \
                    f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
    except AssertionError:
        pass
    cpu_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E3 +
                     resource.getrusage(
                         resource.RUSAGE_CHILDREN).ru_maxrss / 1E3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1E6)
    return f"CPU: (maxrss: {cpu_maxrss:,}mb F: {cpu_free:,}mb) " \
           f"{gpu_str}" \
           f"{torch_str}"

def _sort_by_ratio(bucket: tuple) -> float:
    return bucket[0] / bucket[1]

def _sort_by_area(bucket: tuple) -> float:
    return bucket[0] * bucket[1]

class ImageStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_dir}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        self.image_files = [x for x in self.image_files if self.__valid_file(x)]

    def __len__(self) -> int:
        return len(self.image_files)

    def __valid_file(self, f) -> bool:
        try:
            Image.open(f)
            return True
        except:
            print(f'WARNING: Unable to open file: {f}')
            return False

    # iterator returns images as PIL images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Image.Image, int], None, None]:
        for f in range(len(self)):
            yield Image.open(self.image_files[f]), f

    # get image by index
    def get_image(self, index: int) -> Image.Image:
        return Image.open(self.image_files[index])

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, index: int) -> str:
        filename = re.sub('\.[^/.]+$', '', self.image_files[index]) + '.txt'
        with open(filename, 'r') as f:
            return f.read()


# ====================================== #
# Bucketing code stolen from hasuwoof:   #
# https://github.com/hasuwoof/huskystack #
# ====================================== #

class AspectBucket:
    def __init__(self, store: ImageStore,
                 num_buckets: int,
                 batch_size: int,
                 bucket_side_min: int = 256,
                 bucket_side_max: int = 768,
                 bucket_side_increment: int = 64,
                 max_image_area: int = 512 * 768,
                 max_ratio: float = 2):

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.max_image_area = max_image_area
        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.fill_buckets()

    def init_buckets(self):
        possible_lengths = list(range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))
        possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                        if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)

        buckets_by_ratio = {}

        # group the buckets by their aspect ratios
        for bucket in possible_buckets:
            w, h = bucket
            # use precision to avoid spooky floats messing up your day
            ratio = '{:.4e}'.format(w / h)

            if ratio not in buckets_by_ratio:
                group = set()
                buckets_by_ratio[ratio] = group
            else:
                group = buckets_by_ratio[ratio]

            group.add(bucket)

        # now we take the list of buckets we generated and pick the largest by area for each (the first sorted)
        # then we put all of those in a list, sorted by the aspect ratio
        # the square bucket (LxL) will be the first
        unique_ratio_buckets = sorted([sorted(buckets, key=_sort_by_area)[-1]
                                       for buckets in buckets_by_ratio.values()], key=_sort_by_ratio)

        # how many buckets to create for each side of the distribution
        bucket_count_each = int(np.clip((self.requested_bucket_count + 1) / 2, 1, len(unique_ratio_buckets)))

        # we know that the requested_bucket_count must be an odd number, so the indices we calculate
        # will include the square bucket and some linearly spaced buckets along the distribution
        indices = {*np.linspace(0, len(unique_ratio_buckets) - 1, bucket_count_each, dtype=int)}

        # make the buckets, make sure they are unique (to remove the duplicated square bucket), and sort them by ratio
        # here we add the portrait buckets by reversing the dimensions of the landscape buckets we generated above
        buckets = sorted({*(unique_ratio_buckets[i] for i in indices),
                          *(tuple(reversed(unique_ratio_buckets[i])) for i in indices)}, key=_sort_by_ratio)

        self.buckets = buckets

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = [w / h for w, h in buckets]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(buckets))), assume_sorted=True,
                                       fill_value=None)

        for b in buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({ "buckets": self.buckets, "bucket_ratios": self._bucket_ratios })

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int], List[int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            ((w, h), [image1, image2, ..., image{batch_size}])

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [idx for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, entry: Image.Image, index: int) -> bool:
        aspect = entry.width / entry.height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False

        bucket = self.buckets[round(float(best_bucket))]

        self.bucket_data[bucket].append(index)

        del entry

        return True

class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket, num_replicas: int = 1, rank: int = 0):
        super().__init__(None)
        self.bucket = bucket
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # subsample the bucket to only include the elements that are assigned to this rank
        indices = self.bucket.get_batch_iterator()
        indices = list(indices)[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.bucket.get_batch_count() // self.num_replicas

class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore, tokenizer: CLIPTokenizer, ucg: float = 0.1):
        self.store = store
        self.tokenizer = tokenizer
        self.ucg = ucg

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: int):
        return_dict = {'pixel_values': None, 'input_ids': None}

        image_file = self.store.get_image(item)
        return_dict['pixel_values'] = self.transforms(image_file)
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(item)
        else:
            caption_file = ''
        return_dict['input_ids'] = self.tokenizer(caption_file, max_length=self.tokenizer.model_max_length, padding='do_not_pad', truncation=True).input_ids

        return return_dict

    def collate_fn(self, examples):
            pixel_values = torch.stack([example['pixel_values'] for example in examples if example is not None])
            pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = [example['input_ids'] for example in examples if example is not None]
            padded_tokens = self.tokenizer.pad({'input_ids': input_ids}, return_tensors='pt', padding=True)
            return {
                'pixel_values': pixel_values,
                'input_ids': padded_tokens.input_ids,
                'attention_mask': padded_tokens.attention_mask,
            }

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

def main():
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)

    if args.hf_token is None:
        args.hf_token = os.environ['HF_API_TOKEN']

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

        # remove hf_token from args so sneaky people don't steal it from the wandb logs
        sanitized_args = {k: v for k, v in vars(args).items() if k not in ['hf_token']}
        run = wandb.init(project=args.project_id, name=args.run_name, config=sanitized_args, dir=args.output_path+'/wandb')

    device = torch.device('cuda')

    print("DEVICE:", device)

    # setup fp16 stuff
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Set seed
    torch.manual_seed(args.seed)
    print('RANDOM SEED:', args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer', use_auth_token=args.hf_token)
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder', use_auth_token=args.hf_token)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae', use_auth_token=args.hf_token)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet', use_auth_token=args.hf_token)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.use_8bit_adam: # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print('bitsandbytes not supported, using regular Adam optimizer')
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        num_train_timesteps=1000,
    )

    # load dataset

    store = ImageStore(args.dataset)
    dataset = AspectDataset(store, tokenizer)
    bucket = AspectBucket(store, args.num_buckets, args.batch_size, args.bucket_side_min, args.bucket_side_max, 64, args.resolution * args.resolution, 2.0)
    sampler = AspectBucketSampler(bucket=bucket, num_replicas=world_size, rank=rank)

    print(f'STORE_LEN: {len(store)}')

    if args.output_bucket_info:
        print(bucket.get_bucket_info())
        exit(0)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer
    )

    weight_dtype = torch.float16 if args.fp16 else torch.float32

    # move models to device
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=torch.float32)
    text_encoder = text_encoder.to(device, dtype=weight_dtype)

    #unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[rank], output_device=rank, gradient_as_bucket_view=True)

    # create ema
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    print(get_gpu_ram())

    num_steps_per_epoch = len(train_dataloader)
    progress_bar = tqdm.tqdm(range(args.epochs * num_steps_per_epoch), desc="Total Steps", leave=False)
    global_step = 0

    def save_checkpoint():
        if rank == 0:
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=PNDMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                ),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            pipeline.save_pretrained(args.output_path)
        # barrier
        torch.distributed.barrier()

    # train!
    loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
    for epoch in range(args.epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            b_start = time.perf_counter()
            latents = vae.encode(batch['pixel_values'].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch['input_ids'].to(device), output_hidden_states=True)
            if args.clip_penultimate:
                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
            else:
                encoder_hidden_states = encoder_hidden_states.last_hidden_state

            # Predict the noise residual and compute loss
            with torch.autocast('cuda', enabled=args.fp16):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backprop and all reduce
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update EMA
            if args.use_ema:
                ema_unet.step(unet.parameters())

            # perf
            b_end = time.perf_counter()
            seconds_per_step = b_end - b_start
            steps_per_second = 1 / seconds_per_step
            rank_images_per_second = args.batch_size * steps_per_second
            world_images_per_second = rank_images_per_second * world_size
            samples_seen = global_step * args.batch_size * world_size

            # All reduce loss
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)

            if rank == 0:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "train/loss": loss.detach().item() / world_size,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/samples_seen": samples_seen,
                    "perf/rank_samples_per_second": rank_images_per_second,
                    "perf/global_samples_per_second": world_images_per_second,
                }
                progress_bar.set_postfix(logs)
                run.log(logs)

            if global_step % args.save_steps == 0:
                save_checkpoint()

            if global_step % args.image_log_steps == 0:
                if rank == 0:
                    # get prompt from random batch
                    prompt = tokenizer.decode(batch['input_ids'][random.randint(0, len(batch['input_ids'])-1)].tolist())
                    pipeline = StableDiffusionPipeline(
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet,
                        tokenizer=tokenizer,
                        scheduler=PNDMScheduler(
                            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                        ),
                        safety_checker=None, # display safety checker to save memory
                        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                    ).to(device)
                    # inference
                    images = []
                    with torch.no_grad():
                        with torch.autocast('cuda', enabled=args.fp16):
                            for _ in range(args.image_log_amount):
                                images.append(wandb.Image(pipeline(prompt).images[0], caption=prompt))
                    # log images under single caption
                    run.log({'images': images})

                    # cleanup so we don't run out of memory
                    del pipeline
                    gc.collect()
                torch.distributed.barrier()

    if rank == 0:
        save_checkpoint()

    torch.distributed.barrier()
    cleanup()

    print(get_gpu_ram())
    print('Done!')

if __name__ == "__main__":
    setup()
    main()
