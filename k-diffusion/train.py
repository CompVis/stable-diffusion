#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path

import accelerate
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import trange, tqdm

import k_diffusion as K


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--evaluate-every', type=int, default=10000,
                   help='save a demo grid every this many steps')
    p.add_argument('--evaluate-n', type=int, default=2000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--grow', type=str,
                   help='the checkpoint to grow from')
    p.add_argument('--grow-config', type=str,
                   help='the configuration file of the model to grow from')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--resume', type=str, 
                   help='the checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    args = p.parse_args()

    mp.set_start_method(args.start_method)

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])

    inner_model = K.config.make_model(config)
    if accelerator.is_main_process:
        print('Parameters:', K.utils.n_params(inner_model))

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, save_code=True)

    assert opt_config['type'] == 'adamw'
    opt = optim.AdamW(inner_model.parameters(),
                      lr=opt_config['lr'] if args.lr is None else args.lr,
                      betas=tuple(opt_config['betas']),
                      eps=opt_config['eps'],
                      weight_decay=opt_config['weight_decay'])

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                inv_gamma=sched_config['inv_gamma'],
                                power=sched_config['power'],
                                warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size[0]),
        K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob']),
    ])

    if dataset_config['type'] == 'imagefolder':
        train_set = datasets.ImageFolder(dataset_config['location'], transform=tf)
    elif dataset_config['type'] == 'cifar10':
        train_set = datasets.CIFAR10(dataset_config['location'], train=True, download=True, transform=tf)
    elif dataset_config['type'] == 'mnist':
        train_set = datasets.MNIST(dataset_config['location'], train=True, download=True, transform=tf)
    elif dataset_config['type'] == 'huggingface':
        from datasets import load_dataset
        train_set = load_dataset(dataset_config['location'])
        train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key']))
        train_set = train_set['train']
    else:
        raise ValueError('Invalid dataset type')

    image_key = dataset_config.get('image_key', 0)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True)

    if args.grow:
        if not args.grow_config:
            raise ValueError('--grow requires --grow-config')
        ckpt = torch.load(args.grow, map_location='cpu')
        old_config = K.config.load_config(open(args.grow_config))
        old_inner_model = K.config.make_model(old_config)
        old_inner_model.load_state_dict(ckpt['model_ema'])
        if old_config['model']['skip_stages'] != model_config['skip_stages']:
            old_inner_model.set_skip_stages(model_config['skip_stages'])
        if old_config['model']['patch_size'] != model_config['patch_size']:
            old_inner_model.set_patch_size(model_config['patch_size'])
        inner_model.load_state_dict(old_inner_model.state_dict())
        del ckpt, old_inner_model

    inner_model, opt, train_dl = accelerator.prepare(inner_model, opt, train_dl)
    if use_wandb:
        wandb.watch(inner_model)
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None

    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])
    model_ema = deepcopy(model)

    state_path = Path(f'{args.name}_state.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        accelerator.unwrap_model(model.inner_model).load_state_dict(ckpt['model'])
        accelerator.unwrap_model(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and 'gns_stats' in ckpt and ckpt['gns_stats'] is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        del ckpt
    else:
        epoch = 0
        step = 0

    extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
    train_iter = iter(train_dl)
    if accelerator.is_main_process:
        print('Computing features for reals...')
    reals_features = K.evaluation.compute_features(accelerator, lambda x: next(train_iter)[image_key][1], extractor, args.evaluate_n, args.batch_size)
    if accelerator.is_main_process:
        metrics_log_filepath = Path(f'{args.name}_metrics.csv')
        if metrics_log_filepath.exists():
            metrics_log_file = open(metrics_log_filepath, 'a')
        else:
            metrics_log_file = open(metrics_log_filepath, 'w')
            print('step', 'fid', 'kid', sep=',', file=metrics_log_file, flush=True)
    del train_iter

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def demo():
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        filename = f'{args.name}_demo_{step:08}.png'
        n_per_proc = math.ceil(args.sample_n / accelerator.num_processes)
        x = torch.randn([n_per_proc, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=not accelerator.is_main_process)
        x_0 = accelerator.gather(x_0)[:args.sample_n]
        if accelerator.is_main_process:
            grid = utils.make_grid(x_0, nrow=math.ceil(args.sample_n ** 0.5), padding=0)
            K.utils.to_pil_image(grid).save(filename)
            if use_wandb:
                wandb.log({'demo_grid': wandb.Image(filename)}, step=step)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=True)
            return x_0
        fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size)
        if accelerator.is_main_process:
            fid = K.evaluation.fid(fakes_features, reals_features)
            kid = K.evaluation.kid(fakes_features, reals_features)
            print(f'FID: {fid.item():g}, KID: {kid.item():g}')
            if accelerator.is_main_process:
                print(step, fid.item(), kid.item(), sep=',', file=metrics_log_file, flush=True)
            if use_wandb:
                wandb.log({'FID': fid.item(), 'KID': kid.item()}, step=step)

    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model.inner_model).state_dict(),
            'model_ema': accelerator.unwrap_model(model_ema.inner_model).state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    reals, _, aug_cond = batch[image_key]
                    noise = torch.randn_like(reals)
                    sigma = sample_density([reals.shape[0]], device=device)
                    losses = model.loss(reals, noise, sigma, aug_cond=aug_cond)
                    losses_all = accelerator.gather(losses.detach())
                    loss_local = losses.mean()
                    loss = losses_all.mean()
                    accelerator.backward(loss_local)
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = accelerator.reduce(gns_stats_hook.get_stats(), 'mean').tolist()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    if accelerator.sync_gradients:
                        ema_decay = ema_sched.get_value()
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                if use_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'loss': loss.item(),
                        'lr': sched.get_last_lr()[0],
                        'ema_decay': ema_decay,
                    }
                    if args.gns:
                        log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                    wandb.log(log_dict, step=step)

                if step % args.demo_every == 0:
                    demo()

                if step > 0 and args.evaluate_every > 0 and step % args.evaluate_every == 0:
                    evaluate()

                if step > 0 and step % args.save_every == 0:
                    save()

                step += 1
            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
