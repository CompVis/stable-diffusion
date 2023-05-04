from ldm.util import instantiate_from_config
from ldm.data.base import Txt2ImgIterableBaseDataset
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from PIL import Image
from functools import partial
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import OmegaConf
from packaging import version
import pytorch_lightning as pl
import torchvision
import torch
import time
import numpy as np
import csv
import importlib
import glob
import datetime
import sys
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


makedirs_origin = os.makedirs


def makedirs_pathlib(path, mode=0o777, exist_ok=False):
    p = Path(path)
    try:
        p.mkdir(mode=mode, exist_ok=exist_ok)
    except FileNotFoundError as e:
        print(f"WARNING : {e} \n=> Nested directory creation activates")
        p.mkdir(mode=mode, parents=True, exist_ok=exist_ok)
    except FileExistsError as e:
        print(f"WARNING : {e} \n=> the nested directory already exist")
        pass
    pass


os.makedirs = makedirs_pathlib


def save_gray_image(grid, outfile, colormap):
    plt.imshow(grid, cmap=colormap)
    plt.colorbar()
    plt.savefig(outfile)
    np.save(os.path.split(
        outfile)[0] + "/npy/" + os.path.split(outfile)[1], grid)
    plt.close()


def print_stat(image, inc="=", depht=2, type="numpy"):
    if type == "numpy":

        print("\n"+inc*depht +
              f" grid u  \t m {np.mean(image[0,:,:])},\t s {np.std(image[0,:,:])},\t min  {np.min(image[0,:,:])},\t max {np.max(image[0,:,:])}")
        print(inc*depht +
              f" grid v \t m {np.mean(image[1,:,:])},\t s {np.std(image[1,:,:])},\t min  {np.min(image[1,:,:])},\t max {np.max(image[1,:,:])}")
        print(inc*depht +
              f" grid t \t m {np.mean(image[2,:,:])},\t s {np.std(image[2,:,:])},\t min  {np.min(image[2,:,:])},\t max {np.max(image[2,:,:])}")
    else:
        print("\n"+inc*depht +
              f" grid u  \t m {torch.mean(image[0,:,:])},\t s {torch.std(image[0,:,:])},\t min  {torch.min(image[0,:,:])},\t max {torch.max(image[0,:,:])}")
        print(inc*depht +
              f" grid v \t m {torch.mean(image[1,:,:])},\t s {torch.std(image[1,:,:])},\t min  {torch.min(image[1,:,:])},\t max {torch.max(image[1,:,:])}")
        print(inc*depht +
              f" grid t \t m {torch.mean(image[2,:,:])},\t s {torch.std(image[2,:,:])},\t min  {torch.min(image[2,:,:])},\t max {torch.max(image[2,:,:])}")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="overwrite batch_size",
    )
    parser.add_argument(
        "-nb",
        "--nb_images",
        type=int,
        default=8,
        help="nb samples",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs_sample",
        help="directory for logging dat shit",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`

    sys.path.append(os.getcwd())
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(
            glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    # trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt)

    # data
    if opt.batch_size != None:
        config.data.params.batch_size = opt.batch_size

    import ldm.data.GENS_handler as DSH
    Dl_train = DSH.ISData_Loader_train(config.data.params.batch_size)
    data, dataset = Dl_train.loader()
    means, stds = Dl_train.norm_info()

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = 1 if lightning_config.trainer.gpus == 0 else len(
            lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    print("\n *** ngpu", ngpu)

    for batch_ndx, sample in enumerate(data):

        with torch.no_grad():
            images = model.log_images(sample, only_samples=False)
        for k in images:
            images[k] = images[k]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                # images[k] = torch.clamp(images[k], -1., 1.)

            for i, image in enumerate(images[k]):

                invTrans = transforms.Compose([
                    transforms.Normalize(
                        mean=[0.] * 3, std=[1 / el for el in [2, 2, 2]]),
                    transforms.Normalize(
                        mean=[-el for el in [-1, -1, -1]], std=[1.] * 3),
                    transforms.Normalize(
                        mean=[0.] * 3, std=[1 / el for el in stds]),
                    transforms.Normalize(
                        mean=[-el for el in means], std=[1.] * 3),
                ])
                grid = invTrans(image)
                grid = torch.transpose(grid, 0, 2)
                grid = torch.fliplr(grid)
                grid = torch.rot90(grid, 2)
                grid = grid.numpy()

                filename = f"{k}-{batch_ndx}-{i}.png"
                path = os.path.join(opt.logdir, filename)
                os.makedirs(opt.logdir, exist_ok=True)
                os.makedirs(opt.logdir + "/u", exist_ok=True)
                os.makedirs(opt.logdir + "/u/npy", exist_ok=True)
                os.makedirs(opt.logdir + "/v", exist_ok=True)
                os.makedirs(opt.logdir + "/v/npy", exist_ok=True)
                os.makedirs(opt.logdir + "/t", exist_ok=True)
                os.makedirs(opt.logdir + "/t/npy", exist_ok=True)

                # print_stat(grid, inc="*", depht=2)
                save_gray_image(grid[:, :, 0], opt.logdir +
                                "/u/" + filename, 'viridis')
                save_gray_image(grid[:, :, 1], opt.logdir +
                                "/v/" + filename, 'viridis')
                save_gray_image(grid[:, :, 2], opt.logdir +
                                "/t/" + filename, 'RdBu_r')

        if config.data.params.batch_size * (batch_ndx + 1) > opt.nb_images:
            break
