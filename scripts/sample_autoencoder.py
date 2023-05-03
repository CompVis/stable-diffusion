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


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
# input_tensor= inverse_normalize(tensor=input_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def save_gray_image(grid, outfile, colormap):
    plt.imshow(grid, cmap=colormap)
    plt.colorbar()
    plt.savefig(outfile)
    np.save(outfile, grid)

    plt.close()

    # print(
    #     f"*** grid  \t m {np.mean(grid)},\t s {np.std(grid)},\t min  {np.min(grid)},\t max {np.max(grid)}")


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
    # parser.add_argument(
    #     "-n",
    #     "--name",
    #     type=str,
    #     const=True,
    #     default="",
    #     nargs="?",
    #     help="postfix for logdir",
    # )
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

    # parser.add_argument(
    #     "-t",
    #     "--train",
    #     type=str2bool,
    #     const=True,
    #     default=False,
    #     nargs="?",
    #     help="train",
    # )
    # parser.add_argument(
    #     "--no-test",
    #     type=str2bool,
    #     const=True,
    #     default=False,
    #     nargs="?",
    #     help="disable test",
    # )
    # parser.add_argument(
    #     "-p",
    #     "--project",
    #     help="name of new or path to existing project"
    # )
    # parser.add_argument(
    #     "-d",
    #     "--debug",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="enable post-mortem debugging",
    # )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    # parser.add_argument(
    #     "-f",
    #     "--postfix",
    #     type=str,
    #     default="",
    #     help="post-postfix for default name",
    # )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs_sample",
        help="directory for logging dat shit",
    )
    # parser.add_argument(
    #     "--scale_lr",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=True,
    #     help="scale base-lr by ngpu * batch_size * n_accumulate",
    # )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id *
                                               split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(
            self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(
            self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            # path_temp = self.logdir
            # if not os.path.exists(path_temp) :
            #     os.makedirs(path_temp)

            os.makedirs(self.ckptdir, exist_ok=True)
            # path_temp = self.ckptdir
            # if not os.path.exists(path_temp) :
            #     os.makedirs(path_temp)

            os.makedirs(self.cfgdir, exist_ok=True)
            print("\n\n********************")
            print("self.logdir", self.logdir, Path(self.logdir).is_dir())
            print("self.ckptdir", self.ckptdir, Path(self.ckptdir).is_dir())
            print("self.cfgdir", self.cfgdir, Path(self.cfgdir).is_dir())

            print("\n\n********************")

            # path = self.cfgdir
            # isExist = os.path.exists(path)
            # if not isExist:
            #     os.makedirs(path)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(
                        self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
                    # path_temp = self.ckptdiros.path.join(
                    #     self.ckptdir, 'trainstep_checkpoints')
                    # if not os.path.exists(path_temp) :
                    #     os.makedirs(path_temp)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))

            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                # if not os.path.exists(os.path.split(dst)[0]) :
                # os.makedirs(os.path.split(dst)[0])
                print('\n\n********')
                print('self.logdir', self.logdir)
                print('dst', dst)
                print('\n\n********')
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        print("\n\n rescale", rescale)
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # path_temp = os.path.split(path)[0]
            # if not os.path.exists(path_temp) :
            #     os.makedirs(path_temp)

            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            print("******* log_local ImageLogger ", pl_module.logger.save_dir)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(
            trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    # if opt.name and opt.resume:
    #     raise ValueError(
    #         "-n/--name and -r/--resume cannot be specified both."
    #         "If you want to resume training in a new log folder, "
    #         "use -n/--name in combination with --resume_from_checkpoint"
    #     )
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

    try:
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
        print("ckpt", ckpt)
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
        print("\n\n\n\n ******************     ngpu", ngpu)

        print("\n\n *********  means", means)
        print("\n\n *********  stds", stds)

        for batch_ndx, sample in enumerate(data):

            with torch.no_grad():
                images = model.log_images(sample)
            for k in images:
                images[k] = images[k]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    # images[k] = torch.clamp(images[k], -1., 1.)

                for i, image in enumerate(images[k]):
                    # ******************************
                    # print_stat(image, inc="=", depht=10, type="tensor")
                    var_indexes = ['u', 'v', 't2m']
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

                    # invTransM1 = transforms.Compose([transforms.Normalize(mean=[0.] * len(var_indexes),
                    #                                                       std=[1 / el for el in Dl_train.stds]),
                    #                                 transforms.Normalize(mean=[-el for el in Dl_train.means],
                    #                                                      std=[1.]*len(var_indexes)), ])

                    grid = invTrans(image)
                    grid = torch.transpose(grid, 0, 2)
                    grid = torch.fliplr(grid)
                    grid = torch.rot90(grid, 2)
                    # print_stat(grid, inc="=", depht=8, type="tensor")
                    grid = grid.numpy()
                    # print_stat(grid, inc="=", depht=6)
                    # # grid = grid.astype(np.uint8)
                    # print_stat(grid, inc="*", depht=4)

                    # ******************************
                    # image = torch.clamp(image, -1., 1.)
                    # grid = image
                    # print(
                    #     f"\n*** grid  origin \t m {torch.mean(grid)},\t s {torch.std(grid)},\t min  {torch.min(grid)},\t max {torch.max(grid)}")

                    # print('grid', grid.shape)
                    # grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    # grid = torch.transpose(grid, 0, 2)
                    # grid = torch.fliplr(grid)
                    # grid = torch.rot90(grid, 2)
                    # grid = grid.numpy()
                    # grid = (grid * 255).astype(np.uint8)

                    # print(
                    #     f"\n*** grid final  \t m {np.mean(grid)},\t s {np.std(grid)},\t min  {np.min(grid)},\t max {np.max(grid)}")
                    # ******************************

                    filename = f"{k}-{batch_ndx}-{i}.png"
                    path = os.path.join(opt.logdir, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    # print_stat(grid, inc="*", depht=2)
                    save_gray_image(grid[:, :, 0], os.path.join(
                        opt.logdir, "u_" + filename), 'viridis')
                    save_gray_image(grid[:, :, 1], os.path.join(
                        opt.logdir, "v_" + filename), 'viridis')
                    save_gray_image(grid[:, :, 2], os.path.join(
                        opt.logdir, "t2m_" + filename), plt.cm.get_cmap('RdBu_r'))

                    # plt.imshow(grid[:, :, 0], cmap='viridis',
                    #            interpolation='nearest', aspect='auto')
                    # plt.colorbar()
                    # outfile = os.path.join(opt.logdir, "u_" + filename)
                    # plt.savefig(outfile)
                    # np.save(outfile, grid[:, :, 0])

                    # plt.close()
                    # plt.imshow(grid[:, :, 1], cmap='viridis')
                    # plt.savefig(os.path.join(opt.logdir, "v_" + filename))
                    # np.save(outfile, grid[:, :, 1])
                    # plt.close()

                    # plt.imshow(grid[:, :, 2])
                    # plt.savefig(os.path.join(opt.logdir, "t_" + filename), cmap=plt.cm.get_cmap('RdBu_r'))
                    # np.save(outfile, grid[:, :, 1])
                    # plt.close()

                    # Image.fromarray(grid[:, :, 0]).save(path)

            if config.data.params.batch_size * (batch_ndx + 1) > opt.nb_images:
                break

    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            # path_temp = os.path.split(dst)[0]
            # if not os.path.exists(path_temp) :
            #     os.makedirs(path_temp)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
