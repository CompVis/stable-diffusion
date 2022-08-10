import argparse, os, sys, glob
import clip
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import scann
import time
from multiprocessing import cpu_count

from ldm.util import instantiate_from_config, parallel_data_prefetch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder

DATABASES = [
    "openimages",
    "artbench-art_nouveau",
    "artbench-baroque",
    "artbench-expressionism",
    "artbench-impressionism",
    "artbench-post_impressionism",
    "artbench-realism",
    "artbench-romanticism",
    "artbench-renaissance",
    "artbench-surrealism",
    "artbench-ukiyo_e",
]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class Searcher(object):
    def __init__(self, database, retriever_version='ViT-L/14'):
        assert database in DATABASES
        # self.database = self.load_database(database)
        self.database_name = database
        self.searcher_savedir = f'data/rdm/searchers/{self.database_name}'
        self.database_path = f'data/rdm/retrieval_databases/{self.database_name}'
        self.retriever = self.load_retriever(version=retriever_version)
        self.database = {'embedding': [],
                         'img_id': [],
                         'patch_coords': []}
        self.load_database()
        self.load_searcher()

    def train_searcher(self, k,
                       metric='dot_product',
                       searcher_savedir=None):

        print('Start training searcher')
        searcher = scann.scann_ops_pybind.builder(self.database['embedding'] /
                                                  np.linalg.norm(self.database['embedding'], axis=1)[:, np.newaxis],
                                                  k, metric)
        self.searcher = searcher.score_brute_force().build()
        print('Finish training searcher')

        if searcher_savedir is not None:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            self.searcher.serialize(searcher_savedir)

    def load_single_file(self, saved_embeddings):
        compressed = np.load(saved_embeddings)
        self.database = {key: compressed[key] for key in compressed.files}
        print('Finished loading of clip embeddings.')

    def load_multi_files(self, data_archive):
        out_data = {key: [] for key in self.database}
        for d in tqdm(data_archive, desc=f'Loading datapool from {len(data_archive)} individual files.'):
            for key in d.files:
                out_data[key].append(d[key])

        return out_data

    def load_database(self):

        print(f'Load saved patch embedding from "{self.database_path}"')
        file_content = glob.glob(os.path.join(self.database_path, '*.npz'))

        if len(file_content) == 1:
            self.load_single_file(file_content[0])
        elif len(file_content) > 1:
            data = [np.load(f) for f in file_content]
            prefetched_data = parallel_data_prefetch(self.load_multi_files, data,
                                                     n_proc=min(len(data), cpu_count()), target_data_type='dict')

            self.database = {key: np.concatenate([od[key] for od in prefetched_data], axis=1)[0] for key in
                             self.database}
        else:
            raise ValueError(f'No npz-files in specified path "{self.database_path}" is this directory existing?')

        print(f'Finished loading of retrieval database of length {self.database["embedding"].shape[0]}.')

    def load_retriever(self, version='ViT-L/14', ):
        model = FrozenClipImageEmbedder(model=version)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def load_searcher(self):
        print(f'load searcher for database {self.database_name} from {self.searcher_savedir}')
        self.searcher = scann.scann_ops_pybind.load_searcher(self.searcher_savedir)
        print('Finished loading searcher.')

    def search(self, x, k):
        if self.searcher is None and self.database['embedding'].shape[0] < 2e4:
            self.train_searcher(k)   # quickly fit searcher on the fly for small databases
        assert self.searcher is not None, 'Cannot search with uninitialized searcher'
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if len(x.shape) == 3:
            x = x[:, 0]
        query_embeddings = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

        start = time.time()
        nns, distances = self.searcher.search_batched(query_embeddings, final_num_neighbors=k)
        end = time.time()

        out_embeddings = self.database['embedding'][nns]
        out_img_ids = self.database['img_id'][nns]
        out_pc = self.database['patch_coords'][nns]

        out = {'nn_embeddings': out_embeddings / np.linalg.norm(out_embeddings, axis=-1)[..., np.newaxis],
               'img_ids': out_img_ids,
               'patch_coords': out_pc,
               'queries': x,
               'exec_time': end - start,
               'nns': nns,
               'q_embeddings': query_embeddings}

        return out

    def __call__(self, x, n):
        return self.search(x, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: add n_neighbors and modes (text-only, text-image-retrieval, image-image retrieval etc)
    # TODO: add 'image variation' mode when knn=0 but a single image is given instead of a text prompt?
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--n_repeat",
        type=int,
        default=1,
        help="number of repeats in CLIP latent space",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=768,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=768,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval-augmented-diffusion/768x768.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/rdm/rdm768x768/model.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--clip_type",
        type=str,
        default="ViT-L/14",
        help="which CLIP model to use for retrieval and NN encoding",
    )
    parser.add_argument(
        "--database",
        type=str,
        default='artbench-surrealism',
        choices=DATABASES,
        help="The database used for the search, only applied when --use_neighbors=True",
    )
    parser.add_argument(
        "--use_neighbors",
        default=False,
        action='store_true',
        help="Include neighbors in addition to text prompt for conditioning",
    )
    parser.add_argument(
        "--knn",
        default=10,
        type=int,
        help="The number of included neighbors, only applied when --use_neighbors=True",
    )

    opt = parser.parse_args()

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    clip_text_encoder = FrozenCLIPTextEmbedder(opt.clip_type).to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    print(f"sampling scale for cfg is {opt.scale:.2f}")

    searcher = None
    if opt.use_neighbors:
        searcher = Searcher(opt.database)

    with torch.no_grad():
        with model.ema_scope():
            for n in trange(opt.n_iter, desc="Sampling"):
                all_samples = list()
                for prompts in tqdm(data, desc="data"):
                    print("sampling prompts:", prompts)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = clip_text_encoder.encode(prompts)
                    uc = None
                    if searcher is not None:
                        nn_dict = searcher(c, opt.knn)
                        c = torch.cat([c, torch.from_numpy(nn_dict['nn_embeddings']).cuda()], dim=1)
                    if opt.scale != 1.0:
                        uc = torch.zeros_like(c)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    shape = [16, opt.H // 16, opt.W // 16]  # note: currently hardcoded for f16 model
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=c.shape[0],
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                    all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")
