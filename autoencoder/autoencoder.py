# Thanks to Bryce, and his work on ImaginAIry https://github.com/brycedrennan/imaginAIry
# pylama:ignore=W0613
import logging
import math
from contextlib import contextmanager

import pytorch_lightning as pl
import torch

from autoencoder.feather_tile import rebuild_image, tile_image
from ldm.modules.diffusionmodules.model import Decoder, Encoder
from autoencoder.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config

logger = logging.getLogger(__name__)

class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=None,
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
    ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0.0 < ema_decay < 1.0
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        ks = 128
        stride = 64
        vqf = 8
        self.split_input_params = {
            "ks": (ks, ks),
            "stride": (stride, stride),
            "vqf": vqf,
            "patch_distributed_vq": True,
            "tie_braker": False,
            "clip_max_weight": 0.5,
            "clip_min_weight": 0.01,
            "clip_max_tie_weight": 0.5,
            "clip_min_tie_weight": 0.01,
        }

    def init_from_ckpt(self, path, ignore_keys=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        ignore_keys = [] if ignore_keys is None else ignore_keys
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        try:
            return self.encode_all_at_once(x)
        except:
            # Out of memory, trying sliced encoding.
            try:
                return self.encode_sliced(x, chunk_size=128)
            except:
                # Out of memory, trying smaller slice.
                try:
                    return self.encode_sliced(x, chunk_size=64)
                except:
                    # Out of memory, trying smaller slice.
                    try:
                        return self.encode_sliced(x, chunk_size=32)
                    except:
                        # Out of memory, trying smaller slice.
                        return self.encode_sliced(x, chunk_size=16)
                    

    def encode_all_at_once(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.mode()

    def encode_sliced(self, x, chunk_size=128 * 8):
        """
        encodes the image in slices.
        """
        b, c, h, w = x.size()
        final_tensor = torch.zeros(
            [1, 4, math.ceil(h / 8), math.ceil(w / 8)], device=x.device
        )
        for x_img in x.split(1):
            encoded_chunks = []
            overlap_pct = 0.5
            chunks = tile_image(
                x_img, tile_size=chunk_size, overlap_percent=overlap_pct
            )

            for img_chunk in chunks:
                h = self.encoder(img_chunk)
                moments = self.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                encoded_chunks.append(posterior.sample())
            final_tensor = rebuild_image(
                encoded_chunks,
                base_img=final_tensor,
                tile_size=chunk_size // 8,
                overlap_percent=overlap_pct,
            )

        return final_tensor

    def encode_with_folds(self, x):
        bs, nc, h, w = x.shape
        ks = self.split_input_params["ks"]  # eg. (128, 128)
        stride = self.split_input_params["stride"]  # eg. (64, 64)
        df = self.split_input_params["vqf"]

        if h > ks[0] * df or w > ks[1] * df:
            self.split_input_params["original_image_size"] = x.shape[-2:]
            orig_shape = x.shape

            if ks[0] > h // df or ks[1] > w // df:
                ks = (min(ks[0], h // df), min(ks[1], w // df))
                logger.debug(f"reducing Kernel to {ks}")

            if stride[0] > h // df or stride[1] > w // df:
                stride = (min(stride[0], h // df), min(stride[1], w // df))
                logger.debug("reducing stride")
            bottom_pad = math.ceil(h / (ks[0] * df)) * (ks[0] * df) - h
            right_pad = math.ceil(w / (ks[1] * df)) * (ks[1] * df) - w

            padded_x = torch.zeros(
                (bs, nc, h + bottom_pad, w + right_pad), device=x.device
            )
            padded_x[:, :, :h, :w] = x
            x = padded_x

            fold, unfold, normalization, weighting = self.get_fold_unfold(
                x, ks, stride, df=df
            )
            z = unfold(x)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view(
                (z.shape[0], -1, ks[0] * df, ks[1] * df, z.shape[-1])
            )  # (bn, nc, ks[0], ks[1], L )

            output_list = [
                self.encode_all_at_once(z[:, :, :, :, i]) for i in range(z.shape[-1])
            ]

            o = torch.stack(output_list, axis=-1)
            o = o * weighting

            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            encoded = fold(o)
            encoded = encoded / normalization
            # trim off padding
            encoded = encoded[:, :, : orig_shape[2] // df, : orig_shape[3] // df]

            return encoded

        return self.encode_all_at_once(x)

    def decode(self, z):
        try:
            return self.decode_all_at_once(z)
        except:
            # Out of memory, trying sliced decoding.
            try:
                return self.decode_sliced(z, chunk_size=128)
            except:
                # Out of memory, trying smaller slice.
                try:
                    return self.decode_sliced(z, chunk_size=64)
                except:
                    # Out of memory, trying smaller slice.
                    return self.decode_sliced(z, chunk_size=32)

    def decode_all_at_once(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_sliced(self, z, chunk_size=128):
        """
        decodes the tensor in slices.
        This results in images that don't exactly match, so we overlap, feather, and merge to reduce
        (but not completely elminate) impact.
        """
        b, c, h, w = z.size()
        final_tensor = torch.zeros([1, 3, h * 8, w * 8], device=z.device)
        for z_latent in z.split(1):
            decoded_chunks = []
            overlap_pct = 0.5
            chunks = tile_image(
                z_latent, tile_size=chunk_size, overlap_percent=overlap_pct
            )

            for latent_chunk in chunks:
                latent_chunk = self.post_quant_conv(latent_chunk)
                dec = self.decoder(latent_chunk)
                decoded_chunks.append(dec)
            final_tensor = rebuild_image(
                decoded_chunks,
                base_img=final_tensor,
                tile_size=chunk_size * 8,
                overlap_percent=overlap_pct,
            )

            return final_tensor

    def decode_with_folds(self, z):
        ks = self.split_input_params["ks"]  # eg. (128, 128)
        stride = self.split_input_params["stride"]  # eg. (64, 64)
        uf = self.split_input_params["vqf"]
        orig_shape = z.shape
        bs, nc, h, w = z.shape
        bottom_pad = math.ceil(h / ks[0]) * ks[0] - h
        right_pad = math.ceil(w / ks[1]) * ks[1] - w

        # pad the latent such that the unfolding will cover the whole image
        padded_z = torch.zeros((bs, nc, h + bottom_pad, w + right_pad), device=z.device)
        padded_z[:, :, :h, :w] = z
        z = padded_z

        bs, nc, h, w = z.shape
        if ks[0] > h or ks[1] > w:
            ks = (min(ks[0], h), min(ks[1], w))
            logger.debug("reducing Kernel")

        if stride[0] > h or stride[1] > w:
            stride = (min(stride[0], h), min(stride[1], w))
            logger.debug("reducing stride")

        fold, unfold, normalization, weighting = self.get_fold_unfold(
            z, ks, stride, uf=uf
        )

        z = unfold(z)  # (bn, nc * prod(**ks), L)
        # 1. Reshape to img shape
        z = z.view(
            (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
        )  # (bn, nc, ks[0], ks[1], L )

        # 2. apply model loop over last dim

        output_list = [
            self.decode_all_at_once(z[:, :, :, :, i]) for i in range(z.shape[-1])
        ]

        o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
        o = o * weighting
        # Reverse 1. reshape to img shape
        o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
        # stitch crops together
        decoded = fold(o)
        decoded = decoded / normalization  # norm is shape (1, 1, h, w)
        # trim off padding
        decoded = decoded[:, :, : orig_shape[2] * 8, : orig_shape[3] * 8]
        return decoded

    def forward(self, input, sample_posterior=True):  # noqa
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss
        return None

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(  # noqa
                batch, batch_idx, postfix="_ema"
            )
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
        )
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = {}
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(
                        torch.randn_like(posterior_ema.sample())
                    )
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(
            torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1
        )[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting,
                self.split_input_params["clip_min_tie_weight"],
                self.split_input_params["clip_max_tie_weight"],
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape  # noqa

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = {
                "kernel_size": kernel_size,
                "dilation": 1,
                "padding": 0,
                "stride": stride,
            }
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = {
                "kernel_size": kernel_size,
                "dilation": 1,
                "padding": 0,
                "stride": stride,
            }
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = {
                "kernel_size": (kernel_size[0] * uf, kernel_size[0] * uf),
                "dilation": 1,
                "padding": 0,
                "stride": (stride[0] * uf, stride[1] * uf),
            }
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            Ly = (h - (kernel_size[0] * df)) // (stride[0] * df) + 1
            Lx = (w - (kernel_size[1] * df)) // (stride[1] * df) + 1

            unfold_params = {
                "kernel_size": (kernel_size[0] * df, kernel_size[1] * df),
                "dilation": 1,
                "padding": 0,
                "stride": (stride[0] * df, stride[1] * df),
            }

            unfold = torch.nn.Unfold(**unfold_params)

            fold_params = {
                "kernel_size": kernel_size,
                "dilation": 1,
                "padding": 0,
                "stride": stride,
            }
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params
            )

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    # def to_rgb(self, x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    #     return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


def chunk_latent(tensor, chunk_size=64, overlap_size=8):
    # Get the shape of the tensor
    batch_size, num_channels, height, width = tensor.shape

    # Calculate the number of chunks along each dimension
    num_rows = int(math.ceil(height / chunk_size))
    num_cols = int(math.ceil(width / chunk_size))

    # Initialize a list to store the chunks
    chunks = []

    # Loop over the rows and columns
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the start and end indices for the chunk along each dimension
            row_start = max(row * chunk_size - overlap_size, 0)
            row_end = min(row_start + chunk_size + overlap_size, height)
            col_start = max(col * chunk_size - overlap_size, 0)
            col_end = min(col_start + chunk_size + overlap_size, width)

            # Extract the chunk from the tensor and append it to the list of chunks
            chunk = tensor[:, :, row_start:row_end, col_start:col_end]
            chunks.append((chunk, row_start, col_start))

    return chunks, num_rows, num_cols


def merge_tensors(tensor_list, num_rows, num_cols):
    print(f"num_rows: {num_rows}")
    print(f"num_cols: {num_cols}")
    n, channel, h, w = tensor_list[0].size()
    assert n == 1
    final_width = 0
    final_height = 0
    for col_idx in range(num_cols):
        final_width += tensor_list[col_idx].size()[3]

    for row_idx in range(num_rows):
        final_height += tensor_list[row_idx * num_cols].size()[2]

    final_tensor = torch.zeros([1, channel, final_height, final_width])
    print(f"final size {final_tensor.size()}")
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            list_idx = row_idx * num_cols + col_idx
            chunk = tensor_list[list_idx]
            print(f"chunk size: {chunk.size()}")
            _, _, chunk_h, chunk_w = chunk.size()
            final_tensor[
                :,
                :,
                row_idx * h : row_idx * h + chunk_h,
                col_idx * w : col_idx * w + chunk_w,
            ] = chunk

    return final_tensor