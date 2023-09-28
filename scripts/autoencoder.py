import math
from contextlib import contextmanager

import pytorch_lightning as pl
import torch

from optimization.distributions import DiagonalGaussianDistribution
from ldm.modules import Decoder, Encoder, LitEma
from ldm.util import instantiate_from_config


def mask_tile(tile, overlap, std_overlap, side="bottom"):
    b, c, h, w = tile.shape
    top_overlap, bottom_overlap, right_overlap, left_overlap = overlap
    (
        std_top_overlap,
        std_bottom_overlap,
        std_right_overlap,
        std_left_overlap,
    ) = std_overlap

    if "left" in side:
        lin_mask_left = torch.linspace(0, 1, std_left_overlap, device=tile.device)
        if left_overlap > std_left_overlap:
            zeros_mask = torch.zeros(
                left_overlap - std_left_overlap, device=tile.device
            )
            lin_mask_left = (
                torch.cat([zeros_mask, lin_mask_left], 0)
                .repeat(h, 1)
                .repeat(c, 1, 1)
                .unsqueeze(0)
            )

    if "right" in side:
        lin_mask_right = (
            torch.linspace(1, 0, right_overlap, device=tile.device)
            .repeat(h, 1)
            .repeat(c, 1, 1)
            .unsqueeze(0)
        )
    if "top" in side:
        lin_mask_top = torch.linspace(0, 1, std_top_overlap, device=tile.device)
        if top_overlap > std_top_overlap:
            zeros_mask = torch.zeros(top_overlap - std_top_overlap, device=tile.device)
            lin_mask_top = torch.cat([zeros_mask, lin_mask_top], 0)
        lin_mask_top = lin_mask_top.repeat(w, 1).rot90(3).repeat(c, 1, 1).unsqueeze(0)

    if "bottom" in side:
        lin_mask_bottom = (
            torch.linspace(1, 0, std_bottom_overlap, device=tile.device)
            .repeat(w, 1)
            .rot90(3)
            .repeat(c, 1, 1)
            .unsqueeze(0)
        )

    base_mask = torch.ones_like(tile)

    if "right" in side:
        base_mask[:, :, :, w - right_overlap :] = (
            base_mask[:, :, :, w - right_overlap :] * lin_mask_right
        )
    if "left" in side:
        base_mask[:, :, :, :left_overlap] = (
            base_mask[:, :, :, :left_overlap] * lin_mask_left
        )
    if "bottom" in side:
        base_mask[:, :, h - bottom_overlap :, :] = (
            base_mask[:, :, h - bottom_overlap :, :] * lin_mask_bottom
        )
    if "top" in side:
        base_mask[:, :, :top_overlap, :] = (
            base_mask[:, :, :top_overlap, :] * lin_mask_top
        )
    return tile * base_mask


def get_tile_coords(d, tile_dim, overlap=0):
    move = int(math.ceil(round(tile_dim * (1 - overlap), 10)))
    c, tile_start, coords = 1, 0, [0]
    while tile_start + tile_dim < d:
        tile_start = move * c
        if tile_start + tile_dim >= d:
            coords.append(d - tile_dim)
        else:
            coords.append(tile_start)
        c += 1
    return coords


def get_tiles(img, tile_coords, tile_size):
    tile_list = []
    for y in tile_coords[0]:
        for x in tile_coords[1]:
            tile = img[:, :, y : y + tile_size[0], x : x + tile_size[1]]
            tile_list.append(tile)
    return tile_list


def final_overlap(tile_coords, tile_size):
    last_row, last_col = len(tile_coords[0]) - 1, len(tile_coords[1]) - 1

    f_ovlp = [
        (tile_coords[0][last_row - 1] + tile_size[0]) - (tile_coords[0][last_row]),
        (tile_coords[1][last_col - 1] + tile_size[1]) - (tile_coords[1][last_col]),
    ]
    return f_ovlp


def add_tiles(tiles, base_img, tile_coords, tile_size, overlap):
    f_ovlp = final_overlap(tile_coords, tile_size)
    h, w = tiles[0].size(2), tiles[0].size(3)
    if f_ovlp[0] == h:
        f_ovlp[0] = 0

    if f_ovlp[1] == w:
        f_ovlp[1] = 0

    t = 0
    (column, row) = (0, 0)

    for y in tile_coords[0]:
        for x in tile_coords[1]:
            mask_sides = ""
            c_overlap = overlap.copy()
            if row == 0:
                mask_sides += "bottom"
            elif 0 < row < len(tile_coords[0]) - 2:
                mask_sides += "bottom,top"
            elif row == len(tile_coords[0]) - 2:
                mask_sides += "bottom,top"
            elif row == len(tile_coords[0]) - 1:
                mask_sides += "top"
                if f_ovlp[0] > 0:
                    c_overlap[0] = f_ovlp[0]  # Change top overlap

            if column == 0:
                mask_sides += ",right"
            elif 0 < column < len(tile_coords[1]) - 2:
                mask_sides += ",right,left"
            elif column == len(tile_coords[1]) - 2:
                mask_sides += ",right,left"
            elif column == len(tile_coords[1]) - 1:
                mask_sides += ",left"
                if f_ovlp[1] > 0:
                    c_overlap[3] = f_ovlp[1]  # Change left overlap

            # print(f"mask_tile: tile.shape={tiles[t].shape}, overlap={c_overlap}, side={mask_sides} col={column}, row={row}")

            tile = mask_tile(tiles[t], c_overlap, std_overlap=overlap, side=mask_sides)
            # torch_img_to_pillow_img(tile).show()
            base_img[:, :, y : y + tile_size[0], x : x + tile_size[1]] = (
                base_img[:, :, y : y + tile_size[0], x : x + tile_size[1]] + tile
            )
            # torch_img_to_pillow_img(base_img).show()
            t += 1
            column += 1

        row += 1
        # if row >= 2:
        #     exit()
        column = 0
    return base_img


def tile_setup(tile_size, overlap_percent, base_size):
    if not isinstance(tile_size, (tuple, list)):
        tile_size = (tile_size, tile_size)
    if not isinstance(overlap_percent, (tuple, list)):
        overlap_percent = (overlap_percent, overlap_percent)
    if min(tile_size) < 1:
        raise ValueError("tile_size must be at least 1")

    if max(overlap_percent) > 0.5:
        raise ValueError("overlap_percent must not be greater than 0.5")

    x_coords = get_tile_coords(base_size[1], tile_size[1], overlap_percent[1])
    y_coords = get_tile_coords(base_size[0], tile_size[0], overlap_percent[0])
    y_ovlp = int(math.floor(round(tile_size[0] * overlap_percent[0], 10)))
    x_ovlp = int(math.floor(round(tile_size[1] * overlap_percent[1], 10)))
    if len(x_coords) == 1:
        x_ovlp = 0
    if len(y_coords) == 1:
        y_ovlp = 0

    return (y_coords, x_coords), tile_size, [y_ovlp, y_ovlp, x_ovlp, x_ovlp]


def tile_image(img, tile_size, overlap_percent):
    tile_coords, tile_size, _ = tile_setup(
        tile_size, overlap_percent, (img.size(2), img.size(3))
    )

    return get_tiles(img, tile_coords, tile_size)


def rebuild_image(tiles, base_img, tile_size, overlap_percent):
    if len(tiles) == 1:
        return tiles[0]
    base_img = torch.zeros_like(base_img)
    tile_coords, tile_size, overlap = tile_setup(
        tile_size, overlap_percent, (base_img.size(2), base_img.size(3))
    )
    return add_tiles(tiles, base_img, tile_coords, tile_size, overlap)


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

            if stride[0] > h // df or stride[1] > w // df:
                stride = (min(stride[0], h // df), min(stride[1], w // df))
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
                    try:
                        return self.encode_sliced(z, chunk_size=32)
                    except:
                        # Out of memory, trying smaller slice.
                        return self.encode_sliced(z, chunk_size=16)

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

        if stride[0] > h or stride[1] > w:
            stride = (min(stride[0], h), min(stride[1], w))

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