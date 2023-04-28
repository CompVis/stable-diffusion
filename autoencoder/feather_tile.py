# Thanks to Bryce, and his work on ImaginAIry https://github.com/brycedrennan/imaginAIry

# inspired by https://github.com/ProGamerGov/neural-dream/blob/master/neural_dream/dream_tile.py
# but with all the bugs fixed and lots of simplifications
# MIT License
import math

import torch


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
    (
        column,
        row,
    ) = (
        0,
        0,
    )

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