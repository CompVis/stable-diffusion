import os
from PIL import Image

def climage(file, alignment, *args):
    # Get console bounds with a small margin - better safe than sorry
    twidth, theight = (
        os.get_terminal_size().columns - 1,
        (os.get_terminal_size().lines - 1) * 2,
    )

    # Set up variables
    image = Image.open(file)
    image = image.convert("RGBA")
    iwidth, iheight = min(twidth, image.width), min(theight, image.height)
    line = []
    lines = []

    # Alignment stuff

    margin = 0
    if alignment == "centered":
        margin = int((twidth / 2) - (iwidth / 2))
    elif alignment == "right":
        margin = int(twidth - iwidth)
    elif alignment == "manual":
        margin = args[0]

    # Loop over the height of the image / 2 (because 2 pixels = 1 text character)
    for y2 in range(int(iheight / 2)):
        # Add default colors to the start of the line
        line = ["[white on black]" + " " * margin]
        rgbp, rgb2p = "", ""

        # Loop over width
        for x in range(iwidth):
            # Get the color for the upper and lower half of the text character
            r, g, b, a = image.getpixel((x, (y2 * 2)))
            r2, g2, b2, a2 = image.getpixel((x, (y2 * 2) + 1))

            # Convert to hex colors for Rich to use
            rgb, rgb2 = "#{:02x}{:02x}{:02x}".format(
                r, g, b
            ), "#{:02x}{:02x}{:02x}".format(r2, g2, b2)

            # Lookup table because I was bored
            colorCodes = [
                f"[{rgb2} on {rgb}]",
                f"[{rgb2} on black]",
                f"[black on {rgb}]",
                "[white on black]",
                f"[{rgb}]",
            ]
            # ~It just works~
            color = colorCodes[
                int(a < 200)
                + (int(a2 < 200) * 2)
                + (int(rgb == rgb2 and a + a2 > 400) * 4)
            ]

            # Don't change the color if the color doesn't change...
            if rgb == rgbp and rgb2 == rgb2p:
                color = ""

            # Set text characters, nothing, full block, half block. Half block + background color = 2 pixels
            if a < 200 and a2 < 200:
                line.append(color + " ")
            elif rgb == rgb2:
                line.append(color + "█")
            else:
                line.append(color + "▄")

            rgbp, rgb2p = rgb, rgb2

        # Add default colors to the end of the line
        lines.append("".join(line) + "[white on black]")
    return "\n".join(lines)