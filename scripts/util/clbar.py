import math
import os
import sys
import time

import numpy as np

from scripts.retro_diffusion import rd

def clbar(
    iterable,
    name="",
    printEnd="\r",
    position="",
    unit="it",
    disable=False,
    prefixwidth=1,
    suffixwidth=1,
    total=0,
):
    # Console manipulation stuff
    def up(lines=1):
        for _ in range(lines):
            sys.stdout.write("\x1b[1A")
            sys.stdout.flush()

    def down(lines=1):
        for _ in range(lines):
            sys.stdout.write("\n")
            sys.stdout.flush()

    # Allow the complete disabling of the progress bar
    if not disable:
        # Positions the bar correctly
        down(int(position == "last") * 2)
        up(int(position == "first") * 3)

        # Set up variables
        if total > 0:
            iterable = iterable[0:total]
        else:
            total = max(1, len(iterable))
        name = f"{name}"
        speed = f" {total}/{total} at 100.00 {unit}/s "
        prediction = f" 00:00 < 00:00 "
        prefix = max(len(name), len("100%"), prefixwidth)
        suffix = max(len(speed), len(prediction), suffixwidth)
        barwidth = os.get_terminal_size().columns - (suffix + prefix + 2)

        # Prints the progress bar
        def printProgressBar(iteration, delay):
            # Define progress bar graphic
            line1 = [
                "[#494b9b on #3b1725]▄",
                "[#c4f129 on #494b9b]▄" * int(int(barwidth * iteration // total) > 0),
                "[#ffffff on #494b9b]▄"
                * max(0, int(barwidth * iteration // total) - 2),
                "[#c4f129 on #494b9b]▄" * int(int(barwidth * iteration // total) > 1),
                "[#3b1725 on #494b9b]▄"
                * max(0, barwidth - int(barwidth * iteration // total)),
                "[#494b9b on #3b1725]▄[white on black]",
            ]
            line2 = [
                "[#3b1725 on #494b9b]▄",
                "[#494b9b on #48a971]▄" * int(int(barwidth * iteration // total) > 0),
                "[#494b9b on #c4f129]▄"
                * max(0, int(barwidth * iteration // total) - 2),
                "[#494b9b on #48a971]▄" * int(int(barwidth * iteration // total) > 1),
                "[#494b9b on #3b1725]▄"
                * max(0, barwidth - int(barwidth * iteration // total)),
                "[#3b1725 on #494b9b]▄[white on black]",
            ]

            percent = ("{0:.0f}").format(100 * (iteration / float(total)))

            # Avoid predicting speed until there's enough data
            if len(delay) >= 1:
                delay.append(time.time() - delay[-1])
                del delay[-2]

            # Fancy color stuff and formating
            if iteration == 0:
                speedColor = "[#48a971 on black]"
                measure = f"... {unit}/s"
                passed = f"00:00"
                remaining = f"??:??"
            else:
                if np.mean(delay) <= 1:
                    measure = f"{round(1/max(0.01, np.mean(delay)), 2)} {unit}/s"
                else:
                    measure = f"{round(np.mean(delay), 2)} s/{unit}"

                if np.mean(delay) <= 1:
                    speedColor = "[#c4f129 on black]"
                elif np.mean(delay) <= 10:
                    speedColor = "[#48a971 on black]"
                elif np.mean(delay) <= 30:
                    speedColor = "[#494b9b on black]"
                else:
                    speedColor = "[#ab333d on black]"

                passed = "{:02d}:{:02d}".format(
                    math.floor(sum(delay) / 60), round(sum(delay)) % 60
                )
                remaining = "{:02d}:{:02d}".format(
                    math.floor((total * np.mean(delay) - sum(delay)) / 60),
                    round(total * np.mean(delay) - sum(delay)) % 60,
                )

            speed = f" {iteration}/{total} at {measure} "
            prediction = f" {passed} < {remaining} "

            # Print single bar across two lines
            rd.logger(
                f'\r{f"{name}".center(prefix)} {"".join(line1)}{speedColor}{speed.center(suffix-1)}[white on black]'
            )
            rd.logger(
                f'[#48a971 on black]{f"{percent}%".center(prefix)}[white on black] {"".join(line2)}[#494b9b on black]{prediction.center(suffix-1)}',
                end=printEnd,
            )
            delay.append(time.time())

            return delay

        # Print at 0 progress
        delay = []
        delay = printProgressBar(0, delay)
        down(int(position == "first") * 2)
        # Update the progress bar
        for i, item in enumerate(iterable):
            yield item
            up(int(position == "first") * 2 + 1)
            delay = printProgressBar(i + 1, delay)
            down(int(position == "first") * 2)

        down(int(position != "first"))
    else:
        for i, item in enumerate(iterable):
            yield item