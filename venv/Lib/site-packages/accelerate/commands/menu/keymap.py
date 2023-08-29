# Copyright 2022 The HuggingFace Team and Brian Chao. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities relating to parsing raw characters from the keyboard, based on https://github.com/bchao1/bullet
"""


import os
import string
import sys


ARROW_KEY_FLAG = 1 << 8

KEYMAP = {
    "tab": ord("\t"),
    "newline": ord("\r"),
    "esc": 27,
    "up": 65 + ARROW_KEY_FLAG,
    "down": 66 + ARROW_KEY_FLAG,
    "right": 67 + ARROW_KEY_FLAG,
    "left": 68 + ARROW_KEY_FLAG,
    "mod_int": 91,
    "undefined": sys.maxsize,
    "interrupt": 3,
    "insert": 50,
    "delete": 51,
    "pg_up": 53,
    "pg_down": 54,
}

KEYMAP["arrow_begin"] = KEYMAP["up"]
KEYMAP["arrow_end"] = KEYMAP["left"]

if sys.platform == "win32":
    WIN_CH_BUFFER = []
    WIN_KEYMAP = {
        b"\xe0H": KEYMAP["up"] - ARROW_KEY_FLAG,
        b"\x00H": KEYMAP["up"] - ARROW_KEY_FLAG,
        b"\xe0P": KEYMAP["down"] - ARROW_KEY_FLAG,
        b"\x00P": KEYMAP["down"] - ARROW_KEY_FLAG,
        b"\xe0M": KEYMAP["right"] - ARROW_KEY_FLAG,
        b"\x00M": KEYMAP["right"] - ARROW_KEY_FLAG,
        b"\xe0K": KEYMAP["left"] - ARROW_KEY_FLAG,
        b"\x00K": KEYMAP["left"] - ARROW_KEY_FLAG,
    }

for i in range(10):
    KEYMAP[str(i)] = ord(str(i))


def get_raw_chars():
    "Gets raw characters from inputs"
    if os.name == "nt":
        import msvcrt

        encoding = "mbcs"
        # Flush the keyboard buffer
        while msvcrt.kbhit():
            msvcrt.getch()
        if len(WIN_CH_BUFFER) == 0:
            # Read the keystroke
            ch = msvcrt.getch()

            # If it is a prefix char, get second part
            if ch in (b"\x00", b"\xe0"):
                ch2 = ch + msvcrt.getch()
                # Translate actual Win chars to bullet char types
                try:
                    chx = chr(WIN_KEYMAP[ch2])
                    WIN_CH_BUFFER.append(chr(KEYMAP["mod_int"]))
                    WIN_CH_BUFFER.append(chx)
                    if ord(chx) in (
                        KEYMAP["insert"] - 1 << 9,
                        KEYMAP["delete"] - 1 << 9,
                        KEYMAP["pg_up"] - 1 << 9,
                        KEYMAP["pg_down"] - 1 << 9,
                    ):
                        WIN_CH_BUFFER.append(chr(126))
                    ch = chr(KEYMAP["esc"])
                except KeyError:
                    ch = ch2[1]
            else:
                ch = ch.decode(encoding)
        else:
            ch = WIN_CH_BUFFER.pop(0)
    elif os.name == "posix":
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def get_character():
    "Gets a character from the keyboard and returns the key code"
    char = get_raw_chars()
    if ord(char) in [KEYMAP["interrupt"], KEYMAP["newline"]]:
        return char

    elif ord(char) == KEYMAP["esc"]:
        combo = get_raw_chars()
        if ord(combo) == KEYMAP["mod_int"]:
            key = get_raw_chars()
            if ord(key) >= KEYMAP["arrow_begin"] - ARROW_KEY_FLAG and ord(key) <= KEYMAP["arrow_end"] - ARROW_KEY_FLAG:
                return chr(ord(key) + ARROW_KEY_FLAG)
            else:
                return KEYMAP["undefined"]
        else:
            return get_raw_chars()

    else:
        if char in string.printable:
            return char
        else:
            return KEYMAP["undefined"]
