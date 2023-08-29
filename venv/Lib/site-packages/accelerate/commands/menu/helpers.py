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
A variety of helper functions and constants when dealing with terminal menu choices, based on
https://github.com/bchao1/bullet
"""

import enum
import shutil
import sys


TERMINAL_WIDTH, _ = shutil.get_terminal_size()

CURSOR_TO_CHAR = {"UP": "A", "DOWN": "B", "RIGHT": "C", "LEFT": "D"}


class Direction(enum.Enum):
    UP = 0
    DOWN = 1


def forceWrite(content, end=""):
    sys.stdout.write(str(content) + end)
    sys.stdout.flush()


def writeColor(content, color, end=""):
    forceWrite(f"\u001b[{color}m{content}\u001b[0m", end)


def reset_cursor():
    forceWrite("\r")


def move_cursor(num_lines: int, direction: str):
    forceWrite(f"\033[{num_lines}{CURSOR_TO_CHAR[direction.upper()]}")


def clear_line():
    forceWrite(" " * TERMINAL_WIDTH)
    reset_cursor()


def linebreak():
    reset_cursor()
    forceWrite("-" * TERMINAL_WIDTH)
