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
A utility for showing and hiding the terminal cursor on Windows and Linux, based on https://github.com/bchao1/bullet
"""

import os
import sys
from contextlib import contextmanager


# Windows only
if os.name == "nt":
    import ctypes
    import msvcrt  # noqa

    class CursorInfo(ctypes.Structure):
        # _fields is a specific attr expected by ctypes
        _fields_ = [("size", ctypes.c_int), ("visible", ctypes.c_byte)]


def hide_cursor():
    if os.name == "nt":
        ci = CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = False
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == "posix":
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor():
    if os.name == "nt":
        ci = CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = True
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == "posix":
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


@contextmanager
def hide():
    "Context manager to hide the terminal cursor"
    try:
        hide_cursor()
        yield
    finally:
        show_cursor()
