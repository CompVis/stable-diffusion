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
Main driver for the selection menu, based on https://github.com/bchao1/bullet
"""
import sys

from . import cursor, input
from .helpers import Direction, clear_line, forceWrite, linebreak, move_cursor, reset_cursor, writeColor
from .keymap import KEYMAP


@input.register
class BulletMenu:
    """
    A CLI menu to select a choice from a list of choices using the keyboard.
    """

    def __init__(self, prompt: str = None, choices: list = []):
        self.position = 0
        self.choices = choices
        self.prompt = prompt
        if sys.platform == "win32":
            self.arrow_char = "*"
        else:
            self.arrow_char = "➔ "

    def write_choice(self, index, end: str = ""):
        if sys.platform != "win32":
            writeColor(self.choices[index], 32, end)
        else:
            forceWrite(self.choices[index], end)

    def print_choice(self, index: int):
        "Prints the choice at the given index"
        if index == self.position:
            forceWrite(f" {self.arrow_char} ")
            self.write_choice(index)
        else:
            forceWrite(f"    {self.choices[index]}")
        reset_cursor()

    def move_direction(self, direction: Direction, num_spaces: int = 1):
        "Should not be directly called, used to move a direction of either up or down"
        old_position = self.position
        if direction == Direction.DOWN:
            if self.position + 1 >= len(self.choices):
                return
            self.position += num_spaces
        else:
            if self.position - 1 < 0:
                return
            self.position -= num_spaces
        clear_line()
        self.print_choice(old_position)
        move_cursor(num_spaces, direction.name)
        self.print_choice(self.position)

    @input.mark(KEYMAP["up"])
    def move_up(self):
        self.move_direction(Direction.UP)

    @input.mark(KEYMAP["down"])
    def move_down(self):
        self.move_direction(Direction.DOWN)

    @input.mark(KEYMAP["newline"])
    def select(self):
        move_cursor(len(self.choices) - self.position, "DOWN")
        return self.position

    @input.mark(KEYMAP["interrupt"])
    def interrupt(self):
        move_cursor(len(self.choices) - self.position, "DOWN")
        raise KeyboardInterrupt

    @input.mark_multiple(*[KEYMAP[str(number)] for number in range(10)])
    def select_row(self):
        index = int(chr(self.current_selection))
        movement = index - self.position
        if index == self.position:
            return
        if index < len(self.choices):
            if self.position > index:
                self.move_direction(Direction.UP, -movement)
            elif self.position < index:
                self.move_direction(Direction.DOWN, movement)
            else:
                return
        else:
            return

    def run(self, default_choice: int = 0):
        "Start the menu and return the selected choice"
        if self.prompt:
            linebreak()
            forceWrite(self.prompt, "\n")
            forceWrite("Please select a choice using the arrow or number keys, and selecting with enter", "\n")
        self.position = default_choice
        for i in range(len(self.choices)):
            self.print_choice(i)
            forceWrite("\n")
        move_cursor(len(self.choices) - self.position, "UP")
        with cursor.hide():
            while True:
                choice = self.handle_input()
                if choice is not None:
                    reset_cursor()
                    for _ in range(len(self.choices) + 1):
                        move_cursor(1, "UP")
                        clear_line()
                    self.write_choice(choice, "\n")
                    return choice
