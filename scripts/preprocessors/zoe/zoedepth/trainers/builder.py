# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

from importlib import import_module


def get_trainer(config):
    """Builds and returns a trainer based on the config.

    Args:
        config (dict): the config dict (typically constructed using utils.config.get_config)
            config.trainer (str): the name of the trainer to use. The module named "{config.trainer}_trainer" must exist in trainers root module

    Raises:
        ValueError: If the specified trainer does not exist under trainers/ folder

    Returns:
        Trainer (inherited from zoedepth.trainers.BaseTrainer): The Trainer object
    """
    assert "trainer" in config and config.trainer is not None and config.trainer != '', "Trainer not specified. Config: {0}".format(
        config)
    try:
        Trainer = getattr(import_module(
            f"zoedepth.trainers.{config.trainer}_trainer"), 'Trainer')
    except ModuleNotFoundError as e:
        raise ValueError(f"Trainer {config.trainer}_trainer not found.") from e
    return Trainer
