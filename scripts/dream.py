#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import warnings
import invoke

if __name__ == '__main__':
    warnings.warn("dream.py is deprecated, please run invoke.py instead",
                  DeprecationWarning)
    invoke.main()
