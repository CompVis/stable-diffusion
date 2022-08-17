#!/usr/bin/env python

# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.

# this will preload the Bert tokenizer fles
print("preloading bert tokenizer...",end='')
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
print("...success")

# this will download requirements for Kornia
print("preloading Kornia requirements...",end='')
import kornia
print("...success")

