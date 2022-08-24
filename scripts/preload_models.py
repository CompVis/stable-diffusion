#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
import sys
import transformers

transformers.logging.set_verbosity_error()

# this will preload the Bert tokenizer fles
print("preloading bert tokenizer...")
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
print("...success")

# this will download requirements for Kornia
print("preloading Kornia requirements (ignore the warnings)...")
import kornia
print("...success")

# doesn't work - probably wrong logger
# logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
version='openai/clip-vit-large-patch14'

print('preloading CLIP model (Ignore the warnings)...')
sys.stdout.flush()
import clip
from transformers import CLIPTokenizer, CLIPTextModel
tokenizer  =CLIPTokenizer.from_pretrained(version)
transformer=CLIPTextModel.from_pretrained(version)
print('\n\n...success')


