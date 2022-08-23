from ldm.modules.encoders.modules import BERTTokenizer
from ldm.modules.embedding_manager import EmbeddingManager

import argparse, os
from functools import partial

import torch

def get_placeholder_loop(placeholder_string, tokenizer):
    
    new_placeholder   = None
    
    while True:
        if new_placeholder is None:
            new_placeholder = input(f"Placeholder string {placeholder_string} was already used. Please enter a replacement string: ")
        else:
            new_placeholder = input(f"Placeholder string '{new_placeholder}' maps to more than a single token. Please enter another string: ")

        token = tokenizer(new_placeholder)

        if torch.count_nonzero(token) == 3:
            return new_placeholder, token[0, 1]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manager_ckpts", 
        type=str, 
        nargs="+", 
        required=True,
        help="Paths to a set of embedding managers to be merged."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the merged manager",
    )

    args = parser.parse_args()

    tokenizer = BERTTokenizer(vq_interface=False, max_length=77)
    EmbeddingManager = partial(EmbeddingManager, tokenizer, ["*"])

    string_to_token_dict = {}    
    string_to_param_dict = torch.nn.ParameterDict()

    placeholder_to_src = {}

    for manager_ckpt in args.manager_ckpts:
        print(f"Parsing {manager_ckpt}...")

        manager = EmbeddingManager()
        manager.load(manager_ckpt)

        for placeholder_string in manager.string_to_token_dict:
            if not placeholder_string in string_to_token_dict:
                string_to_token_dict[placeholder_string] = manager.string_to_token_dict[placeholder_string]
                string_to_param_dict[placeholder_string] = manager.string_to_param_dict[placeholder_string]

                placeholder_to_src[placeholder_string] = manager_ckpt
            else:
                new_placeholder, new_token = get_placeholder_loop(placeholder_string, tokenizer)
                string_to_token_dict[new_placeholder] = new_token
                string_to_param_dict[new_placeholder] = manager.string_to_param_dict[placeholder_string]

                placeholder_to_src[new_placeholder] = manager_ckpt

    print("Saving combined manager...")
    merged_manager = EmbeddingManager()
    merged_manager.string_to_param_dict = string_to_param_dict
    merged_manager.string_to_token_dict = string_to_token_dict
    merged_manager.save(args.output_path)

    print("Managers merged. Final list of placeholders: ")
    print(placeholder_to_src)



                
