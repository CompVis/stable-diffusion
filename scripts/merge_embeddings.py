from ldm.modules.encoders.modules import FrozenCLIPEmbedder, BERTEmbedder
from ldm.modules.embedding_manager import EmbeddingManager

import argparse, os
from functools import partial

import torch

def get_placeholder_loop(placeholder_string, embedder, use_bert):
    
    new_placeholder   = None
    
    while True:
        if new_placeholder is None:
            new_placeholder = input(f"Placeholder string {placeholder_string} was already used. Please enter a replacement string: ")
        else:
            new_placeholder = input(f"Placeholder string '{new_placeholder}' maps to more than a single token. Please enter another string: ")

        token = get_bert_token_for_string(embedder.tknz_fn, new_placeholder) if use_bert else get_clip_token_for_string(embedder.tokenizer, new_placeholder)

        if token is not None:
            return new_placeholder, token
            
def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(
        string,
        truncation=True,
        max_length=77,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt"
    )

    tokens = batch_encoding["input_ids"]

    if torch.count_nonzero(tokens - 49407) == 2:
        return tokens[0, 1]
    
    return None

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    if torch.count_nonzero(token) == 3:
        return token[0, 1]

    return None


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

    parser.add_argument(
        "-sd", "--use_bert",
        action="store_true",
        help="Flag to denote that we are not merging stable diffusion embeddings"
    )

    args = parser.parse_args()

    if args.use_bert:
        embedder = BERTEmbedder(n_embed=1280, n_layer=32).cuda()
    else:
        embedder = FrozenCLIPEmbedder().cuda()

    EmbeddingManager = partial(EmbeddingManager, embedder, ["*"])

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
                new_placeholder, new_token = get_placeholder_loop(placeholder_string, embedder, use_bert=args.use_bert)
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
