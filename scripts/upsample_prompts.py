from pprint import pprint
from typing import Dict, List, Tuple

from transformers import pipeline, set_seed

def load_chat_pipeline():
    pipe = pipeline(
        "text-generation",
        model="TheBloke/zephyr-7B-beta-GPTQ",
        device_map = "auto",
        use_fast=True
    )
    return pipe


def get_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    # Base system message
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

    There are a few rules to follow:

    - You will only ever output a single image description per user request.
    - Do not respond with more than one image description per request.
    - Do not use any colors in your image descriptions.
    - The description must be relevant to the user request.
    - Image descriptions must be between 15-60 words. Extra words will be ignored.
    """,
    }
    rest_of_the_message = [
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'a raven and mountains'",
        },
        {
            "role": "assistant",
            "content": "a raven perched on a branch with dramatic mountains in the background, clouds, mountains, beautiful landscape",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'woman, asian, city street, trending on artstation'",
        },
        {
            "role": "assistant",
            "content": "an asian woman standing in a city street lined with street merchants, busy, dynamic composition",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'In the foreground is a blond lady with her two dog in a rain suit on a shiny day. beautiful background'",
        },
        {
            "role": "assistant",
            "content": "a blonde woman wearing a raincoat on a sunny day walking her dogs on a dirt path, fields and flowers, nature, beautiful landscape",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a frog sitting on a stump playing dominoes in the forest, dense trees, forest clearing",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'robot holding a gun, no background'",
        },
        {
            "role": "assistant",
            "content": "a complex robot with heavy armor plating holding a large rifle, mech, sci-fi, blank background",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{start_prompt}'",
        },
    ]
    return system_message, rest_of_the_message


def make_final_message(system_message: Dict[str, str], rest_of_the_message: List[Dict[str, str]], debug=False):
    # Combines system message and final message to prepare the LLM for inference
    final_message = [system_message]
    final_message.extend(rest_of_the_message)
    if debug:
        pprint(final_message)
    return final_message


def upsample_caption(pipeline, message, seed):
    system_message, rest_of_the_message = get_messages_for_chat()
    updated_prompt = rest_of_the_message[-1]["content"].format(start_prompt=message)
    rest_of_the_message[-1]["content"] = updated_prompt
    final_message = make_final_message(system_message, rest_of_the_message, debug=False)
    
    set_seed(seed)
    prompt = pipeline.tokenizer.apply_chat_template(
        final_message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def collect_response(assistant_output):
    # Collect response
    output = assistant_output[0]["generated_text"]
    parts = output.rsplit("<|assistant|>", 1)
    assistant_reply = parts[1].strip() if len(parts) > 1 else None
    return assistant_reply.splitlines()[0]