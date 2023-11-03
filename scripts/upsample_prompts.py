from pprint import pprint
from typing import Dict, List, Tuple

from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

def load_chat_pipeline(modelPath):
    model = AutoModelForCausalLM.from_pretrained(modelPath, device_map="auto", trust_remote_code=False)

    tokenizer = AutoTokenizer.from_pretrained(modelPath, use_fast=True)

    #model = BetterTransformer.transform(model) 

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
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
    - Do not use words like "realistic", "photograph", "realism", "photography" or any other words indicating the style of the image.
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
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'ultra detailed photo of a detailed corgi dog in a field, detailed fur, natural grass, good composition, 8k render, hyper quality, best image, masterpiece, soft natural volumetric cinematic perfect light, trending on artstation, perfect composition'",
        },
        {
            "role": "assistant",
            "content": "a happy corgi in a grassy field with a meadow in the background, detailed, soft lighting, nature",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'a man walking with detailed background'",
        },
        {
            "role": "assistant",
            "content": "a man wearing hiking clothing walking down a forest path with trees and a stream, mushrooms, wildlife, gravel path, nature, hiking",
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
            "content": "a complex robot with heavy armor plating holding a large rifle, gears exposed, mech, sci-fi, blank background",
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

def enhance_prompts(pipeline, prompts, seed = 0):
    upsampled_captions = []
    for i, prompt in enumerate(prompts):
        # Try to generate a response, if no response is identified after retrys, set upsampled prompt to initial prompt
        upsampled_caption = None
        retrys = 5
        while upsampled_caption == None and retrys > 0:
            outputs = upsample_caption(pipeline, prompt, seed+i)
            upsampled_caption = collect_response(outputs)
            retrys -= 1

        if upsampled_caption == None:
            upsampled_caption = prompt
        
        upsampled_captions.append(upsampled_caption)

    del outputs, pipeline
    return upsampled_captions

# How to generate prompts
'''
pipeline = load_chat_pipeline(path_to_model_folder)

prompts = enhance_prompts(pipeline, ["a butterfly styled robot, no background"])
'''