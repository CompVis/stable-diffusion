import random
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
#device = "cuda"
device = "mps" #torch.device("mps")


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token="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")
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of a red panda with a hat playing table tennis"
prompt = "a photorealistic portrait of " + random.choice(["Mary Cury", "Scarlett Johansson", "Marilyn Monroe", "Poison Ivy", "Black Widow", "Medusa", "Batman", "Albert Einstein", "Louis XIV", "Tarzan"]) + random.choice([" with glasses", " with a hat", " with a cigarette", "with a scarf"])
prompt = "a photorealistic portrait of " + random.choice(["Nelson Mandela", "Superman", "Superwoman", "Volodymyr Zelenskyy", "Tsai Ing-Wen", "Lzzy Hale", "Meg Myers"]) + random.choice([" with glasses", " with a hat", " with a cigarette", "with a scarf"])
prompt = random.choice(["A woman with three eyes", "Meg Myers", "The rock band Ankor", "Miley Cyrus", "The man named Rahan", "A murder", "Rambo playing table tennis"])
prompt = "Photo of a female Terminator."
prompt = random.choice([
     "Photo of Tarzan as a lawyer with a tie",
     "Photo of Scarlett Johansson as a sumo-tori",
     "Photo of the little mermaid as a young black girl",
     "Photo of Schwarzy with tentacles",
     "Photo of Meg Myers with an Egyptian dress",
     "Photo of Schwarzy as a ballet dancer",
    ])

name = random.choice(["Mark Zuckerbeg", "Zendaya", "Yann LeCun", "Scarlett Johansson", "Superman", "Meg Myers"])
name = "Zendaya"
prompt = f"Photo of {name} as a sumo-tori."

prompt = "Full length portrait of Mark Zuckerberg as a Sumo-Tori."
prompt = "Full length portrait of Scarlett Johansson as a Sumo-Tori."
prompt = "A close up photographic portrait of a young woman with uniformly colored hair."
prompt = "Zombies raising and worshipping a flying human."
prompt = "Zombies trying to kill Meg Myers."
prompt = "Meg Myers with an Egyptian dress killing a vampire with a gun."
prompt = "Meg Myers grabbing a vampire by the scruff of the neck."
prompt = "Mark Zuckerberg chokes a vampire to death."
prompt = "Mark Zuckerberg riding an animal."
prompt = "A giant cute animal worshipped by zombies."



import os
prompt = os.environ.get("prompt", prompt)
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
      
sentinel = random.randint(0,100000)
image.save(f"SD_{prompt.replace(' ','_')}_image_{sentinel}.png")
latent = eval((os.environ["latent_sd"]))
with open(f"SD_{prompt.replace(' ','_')}_latent_{sentinel}.txt", 'w') as f:
    f.write(f"{latent}")

