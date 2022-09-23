import random
import os
import torch
import numpy as np
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
#device = "cuda"
device = "mps" #torch.device("mps")

os.environ["skl"] = "nn"
os.environ["epsilon"] = "0.005"
os.environ["decay"] = "0."
os.environ["ngoptim"] = "DiscreteLenglerOnePlusOne"
os.environ["forcedlatent"] = ""
os.environ["good"] = "[]"
os.environ["bad"] = "[]"

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


prompt = "Several faces."

prompt = "An armoured Yann LeCun fighting tentacles in the jungle."
prompt = "Tentacles everywhere."
prompt = "A photo of a smiling Medusa."
prompt = "Medusa."
prompt = "Meg Myers in bloody armor fending off tentacles with a sword."

import os
import pygame
from os import listdir
from os.path import isfile, join
      
sentinel = str(random.randint(0,100000)) + "XX" +  str(random.randint(0,100000))

all_files = []

llambda = 15

assert llambda < 16, "lambda < 16 for convenience in pygame."

bad = []
five_best = []
latent = []
images = []
for iteration in range(30):
    onlyfiles = []
    latent = [latent[f] for f in five_best]
    images = [images[f] for f in five_best]
    for k in range(llambda):
        if k < len(five_best):
            continue
        os.environ["earlystop"] = "False" if k > len(five_best) else "True"
        os.environ["epsilon"] = str(0. if k == len(five_best) else (k - len(five_best)) / llambda)
        os.environ["budget"] = str(300 if k > len(five_best) else 2)
        os.environ["skl"] = {0: "nn", 1: "tree", 2: "logit"}[k % 3]
        if iteration > 0:
            os.environ["forcedlatent"] = str(list(forcedlatents[k].flatten()))            
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=7.5)["sample"][0]
            images += [image]
        filename = f"SD_{prompt.replace(' ','_')}_image_{sentinel}_{iteration}_{k}.png"  
        image.save(filename)
        onlyfiles += [filename]
        str_latent = eval((os.environ["latent_sd"]))
        array_latent = eval(f"np.array(str_latent).reshape(4, 64, 64)")
        print(f"array_latent sumsq/var {sum(array_latent.flatten() ** 2) / len(array_latent.flatten())}")
        latent += [array_latent]
        with open(f"SD_{prompt.replace(' ','_')}_latent_{sentinel}_{k}.txt", 'w') as f:
            f.write(f"{latent}")
    
    # importing required library
    
    #mypath = "./"
    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #onlyfiles = [str(f) for f in onlyfiles if "SD_" in str(f) and ".png" in str(f) and str(f) not in all_files and sentinel in str(f)]
    #print()
    # activate the pygame library .
    pygame.init()
    X = 1500
    Y = 900
     
    # create the display surface object
    # of specific dimension..e(X, Y).
    scrn = pygame.display.set_mode((X, Y))
    
    for idx in range(llambda):
        # set the pygame window name
        pygame.display.set_caption('images')
         
        # create a surface object, image is drawn on it.
        imp = pygame.transform.scale(images[idx].convert(), (300, 300))
        #imp = pygame.transform.scale(pygame.image.load(onlyfiles[idx]).convert(), (300, 300))
         
        # Using blit to copy content from one surface to other
        scrn.blit(imp, (300 * (idx // 3), 300 * (idx % 3)))
     
    # paint screen one time
    pygame.display.flip()
    status = True
    indices = []
    good = []
    five_best = []
    while (status):
     
      # iterate over the list of Event objects
      # that was returned by pygame.event.get() method.
        for i in pygame.event.get():
            if i.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos() 
                print(pos)
                index = 3 * (pos[0] // 300) + (pos[1] // 300)
                if index not in five_best and len(five_best) < 5:
                    five_best += [index]
                indices += [[index, (pos[0] - (pos[0] // 300) * 300) / 300, (pos[1] - (pos[1] // 300) * 300) / 300]]
                good += [list(latent[index].flatten())]
    
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if i.type == pygame.QUIT:
                status = False
     
    # deactivates the pygame library
    pygame.quit()
    print(indices)
    os.environ["mu"] = str(len(indices))
    forcedlatents = []
    bad += [list(latent[u].flatten()) for u in range(len(onlyfiles)) if u not in [i[0] for i in indices]]
    for a in range(llambda):
        forcedlatent = np.zeros((4, 64, 64))
        os.environ["good"] = str(good)
        os.environ["bad"] = str(bad)
        coefficients = np.zeros(len(indices))
        for i in range(len(indices)):
            coefficients[i] = np.exp(np.random.randn())
        for i in range(64):
            x = i / 63.
            for j in range(64):
                y = j / 63
                mindistances = 10000000000.
                for u in range(len(indices)):
                    distance = coefficients[u] * np.linalg.norm( np.array((x, y)) - np.array((indices[u][1], indices[u][2])) )
                    if distance < mindistances:
                        mindistances = distance
                        uu = indices[u][0]
                for k in range(4):
                    assert k < len(forcedlatent), k
                    assert i < len(forcedlatent[k]), i
                    assert j < len(forcedlatent[k][i]), j
                    assert uu < len(latent)
                    assert k < len(latent[uu]), k
                    assert i < len(latent[uu][k]), i
                    assert j < len(latent[uu][k][i]), j
                    forcedlatent[k][i][j] = latent[uu][k][i][j]
        forcedlatents += [forcedlatent]
    #for uu in range(len(latent)):
    #    print(f"--> latent[{uu}] sum of sq / variable = {np.sum(latent[uu].flatten()**2) / len(latent[uu].flatten())}")
            
