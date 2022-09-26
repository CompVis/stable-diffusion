import random
import os
import torch
import numpy as np
from torch import autocast
from diffusers import StableDiffusionPipeline
import webbrowser
from deep_translator import GoogleTranslator


model_id = "CompVis/stable-diffusion-v1-4"
#device = "cuda"
device = "mps" #torch.device("mps")

white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)

os.environ["skl"] = "nn"
os.environ["epsilon"] = "0.005"
os.environ["decay"] = "0."
os.environ["ngoptim"] = "DiscreteLenglerOnePlusOne"
os.environ["forcedlatent"] = ""
os.environ["enforcedlatent"] = ""
os.environ["good"] = "[]"
os.environ["bad"] = "[]"
num_iterations = 50
gs = 7.5

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
prompt = "A red-haired woman with red hair. Her head is tilted."
prompt = "A bloody heavy-metal zombie with a chainsaw."
prompt = "Tentacles attacking a bloody Meg Myers in Eyptian dress. Meg Myers has a chainsaw."
prompt = "Bizarre art."
print(f"The prompt is {prompt}")
user_prompt = input("Enter a new prompt if you prefer\n")
if len(user_prompt) > 2:
    prompt = user_prompt

# On the fly translation.
english_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)


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
onlyfiles = []

# activate the pygame library .
pygame.init()
X = 2000  # > 1500 = buttons
Y = 900  
scrn = pygame.display.set_mode((1700, Y))
font = pygame.font.Font('freesansbold.ttf', 22)


for iteration in range(30):
    latent = [latent[f] for f in five_best]
    images = [images[f] for f in five_best]
    onlyfiles = [onlyfiles[f] for f in five_best]
    for k in range(llambda):
        if k < len(five_best):
            imp = pygame.transform.scale(pygame.image.load(onlyfiles[k]).convert(), (300, 300))
            # Using blit to copy content from one surface to other
            scrn.blit(imp, (300 * (k // 3), 300 * (k % 3)))
            pygame.display.flip()
            continue
        text0 = font.render(f'Please wait !!! {k} / {llambda}', True, green, blue)
        scrn.blit(text0, ((X*3/4)/2 - X/32, Y/2))
        pygame.display.flip()
        os.environ["earlystop"] = "False" if k > len(five_best) else "True"
        os.environ["epsilon"] = str(0. if k == len(five_best) else (k - len(five_best)) / llambda)
        os.environ["budget"] = str(300 if k > len(five_best) else 2)
        os.environ["skl"] = {0: "nn", 1: "tree", 2: "logit"}[k % 3]
        if iteration > 0:
            os.environ["forcedlatent"] = str(list(forcedlatents[k].flatten()))            
        enforcedlatent = os.environ.get("enforcedlatent", "")
        if len(enforcedlatent) > 2:
            os.environ["forcedlatent"] = enforcedlatent
        with autocast("cuda"):
            image = pipe(english_prompt, guidance_scale=gs, num_inference_steps=num_iterations)["sample"][0]
            images += [image]
        filename = f"SD_{prompt.replace(' ','_')}_image_{sentinel}_{iteration}_{k}.png"  
        image.save(filename)
        onlyfiles += [filename]
        imp = pygame.transform.scale(pygame.image.load(onlyfiles[-1]).convert(), (300, 300))
        # Using blit to copy content from one surface to other
        scrn.blit(imp, (300 * (k // 3), 300 * (k % 3)))
        pygame.display.flip()
        str_latent = eval((os.environ["latent_sd"]))
        array_latent = eval(f"np.array(str_latent).reshape(4, 64, 64)")
        print(f"Debug info: array_latent sumsq/var {sum(array_latent.flatten() ** 2) / len(array_latent.flatten())}")
        latent += [array_latent]
        with open(f"SD_{prompt.replace(' ','_')}_latent_{sentinel}_{k}.txt", 'w') as f:
            f.write(f"{latent}")
    
    # Stop the forcing from disk!
    os.environ["enforcedlatent"] = ""
    # importing required library
    
    #mypath = "./"
    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #onlyfiles = [str(f) for f in onlyfiles if "SD_" in str(f) and ".png" in str(f) and str(f) not in all_files and sentinel in str(f)]
    #print()
     
    # create the display surface object
    # of specific dimension..e(X, Y).

    # Button for loading a starting point
    text1 = font.render('Load image    ', True, green, blue)
    text1 = pygame.transform.rotate(text1, 90)
    scrn.blit(text1, (X*3/4+X/16 - X/32, 0))
    text1 = font.render('& latent    ', True, green, blue)
    text1 = pygame.transform.rotate(text1, 90)
    scrn.blit(text1, (X*3/4+X/16+X/32 - X/32, 0))
    # Button for creating a meme
    text2 = font.render('Create', True, green, blue)
    text2 = pygame.transform.rotate(text2, 90)
    scrn.blit(text2, (X*3/4+X/16 - X/32, Y/3))
    text2 = font.render('a meme', True, green, blue)
    text2 = pygame.transform.rotate(text2, 90)
    scrn.blit(text2, (X*3/4+X/16+X/32 - X/32, Y/3))
    # Button for new generation
    text3 = font.render(f"I don't want to", True, green, blue)
    text3 = pygame.transform.rotate(text3, 90)
    scrn.blit(text3, (X*3/4+X/16 - X/32, Y*2/3))
    text3 = font.render(f"select images! Just rerun.", True, green, blue)
    text3 = pygame.transform.rotate(text3, 90)
    scrn.blit(text3, (X*3/4+X/16+X/32 - X/32, Y*2/3))
    text4 = font.render(f"Modify parameters !", True, green, blue)
    scrn.blit(text4, (300, Y + 30))
    pygame.display.flip()

    for idx in range(llambda):
        # set the pygame window name
        pygame.display.set_caption('images')
        imp = pygame.transform.scale(pygame.image.load(onlyfiles[idx]).convert(), (300, 300))
        scrn.blit(imp, (300 * (idx // 3), 300 * (idx % 3)))
     
    # paint screen one time
    pygame.display.flip()
    status = True
    indices = []
    good = []
    five_best = []
    for i in pygame.event.get():
        if i.type == pygame.MOUSEBUTTONUP:
            print("too early for clicking !!!!")
    while (status):
     
      # iterate over the list of Event objects
      # that was returned by pygame.event.get() method.
        for i in pygame.event.get():
            if i.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos() 
                print(f"Click at {pos}")
                if pos[1] > Y:
                    text4 = font.render(f"ok, go to shell !", True, green, blue)
                    scrn.blit(text4, (300, Y + 30))
                    pygame.display.flip()
                    num_iterations = int(input(f"Number of iterations ? (current = {num_iterations})\n"))
                    gs = float(input(f"Guidance scale ? (current = {gs})\n"))
                    text4 = font.render(f"Ok! parameters changed!", True, green, blue)
                    scrn.blit(text4, (300, Y + 30))
                    pygame.display.flip()
                elif pos[0] > 1500:  # Not in the images.
                    if pos[1] < Y/3:
                        filename = input("Filename (please provide the latent file, of the format SD*latent*.txt) ?\n")
                        status = False
                        with open(filename, 'r') as f:
                             latent = f.read()
                        break
                    if pos[1] < 2*Y/3:
                        url = 'https://imgflip.com/memegenerator'
                        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
                        onlyfiles = [str(f) for f in onlyfiles if "SD_" in str(f) and ".png" in str(f) and str(f) not in all_files and sentinel in str(f)]
                        print("Your generated images:")
                        print(onlyfiles)
                        webbrowser.open(url)
                        exit()
                    status = False
                    break
                index = 3 * (pos[0] // 300) + (pos[1] // 300)
                if index not in five_best and len(five_best) < 5:
                    five_best += [index]
                indices += [[index, (pos[0] - (pos[0] // 300) * 300) / 300, (pos[1] - (pos[1] // 300) * 300) / 300]]
                # Update the button for new generation.
                text3 = font.render(f"  I have chosen {len(indices)} images:", True, green, blue)
                text3 = pygame.transform.rotate(text3, 90)
                scrn.blit(text3, (X*3/4+X/16 - X/32, Y*2/3))
                text3 = font.render(f"        New generation!", True, green, blue)
                text3 = pygame.transform.rotate(text3, 90)
                scrn.blit(text3, (X*3/4+X/16+X/32 - X/32, Y*2/3))
                pygame.display.flip()
                #text3Rect = text3.get_rect()
                #text3Rect.center = (750+750*3/4, 1000)
                good += [list(latent[index].flatten())]
    
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if i.type == pygame.QUIT:
                status = False
     
    # Using draw.rect module of
    # pygame to draw the solid circle
    for _ in range(123):
        x = np.random.randint(1500)
        y = np.random.randint(900)
        pygame.draw.circle(scrn, (0, 255, 0),
                           [x, y], 17, 0)
     
    # Draws the surface object to the screen.
    pygame.display.update()
    if len(indices) == 0:
        print("The user did not like anything! Rerun :-(")
        continue
    print(f"Clicks at {indices}")
    os.environ["mu"] = str(len(indices))
    forcedlatents = []
    bad += [list(latent[u].flatten()) for u in range(len(onlyfiles)) if u not in [i[0] for i in indices]]
    if len(bad) > 200:
        bad = bad[(len(bad) - 200):]
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
            
pygame.quit()
