import random
import os
import time
import torch
import numpy as np
import shutil
import PIL
from PIL import Image
from einops import rearrange, repeat
from torch import autocast
from diffusers import StableDiffusionPipeline
import webbrowser
from deep_translator import GoogleTranslator
from langdetect import detect
from joblib import Parallel, delayed

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
model_id = "CompVis/stable-diffusion-v1-4"
#device = "cuda"
device = "mps" #torch.device("mps")

white = (255, 255, 255)
green = (0, 255, 0)
darkgreen = (0, 128, 0)
red = (255, 0, 0)
blue = (0, 0, 128)
black = (0, 0, 0)

os.environ["skl"] = "nn"
os.environ["epsilon"] = "0.005"
os.environ["decay"] = "0."
os.environ["ngoptim"] = "DiscreteLenglerOnePlusOne"
os.environ["forcedlatent"] = ""
latent_forcing = ""
#os.environ["enforcedlatent"] = ""
os.environ["good"] = "[]"
os.environ["bad"] = "[]"
num_iterations = 50
gs = 7.5
voronoi_in_images = True



import pyttsx3

noise = pyttsx3.init()
noise.setProperty("rate", 240)
noise.setProperty('voice', 'mb-us1')                                            

#voice = noise.getProperty('voices')
#for v in voice:
#    if v.name == "Kyoko":
#        noise.setProperty('voice', v.id)


all_selected = []
all_selected_latent = []
final_selection = []
forcedlatents = []
forcedgs = []



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

prompt = "Beautiful bizarre woman."
prompt = "Yann LeCun as the grim reaper: bizarre art."
prompt = "A star with flashy colors."
prompt = "Un chat en sang et en armure joue de la batterie."
prompt = "Photo of a cyberpunk Mark Zuckerberg killing Cthulhu with a light saber."
prompt = "A ferocious cyborg bear."
prompt = "Photo of Mark Zuckerberg killing Cthulhu with a light saber."
prompt = "A bear with horns and blood and big teeth."
prompt = "A photo of a bear and Yoda, good friends."
prompt = "A photo of Yoda on the left, a blue octopus on the right, an explosion in the center."
prompt = "A bird is on a hippo. They fight a black and red octopus. Jungle in the background."
prompt = "A flying white owl above 4 colored pots with fire. The owl has a hat."
prompt = "A flying white owl above 4 colored pots with fire."
prompt = "Yann LeCun rides a dragon which spits fire on a cherry on a cake."
prompt = "An armored Mark Zuckerberg fighting off a monster with bloody tentacles in the jungle with a light saber."
prompt = "Cute woman, portrait, photo, red hair, green eyes, smiling."
prompt = "Photo of Tarzan as a lawyer with a tie and an octopus on his head."
prompt = "An armored bloody Yann Lecun has a lightsabar and fights a red tentacular monster."
prompt = "Photo of a giant armored insect attacking a building. The building is broken. There are flames."
prompt = "Photo of Meg Myers, on the left, in Egyptian dress, fights Cthulhu (on the right) with a light saber. They stare at each other."
prompt = "Photo of a cute red panda."
prompt = "Photo of a cute smiling white-haired woman with pink eyes."
prompt = "A muscular Jesus with and assault rifle, a cap and and a light saber."
prompt = "A portrait of a cute smiling woman."
prompt = "A woman with black skin, red hair, egyptian dress, yellow eyes."
prompt = "Photo of a young cute black woman."
prompt = "Photo of a woman with cyborg implants."
prompt = "Photo of a man and a woman. Cats and drums and computers on shelves in the background."
prompt = "A 40yo smiling woman."
print(f"The prompt is {prompt}")


import pyfiglet
print(pyfiglet.figlet_format("Welcome in Genetic Stable Diffusion !"))
print(pyfiglet.figlet_format("First, let us choose the text :-)!"))



print(f"Francais: Proposez un nouveau texte si vous ne voulez pas dessiner << {prompt} >>.\n")
noise.say("Hey!")
noise.runAndWait()
user_prompt = "" #input(f"English: Enter a new prompt if you prefer something else than << {prompt} >>.\n")
if len(user_prompt) > 2:
    prompt = user_prompt

# On the fly translation.
language = detect(prompt)
english_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)

def to_native(stri):
    return GoogleTranslator(source='en', target=language).translate(stri)

def pretty_print(stri):
    print(pyfiglet.figlet_format(to_native(stri)))

print(f"{to_native('Working on')} {english_prompt}, a.k.a {prompt}.")

def latent_to_image(latent,gs,num_iterations):
    os.environ["forcedlatent"] = str(list(latent.flatten()))  #str(list(forcedlatents[k].flatten()))            
    with autocast("cuda"):
         image = pipe(english_prompt, guidance_scale=gs, num_inference_steps=num_iterations)["sample"][0]
    return image

import torch
from PIL import Image
from RealESRGAN import RealESRGAN

sr_device = torch.device('cpu') #device #('mps')   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esrmodel = RealESRGAN(sr_device, scale=4)
esrmodel.load_weights('weights/RealESRGAN_x4.pth', download=True)
esrmodel2 = RealESRGAN(sr_device, scale=2)
esrmodel2.load_weights('weights/RealESRGAN_x2.pth', download=True)

def singleeg(path_to_image):
    image = Image.open(path_to_image).convert('RGB')
    sr_device = device #('mps')   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Type before SR = {type(image)}")
    sr_image = esrmodel.predict(image)
    print(f"Type after SR = {type(sr_image)}")
    output_filename = path_to_image + ".SR.png"
    sr_image.save(output_filename)
    return output_filename

def singleeg2(path_to_image):
    time.sleep(0.5*np.random.rand())
    image = Image.open(path_to_image).convert('RGB')
    sr_device = device #('mps')   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Type before SR = {type(image)}")
    sr_image = esrmodel2.predict(image)
    print(f"Type after SR = {type(sr_image)}")
    output_filename = path_to_image + ".SR.png"
    sr_image.save(output_filename)
    return output_filename


def eg(list_of_files):
    pretty_print("Should I convert images below to high resolution ?")
    print(list_of_files)
    noise.say("Go to the text window!")
    noise.runAndWait()
    answer = input(" [y]es / [n]o ?")
    if "y" in answer or "Y" in answer:
        #images = Parallel(n_jobs=12)(delayed(singleeg)(image) for image in list_of_files)
        #print(to_native(f"Created the super-resolution files {images}")) 
        for path_to_image in list_of_files:
            output_filename = singleeg(path_to_image)
            print(to_native(f"Created the super-resolution file {output_filename}")) 

def stop_all(list_of_files, list_of_latent, last_list_of_latent):
    print(to_native("Your selected images and the last generation:"))
    print(list_of_files)
    eg(list_of_files)
    pretty_print("Should we create animations ?")
    answer = input(" [y]es or [n]o or [j]ust the selection on the last panel ?")
    if "y" in answer or "Y" in answer or "j" in answer or "J" in answer:
        assert len(list_of_files) == len(list_of_latent)
        if "j" in answer or "J" in answer:
            list_of_latent = last_list_of_latent
        pretty_print("Let us create animations!")
        for c in sorted([0.5, 0.25, 0.125, 0.0625, 0.05, 0.04,0.03125]):
            for idx in range(len(list_of_files)):
                images = []
                l = list_of_latent[idx].reshape(1,4,64,64)
                l = np.sqrt(len(l.flatten()) / np.sum(l**2)) * l
                l1 = l + c * np.random.randn(len(l.flatten())).reshape(1,4,64,64)
                l1 = np.sqrt(len(l1.flatten()) / np.sum(l1**2)) * l1
                l2 = l + c * np.random.randn(len(l.flatten())).reshape(1,4,64,64)
                l2 = np.sqrt(len(l2.flatten()) / np.sum(l2**2)) * l2
                num_animation_steps = 13
                index = 0
                for u in np.linspace(0., 2*3.14159 * (1-1/30), 30):
                     cc = np.cos(u)
                     ss = np.sin(u*2)
                     index += 1
                     image = latent_to_image(l + cc * (l1 - l) + ss * (l2 - l))
                     image_name = f"imgA{index}.png"
                     image.save(image_name)
                     images += [image_name]
                     
#                for u in np.linspace(0., 1., num_animation_steps):
#                    index += 1
#                    image = latent_to_image(u*l1 + (1-u)*l)
#                    image_name = f"imgA{index}.png"
#                    image.save(image_name)
#                    images += [image_name]
#                for u in np.linspace(0., 1., num_animation_steps):
#                    index += 1
#                    image = latent_to_image(u*l2 + (1-u)*l1)
#                    image_name = f"imgB{index}.png"
#                    image.save(image_name)
#                    images += [image_name]
#                for u in np.linspace(0., 1.,num_animation_steps):
#                    index += 1
#                    image = latent_to_image(u*l + (1-u)*l2)
#                    image_name = f"imgC{index}.png"
#                    image.save(image_name)
#                    images += [image_name]
                print(to_native(f"Base images created for perturbation={c} and file {list_of_files[idx]}"))
                #images = Parallel(n_jobs=8)(delayed(process)(i) for i in range(10))
                images = Parallel(n_jobs=10)(delayed(singleeg2)(image) for image in images)

                frames = [Image.open(image) for image in images]
                frame_one = frames[0]
                gif_name = list_of_files[idx] + "_" + str(c) + ".gif"
                frame_one.save(gif_name, format="GIF", append_images=frames,
                      save_all=True, duration=100, loop=0)    
                webbrowser.open(os.environ["PWD"] + "/" + gif_name)
    
    pretty_print("Should we create a meme ?")
    answer = input(" [y]es or [n]o ?")
    if "y" in answer or "Y" in answer:
        url = 'https://imgflip.com/memegenerator'
        webbrowser.open(url)
    pretty_print("Good bye!")
    exit()


import os
from os import listdir
from os.path import isfile, join
      
sentinel = str(random.randint(0,100000)) + "XX" +  str(random.randint(0,100000))

all_files = []

llambda = 15


bad = []
five_best = []
latent = []
images = []
onlyfiles = []

pretty_print("Now let us choose (if you want) an image as a start.")
#image_name = input(to_native("Name of image for starting ? (enter if no start image)"))


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(to_native(f"loaded input image of size ({w}, {h}) from {path}"))
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    #image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

model = pipe.vae

def img_to_latent(path):
    init_image = load_img(path)
    init_image = init_image.to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=1)
    forced_latent = model.encode(init_image.to(device)).latent_dist.sample()
    new_fl = forced_latent.cpu().detach().numpy().flatten()
    new_fl = np.sqrt(len(new_fl)) * new_fl / np.sqrt(np.sum(new_fl ** 2))
    return new_fl

image_name = "SelectedSD_Photo_of_a_young_cute_black_woman._image_30106XX76830_00000_00000.png"
image_name = "cyb.jpeg"
image_name = "ln.png"
def randomized_image_to_latent(image_name, scale=None, epsilon=None, c=None, f=None):
    base_init_image = load_img(image_name).to(device)
    new_base_init_image = base_init_image
    c = np.exp(np.random.randn()) if c is None else c
    f = np.exp(-3. * np.random.rand()) if f is None else f
    init_image_shape = base_init_image.cpu().numpy().shape
    init_image = c * new_base_init_image
    init_image = repeat(init_image, '1 ... -> b ...', b=1)
    forced_latent = 1. * model.encode(init_image.to(device)).latent_dist.sample()
    new_fl = forced_latent.cpu().detach().numpy().flatten()
    basic_new_fl = new_fl  #np.sqrt(len(new_fl) / sum(new_fl ** 2)) * new_fl
    basic_new_fl = f * np.sqrt(len(new_fl) / np.sum(basic_new_fl**2)) * basic_new_fl
    epsilon = 0.1 * np.exp(-3 * np.random.rand()) if epsilon is None else epsilon
    new_fl = (1. - epsilon) * basic_new_fl + epsilon * np.random.randn(1*4*64*64)
    scale = 2.8 + 3.6 * np.random.rand() if scale is None else scale
    new_fl = scale * np.sqrt(len(new_fl)) * new_fl / np.sqrt(np.sum(new_fl ** 2))
    gs = np.random.rand()*50.
    num_iterations = np.random.choice([10,20, 40, 80, 160])
    image = latent_to_image(np.asarray(new_fl), gs, num_iterations) #eval(os.environ["forcedlatent"])))
    image.save(f"ln3_rebuild_ni{num_iterations}_gs{gs}_f{f}_scale{scale}_epsilon{epsilon}_c{c}.png")
    return new_fl

for i in range(800):
    latent =np.random.randn(4*64*64)
    str_latent = str(list(latent))
    filename=f"build{i}_{prompt.replace(' ', '_')}.png"
    print(f"Creating {filename}")
    latent_to_image(latent, 7.5, num_iterations=50).save(filename)
    with open(filename + ".latent.txt", 'w') as f:
        f.write(f"{str_latent}")
