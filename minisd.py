assert False, "Deprecated! Use geneticsd.py instead."
###############   DEPRECATED: see geneticsd.py      import random
###############   DEPRECATED: see geneticsd.py      import os
###############   DEPRECATED: see geneticsd.py      import time
###############   DEPRECATED: see geneticsd.py      import torch
###############   DEPRECATED: see geneticsd.py      import numpy as np
###############   DEPRECATED: see geneticsd.py      import shutil
###############   DEPRECATED: see geneticsd.py      import PIL
###############   DEPRECATED: see geneticsd.py      from PIL import Image
###############   DEPRECATED: see geneticsd.py      from einops import rearrange, repeat
###############   DEPRECATED: see geneticsd.py      from torch import autocast
###############   DEPRECATED: see geneticsd.py      from diffusers import StableDiffusionPipeline
###############   DEPRECATED: see geneticsd.py      import webbrowser
###############   DEPRECATED: see geneticsd.py      from deep_translator import GoogleTranslator
###############   DEPRECATED: see geneticsd.py      from langdetect import detect
###############   DEPRECATED: see geneticsd.py      from joblib import Parallel, delayed
###############   DEPRECATED: see geneticsd.py      import torch
###############   DEPRECATED: see geneticsd.py      from PIL import Image
###############   DEPRECATED: see geneticsd.py      from RealESRGAN import RealESRGAN
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
###############   DEPRECATED: see geneticsd.py      model_id = "CompVis/stable-diffusion-v1-4"
###############   DEPRECATED: see geneticsd.py      #device = "cuda"
###############   DEPRECATED: see geneticsd.py      device = "mps" #torch.device("mps")
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      white = (255, 255, 255)
###############   DEPRECATED: see geneticsd.py      green = (0, 255, 0)
###############   DEPRECATED: see geneticsd.py      darkgreen = (0, 128, 0)
###############   DEPRECATED: see geneticsd.py      red = (255, 0, 0)
###############   DEPRECATED: see geneticsd.py      blue = (0, 0, 128)
###############   DEPRECATED: see geneticsd.py      black = (0, 0, 0)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      os.environ["skl"] = "nn"
###############   DEPRECATED: see geneticsd.py      os.environ["epsilon"] = "0.005"
###############   DEPRECATED: see geneticsd.py      os.environ["decay"] = "0."
###############   DEPRECATED: see geneticsd.py      os.environ["ngoptim"] = "DiscreteLenglerOnePlusOne"
###############   DEPRECATED: see geneticsd.py      os.environ["forcedlatent"] = ""
###############   DEPRECATED: see geneticsd.py      latent_forcing = ""
###############   DEPRECATED: see geneticsd.py      #os.environ["enforcedlatent"] = ""
###############   DEPRECATED: see geneticsd.py      os.environ["good"] = "[]"
###############   DEPRECATED: see geneticsd.py      os.environ["bad"] = "[]"
###############   DEPRECATED: see geneticsd.py      num_iterations = 50
###############   DEPRECATED: see geneticsd.py      gs = 7.5
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      import pyttsx3
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      noise = pyttsx3.init()
###############   DEPRECATED: see geneticsd.py      noise.setProperty("rate", 240)
###############   DEPRECATED: see geneticsd.py      noise.setProperty('voice', 'mb-us1')                                            
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      #voice = noise.getProperty('voices')
###############   DEPRECATED: see geneticsd.py      #for v in voice:
###############   DEPRECATED: see geneticsd.py      #    if v.name == "Kyoko":
###############   DEPRECATED: see geneticsd.py      #        noise.setProperty('voice', v.id)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      all_selected = []
###############   DEPRECATED: see geneticsd.py      all_selected_latent = []
###############   DEPRECATED: see geneticsd.py      final_selection = []
###############   DEPRECATED: see geneticsd.py      final_selection_latent = []
###############   DEPRECATED: see geneticsd.py      forcedlatents = []
###############   DEPRECATED: see geneticsd.py      forcedgs = []
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token="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")
###############   DEPRECATED: see geneticsd.py      pipe = pipe.to(device)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      prompt = "a photo of an astronaut riding a horse on mars"
###############   DEPRECATED: see geneticsd.py      prompt = "a photo of a red panda with a hat playing table tennis"
###############   DEPRECATED: see geneticsd.py      prompt = "a photorealistic portrait of " + random.choice(["Mary Cury", "Scarlett Johansson", "Marilyn Monroe", "Poison Ivy", "Black Widow", "Medusa", "Batman", "Albert Einstein", "Louis XIV", "Tarzan"]) + random.choice([" with glasses", " with a hat", " with a cigarette", "with a scarf"])
###############   DEPRECATED: see geneticsd.py      prompt = "a photorealistic portrait of " + random.choice(["Nelson Mandela", "Superman", "Superwoman", "Volodymyr Zelenskyy", "Tsai Ing-Wen", "Lzzy Hale", "Meg Myers"]) + random.choice([" with glasses", " with a hat", " with a cigarette", "with a scarf"])
###############   DEPRECATED: see geneticsd.py      prompt = random.choice(["A woman with three eyes", "Meg Myers", "The rock band Ankor", "Miley Cyrus", "The man named Rahan", "A murder", "Rambo playing table tennis"])
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of a female Terminator."
###############   DEPRECATED: see geneticsd.py      prompt = random.choice([
###############   DEPRECATED: see geneticsd.py           "Photo of Tarzan as a lawyer with a tie",
###############   DEPRECATED: see geneticsd.py           "Photo of Scarlett Johansson as a sumo-tori",
###############   DEPRECATED: see geneticsd.py           "Photo of the little mermaid as a young black girl",
###############   DEPRECATED: see geneticsd.py           "Photo of Schwarzy with tentacles",
###############   DEPRECATED: see geneticsd.py           "Photo of Meg Myers with an Egyptian dress",
###############   DEPRECATED: see geneticsd.py           "Photo of Schwarzy as a ballet dancer",
###############   DEPRECATED: see geneticsd.py          ])
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      name = random.choice(["Mark Zuckerbeg", "Zendaya", "Yann LeCun", "Scarlett Johansson", "Superman", "Meg Myers"])
###############   DEPRECATED: see geneticsd.py      name = "Zendaya"
###############   DEPRECATED: see geneticsd.py      prompt = f"Photo of {name} as a sumo-tori."
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      prompt = "Full length portrait of Mark Zuckerberg as a Sumo-Tori."
###############   DEPRECATED: see geneticsd.py      prompt = "Full length portrait of Scarlett Johansson as a Sumo-Tori."
###############   DEPRECATED: see geneticsd.py      prompt = "A close up photographic portrait of a young woman with uniformly colored hair."
###############   DEPRECATED: see geneticsd.py      prompt = "Zombies raising and worshipping a flying human."
###############   DEPRECATED: see geneticsd.py      prompt = "Zombies trying to kill Meg Myers."
###############   DEPRECATED: see geneticsd.py      prompt = "Meg Myers with an Egyptian dress killing a vampire with a gun."
###############   DEPRECATED: see geneticsd.py      prompt = "Meg Myers grabbing a vampire by the scruff of the neck."
###############   DEPRECATED: see geneticsd.py      prompt = "Mark Zuckerberg chokes a vampire to death."
###############   DEPRECATED: see geneticsd.py      prompt = "Mark Zuckerberg riding an animal."
###############   DEPRECATED: see geneticsd.py      prompt = "A giant cute animal worshipped by zombies."
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      prompt = "Several faces."
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      prompt = "An armoured Yann LeCun fighting tentacles in the jungle."
###############   DEPRECATED: see geneticsd.py      prompt = "Tentacles everywhere."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of a smiling Medusa."
###############   DEPRECATED: see geneticsd.py      prompt = "Medusa."
###############   DEPRECATED: see geneticsd.py      prompt = "Meg Myers in bloody armor fending off tentacles with a sword."
###############   DEPRECATED: see geneticsd.py      prompt = "A red-haired woman with red hair. Her head is tilted."
###############   DEPRECATED: see geneticsd.py      prompt = "A bloody heavy-metal zombie with a chainsaw."
###############   DEPRECATED: see geneticsd.py      prompt = "Tentacles attacking a bloody Meg Myers in Eyptian dress. Meg Myers has a chainsaw."
###############   DEPRECATED: see geneticsd.py      prompt = "Bizarre art."
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      prompt = "Beautiful bizarre woman."
###############   DEPRECATED: see geneticsd.py      prompt = "Yann LeCun as the grim reaper: bizarre art."
###############   DEPRECATED: see geneticsd.py      prompt = "Un chat en sang et en armure joue de la batterie."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of a cyberpunk Mark Zuckerberg killing Cthulhu with a light saber."
###############   DEPRECATED: see geneticsd.py      prompt = "A ferocious cyborg bear."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of Mark Zuckerberg killing Cthulhu with a light saber."
###############   DEPRECATED: see geneticsd.py      prompt = "A bear with horns and blood and big teeth."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of a bear and Yoda, good friends."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of Yoda on the left, a blue octopus on the right, an explosion in the center."
###############   DEPRECATED: see geneticsd.py      prompt = "A bird is on a hippo. They fight a black and red octopus. Jungle in the background."
###############   DEPRECATED: see geneticsd.py      prompt = "A flying white owl above 4 colored pots with fire. The owl has a hat."
###############   DEPRECATED: see geneticsd.py      prompt = "A flying white owl above 4 colored pots with fire."
###############   DEPRECATED: see geneticsd.py      prompt = "Yann LeCun rides a dragon which spits fire on a cherry on a cake."
###############   DEPRECATED: see geneticsd.py      prompt = "An armored Mark Zuckerberg fighting off a monster with bloody tentacles in the jungle with a light saber."
###############   DEPRECATED: see geneticsd.py      prompt = "Cute woman, portrait, photo, red hair, green eyes, smiling."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of Tarzan as a lawyer with a tie and an octopus on his head."
###############   DEPRECATED: see geneticsd.py      prompt = "An armored bloody Yann Lecun has a lightsabar and fights a red tentacular monster."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of a giant armored insect attacking a building. The building is broken. There are flames."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of Meg Myers, on the left, in Egyptian dress, fights Cthulhu (on the right) with a light saber. They stare at each other."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of a cute red panda."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of a cute smiling white-haired woman with pink eyes."
###############   DEPRECATED: see geneticsd.py      prompt = "A muscular Jesus with and assault rifle, a cap and and a light saber."
###############   DEPRECATED: see geneticsd.py      prompt = "A portrait of a cute smiling woman."
###############   DEPRECATED: see geneticsd.py      prompt = "A woman with black skin, red hair, egyptian dress, yellow eyes."
###############   DEPRECATED: see geneticsd.py      prompt = "Photo of a red haired man with tilted head."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of Cleopatra with Egyptian Dress kissing Yoda."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of Yoda fighting Meg Myers with light sabers."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of Meg Myers, laughing, pulling Gandalf's hair."
###############   DEPRECATED: see geneticsd.py      prompt = "A photo of Meg Myers laughing and pulling Gandalf's hair. Gandalf is stooping."
###############   DEPRECATED: see geneticsd.py      prompt = "A star with flashy colors."
###############   DEPRECATED: see geneticsd.py      prompt = "Portrait of a green haired woman with blue eyes."
###############   DEPRECATED: see geneticsd.py      prompt = "Portrait of a female kung-fu master."
###############   DEPRECATED: see geneticsd.py      prompt = "In a dark cave, in the middle of computers, a geek meets the devil."
###############   DEPRECATED: see geneticsd.py      print(f"The prompt is {prompt}")
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      import pyfiglet
###############   DEPRECATED: see geneticsd.py      print(pyfiglet.figlet_format("Welcome in Genetic Stable Diffusion !"))
###############   DEPRECATED: see geneticsd.py      print(pyfiglet.figlet_format("First, let us choose the text :-)!"))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      print(f"Francais: Proposez un nouveau texte si vous ne voulez pas dessiner << {prompt} >>.\n")
###############   DEPRECATED: see geneticsd.py      noise.say("Hey!")
###############   DEPRECATED: see geneticsd.py      noise.runAndWait()
###############   DEPRECATED: see geneticsd.py      user_prompt = input(f"English: Enter a new prompt if you prefer something else than << {prompt} >>.\n")
###############   DEPRECATED: see geneticsd.py      if len(user_prompt) > 2:
###############   DEPRECATED: see geneticsd.py          prompt = user_prompt
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      # On the fly translation.
###############   DEPRECATED: see geneticsd.py      language = detect(prompt)
###############   DEPRECATED: see geneticsd.py      english_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def to_native(stri):
###############   DEPRECATED: see geneticsd.py          return GoogleTranslator(source='en', target=language).translate(stri)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def pretty_print(stri):
###############   DEPRECATED: see geneticsd.py          print(pyfiglet.figlet_format(to_native(stri)))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      print(f"{to_native('Working on')} {english_prompt}, a.k.a {prompt}.")
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def latent_to_image(latent):
###############   DEPRECATED: see geneticsd.py          os.environ["forcedlatent"] = str(list(latent.flatten()))  #str(list(forcedlatents[k].flatten()))            
###############   DEPRECATED: see geneticsd.py          with autocast("cuda"):
###############   DEPRECATED: see geneticsd.py               image = pipe(english_prompt, guidance_scale=gs, num_inference_steps=num_iterations)["sample"][0]
###############   DEPRECATED: see geneticsd.py          os.environ["forcedlatent"] = "[]"
###############   DEPRECATED: see geneticsd.py          return image
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      sr_device = torch.device('cpu') #device #('mps')   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############   DEPRECATED: see geneticsd.py      esrmodel = RealESRGAN(sr_device, scale=4)
###############   DEPRECATED: see geneticsd.py      esrmodel.load_weights('weights/RealESRGAN_x4.pth', download=True)
###############   DEPRECATED: see geneticsd.py      esrmodel2 = RealESRGAN(sr_device, scale=2)
###############   DEPRECATED: see geneticsd.py      esrmodel2.load_weights('weights/RealESRGAN_x2.pth', download=True)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def singleeg(path_to_image):
###############   DEPRECATED: see geneticsd.py          image = Image.open(path_to_image).convert('RGB')
###############   DEPRECATED: see geneticsd.py          sr_device = device #('mps')   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############   DEPRECATED: see geneticsd.py          print(f"Type before SR = {type(image)}")
###############   DEPRECATED: see geneticsd.py          sr_image = esrmodel.predict(image)
###############   DEPRECATED: see geneticsd.py          print(f"Type after SR = {type(sr_image)}")
###############   DEPRECATED: see geneticsd.py          output_filename = path_to_image + ".SR.png"
###############   DEPRECATED: see geneticsd.py          sr_image.save(output_filename)
###############   DEPRECATED: see geneticsd.py          return output_filename
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def singleeg2(path_to_image):
###############   DEPRECATED: see geneticsd.py          time.sleep(0.5*np.random.rand())
###############   DEPRECATED: see geneticsd.py          image = Image.open(path_to_image).convert('RGB')
###############   DEPRECATED: see geneticsd.py          sr_device = device #('mps')   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############   DEPRECATED: see geneticsd.py          print(f"Type before SR = {type(image)}")
###############   DEPRECATED: see geneticsd.py          sr_image = esrmodel2.predict(image)
###############   DEPRECATED: see geneticsd.py          print(f"Type after SR = {type(sr_image)}")
###############   DEPRECATED: see geneticsd.py          output_filename = path_to_image + ".SR.png"
###############   DEPRECATED: see geneticsd.py          sr_image.save(output_filename)
###############   DEPRECATED: see geneticsd.py          return output_filename
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def eg(list_of_files, last_list_of_files):
###############   DEPRECATED: see geneticsd.py          pretty_print("Should I convert images below to high resolution ?")
###############   DEPRECATED: see geneticsd.py          print(list_of_files)
###############   DEPRECATED: see geneticsd.py          noise.say("Go to the text window!")
###############   DEPRECATED: see geneticsd.py          noise.runAndWait()
###############   DEPRECATED: see geneticsd.py          answer = input(" [y]es / [n]o / [j]ust the last batch of {len(last_list_of_files)} images ?")
###############   DEPRECATED: see geneticsd.py          if "y" in answer or "Y" in answer or "j" in answer or "J" in answer:
###############   DEPRECATED: see geneticsd.py              if j in answer or "J" in answer:
###############   DEPRECATED: see geneticsd.py                  list_of_files = last_list_of_files
###############   DEPRECATED: see geneticsd.py              #images = Parallel(n_jobs=12)(delayed(singleeg)(image) for image in list_of_files)
###############   DEPRECATED: see geneticsd.py              #print(to_native(f"Created the super-resolution files {images}")) 
###############   DEPRECATED: see geneticsd.py              for path_to_image in list_of_files:
###############   DEPRECATED: see geneticsd.py                  output_filename = singleeg(path_to_image)
###############   DEPRECATED: see geneticsd.py                  print(to_native(f"Created the super-resolution file {output_filename}")) 
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def stop_all(list_of_files, list_of_latent, last_list_of_files, last_list_of_latent):
###############   DEPRECATED: see geneticsd.py          print(to_native("Your selected images and the last generation:"))
###############   DEPRECATED: see geneticsd.py          print(list_of_files)
###############   DEPRECATED: see geneticsd.py          eg(list_of_files, last_list_of_files)
###############   DEPRECATED: see geneticsd.py          pretty_print("Should we create animations ?")
###############   DEPRECATED: see geneticsd.py          answer = input(" [y]es or [n]o or [j]ust the selection on the last panel ?")
###############   DEPRECATED: see geneticsd.py          if "y" in answer or "Y" in answer or "j" in answer or "J" in answer:
###############   DEPRECATED: see geneticsd.py              assert len(list_of_files) == len(list_of_latent)
###############   DEPRECATED: see geneticsd.py              if "j" in answer or "J" in answer:
###############   DEPRECATED: see geneticsd.py                  list_of_latent = last_list_of_latent
###############   DEPRECATED: see geneticsd.py              pretty_print("Let us create animations!")
###############   DEPRECATED: see geneticsd.py              for c in sorted([0.05, 0.04,0.03,0.02,0.01]):
###############   DEPRECATED: see geneticsd.py                  for idx in range(len(list_of_files)):
###############   DEPRECATED: see geneticsd.py                      images = []
###############   DEPRECATED: see geneticsd.py                      l = list_of_latent[idx].reshape(1,4,64,64)
###############   DEPRECATED: see geneticsd.py                      l = np.sqrt(len(l.flatten()) / np.sum(l**2)) * l
###############   DEPRECATED: see geneticsd.py                      l1 = l + c * np.random.randn(len(l.flatten())).reshape(1,4,64,64)
###############   DEPRECATED: see geneticsd.py                      l1 = np.sqrt(len(l1.flatten()) / np.sum(l1**2)) * l1
###############   DEPRECATED: see geneticsd.py                      l2 = l + c * np.random.randn(len(l.flatten())).reshape(1,4,64,64)
###############   DEPRECATED: see geneticsd.py                      l2 = np.sqrt(len(l2.flatten()) / np.sum(l2**2)) * l2
###############   DEPRECATED: see geneticsd.py                      num_animation_steps = 13
###############   DEPRECATED: see geneticsd.py                      index = 0
###############   DEPRECATED: see geneticsd.py                      for u in np.linspace(0., 2*3.14159 * (1-1/30), 30):
###############   DEPRECATED: see geneticsd.py                           cc = np.cos(u)
###############   DEPRECATED: see geneticsd.py                           ss = np.sin(u*2)
###############   DEPRECATED: see geneticsd.py                           index += 1
###############   DEPRECATED: see geneticsd.py                           image = latent_to_image(l + cc * (l1 - l) + ss * (l2 - l))
###############   DEPRECATED: see geneticsd.py                           image_name = f"imgA{index}.png"
###############   DEPRECATED: see geneticsd.py                           image.save(image_name)
###############   DEPRECATED: see geneticsd.py                           images += [image_name]
###############   DEPRECATED: see geneticsd.py                           
###############   DEPRECATED: see geneticsd.py      #                for u in np.linspace(0., 1., num_animation_steps):
###############   DEPRECATED: see geneticsd.py      #                    index += 1
###############   DEPRECATED: see geneticsd.py      #                    image = latent_to_image(u*l1 + (1-u)*l)
###############   DEPRECATED: see geneticsd.py      #                    image_name = f"imgA{index}.png"
###############   DEPRECATED: see geneticsd.py      #                    image.save(image_name)
###############   DEPRECATED: see geneticsd.py      #                    images += [image_name]
###############   DEPRECATED: see geneticsd.py      #                for u in np.linspace(0., 1., num_animation_steps):
###############   DEPRECATED: see geneticsd.py      #                    index += 1
###############   DEPRECATED: see geneticsd.py      #                    image = latent_to_image(u*l2 + (1-u)*l1)
###############   DEPRECATED: see geneticsd.py      #                    image_name = f"imgB{index}.png"
###############   DEPRECATED: see geneticsd.py      #                    image.save(image_name)
###############   DEPRECATED: see geneticsd.py      #                    images += [image_name]
###############   DEPRECATED: see geneticsd.py      #                for u in np.linspace(0., 1.,num_animation_steps):
###############   DEPRECATED: see geneticsd.py      #                    index += 1
###############   DEPRECATED: see geneticsd.py      #                    image = latent_to_image(u*l + (1-u)*l2)
###############   DEPRECATED: see geneticsd.py      #                    image_name = f"imgC{index}.png"
###############   DEPRECATED: see geneticsd.py      #                    image.save(image_name)
###############   DEPRECATED: see geneticsd.py      #                    images += [image_name]
###############   DEPRECATED: see geneticsd.py                      print(to_native(f"Base images created for perturbation={c} and file {list_of_files[idx]}"))
###############   DEPRECATED: see geneticsd.py                      #images = Parallel(n_jobs=8)(delayed(process)(i) for i in range(10))
###############   DEPRECATED: see geneticsd.py                      images = Parallel(n_jobs=10)(delayed(singleeg2)(image) for image in images)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py                      frames = [Image.open(image) for image in images]
###############   DEPRECATED: see geneticsd.py                      frame_one = frames[0]
###############   DEPRECATED: see geneticsd.py                      gif_name = list_of_files[idx] + "_" + str(c) + ".gif"
###############   DEPRECATED: see geneticsd.py                      frame_one.save(gif_name, format="GIF", append_images=frames,
###############   DEPRECATED: see geneticsd.py                            save_all=True, duration=100, loop=0)    
###############   DEPRECATED: see geneticsd.py                      webbrowser.open(os.environ["PWD"] + "/" + gif_name)
###############   DEPRECATED: see geneticsd.py          
###############   DEPRECATED: see geneticsd.py          pretty_print("Should we create a meme ?")
###############   DEPRECATED: see geneticsd.py          answer = input(" [y]es or [n]o ?")
###############   DEPRECATED: see geneticsd.py          if "y" in answer or "Y" in answer:
###############   DEPRECATED: see geneticsd.py              url = 'https://imgflip.com/memegenerator'
###############   DEPRECATED: see geneticsd.py              webbrowser.open(url)
###############   DEPRECATED: see geneticsd.py          pretty_print("Good bye!")
###############   DEPRECATED: see geneticsd.py          exit()
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      import os
###############   DEPRECATED: see geneticsd.py      import pygame
###############   DEPRECATED: see geneticsd.py      from os import listdir
###############   DEPRECATED: see geneticsd.py      from os.path import isfile, join
###############   DEPRECATED: see geneticsd.py            
###############   DEPRECATED: see geneticsd.py      sentinel = str(random.randint(0,100000)) + "XX" +  str(random.randint(0,100000))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      all_files = []
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      llambda = 15
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      assert llambda < 16, "lambda < 16 for convenience in pygame."
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      bad = []
###############   DEPRECATED: see geneticsd.py      five_best = []
###############   DEPRECATED: see geneticsd.py      latent = []
###############   DEPRECATED: see geneticsd.py      images = []
###############   DEPRECATED: see geneticsd.py      onlyfiles = []
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      pretty_print("Now let us choose (if you want) an image as a start.")
###############   DEPRECATED: see geneticsd.py      image_name = input(to_native("Name of image for starting ? (enter if no start image)"))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      # activate the pygame library .
###############   DEPRECATED: see geneticsd.py      pygame.init()
###############   DEPRECATED: see geneticsd.py      X = 2000  # > 1500 = buttons
###############   DEPRECATED: see geneticsd.py      Y = 900  
###############   DEPRECATED: see geneticsd.py      scrn = pygame.display.set_mode((1700, Y + 100))
###############   DEPRECATED: see geneticsd.py      font = pygame.font.Font('freesansbold.ttf', 22)
###############   DEPRECATED: see geneticsd.py      bigfont = pygame.font.Font('freesansbold.ttf', 44)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def load_img(path):
###############   DEPRECATED: see geneticsd.py          image = Image.open(path).convert("RGB")
###############   DEPRECATED: see geneticsd.py          w, h = image.size
###############   DEPRECATED: see geneticsd.py          print(to_native(f"loaded input image of size ({w}, {h}) from {path}"))
###############   DEPRECATED: see geneticsd.py          w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
###############   DEPRECATED: see geneticsd.py          image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
###############   DEPRECATED: see geneticsd.py          #image = image.resize((w, h), resample=PIL.Image.LANCZOS)
###############   DEPRECATED: see geneticsd.py          image = np.array(image).astype(np.float32) / 255.0
###############   DEPRECATED: see geneticsd.py          image = image[None].transpose(0, 3, 1, 2)
###############   DEPRECATED: see geneticsd.py          image = torch.from_numpy(image)
###############   DEPRECATED: see geneticsd.py          return 2.*image - 1.
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      model = pipe.vae
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def img_to_latent(path):
###############   DEPRECATED: see geneticsd.py          #init_image = 1.8 * load_img(path).to(device)
###############   DEPRECATED: see geneticsd.py          init_image = load_img(path).to(device)
###############   DEPRECATED: see geneticsd.py          init_image = repeat(init_image, '1 ... -> b ...', b=1)
###############   DEPRECATED: see geneticsd.py          forced_latent = model.encode(init_image.to(device)).latent_dist.sample()
###############   DEPRECATED: see geneticsd.py          new_fl = forced_latent.cpu().detach().numpy().flatten()
###############   DEPRECATED: see geneticsd.py          new_fl = np.sqrt(len(new_fl)) * new_fl / np.sqrt(np.sum(new_fl ** 2))
###############   DEPRECATED: see geneticsd.py          return new_fl
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      def randomized_image_to_latent(image_name, scale=None, epsilon=None, c=None, f=None):
###############   DEPRECATED: see geneticsd.py          base_init_image = load_img(image_name).to(device)
###############   DEPRECATED: see geneticsd.py          new_base_init_image = base_init_image
###############   DEPRECATED: see geneticsd.py          c = np.exp(np.random.randn()) if c is None else c
###############   DEPRECATED: see geneticsd.py          f = np.exp(-3. * np.random.rand()) if f is None else f
###############   DEPRECATED: see geneticsd.py          init_image_shape = base_init_image.cpu().numpy().shape
###############   DEPRECATED: see geneticsd.py          init_image = c * new_base_init_image
###############   DEPRECATED: see geneticsd.py          init_image = repeat(init_image, '1 ... -> b ...', b=1)
###############   DEPRECATED: see geneticsd.py          forced_latent = 1. * model.encode(init_image.to(device)).latent_dist.sample()
###############   DEPRECATED: see geneticsd.py          new_fl = forced_latent.cpu().detach().numpy().flatten()
###############   DEPRECATED: see geneticsd.py          basic_new_fl = new_fl  #np.sqrt(len(new_fl) / sum(new_fl ** 2)) * new_fl
###############   DEPRECATED: see geneticsd.py          basic_new_fl = f * np.sqrt(len(new_fl) / np.sum(basic_new_fl**2)) * basic_new_fl
###############   DEPRECATED: see geneticsd.py          epsilon = 0.1 * np.exp(-3 * np.random.rand()) if epsilon is None else epsilon
###############   DEPRECATED: see geneticsd.py          new_fl = (1. - epsilon) * basic_new_fl + epsilon * np.random.randn(1*4*64*64)
###############   DEPRECATED: see geneticsd.py          scale = 2.8 + 3.6 * np.random.rand() if scale is None else scale
###############   DEPRECATED: see geneticsd.py          new_fl = scale * np.sqrt(len(new_fl)) * new_fl / np.sqrt(np.sum(new_fl ** 2))
###############   DEPRECATED: see geneticsd.py          #image = latent_to_image(np.asarray(new_fl)) #eval(os.environ["forcedlatent"])))
###############   DEPRECATED: see geneticsd.py          #image.save(f"rebuild_{f}_{scale}_{epsilon}_{c}.png")
###############   DEPRECATED: see geneticsd.py          return new_fl
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      if len(image_name) > 0:
###############   DEPRECATED: see geneticsd.py          pretty_print("Importing an image !")
###############   DEPRECATED: see geneticsd.py          try:
###############   DEPRECATED: see geneticsd.py              init_image = load_img(image_name).to(device)
###############   DEPRECATED: see geneticsd.py          except:
###############   DEPRECATED: see geneticsd.py              pretty_print("Try again!")
###############   DEPRECATED: see geneticsd.py              pretty_print("Loading failed!!")
###############   DEPRECATED: see geneticsd.py              image_name = input(to_native("Name of image for starting ? (enter if no start image)"))
###############   DEPRECATED: see geneticsd.py              
###############   DEPRECATED: see geneticsd.py          base_init_image = load_img(image_name).to(device)
###############   DEPRECATED: see geneticsd.py          noise.say("Image loaded")
###############   DEPRECATED: see geneticsd.py          noise.runAndWait()
###############   DEPRECATED: see geneticsd.py          print(base_init_image.shape)
###############   DEPRECATED: see geneticsd.py          print(np.max(base_init_image.cpu().detach().numpy().flatten()))
###############   DEPRECATED: see geneticsd.py          print(np.min(base_init_image.cpu().detach().numpy().flatten()))
###############   DEPRECATED: see geneticsd.py          
###############   DEPRECATED: see geneticsd.py          forcedlatents = []
###############   DEPRECATED: see geneticsd.py          divider = 1.5
###############   DEPRECATED: see geneticsd.py          latent_found = False
###############   DEPRECATED: see geneticsd.py          try:
###############   DEPRECATED: see geneticsd.py              latent_file = image_name + ".latent.txt"
###############   DEPRECATED: see geneticsd.py              print(to_native(f"Trying to load latent variables in {latent_file}."))
###############   DEPRECATED: see geneticsd.py              f = open(latent_file, "r")
###############   DEPRECATED: see geneticsd.py              print(to_native("File opened."))
###############   DEPRECATED: see geneticsd.py              latent_str = f.read()
###############   DEPRECATED: see geneticsd.py              print("Latent string read.")
###############   DEPRECATED: see geneticsd.py              latent_found = True
###############   DEPRECATED: see geneticsd.py          except:
###############   DEPRECATED: see geneticsd.py              print(to_native("No latent file: guessing."))
###############   DEPRECATED: see geneticsd.py          for i in range(llambda):
###############   DEPRECATED: see geneticsd.py              new_base_init_image = base_init_image
###############   DEPRECATED: see geneticsd.py              if not latent_found: # In case of latent vars we need less exploration.
###############   DEPRECATED: see geneticsd.py                  if (i % 7)  == 1:
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,0,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                  if (i % 7) == 2:
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,1,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                  if (i % 7) == 3:
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,2,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                  if (i % 7) == 4:
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,0,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,1,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                  if (i % 7) == 5:
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,1,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,2,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                  if (i % 7) == 6:
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,0,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                      new_base_init_image[0,2,:,:] /= divider
###############   DEPRECATED: see geneticsd.py                 
###############   DEPRECATED: see geneticsd.py              c = np.exp(np.random.randn() - 5)
###############   DEPRECATED: see geneticsd.py              f = np.exp(-3. * np.random.rand())
###############   DEPRECATED: see geneticsd.py              init_image_shape = base_init_image.cpu().numpy().shape
###############   DEPRECATED: see geneticsd.py              if i > 0 and not latent_found:
###############   DEPRECATED: see geneticsd.py                  init_image = new_base_init_image + torch.from_numpy(c * np.random.randn(np.prod(init_image_shape))).reshape(init_image_shape).float().to(device)
###############   DEPRECATED: see geneticsd.py              else:
###############   DEPRECATED: see geneticsd.py                  init_image = new_base_init_image
###############   DEPRECATED: see geneticsd.py              init_image = repeat(init_image, '1 ... -> b ...', b=1)
###############   DEPRECATED: see geneticsd.py              if latent_found:
###############   DEPRECATED: see geneticsd.py                  new_fl = np.asarray(eval(latent_str))
###############   DEPRECATED: see geneticsd.py                  assert len(new_fl) > 1
###############   DEPRECATED: see geneticsd.py              else:
###############   DEPRECATED: see geneticsd.py                  forced_latent = 1. * model.encode(init_image.to(device)).latent_dist.sample()
###############   DEPRECATED: see geneticsd.py                  new_fl = forced_latent.cpu().detach().numpy().flatten()
###############   DEPRECATED: see geneticsd.py              basic_new_fl = new_fl  #np.sqrt(len(new_fl) / sum(new_fl ** 2)) * new_fl
###############   DEPRECATED: see geneticsd.py              #new_fl = forced_latent + (1. / 1.1**(llambda-i)) * torch.from_numpy(np.random.randn(1*4*64*64).reshape(1,4,64,64)).float().to(device)
###############   DEPRECATED: see geneticsd.py              #forcedlatents += [new_fl.cpu().detach().numpy()]
###############   DEPRECATED: see geneticsd.py              if i > 0:
###############   DEPRECATED: see geneticsd.py                  #epsilon = 0.3 / 1.1**i
###############   DEPRECATED: see geneticsd.py                  basic_new_fl = f * np.sqrt(len(new_fl) / np.sum(basic_new_fl**2)) * basic_new_fl
###############   DEPRECATED: see geneticsd.py                  epsilon = .7 * ((i-1)/(llambda-1)) #1.0 / 2**(2 + (llambda - i) / 6)
###############   DEPRECATED: see geneticsd.py                  print(f"{i} -- {i % 7} {c} {f} {epsilon}")
###############   DEPRECATED: see geneticsd.py                  # 1 -- 1 0.050020045300292804 0.0790648688521246 0.0
###############   DEPRECATED: see geneticsd.py                  new_fl = (1. - epsilon) * basic_new_fl + epsilon * np.random.randn(1*4*64*64)
###############   DEPRECATED: see geneticsd.py              else:
###############   DEPRECATED: see geneticsd.py                  new_fl = basic_new_fl
###############   DEPRECATED: see geneticsd.py              new_fl = 6. * np.sqrt(len(new_fl)) * new_fl / np.sqrt(np.sum(new_fl ** 2))
###############   DEPRECATED: see geneticsd.py              forcedlatents += [new_fl] #np.clip(new_fl, -3., 3.)] #np.sqrt(len(new_fl) / sum(new_fl ** 2)) * new_fl]
###############   DEPRECATED: see geneticsd.py              forcedgs += [7.5]  #np.random.choice([7.5, 15.0, 30.0, 60.0])] TODO
###############   DEPRECATED: see geneticsd.py              #forcedlatents += [np.sqrt(len(new_fl) / sum(new_fl ** 2)) * new_fl]
###############   DEPRECATED: see geneticsd.py              #print(f"{i} --> {forcedlatents[i][:10]}")
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      # We start the big time consuming loop!
###############   DEPRECATED: see geneticsd.py      for iteration in range(30):
###############   DEPRECATED: see geneticsd.py          latent = [latent[f] for f in five_best]
###############   DEPRECATED: see geneticsd.py          images = [images[f] for f in five_best]
###############   DEPRECATED: see geneticsd.py          onlyfiles = [onlyfiles[f] for f in five_best]
###############   DEPRECATED: see geneticsd.py          early_stop = []
###############   DEPRECATED: see geneticsd.py          noise.say("WAIT!")
###############   DEPRECATED: see geneticsd.py          noise.runAndWait()
###############   DEPRECATED: see geneticsd.py          final_selection = []
###############   DEPRECATED: see geneticsd.py          final_selection_latent = []
###############   DEPRECATED: see geneticsd.py          for k in range(llambda):
###############   DEPRECATED: see geneticsd.py              if len(early_stop) > 0:
###############   DEPRECATED: see geneticsd.py                  break
###############   DEPRECATED: see geneticsd.py              max_created_index = k
###############   DEPRECATED: see geneticsd.py              if len(forcedlatents) > 0 and k < len(forcedlatents):
###############   DEPRECATED: see geneticsd.py                  #os.environ["forcedlatent"] = str(list(forcedlatents[k].flatten()))            
###############   DEPRECATED: see geneticsd.py                  latent_forcing = str(list(forcedlatents[k].flatten()))
###############   DEPRECATED: see geneticsd.py                  print(f"We play with {latent_forcing[:20]}")
###############   DEPRECATED: see geneticsd.py              if k < len(five_best):
###############   DEPRECATED: see geneticsd.py                  imp = pygame.transform.scale(pygame.image.load(onlyfiles[k]).convert(), (300, 300))
###############   DEPRECATED: see geneticsd.py                  # Using blit to copy content from one surface to other
###############   DEPRECATED: see geneticsd.py                  scrn.blit(imp, (300 * (k // 3), 300 * (k % 3)))
###############   DEPRECATED: see geneticsd.py                  pygame.display.flip()
###############   DEPRECATED: see geneticsd.py                  continue
###############   DEPRECATED: see geneticsd.py              pygame.draw.rect(scrn, black, pygame.Rect(0, Y, 1700, Y+100))
###############   DEPRECATED: see geneticsd.py              pygame.draw.rect(scrn, black, pygame.Rect(1500, 0, 2000, Y+100))
###############   DEPRECATED: see geneticsd.py              text0 = bigfont.render(to_native(f'Please wait !!! {k} / {llambda}'), True, green, blue)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text0, ((X*3/4)/2 - X/32, Y/2-Y/4))
###############   DEPRECATED: see geneticsd.py              text0 = font.render(to_native(f'Or, for an early stopping,'), True, green, blue)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text0, ((X*3/4)/3 - X/32, Y/2-Y/8))
###############   DEPRECATED: see geneticsd.py              text0 = font.render(to_native(f'click <here> and WAIT a bit'), True, green, blue)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text0, ((X*3/4)/3 - X/32, Y/2))
###############   DEPRECATED: see geneticsd.py              text0 = font.render(to_native(f'... ... ... '), True, green, blue)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text0, ((X*3/4)/2 - X/32, Y/2+Y/8))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py              # Button for early stopping
###############   DEPRECATED: see geneticsd.py              text2 = font.render(to_native(f'Total: {len(all_selected)} chosen images! '), True, green, blue)
###############   DEPRECATED: see geneticsd.py              text2 = pygame.transform.rotate(text2, 90)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text2, (X*3/4+X/16      - X/32, Y/3))
###############   DEPRECATED: see geneticsd.py              text2 = font.render(to_native('Click <here> for stopping,'), True, green, blue)
###############   DEPRECATED: see geneticsd.py              text2 = pygame.transform.rotate(text2, 90)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text2, (X*3/4+X/16+X/64 - X/32, Y/3))
###############   DEPRECATED: see geneticsd.py              text2 = font.render(to_native('and get the effects.'), True, green, blue)
###############   DEPRECATED: see geneticsd.py              text2 = pygame.transform.rotate(text2, 90)
###############   DEPRECATED: see geneticsd.py              scrn.blit(text2, (X*3/4+X/16+X/32 - X/32, Y/3))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py              pygame.display.flip()
###############   DEPRECATED: see geneticsd.py              os.environ["earlystop"] = "False" if k > len(five_best) else "True"
###############   DEPRECATED: see geneticsd.py              os.environ["epsilon"] = str(0. if k == len(five_best) else (k - len(five_best)) / llambda)
###############   DEPRECATED: see geneticsd.py              os.environ["budget"] = str(np.random.randint(400) if k > len(five_best) else 2)
###############   DEPRECATED: see geneticsd.py              os.environ["skl"] = {0: "nn", 1: "tree", 2: "logit"}[k % 3]
###############   DEPRECATED: see geneticsd.py              #enforcedlatent = os.environ.get("enforcedlatent", "")
###############   DEPRECATED: see geneticsd.py              #if len(enforcedlatent) > 2:
###############   DEPRECATED: see geneticsd.py              #    os.environ["forcedlatent"] = enforcedlatent
###############   DEPRECATED: see geneticsd.py              #    os.environ["enforcedlatent"] = ""
###############   DEPRECATED: see geneticsd.py              #with autocast("cuda"):
###############   DEPRECATED: see geneticsd.py              #    image = pipe(english_prompt, guidance_scale=gs, num_inference_steps=num_iterations)["sample"][0]
###############   DEPRECATED: see geneticsd.py              previous_gs = gs
###############   DEPRECATED: see geneticsd.py              if k < len(forcedgs):
###############   DEPRECATED: see geneticsd.py                  gs = forcedgs[k]
###############   DEPRECATED: see geneticsd.py              image = latent_to_image(np.asarray(latent_forcing)) #eval(os.environ["forcedlatent"])))
###############   DEPRECATED: see geneticsd.py              gs = previous_gs
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py              images += [image]
###############   DEPRECATED: see geneticsd.py              filename = f"SD_{prompt.replace(' ','_')}_image_{sentinel}_{iteration:05d}_{k:05d}.png"  
###############   DEPRECATED: see geneticsd.py              image.save(filename)
###############   DEPRECATED: see geneticsd.py              onlyfiles += [filename]
###############   DEPRECATED: see geneticsd.py              imp = pygame.transform.scale(pygame.image.load(onlyfiles[-1]).convert(), (300, 300))
###############   DEPRECATED: see geneticsd.py              # Using blit to copy content from one surface to other
###############   DEPRECATED: see geneticsd.py              scrn.blit(imp, (300 * (k // 3), 300 * (k % 3)))
###############   DEPRECATED: see geneticsd.py              pygame.display.flip()
###############   DEPRECATED: see geneticsd.py              #noise.say("Dong")
###############   DEPRECATED: see geneticsd.py              #noise.runAndWait()
###############   DEPRECATED: see geneticsd.py              print('\a')
###############   DEPRECATED: see geneticsd.py              str_latent = eval((os.environ["latent_sd"]))
###############   DEPRECATED: see geneticsd.py              array_latent = eval(f"np.array(str_latent).reshape(4, 64, 64)")
###############   DEPRECATED: see geneticsd.py              print(f"Debug info: array_latent sumsq/var {sum(array_latent.flatten() ** 2) / len(array_latent.flatten())}")
###############   DEPRECATED: see geneticsd.py              latent += [array_latent]
###############   DEPRECATED: see geneticsd.py              with open(filename + ".latent.txt", 'w') as f:
###############   DEPRECATED: see geneticsd.py                  f.write(f"{str_latent}")
###############   DEPRECATED: see geneticsd.py              # In case of early stopping.
###############   DEPRECATED: see geneticsd.py              first_event = True
###############   DEPRECATED: see geneticsd.py              for i in pygame.event.get():
###############   DEPRECATED: see geneticsd.py                  if i.type == pygame.MOUSEBUTTONUP:
###############   DEPRECATED: see geneticsd.py                      if first_event:
###############   DEPRECATED: see geneticsd.py                          noise.say("Ok I stop")
###############   DEPRECATED: see geneticsd.py                          noise.runAndWait()
###############   DEPRECATED: see geneticsd.py                          first_event = False
###############   DEPRECATED: see geneticsd.py                      pos = pygame.mouse.get_pos()
###############   DEPRECATED: see geneticsd.py                      index = 3 * (pos[0] // 300) + (pos[1] // 300)
###############   DEPRECATED: see geneticsd.py                      if pos[0] > X and pos[1] > Y /3 and pos[1] < 2*Y/3:
###############   DEPRECATED: see geneticsd.py                          stop_all(all_selected, all_selected_latent, final_selection, final_selection_latent)
###############   DEPRECATED: see geneticsd.py                          exit()
###############   DEPRECATED: see geneticsd.py                      if index <= k:
###############   DEPRECATED: see geneticsd.py                          pretty_print(("You clicked for requesting an early stopping."))
###############   DEPRECATED: see geneticsd.py                          early_stop = [pos]
###############   DEPRECATED: see geneticsd.py                          break
###############   DEPRECATED: see geneticsd.py                      early_stop = [(1,1)]
###############   DEPRECATED: see geneticsd.py                      satus = False
###############   DEPRECATED: see geneticsd.py          forcedgs = []
###############   DEPRECATED: see geneticsd.py          # Stop the forcing from disk!
###############   DEPRECATED: see geneticsd.py          #os.environ["enforcedlatent"] = ""
###############   DEPRECATED: see geneticsd.py          # importing required library
###############   DEPRECATED: see geneticsd.py          
###############   DEPRECATED: see geneticsd.py          #mypath = "./"
###############   DEPRECATED: see geneticsd.py          #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
###############   DEPRECATED: see geneticsd.py          #onlyfiles = [str(f) for f in onlyfiles if "SD_" in str(f) and ".png" in str(f) and str(f) not in all_files and sentinel in str(f)]
###############   DEPRECATED: see geneticsd.py          #print()
###############   DEPRECATED: see geneticsd.py           
###############   DEPRECATED: see geneticsd.py          # create the display surface object
###############   DEPRECATED: see geneticsd.py          # of specific dimension..e(X, Y).
###############   DEPRECATED: see geneticsd.py          noise.say("Ok I'm ready! Choose")
###############   DEPRECATED: see geneticsd.py          noise.runAndWait()
###############   DEPRECATED: see geneticsd.py          pretty_print("Please choose your images.")
###############   DEPRECATED: see geneticsd.py          text0 = bigfont.render(to_native(f'Choose your favorite images !!!========='), True, green, blue)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text0, ((X*3/4)/2 - X/32, Y/2-Y/4))
###############   DEPRECATED: see geneticsd.py          text0 = font.render(to_native(f'=================================='), True, green, blue)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text0, ((X*3/4)/3 - X/32, Y/2-Y/8))
###############   DEPRECATED: see geneticsd.py          text0 = font.render(to_native(f'=================================='), True, green, blue)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text0, ((X*3/4)/3 - X/32, Y/2))
###############   DEPRECATED: see geneticsd.py          # Add rectangles
###############   DEPRECATED: see geneticsd.py          pygame.draw.rect(scrn, red, pygame.Rect(X*3/4, 0, X*3/4+X/16+X/32, Y/3), 2)
###############   DEPRECATED: see geneticsd.py          pygame.draw.rect(scrn, red, pygame.Rect(X*3/4, Y/3, X*3/4+X/16+X/32, 2*Y/3), 2)
###############   DEPRECATED: see geneticsd.py          pygame.draw.rect(scrn, red, pygame.Rect(X*3/4, 2*Y/3, X*3/4+X/16+X/32, Y), 2)
###############   DEPRECATED: see geneticsd.py          pygame.draw.rect(scrn, red, pygame.Rect(0, Y, X/2, Y+100), 2)
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py          # Button for loading a starting point
###############   DEPRECATED: see geneticsd.py          text1 = font.render('Manually edit an image.', True, green, blue)
###############   DEPRECATED: see geneticsd.py          text1 = pygame.transform.rotate(text1, 90)
###############   DEPRECATED: see geneticsd.py          #scrn.blit(text1, (X*3/4+X/16 - X/32, 0))
###############   DEPRECATED: see geneticsd.py          #text1 = font.render('& latent    ', True, green, blue)
###############   DEPRECATED: see geneticsd.py          #text1 = pygame.transform.rotate(text1, 90)
###############   DEPRECATED: see geneticsd.py          #scrn.blit(text1, (X*3/4+X/16+X/32 - X/32, 0))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py          # Button for creating a meme
###############   DEPRECATED: see geneticsd.py          text2 = font.render(to_native('Click <here>,'), True, green, blue)
###############   DEPRECATED: see geneticsd.py          text2 = pygame.transform.rotate(text2, 90)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text2, (X*3/4+X/16 - X/32, Y/3+10))
###############   DEPRECATED: see geneticsd.py          text2 = font.render(to_native('for finishing with effects.'), True, green, blue)
###############   DEPRECATED: see geneticsd.py          text2 = pygame.transform.rotate(text2, 90)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text2, (X*3/4+X/16+X/32 - X/32, Y/3+10))
###############   DEPRECATED: see geneticsd.py          # Button for new generation
###############   DEPRECATED: see geneticsd.py          text3 = font.render(to_native(f"I don't want to select images"), True, green, blue)
###############   DEPRECATED: see geneticsd.py          text3 = pygame.transform.rotate(text3, 90)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text3, (X*3/4+X/16 - X/32, Y*2/3+10))
###############   DEPRECATED: see geneticsd.py          text3 = font.render(to_native(f"Just rerun."), True, green, blue)
###############   DEPRECATED: see geneticsd.py          text3 = pygame.transform.rotate(text3, 90)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text3, (X*3/4+X/16+X/32 - X/32, Y*2/3+10))
###############   DEPRECATED: see geneticsd.py          text4 = font.render(to_native(f"Modify parameters or text!"), True, green, blue)
###############   DEPRECATED: see geneticsd.py          scrn.blit(text4, (300, Y + 30))
###############   DEPRECATED: see geneticsd.py          pygame.display.flip()
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py          for idx in range(max_created_index + 1):
###############   DEPRECATED: see geneticsd.py              # set the pygame window name
###############   DEPRECATED: see geneticsd.py              pygame.display.set_caption(prompt)
###############   DEPRECATED: see geneticsd.py              print(to_native(f"Pasting image {onlyfiles[idx]}..."))
###############   DEPRECATED: see geneticsd.py              imp = pygame.transform.scale(pygame.image.load(onlyfiles[idx]).convert(), (300, 300))
###############   DEPRECATED: see geneticsd.py              scrn.blit(imp, (300 * (idx // 3), 300 * (idx % 3)))
###############   DEPRECATED: see geneticsd.py           
###############   DEPRECATED: see geneticsd.py          # paint screen one time
###############   DEPRECATED: see geneticsd.py          pygame.display.flip()
###############   DEPRECATED: see geneticsd.py          status = True
###############   DEPRECATED: see geneticsd.py          indices = []
###############   DEPRECATED: see geneticsd.py          good = []
###############   DEPRECATED: see geneticsd.py          five_best = []
###############   DEPRECATED: see geneticsd.py          for i in pygame.event.get():
###############   DEPRECATED: see geneticsd.py              if i.type == pygame.MOUSEBUTTONUP:
###############   DEPRECATED: see geneticsd.py                  print(to_native(".... too early for clicking !!!!"))
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py      
###############   DEPRECATED: see geneticsd.py          pretty_print("Please click on your favorite elements!")
###############   DEPRECATED: see geneticsd.py          print(to_native("You might just click on one image and we will provide variations."))
###############   DEPRECATED: see geneticsd.py          print(to_native("Or you can click on the top of an image and the bottom of another one."))
###############   DEPRECATED: see geneticsd.py          print(to_native("Click on the << new generation >> when you're done.")) 
###############   DEPRECATED: see geneticsd.py          while (status):
###############   DEPRECATED: see geneticsd.py           
###############   DEPRECATED: see geneticsd.py            # iterate over the list of Event objects
###############   DEPRECATED: see geneticsd.py            # that was returned by pygame.event.get() method.
###############   DEPRECATED: see geneticsd.py              for i in pygame.event.get():
###############   DEPRECATED: see geneticsd.py                  if hasattr(i, "type") and i.type == pygame.MOUSEBUTTONUP:
###############   DEPRECATED: see geneticsd.py                      pos = pygame.mouse.get_pos() 
###############   DEPRECATED: see geneticsd.py                      pretty_print(f"Detected! Click at {pos}")
###############   DEPRECATED: see geneticsd.py                      if pos[1] > Y:
###############   DEPRECATED: see geneticsd.py                          pretty_print("Let us update parameters!")
###############   DEPRECATED: see geneticsd.py                          text4 = font.render(to_native(f"ok, go to text window!"), True, green, blue)
###############   DEPRECATED: see geneticsd.py                          scrn.blit(text4, (300, Y + 30))
###############   DEPRECATED: see geneticsd.py                          pygame.display.flip()
###############   DEPRECATED: see geneticsd.py                          try:
###############   DEPRECATED: see geneticsd.py                              num_iterations = int(input(to_native(f"Number of iterations ? (current = {num_iterations})\n")))
###############   DEPRECATED: see geneticsd.py                          except:
###############   DEPRECATED: see geneticsd.py                              num_iterations = int(input(to_native(f"Number of iterations ? (current = {num_iterations})\n")))
###############   DEPRECATED: see geneticsd.py                          gs = float(input(to_native(f"Guidance scale ? (current = {gs})\n")))
###############   DEPRECATED: see geneticsd.py                          print(to_native(f"The current text is << {prompt} >>."))
###############   DEPRECATED: see geneticsd.py                          print(to_native("Start your answer with a symbol << + >> if this is an edit and not a new text.")) 
###############   DEPRECATED: see geneticsd.py                          new_prompt = str(input(to_native(f"Enter a text if you want to change from ") + prompt))
###############   DEPRECATED: see geneticsd.py                          if len(new_prompt) > 2:
###############   DEPRECATED: see geneticsd.py                              if new_prompt[0] == "+":
###############   DEPRECATED: see geneticsd.py                                  prompt += new_prompt[1:]
###############   DEPRECATED: see geneticsd.py                              else:
###############   DEPRECATED: see geneticsd.py                                  prompt = new_prompt
###############   DEPRECATED: see geneticsd.py                              language = detect(prompt)
###############   DEPRECATED: see geneticsd.py                              english_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
###############   DEPRECATED: see geneticsd.py                          pretty_print("Ok! Parameters updated.")
###############   DEPRECATED: see geneticsd.py                          pretty_print("==> go back to the window!")
###############   DEPRECATED: see geneticsd.py                          text4 = font.render(to_native(f"Ok! parameters changed!"), True, green, blue)
###############   DEPRECATED: see geneticsd.py                          scrn.blit(text4, (300, Y + 30))
###############   DEPRECATED: see geneticsd.py                          pygame.display.flip()
###############   DEPRECATED: see geneticsd.py                      elif pos[0] > 1500:  # Not in the images.
###############   DEPRECATED: see geneticsd.py                          if pos[1] < Y/3:
###############   DEPRECATED: see geneticsd.py                              #filename = input(to_native("Filename (please provide the latent file, of the format SD*latent*.txt) ?\n"))
###############   DEPRECATED: see geneticsd.py                              #status = False
###############   DEPRECATED: see geneticsd.py                              #with open(filename, 'r') as f:
###############   DEPRECATED: see geneticsd.py                              #     latent = f.read()
###############   DEPRECATED: see geneticsd.py                              #break
###############   DEPRECATED: see geneticsd.py                              pretty_print("Easy! I exit now, you edit the file and you save it.")
###############   DEPRECATED: see geneticsd.py                              pretty_print("Then just relaunch me and provide the text and the image.")
###############   DEPRECATED: see geneticsd.py                              exit()
###############   DEPRECATED: see geneticsd.py                          if pos[1] < 2*Y/3:
###############   DEPRECATED: see geneticsd.py                              #onlyfiles = [f for f in listdir(".") if isfile(join(mypath, f))]
###############   DEPRECATED: see geneticsd.py                              #onlyfiles = [str(f) for f in onlyfiles if "SD_" in str(f) and ".png" in str(f) and str(f) not in all_files and sentinel in str(f)]
###############   DEPRECATED: see geneticsd.py                              assert len(onlyfiles) == len(latent)
###############   DEPRECATED: see geneticsd.py                              assert len(all_selected) == len(all_selected_latent)
###############   DEPRECATED: see geneticsd.py                              stop_all(all_selected, all_selected_latent, final_selection, final_selection_latent) # + onlyfiles, all_selected_latent + latent)
###############   DEPRECATED: see geneticsd.py                              exit()
###############   DEPRECATED: see geneticsd.py                          status = False
###############   DEPRECATED: see geneticsd.py                          break
###############   DEPRECATED: see geneticsd.py                      index = 3 * (pos[0] // 300) + (pos[1] // 300)
###############   DEPRECATED: see geneticsd.py                      pygame.draw.circle(scrn, red, [pos[0], pos[1]], 13, 0)
###############   DEPRECATED: see geneticsd.py                      if index <= max_created_index:
###############   DEPRECATED: see geneticsd.py                          selected_filename = to_native("Selected") + onlyfiles[index]
###############   DEPRECATED: see geneticsd.py                          shutil.copyfile(onlyfiles[index], selected_filename)
###############   DEPRECATED: see geneticsd.py                          assert len(onlyfiles) == len(latent), f"{len(onlyfiles)} != {len(latent)}"
###############   DEPRECATED: see geneticsd.py                          all_selected += [selected_filename]
###############   DEPRECATED: see geneticsd.py                          all_selected_latent += [latent[index]]
###############   DEPRECATED: see geneticsd.py                          final_selection += [selected_filename]
###############   DEPRECATED: see geneticsd.py                          final_selection_latent += [latent[index]]
###############   DEPRECATED: see geneticsd.py                          text2 = font.render(to_native(f'==> {len(all_selected)} chosen images! '), True, green, blue)
###############   DEPRECATED: see geneticsd.py                          text2 = pygame.transform.rotate(text2, 90)
###############   DEPRECATED: see geneticsd.py                          scrn.blit(text2, (X*3/4+X/16      - X/32, Y/3))
###############   DEPRECATED: see geneticsd.py                          if index not in five_best and len(five_best) < 5:
###############   DEPRECATED: see geneticsd.py                              five_best += [index]
###############   DEPRECATED: see geneticsd.py                          indices += [[index, (pos[0] - (pos[0] // 300) * 300) / 300, (pos[1] - (pos[1] // 300) * 300) / 300]]
###############   DEPRECATED: see geneticsd.py                          # Update the button for new generation.
###############   DEPRECATED: see geneticsd.py                          pygame.draw.rect(scrn, black, pygame.Rect(X*3/4, 2*Y/3, X*3/4+X/16+X/32, Y))
###############   DEPRECATED: see geneticsd.py                          pygame.draw.rect(scrn, red, pygame.Rect(X*3/4, 2*Y/3, X*3/4+X/16+X/32, Y), 2)
###############   DEPRECATED: see geneticsd.py                          text3 = font.render(to_native(f"  You have chosen {len(indices)} images:"), True, green, blue)
###############   DEPRECATED: see geneticsd.py                          text3 = pygame.transform.rotate(text3, 90)
###############   DEPRECATED: see geneticsd.py                          scrn.blit(text3, (X*3/4+X/16 - X/32, Y*2/3))
###############   DEPRECATED: see geneticsd.py                          text3 = font.render(to_native(f"  Click <here> for new generation!"), True, green, blue)
###############   DEPRECATED: see geneticsd.py                          text3 = pygame.transform.rotate(text3, 90)
###############   DEPRECATED: see geneticsd.py                          scrn.blit(text3, (X*3/4+X/16+X/32 - X/32, Y*2/3))
###############   DEPRECATED: see geneticsd.py                          pygame.display.flip()
###############   DEPRECATED: see geneticsd.py                          #text3Rect = text3.get_rect()
###############   DEPRECATED: see geneticsd.py                          #text3Rect.center = (750+750*3/4, 1000)
###############   DEPRECATED: see geneticsd.py                          good += [list(latent[index].flatten())]
###############   DEPRECATED: see geneticsd.py                      else:
###############   DEPRECATED: see geneticsd.py                          noise.say("Bad click! Click on image.")
###############   DEPRECATED: see geneticsd.py                          noise.runAndWait()
###############   DEPRECATED: see geneticsd.py                          pretty_print("Bad click! Click on image.")
###############   DEPRECATED: see geneticsd.py          
###############   DEPRECATED: see geneticsd.py                  if i.type == pygame.QUIT:
###############   DEPRECATED: see geneticsd.py                      status = False
###############   DEPRECATED: see geneticsd.py           
###############   DEPRECATED: see geneticsd.py          # Covering old images with full circles.
###############   DEPRECATED: see geneticsd.py          for _ in range(123):
###############   DEPRECATED: see geneticsd.py              x = np.random.randint(1500)
###############   DEPRECATED: see geneticsd.py              y = np.random.randint(900)
###############   DEPRECATED: see geneticsd.py              pygame.draw.circle(scrn, darkgreen,
###############   DEPRECATED: see geneticsd.py                                 [x, y], 17, 0)
###############   DEPRECATED: see geneticsd.py          pygame.display.update()
###############   DEPRECATED: see geneticsd.py          if len(indices) == 0:
###############   DEPRECATED: see geneticsd.py              print("The user did not like anything! Rerun :-(")
###############   DEPRECATED: see geneticsd.py              continue
###############   DEPRECATED: see geneticsd.py          print(f"Clicks at {indices}")
###############   DEPRECATED: see geneticsd.py          os.environ["mu"] = str(len(indices))
###############   DEPRECATED: see geneticsd.py          forcedlatents = []
###############   DEPRECATED: see geneticsd.py          bad += [list(latent[u].flatten()) for u in range(len(onlyfiles)) if u not in [i[0] for i in indices]]
###############   DEPRECATED: see geneticsd.py          #sauron = 0 * latent[0]
###############   DEPRECATED: see geneticsd.py          #for u in [u for u in range(len(onlyfiles)) if u not in [i[0] for i in indices]]:
###############   DEPRECATED: see geneticsd.py          #    sauron += latent[u]
###############   DEPRECATED: see geneticsd.py          #sauron = (1 / len([u for u in range(len(onlyfiles)) if u not in [i[0] for i in indices]])) * sauron
###############   DEPRECATED: see geneticsd.py          if len(bad) > 500:
###############   DEPRECATED: see geneticsd.py              bad = bad[(len(bad) - 500):]
###############   DEPRECATED: see geneticsd.py          print(to_native(f"{len(indices)} indices are selected."))
###############   DEPRECATED: see geneticsd.py          #print(f"indices = {indices}")
###############   DEPRECATED: see geneticsd.py          os.environ["good"] = str(good)
###############   DEPRECATED: see geneticsd.py          os.environ["bad"] = str(bad)
###############   DEPRECATED: see geneticsd.py          coefficients = np.zeros(len(indices))
###############   DEPRECATED: see geneticsd.py          numpy_images = [np.array(image) for image in images]
###############   DEPRECATED: see geneticsd.py          for a in range(llambda):
###############   DEPRECATED: see geneticsd.py              voronoi_in_images = False #(a % 2 == 1) and len(good) > 1
###############   DEPRECATED: see geneticsd.py              if voronoi_in_images:
###############   DEPRECATED: see geneticsd.py                  image = np.array(numpy_images[0])
###############   DEPRECATED: see geneticsd.py                  print(f"Voronoi in the image space! {a} / {llambda}")
###############   DEPRECATED: see geneticsd.py                  for i in range(len(indices)):
###############   DEPRECATED: see geneticsd.py                      coefficients[i] = np.exp(np.random.randn())
###############   DEPRECATED: see geneticsd.py                  # Creating a forcedlatent.
###############   DEPRECATED: see geneticsd.py                  for i in range(512):
###############   DEPRECATED: see geneticsd.py                      x = i / 511.
###############   DEPRECATED: see geneticsd.py                      for j in range(512):
###############   DEPRECATED: see geneticsd.py                          y = j / 511 
###############   DEPRECATED: see geneticsd.py                          mindistances = 10000000000.
###############   DEPRECATED: see geneticsd.py                          for u in range(len(indices)):
###############   DEPRECATED: see geneticsd.py                              distance = coefficients[u] * np.linalg.norm( np.array((x, y)) - np.array((indices[u][2], indices[u][1])) )
###############   DEPRECATED: see geneticsd.py                              if distance < mindistances:
###############   DEPRECATED: see geneticsd.py                                  mindistances = distance
###############   DEPRECATED: see geneticsd.py                                  uu = indices[u][0]
###############   DEPRECATED: see geneticsd.py                          image[i][j][:] = numpy_images[uu][i][j][:]
###############   DEPRECATED: see geneticsd.py                  # Conversion before using img2latent
###############   DEPRECATED: see geneticsd.py                  pil_image = Image.fromarray(image)
###############   DEPRECATED: see geneticsd.py                  voronoi_name = f"voronoi{a}_iteration{iteration}.png"
###############   DEPRECATED: see geneticsd.py                  pil_image.save(voronoi_name)
###############   DEPRECATED: see geneticsd.py                  #timage = np.array([image]).astype(np.float32) / 255.0
###############   DEPRECATED: see geneticsd.py                  #timage = timage.transpose(0, 3, 1, 2)
###############   DEPRECATED: see geneticsd.py                  #timage = torch.from_numpy(timage).to(device)
###############   DEPRECATED: see geneticsd.py                  #timage = repeat(timage, '1 ... -> b ...', b=1)
###############   DEPRECATED: see geneticsd.py                  #timage = 2.*timage - 1.
###############   DEPRECATED: see geneticsd.py                  #forcedlatent = model.encode(timage).latent_dist.sample().cpu().detach().numpy().flatten()
###############   DEPRECATED: see geneticsd.py                  #basic_new_fl = np.sqrt(len(forcedlatent) / np.sum(forcedlatent**2)) * forcedlatent
###############   DEPRECATED: see geneticsd.py                  basic_new_fl = randomized_image_to_latent(voronoi_name)  #img_to_latent(voronoi_name)
###############   DEPRECATED: see geneticsd.py                  basic_new_fl = np.sqrt(len(basic_new_fl) / np.sum(basic_new_fl**2)) * basic_new_fl
###############   DEPRECATED: see geneticsd.py                  #basic_new_fl = 0.8 * np.sqrt(len(basic_new_fl) / np.sum(basic_new_fl**2)) * basic_new_fl
###############   DEPRECATED: see geneticsd.py                  if len(good) > 1:
###############   DEPRECATED: see geneticsd.py                      print("Directly copying latent vars !!!")
###############   DEPRECATED: see geneticsd.py                      #forcedlatents += [4.6 * basic_new_fl]
###############   DEPRECATED: see geneticsd.py                      forcedlatents += [basic_new_fl]
###############   DEPRECATED: see geneticsd.py                  else:
###############   DEPRECATED: see geneticsd.py                      epsilon = 1.0 * (((a + .5 - len(good)) / (llambda - len(good) - 1)) ** 2)
###############   DEPRECATED: see geneticsd.py                      forcedlatent = (1. - epsilon) * basic_new_fl.flatten() + epsilon * np.random.randn(4*64*64)
###############   DEPRECATED: see geneticsd.py                      forcedlatent = np.sqrt(len(forcedlatent) / np.sum(forcedlatent**2)) * forcedlatent
###############   DEPRECATED: see geneticsd.py                      forcedlatents += [forcedlatent]
###############   DEPRECATED: see geneticsd.py                      #forcedlatents += [4.6 * forcedlatent]
###############   DEPRECATED: see geneticsd.py              else:
###############   DEPRECATED: see geneticsd.py                  print(f"Voronoi in the latent space! {a} / {llambda}")
###############   DEPRECATED: see geneticsd.py                  forcedlatent = np.zeros((4, 64, 64))
###############   DEPRECATED: see geneticsd.py                      #print(type(numpy_image))
###############   DEPRECATED: see geneticsd.py                      #print(numpy_image.shape)
###############   DEPRECATED: see geneticsd.py                      #print(np.max(numpy_image))
###############   DEPRECATED: see geneticsd.py                      #print(np.min(numpy_image))
###############   DEPRECATED: see geneticsd.py                      #assert False
###############   DEPRECATED: see geneticsd.py                  for i in range(len(indices)):
###############   DEPRECATED: see geneticsd.py                      coefficients[i] = np.exp(np.random.randn())
###############   DEPRECATED: see geneticsd.py                  for i in range(64):
###############   DEPRECATED: see geneticsd.py                      x = i / 63.
###############   DEPRECATED: see geneticsd.py                      for j in range(64):
###############   DEPRECATED: see geneticsd.py                          y = j / 63
###############   DEPRECATED: see geneticsd.py                          mindistances = 10000000000.
###############   DEPRECATED: see geneticsd.py                          for u in range(len(indices)):
###############   DEPRECATED: see geneticsd.py                              #print(a, i, x, j, y, u)
###############   DEPRECATED: see geneticsd.py                              #print(indices[u][1])
###############   DEPRECATED: see geneticsd.py                              #print(indices[u][2])
###############   DEPRECATED: see geneticsd.py                              #print(f"  {coefficients[u]}* np.linalg.norm({np.array((x, y))}-{np.array((indices[u][1], indices[u][2]))}")
###############   DEPRECATED: see geneticsd.py                              distance = coefficients[u] * np.linalg.norm( np.array((x, y)) - np.array((indices[u][2], indices[u][1])) )
###############   DEPRECATED: see geneticsd.py                              if distance < mindistances:
###############   DEPRECATED: see geneticsd.py                                  mindistances = distance
###############   DEPRECATED: see geneticsd.py                                  uu = indices[u][0]
###############   DEPRECATED: see geneticsd.py                          for k in range(4):
###############   DEPRECATED: see geneticsd.py                              assert k < len(forcedlatent), k
###############   DEPRECATED: see geneticsd.py                              assert i < len(forcedlatent[k]), i
###############   DEPRECATED: see geneticsd.py                              assert j < len(forcedlatent[k][i]), j
###############   DEPRECATED: see geneticsd.py                              assert uu < len(latent)
###############   DEPRECATED: see geneticsd.py                              assert k < len(latent[uu]), k
###############   DEPRECATED: see geneticsd.py                              assert i < len(latent[uu][k]), i
###############   DEPRECATED: see geneticsd.py                              assert j < len(latent[uu][k][i]), j
###############   DEPRECATED: see geneticsd.py                              forcedlatent[k][i][j] = float(latent[uu][k][i][j])
###############   DEPRECATED: see geneticsd.py                  #if a % 2 == 0:
###############   DEPRECATED: see geneticsd.py                  #    forcedlatent -= np.random.rand() * sauron
###############   DEPRECATED: see geneticsd.py                  forcedlatent = forcedlatent.flatten()
###############   DEPRECATED: see geneticsd.py                  basic_new_fl = np.sqrt(len(forcedlatent) / np.sum(forcedlatent**2)) * forcedlatent
###############   DEPRECATED: see geneticsd.py                  if len(good) > 1 or len(forcedlatents) < len(good) + 1:
###############   DEPRECATED: see geneticsd.py                      forcedlatents += [basic_new_fl]
###############   DEPRECATED: see geneticsd.py                  else:
###############   DEPRECATED: see geneticsd.py                      epsilon = ((0.5 * (a + .5 - len(good)) / (llambda - len(good) - 1)) ** 2)
###############   DEPRECATED: see geneticsd.py                      forcedlatent = (1. - epsilon) * basic_new_fl.flatten() + epsilon * np.random.randn(4*64*64)
###############   DEPRECATED: see geneticsd.py                      #forcedlatent = np.sqrt(len(forcedlatent) / np.sum(forcedlatent**2)) * forcedlatent
###############   DEPRECATED: see geneticsd.py                      forcedlatents += [forcedlatent]
###############   DEPRECATED: see geneticsd.py          #for uu in range(len(latent)):
###############   DEPRECATED: see geneticsd.py          #    print(f"--> latent[{uu}] sum of sq / variable = {np.sum(latent[uu].flatten()**2) / len(latent[uu].flatten())}")
###############   DEPRECATED: see geneticsd.py          os.environ["good"] = "[]"
###############   DEPRECATED: see geneticsd.py          os.environ["bad"] = "[]"
###############   DEPRECATED: see geneticsd.py                  
###############   DEPRECATED: see geneticsd.py      pygame.quit()
