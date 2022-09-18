---
title: Prompting Features
---

# :octicons-command-palette-24: Prompting Features

## **Reading Prompts from a File**

You can automate `dream.py` by providing a text file with the prompts you want to run, one line per
prompt. The text file must be composed with a text editor (e.g. Notepad) and not a word processor.
Each line should look like what you would type at the dream> prompt:

```bash
a beautiful sunny day in the park, children playing -n4 -C10
stormy weather on a mountain top, goats grazing     -s100
innovative packaging for a squid's dinner           -S137038382
```

Then pass this file's name to `dream.py` when you invoke it:

```bash
(ldm) ~/stable-diffusion$ python3 scripts/dream.py --from_file "path/to/prompts.txt"
```

You may read a series of prompts from standard input by providing a filename of `-`:

```bash
(ldm) ~/stable-diffusion$ echo "a beautiful day" | python3 scripts/dream.py --from_file -
```

---

## **Weighted Prompts**

You may weight different sections of the prompt to tell the sampler to attach different levels of
priority to them, by adding `:(number)` to the end of the section you wish to up- or downweight. For
example consider this prompt:

```bash
tabby cat:0.25 white duck:0.75 hybrid
```

This will tell the sampler to invest 25% of its effort on the tabby cat aspect of the image and 75%
on the white duck aspect (surprisingly, this example actually works). The prompt weights can use any
combination of integers and floating point numbers, and they do not need to add up to 1.

---

## **Negative and Unconditioned Prompts**

Any words between a pair of square brackets will try and be ignored by Stable Diffusion's model during generation of images.

```bash
this is a test prompt [not really] to make you understand [cool] how this works.
```

In the above statement, the words 'not really cool` will be ignored by Stable Diffusion.

Here's a prompt that depicts what it does.

original prompt:

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

![step1](../assets/negative_prompt_walkthru/step1.png)

That image has a woman, so if we want the horse without a rider, we can influence the image not to have a woman by putting [woman] in the prompt, like this:

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve [woman]" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

![step2](../assets/negative_prompt_walkthru/step2.png)

That's nice - but say we also don't want the image to be quite so blue. We can add "blue" to the list of negative prompts, so it's now [woman blue]:

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve [woman blue]" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

![step3](../assets/negative_prompt_walkthru/step3.png)

Getting close - but there's no sense in having a saddle when our horse doesn't have a rider, so we'll add one more negative prompt: [woman blue saddle].

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve [woman blue saddle]" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

![step4](../assets/negative_prompt_walkthru/step4.png)

!!! notes "Notes about this feature:"

    * The only requirement for words to be ignored is that they are in between a pair of square brackets.
    * You can provide multiple words within the same bracket.
    * You can provide multiple brackets with multiple words in different places of your prompt. That works just fine.
    * To improve typical anatomy problems, you can add negative prompts like `[bad anatomy, extra legs, extra arms, extra fingers, poorly drawn hands, poorly drawn feet, disfigured, out of frame, tiling, bad art, deformed, mutated]`.
