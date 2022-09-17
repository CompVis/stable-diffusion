---
title: TEXTUAL_INVERSION
---

## **Personalizing Text-to-Image Generation**

You may personalize the generated images to provide your own styles or objects
by training a new LDM checkpoint and introducing a new vocabulary to the fixed
model as a (.pt) embeddings file. Alternatively, you may use or train
HuggingFace Concepts embeddings files (.bin) from
<https://huggingface.co/sd-concepts-library> and its associated notebooks.

## **Training**

To train, prepare a folder that contains images sized at 512x512 and execute the
following:

### WINDOWS

As the default backend is not available on Windows, if you're using that
platform, set the environment variable `PL_TORCH_DISTRIBUTED_BACKEND` to `gloo`

```bash
python3 ./main.py --base ./configs/stable-diffusion/v1-finetune.yaml \
                  --actual_resume ./models/ldm/stable-diffusion-v1/model.ckpt \
                  -t \
                  -n my_cat \
                  --gpus 0 \
                  --data_root D:/textual-inversion/my_cat \
                  --init_word 'cat'
```

During the training process, files will be created in
`/logs/[project][time][project]/` where you can see the process.

Conditioning contains the training prompts inputs, reconstruction the input
images for the training epoch samples, samples scaled for a sample of the prompt
and one with the init word provided.

On a RTX3090, the process for SD will take ~1h @1.6 iterations/sec.

!!! Info _Note_

    According to the associated paper, the optimal number of
    images is 3-5. Your model may not converge if you use more images than
    that.

Training will run indefinitely, but you may wish to stop it (with ctrl-c) before
the heat death of the universe, when you find a low loss epoch or around ~5000
iterations. Note that you can set a fixed limit on the number of training steps
by decreasing the "max_steps" option in
configs/stable_diffusion/v1-finetune.yaml (currently set to 4000000)

## **Run the Model**

Once the model is trained, specify the trained .pt or .bin file when starting
dream using

```bash
python3 ./scripts/dream.py --embedding_path /path/to/embedding.pt
```

Then, to utilize your subject at the dream prompt

```bash
dream> "a photo of *"
```

This also works with image2image

```bash
dream> "waterfall and rainbow in the style of *" --init_img=./init-images/crude_drawing.png --strength=0.5 -s100 -n4
```

For .pt files it's also possible to train multiple tokens (modify the
placeholder string in `configs/stable-diffusion/v1-finetune.yaml`) and combine
LDM checkpoints using:

```bash
python3 ./scripts/merge_embeddings.py \
        --manager_ckpts /path/to/first/embedding.pt \
        [</path/to/second/embedding.pt>,[...]] \
        --output_path /path/to/output/embedding.pt
```

Credit goes to rinongal and the repository

Please see [the repository](https://github.com/rinongal/textual_inversion) and
associated paper for details and limitations.
