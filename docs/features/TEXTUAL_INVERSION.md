# **Personalizing Text-to-Image Generation**

You may personalize the generated images to provide your own styles or objects by training a new LDM checkpoint and introducing a new vocabulary to the fixed model as a (.pt) embeddings file. Alternatively, you may use or train HuggingFace Concepts embeddings files (.bin) from https://huggingface.co/sd-concepts-library and its associated notebooks.

**Training**

To train, prepare a folder that contains images sized at 512x512 and execute the following:

**WINDOWS**: As the default backend is not available on Windows, if you're using that platform, set the environment variable `PL_TORCH_DISTRIBUTED_BACKEND=gloo`

```
(ldm) ~/stable-diffusion$ python3 ./main.py --base ./configs/stable-diffusion/v1-finetune.yaml \
                                            -t \
                                            --actual_resume ./models/ldm/stable-diffusion-v1/model.ckpt \
                                            -n my_cat \
                                            --gpus 0, \
                                            --data_root D:/textual-inversion/my_cat \
                                            --init_word 'cat'
```

During the training process, files will be created in
/logs/[project][time][project]/ where you can see the process.

Conditioning contains the training prompts inputs, reconstruction the
input images for the training epoch samples, samples scaled for a
sample of the prompt and one with the init word provided.

On a RTX3090, the process for SD will take ~1h @1.6 iterations/sec.

_Note_: According to the associated paper, the optimal number of
images is 3-5. Your model may not converge if you use more images than
that.

Training will run indefinitely, but you may wish to stop it (with
ctrl-c) before the heat death of the universe, when you find a low
loss epoch or around ~5000 iterations. Note that you can set a fixed
limit on the number of training steps by decreasing the "max_steps"
option in configs/stable_diffusion/v1-finetune.yaml (currently set to
4000000)

**Running**

Once the model is trained, specify the trained .pt or .bin file when
starting dream using

```
(ldm) ~/stable-diffusion$ python3 ./scripts/dream.py --embedding_path /path/to/embedding.pt --full_precision
```

Then, to utilize your subject at the dream prompt

```
dream> "a photo of *"
```

This also works with image2image

```
dream> "waterfall and rainbow in the style of *" --init_img=./init-images/crude_drawing.png --strength=0.5 -s100 -n4
```

For .pt files it's also possible to train multiple tokens (modify the placeholder string in `configs/stable-diffusion/v1-finetune.yaml`) and combine LDM checkpoints using:

```
(ldm) ~/stable-diffusion$ python3 ./scripts/merge_embeddings.py \
                                            --manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...] \
                                            --output_path /path/to/output/embedding.pt
```

Credit goes to rinongal and the repository located at https://github.com/rinongal/textual_inversion Please see the repository and associated paper for details and limitations.
