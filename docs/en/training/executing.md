# 3. Executing

There are two modes of executing the training:
1. Using docker image. This is the fastest way to get started.
2. Using system python install. Allows more customization.

Note: You will need to provide the initial checkpoint for resuming the training. This must be a version with the full EMA. Otherwise you will get this error:
```
RuntimeError: Error(s) in loading state_dict for LatentDiffusion:
  Missing key(s) in state_dict: "model_ema.diffusion_modeltime_embed0weight", "model_ema.diffusion_modeltime_embed0bias".... (Many lines of similar outputs)
```

## 1. Using docker image

An image is provided at `ghcr.io/derfred/waifu-diffusion`. Execute it using by adjusting the NUM_GPU variable:
```
docker run -it -e NUM_GPU=x ghcr.io/derfred/waifu-diffusion
```

Next you will want to download the starting checkpoint into the file `model.ckpt` and copy the training data in the directory `/waifu/danbooru-aesthetic`.

Finally execute the training using:
```
sh train.sh -t -n "aesthetic" --resume_from_checkpoint model.ckpt --base ./configs/stable-diffusion/v1-finetune-4gpu.yaml --no-test --seed 25 --scale_lr False --data_root "./danbooru-aesthetic"
```

## 2. system python install

First install the dependencies:
```bash
pip install -r requirements.txt
```

Next you will want to download the starting checkpoint into the file `model.ckpt` and copy the training data in the directory `/waifu/danbooru-aesthetic`.

Also you will need to edit the configuration in `./configs/stable-diffusion/v1-finetune-4gpu.yaml`. In the `data` section (around line 70) change the `batch_size` and `num_workers` to the number of GPUs you are using:
```
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 4
    num_workers: 4
    wrap: false
```

Finally execute the training using the following command. You need to adjust the `--gpu` parameter according to your GPU settings.
```bash
sh train.sh -t -n "aesthetic" --resume_from_checkpoint model.ckpt --base ./configs/stable-diffusion/v1-finetune-4gpu.yaml --no-test --seed 25 --scale_lr False --data_root "./danbooru-aesthetic" --gpu=0,1,2,3,
```

In case you get an error stating `KeyError: 'Trying to restore optimizer state but checkpoint contains only the model. This is probably due to ModelCheckpoint.save_weights_only being set to True.'` follow these instructions: https://discord.com/channels/930499730843250783/953132470528798811/1018668937052962908
