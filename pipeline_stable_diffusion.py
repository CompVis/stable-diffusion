# Modification of the original file by O. Teytaud for facilitating genetic stable diffusion.

import inspect
import os
import numpy as np
import random
import warnings
from typing import List, Optional, Union

import torch

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker


class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

#    def get_latent(self, image):
#        return self.vae.encode(image)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        latents_intermediate_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_intermediate_shape,
                generator=generator,
                device=latents_device,
            )
            if len(os.environ["forcedlatent"]) > 0:
                print("we get a forcing for the latent z.")
                latents = np.array(eval(os.environ["forcedlatent"]))
                latents = np.sqrt(len(latents)) * latents / np.sqrt(np.sum(latents ** 2))
                latents = torch.from_numpy(np.array(eval(os.environ["forcedlatent"])).reshape((1,4,64,64)))
            good = eval(os.environ["good"])
            bad = eval(os.environ["bad"])
            print(f"{len(good)} good and {len(bad)} bad")
            i_believe_in_evolution = len(good) > 0 and len(bad) > 0
            #i_believe_in_evolution = False
            print(f"I believe in evolution = {i_believe_in_evolution}")
            if i_believe_in_evolution: 
                from sklearn import tree
                from sklearn.neural_network import MLPClassifier
                #from sklearn.neighbors import NearestCentroid
                from sklearn.linear_model import LogisticRegression
                #z = (np.random.randn(4*64*64))
                z = latents.cpu().numpy().flatten()
                if os.environ.get("skl", "tree") == "tree":
                    clf = tree.DecisionTreeClassifier()#min_samples_split=0.1)
                elif os.environ.get("skl", "tree") == "logit":
                    clf = LogisticRegression()
                else:
                    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
                #clf = NearestCentroid()
                
                
                
                X=good + bad
                Y = [1] * len(good) + [0] * len(bad)
                clf = clf.fit(X,Y)
                epsilon = 0.0001 # for astronauts
                epsilon = 1.0
                
                def loss(x):
                        return clf.predict_proba([x])[0][0]  # for astronauts
                        #return clf.predict_proba([(1-epsilon)*z+epsilon*x])[0][0]  # for astronauts
                        #return clf.predict_proba([z+epsilon*x])[0][0]
                
                
                if i_believe_in_evolution:
                    import nevergrad as ng
                    budget = int(os.environ.get("budget", "300"))
                    #nevergrad_optimizer = ng.optimizers.RandomSearch(len(z), budget)
                    #nevergrad_optimizer = ng.optimizers.RandomSearch(len(z), budget)
                    optim_class = ng.optimizers.registry[os.environ.get("ngoptim", "DiscreteLenglerOnePlusOne")]
                    #nevergrad_optimizer = ng.optimizers.DiscreteLenglerOnePlusOne(len(z), budget)
                    nevergrad_optimizer = optim_class(len(z), budget)
                    #nevergrad_optimizer = ng.optimizers.DiscreteOnePlusOne(len(z), budget)
#                    for k in range(5):
#                        z1 = np.array(random.choice(good))
#                        z2 = np.array(random.choice(good))
#                        z3 = np.array(random.choice(good))
#                        z4 = np.array(random.choice(good))
#                        z5 = np.array(random.choice(good))
#                        #z = 0.99 * z1 + 0.01 * (z2+z3+z4+z5)/4.
#                        z = 0.2 * (z1 + z2 + z3 + z4 + z5)
#                        mu = int(os.environ.get("mu", "5"))
#                        parents = [z1, z2, z3, z4, z5]
#                        weights = [np.exp(np.random.randn() - i * float(os.environ.get("decay", "1."))) for i in range(5)]
#                        z = weights[0] * z1
#                        for u in range(mu):
#                            if u > 0:
#                                z += weights[u] * parents[u]
#                        z = (1. / sum(weights[:mu])) * z
#                        z = np.sqrt(len(z)) * z / np.linalg.norm(z)
#
#                        #for u in range(len(z)):
#                        #    z[u] = random.choice([z1[u],z2[u],z3[u],z4[u],z5[u]])
#                        nevergrad_optimizer.suggest
                    if len(os.environ["forcedlatent"]) > 0:
                        print("we get a forcing for the latent z.")
                        z0 = eval(os.environ["forcedlatent"])
                        #nevergrad_optimizer.suggest(eval(os.environ["forcedlatent"]))
                    else:
                        z0 = z
                    for i in range(budget):
                        x = nevergrad_optimizer.ask()
                        z = z0 + float(os.environ.get("epsilon", "0.001")) * x.value
                        z = np.sqrt(len(z)) * z / np.linalg.norm(z)
                        l = loss(z)
                        nevergrad_optimizer.tell(x, l)
                        if np.log2(i+1) == int(np.log2(i+1)):
                          print(f"iteration {i} --> {l}")
                          print("var/variable = ", sum(z**2)/len(z))
                        #z = (1.-epsilon) * z + epsilon * x / np.sqrt(np.sum(x ** 2))
                        if l < 0.0000001 and os.environ.get("earlystop", "False") in ["true", "True"]:
                                print(f"we find proba(bad)={l}")
                                break
                    x = nevergrad_optimizer.recommend().value
                    z = z0 + float(os.environ.get("epsilon", "0.001")) * x
                    z = np.sqrt(len(z)) * z / np.linalg.norm(z)
                    latents = torch.from_numpy(z.reshape(latents_intermediate_shape)).float() #.half()
        else:
            if latents.shape != latents_intermediate_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_intermediate_shape}")
        print(f"latent ==> {sum(latents.flatten()**2) / len(latents.flatten())}")
        os.environ["latent_sd"] = str(list(latents.flatten().cpu().numpy()))
        for i in [2, 3]:
            latents = torch.repeat_interleave(latents, repeats=latents_shape[i] // latents_intermediate_shape[i], dim=i) #/ np.sqrt(np.sqrt(latents_shape[i] // latents_intermediate_shape[i]))
        latents = latents.float().to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
