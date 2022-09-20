def parameters_to_metadata(opt,
                   seeds=[],
                   model_hash=None,
                   postprocessing=None):
    '''
    Given an Args object, returns a dict containing the keys and
    structure of the proposed stable diffusion metadata standard
    https://github.com/lstein/stable-diffusion/discussions/392
    This is intended to be turned into JSON and stored in the 
    "sd
    '''

    # top-level metadata minus `image` or `images`
    metadata = {
        'model'       : 'stable diffusion',
        'model_id'    : opt.model,
        'model_hash'  : model_hash,
        'app_id'      : APP_ID,
        'app_version' : APP_VERSION,
    }

    # add some RFC266 fields that are generated internally, and not as
    # user args
    image_dict = opt.to_dict(
        postprocessing=postprocessing
    )

    # 'postprocessing' is either null or an array of postprocessing metadatal
    if postprocessing:
        # TODO: This is just a hack until postprocessing pipeline work completed
        image_dict['postprocessing'] = []

        if image_dict['gfpgan_strength'] and image_dict['gfpgan_strength'] > 0:
            image_dict['postprocessing'].append('GFPGAN (not RFC compliant)')
        if image_dict['upscale'] and image_dict['upscale'][0] > 0:
            image_dict['postprocessing'].append('ESRGAN (not RFC compliant)')
    else:
        image_dict['postprocessing'] = None

    # remove any image keys not mentioned in RFC #266
    rfc266_img_fields = ['type','postprocessing','sampler','prompt','seed','variations','steps',
                         'cfg_scale','step_number','width','height','extra','strength']

    rfc_dict ={}

    for item in image_dict.items():
        key,value = item
        if key in rfc266_img_fields:
            rfc_dict[key] = value

    # semantic drift
    rfc_dict['sampler']  = image_dict.get('sampler_name',None)

    # display weighted subprompts (liable to change)
    if opt.prompt:
        subprompts = split_weighted_subprompts(opt.prompt)
        subprompts = [{'prompt':x[0],'weight':x[1]} for x in subprompts]
        rfc_dict['prompt'] = subprompts

    # 'variations' should always exist and be an array, empty or consisting of {'seed': seed, 'weight': weight} pairs
    rfc_dict['variations'] = [{'seed':x[0],'weight':x[1]} for x in opt.with_variations] if opt.with_variations else []

    if opt.init_img:
        rfc_dict['type']           = 'img2img'
        rfc_dict['strength_steps'] = rfc_dict.pop('strength')
        rfc_dict['orig_hash']      = calculate_init_img_hash(opt.init_img)
        rfc_dict['sampler']        = 'ddim'  # TODO: FIX ME WHEN IMG2IMG SUPPORTS ALL SAMPLERS
    else:
        rfc_dict['type']  = 'txt2img'
        rfc_dict.pop('strength')

    if len(seeds)==0 and opt.seed:
        seeds=[seed]

    if opt.grid:
        images = []
        for seed in seeds:
            rfc_dict['seed'] = seed
            images.append(copy.copy(rfc_dict))
        metadata['images'] = images
    else:
        # there should only ever be a single seed if we did not generate a grid
        assert len(seeds) == 1, 'Expected a single seed'
        rfc_dict['seed'] = seeds[0]
        metadata['image'] = rfc_dict

    return metadata
