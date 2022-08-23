conda activate ldm
pip install -e .
$env:PL_TORCH_DISTRIBUTED_BACKEND="gloo"
python ./main.py --base ./configs/stable-diffusion/v1-finetune.yaml `
                 -t `
                 --actual_resume ./models/ldm/stable-diffusion-v1/model.ckpt `
                 -n my_cat `
                 --gpus 0, `
                 --data_root D:/textual-inversion/my_cat `
                 --init_Word 'Isla Fisher'
# python ./scripts/train_personalization.py --base ./configs/stable-diffusion/v1-finetune.yaml `
#                                           -t `
#                                           --actual_resume ../stable-diffusion-dream/models/ldm/stable-diffusion-v1/model.ckpt `
#                                           --gpus 1 `
#                                           --data_root D:/textual-inversion/isla_fisher `
#                                           --resume 'logs/my_cat2022-08-23T01-46-37_my_cat' `
#                                           --init_Word 'cat'
