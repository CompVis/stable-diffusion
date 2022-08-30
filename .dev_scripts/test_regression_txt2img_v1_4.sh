# generate an image
PROMPT="a photograph of an astronaut riding a horse"
OUT_DIR="outputs/txt2img-samples/test_regression_txt2img_v1_4"
SAMPLES_DIR="outputs/txt2img-samples/test_regression_txt2img_v1_4/samples"
python scripts/orig_scripts/txt2img.py \
    --prompt "${PROMPT}" \
    --outdir ${OUT_DIR} \
    --plms \
    --ddim_steps 50 \
    --n_samples 1 \
    --n_iter 1 \
    --seed 42

# original output by CompVis/stable-diffusion
IMAGE1=".dev_scripts/images/v1_4_astronaut_rides_horse_plms_step50_seed42.png"
# new output
IMAGE2=`ls -A ${SAMPLES_DIR}/*.png | sort | tail -n 1`

echo ""
echo "comparing the following two images"
echo "IMAGE1: ${IMAGE1}"
echo "IMAGE2: ${IMAGE2}"
python .dev_scripts/diff_images.py ${IMAGE1} ${IMAGE2}
