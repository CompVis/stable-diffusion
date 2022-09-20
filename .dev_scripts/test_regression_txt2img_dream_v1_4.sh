# generate an image
PROMPT_FILE=".dev_scripts/sample_command.txt"
OUT_DIR="outputs/img-samples/test_regression_txt2img_v1_4"
SAMPLES_DIR=${OUT_DIR}
python scripts/dream.py \
    --from_file ${PROMPT_FILE} \
    --outdir ${OUT_DIR} \
    --sampler plms

# original output by CompVis/stable-diffusion
IMAGE1=".dev_scripts/images/v1_4_astronaut_rides_horse_plms_step50_seed42.png"
# new output
IMAGE2=`ls -A ${SAMPLES_DIR}/*.png | sort | tail -n 1`

echo ""
echo "comparing the following two images"
echo "IMAGE1: ${IMAGE1}"
echo "IMAGE2: ${IMAGE2}"
python .dev_scripts/diff_images.py ${IMAGE1} ${IMAGE2}
