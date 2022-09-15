---
title: Linux
---

1. You will need to install the following prerequisites if they are not already
   available. Use your operating system's preferred installer.

    - Python (version 3.8.5 recommended; higher may work)
    - git

2. Install the Python Anaconda environment manager.

    ```bash
    ~$  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    ~$  chmod +x Anaconda3-2022.05-Linux-x86_64.sh
    ~$  ./Anaconda3-2022.05-Linux-x86_64.sh
    ```

    After installing anaconda, you should log out of your system and log back in. If
    the installation worked, your command prompt will be prefixed by the name of the
    current anaconda environment - `(base)`.

3. Copy the stable-diffusion source code from GitHub:

    ```bash
    (base) ~$ git clone https://github.com/lstein/stable-diffusion.git
    ```

    This will create stable-diffusion folder where you will follow the rest of the
    steps.

4. Enter the newly-created stable-diffusion folder. From this step forward make
   sure that you are working in the stable-diffusion directory!

    ```bash
    (base) ~$ cd stable-diffusion
    (base) ~/stable-diffusion$
    ```

5. Use anaconda to copy necessary python packages, create a new python
   environment named `ldm` and activate the environment.

    ```bash
    (base) ~/stable-diffusion$ conda env create -f environment.yaml
    (base) ~/stable-diffusion$ conda activate ldm
    (ldm) ~/stable-diffusion$
    ```

    After these steps, your command prompt will be prefixed by `(ldm)` as shown
    above.

6. Load a couple of small machine-learning models required by stable diffusion:

    ```bash
    (ldm) ~/stable-diffusion$ python3 scripts/preload_models.py
    ```

    Note that this step is necessary because I modified the original just-in-time
    model loading scheme to allow the script to work on GPU machines that are not
    internet connected. See [Preload Models](../features/OTHER.md#preload-models)

7. Now you need to install the weights for the stable diffusion model.

      - For running with the released weights, you will first need to set up an acount
        with [Hugging Face](https://huggingface.co).
      - Use your credentials to log in, and then point your browser [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original.)
      - You may be asked to sign a license agreement at this point.
      - Click on "Files and versions" near the top of the page, and then click on the
        file named "sd-v1-4.ckpt". You'll be taken to a page that prompts you to click
        the "download" link. Save the file somewhere safe on your local machine.

      Now run the following commands from within the stable-diffusion directory.
      This will create a symbolic link from the stable-diffusion model.ckpt file, to
      the true location of the `sd-v1-4.ckpt` file.

    ```bash
    (ldm) ~/stable-diffusion$ mkdir -p models/ldm/stable-diffusion-v1
    (ldm) ~/stable-diffusion$ ln -sf /path/to/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
    ```

8. Start generating images!

    ```bash
    # for the pre-release weights use the -l or --liaon400m switch
    (ldm) ~/stable-diffusion$ python3 scripts/dream.py -l

    # for the post-release weights do not use the switch
    (ldm) ~/stable-diffusion$ python3 scripts/dream.py

    # for additional configuration switches and arguments, use -h or --help
    (ldm) ~/stable-diffusion$ python3 scripts/dream.py -h
    ```

9. Subsequently, to relaunch the script, be sure to run "conda activate ldm"
   (step 5, second command), enter the `stable-diffusion` directory, and then
   launch the dream script (step 8). If you forget to activate the ldm
   environment, the script will fail with multiple `ModuleNotFound` errors.

    ### Updating to newer versions of the script

    This distribution is changing rapidly. If you used the `git clone` method
    (step 5) to download the stable-diffusion directory, then to update to the
    latest and greatest version, launch the Anaconda window, enter
    `stable-diffusion` and type:

    ```bash
    (ldm) ~/stable-diffusion$ git pull
    ```

    This will bring your local copy into sync with the remote one.
