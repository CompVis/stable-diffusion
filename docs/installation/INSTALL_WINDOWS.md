# **Windows Installation**

## **Notebook install (semi-automated)**

We have a [Jupyter
notebook](https://github.com/lstein/stable-diffusion/blob/main/notebooks/Stable-Diffusion-local-Windows.ipynb)
with cell-by-cell installation steps. It will download the code in
this repo as one of the steps, so instead of cloning this repo, simply
download the notebook from the link above and load it up in VSCode
(with the appropriate extensions installed)/Jupyter/JupyterLab and
start running the cells one-by-one.

Note that you will need NVIDIA drivers, Python 3.10, and Git installed
beforehand - simplified [step-by-step
instructions](https://github.com/lstein/stable-diffusion/wiki/Easy-peasy-Windows-install)
are available in the wiki (you'll only need steps 1, 2, & 3 ).

## **Manual Install**

### **pip**

See [Easy-peasy Windows install](https://github.com/lstein/stable-diffusion/wiki/Easy-peasy-Windows-install)
in the wiki

### **Conda**

1. Install Anaconda3 (miniconda3 version) from here: https://docs.anaconda.com/anaconda/install/windows/

2. Install Git from here: https://git-scm.com/download/win

3. Launch Anaconda from the Windows Start menu. This will bring up a command window. Type all the remaining commands in this window.

4. Run the command:

```
git clone https://github.com/lstein/stable-diffusion.git
```

This will create stable-diffusion folder where you will follow the rest of the steps.

5. Enter the newly-created stable-diffusion folder. From this step forward make sure that you are working in the stable-diffusion directory!

```
cd stable-diffusion
```

6. Run the following two commands:

```
conda env create -f environment.yaml    (step 6a)
conda activate ldm                      (step 6b)
```

This will install all python requirements and activate the "ldm"
environment which sets PATH and other environment variables properly.

7. Run the command:

```
python scripts\preload_models.py
```

This installs several machine learning models that stable diffusion requires.

Note: This step is required. This was done because some users may might be blocked by firewalls or have limited internet connectivity for the models to be downloaded just-in-time.

8. Now you need to install the weights for the big stable diffusion model.

- For running with the released weights, you will first need to set up an acount with Hugging Face (https://huggingface.co).
- Use your credentials to log in, and then point your browser at https://huggingface.co/CompVis/stable-diffusion-v-1-4-original.
- You may be asked to sign a license agreement at this point.
- Click on "Files and versions" near the top of the page, and then click on the file named `sd-v1-4.ckpt`. You'll be taken to a page that
  prompts you to click the "download" link. Now save the file somewhere safe on your local machine.
- The weight file is >4 GB in size, so
  downloading may take a while.

Now run the following commands from **within the stable-diffusion directory** to copy the weights file to the right place:

```
mkdir -p models\ldm\stable-diffusion-v1
copy C:\path\to\sd-v1-4.ckpt models\ldm\stable-diffusion-v1\model.ckpt
```

Please replace `C:\path\to\sd-v1.4.ckpt` with the correct path to wherever you stashed this file. If you prefer not to copy or move the .ckpt file,
you may instead create a shortcut to it from within `models\ldm\stable-diffusion-v1\`.

9. Start generating images!

```
# for the pre-release weights
python scripts\dream.py -l

# for the post-release weights
python scripts\dream.py
```

10. Subsequently, to relaunch the script, first activate the Anaconda command window (step 3),enter the stable-diffusion directory (step 5, `cd \path\to\stable-diffusion`), run `conda activate ldm` (step 6b), and then launch the dream script (step 9).

**Note:** Tildebyte has written an alternative ["Easy peasy Windows
install"](https://github.com/lstein/stable-diffusion/wiki/Easy-peasy-Windows-install)
which uses the Windows Powershell and pew. If you are having trouble with Anaconda on Windows, give this a try (or try it first!)

### Updating to newer versions of the script

This distribution is changing rapidly. If you used the `git clone` method (step 5) to download the stable-diffusion directory, then to update to the latest and greatest version, launch the Anaconda window, enter `stable-diffusion`, and type:

```
git pull
conda env update -f environment.yaml
```

This will bring your local copy into sync with the remote one.
