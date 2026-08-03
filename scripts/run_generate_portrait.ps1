# Run this from a PowerShell prompt in the repository root.
# Usage: Open PowerShell as Administrator (if needed) and run:
#   .\scripts\run_generate_portrait.ps1

Set-StrictMode -Version Latest

function Check-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Error "Python not found. Install Python 3.10+ from https://www.python.org/downloads/ and re-run this script."
        exit 1
    }
    return $py.Path
}

$python = Check-Python
$venvDir = ".\.venv"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment in $venvDir..."
    & $python -m venv $venvDir
} else {
    Write-Host "Using existing virtual environment: $venvDir"
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtualenv python not found at $venvPython"
    exit 1
}

Write-Host "Upgrading pip in venv..."
& $venvPython -m pip install --upgrade pip

# Install common packages. NOTE: torch installation often requires choosing the correct CUDA wheel.
$commonPkgs = @(
    "accelerate",
    "transformers",
    "diffusers[torch]",
    "safetensors",
    "xformers",
    "Pillow"
)

Write-Host "Installing common Python packages (may take a while)..."
try {
    & $venvPython -m pip install $commonPkgs
} catch {
    Write-Warning "Automatic installation of some packages failed. You may need to install torch manually for your CUDA/cuDNN setup."
    Write-Host "If you have an NVIDIA GPU, follow https://pytorch.org/get-started/locally/ to install a matching torch+torchvision+torchaudio wheel."
}

# Prepared prompt and negative prompt (artistic, adult, non-explicit)
$prompt = "A hyper-realistic oil-painting style full-length vertical portrait of a stunning adult Indian woman, standing before an ornate floor-to-ceiling mirror, viewed from behind in a graceful over-the-shoulder pose; ultra-fine sheer lace lingerie (artistic, non-explicit), cinematic golden-hour window light, elegant shadows, rich texture on skin and fabric, sophisticated dim boutique bedroom, shallow depth of field, ultra-detailed, photorealistic, oil painting finish"

$negative = "explicit nudity, genitals, sexual act, minors, lowres, text, watermark, blurred, nsfw"

# Output settings
$width = 512
$height = 768
$steps = 30
$cfg = 7.5
$sampler = "DPM++ 2M Karras"
$seed = -1
$outdir = "outputs"

if (-not (Test-Path $outdir)) { New-Item -ItemType Directory -Path $outdir | Out-Null }

# Command to run the txt2img script. Adjust argument names if your local script expects different flags.
$cmd = @(
    "$venvPython",
    ".\scripts\txt2img.py",
    "--prompt", "`"$prompt`"",
    "--negative_prompt", "`"$negative`"",
    "--W", $width,
    "--H", $height,
    "--steps", $steps,
    "--scale", $cfg,
    "--sampler", "`"$sampler`"",
    "--seed", $seed,
    "--outdir", "`"$outdir`"",
    "--n_samples", 1
) -join ' '

Write-Host "\nReady to generate. About to run txt2img with the prepared prompt."
Write-Host "If your local txt2img script uses different argument names, open scripts\run_generate_portrait.ps1 and edit the command string in the \$cmd variable."
Write-Host "Running command:\n$cmd\n"

# Run the command
Invoke-Expression $cmd

Write-Host "Done. Check the $outdir folder for output images."