# Clean Before You Code: Denoising in Neural Audio Codec Continuous Latent Space

Code from my MSc dissertation on speech denoising in the neural audio codec latent space, using Descript Audio Codec (DAC).

- Denoising is performed in the **pre-quantised codec latent space**, using a pretrained DAC encoder that already provides speech compression and a task-relevant representation (computationally efficient).
- To the best of my knowledge (as of Aug 2025), no prior work uses **adversarial (GAN) objectives** to train a denoiser that operates directly on a codec’s pre-quantised latents.
- I show that **GAN-based latent-only training** can produce effective denoising while keeping training **memory requirements low**.

DAC: [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)


## What’s in this repo
- Latent denoiser: `src/denoise_module.py`
- Training: `src/train.py`, `src/dataset.py`
- Inference script: `src/denoise.py` (denoise a folder of noisy WAVs)

## Example
One example from the [2020 DNS](https://arxiv.org/pdf/2005.13981) dev test set illustrating denoising quality.

### Clean
![Clean spectrogram](assets/babble_CLEAN.png)  
[Download clean sample (WAV)](assets/babble_CLEAN.wav)

### Noisy (noise-added)
![Noisy spectrogram](assets/babble_NOISY.png)  
[Download noisy sample (WAV)](assets/babble_NOISY.wav)

### Denoised (noise-removed)
![Denoised spectrogram](assets/babble_DENOISED.png)  
[Download denoised sample (WAV)](assets/babble_DENOISED.wav)

## Denoising performance
[DNSMOS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746108) comparison against published results (and speaker similarity) on the DNS2020 dev test set. DNS scores from 1-5, and Speaker similarity from 0-1.

![DNSMOS comparison](assets/DNSMOS_compare.png)

Our (Large) denoising method compared with:
- DEMUCS (Defossez et al., 2020) — [paper](https://arxiv.org/pdf/2006.12847)
- FRCRN (Zhao et al., 2022) — [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747578)
- SELM (Wang et al., 2024) — [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10447464)
- MaskSR (Li et al., 2024) — [paper](https://arxiv.org/pdf/2406.02092)
- LatentSE (Li et al., 2025) — [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10890379)

## Requirements
- Python 3.7
- All dependencies in `requirements.txt`
- A directory of (noisy) 16 kHz WAV files

## Quickstart example: denoise a folder
1) **Create a Python 3.7 environment and install dependencies (example uses `uv`)**
- (install uv (if needed): https://docs.astral.sh/uv/)
```bash
# create and activate a venv
uv venv --python 3.7

# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# install dependencies
uv pip install -r requirements.txt
```

2) **Download weights**
- `codec_weights`: DAC 16 kHz weights (see DAC repo above)
- `denoiser_ckpt`: pretrained weights for Medium (9.5M params) and Large (39.4M params) available on request

3) **Prepare a config file (example: `conf/denoise.yml`)**

- Input WAVs must be 16 kHz; the script raises an error if sample rate differs

4) **Run:**
```bash
python src/denoise.py --config conf/denoise.yml
```

Outputs are written to `output_dir` with the same filenames as the inputs.

## Training (research code)
Training is included mainly for reproducibility and is not "turn key". 
