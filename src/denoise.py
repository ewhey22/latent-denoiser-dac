import math
import argparse
import yaml
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from denoise_module import Denoiser
from dac.model.dac import DAC


# Constants tied to the pre-trained DAC configuration
SAMPLE_RATE = 16000
HOP_LENGTH = 320
LATENT_CHANNELS = 1024


def parse_args():
    p = argparse.ArgumentParser(description="Do denoising")
    p.add_argument("--config", "-c",
        type=str, required=True,
        help="Path to YAML config file")
    return p.parse_args()


class FullDenoiser(nn.Module):
    
    def __init__(self, codec, denoiser):
        super().__init__()
        self.codec    = codec
        self.denoiser = denoiser

    def forward(self, wav):
        # encode audio into continuous latent representations. DAC-16 kHz transforms 1 channel waveform into 1024-channel latent frames.
        latents = self.codec.encoder(wav)
       
        # denoise latents
        g_out = self.denoiser(latents)

        # quantize
        z, *_ = self.codec.quantizer(g_out)

        # decode
        recon = self.codec.decoder(z)
        return recon


@torch.no_grad()
def run_denoise(full_forward, wav, codec, bypass_denoiser=False):
    """Run a single waveform through the codec + denoiser stack."""
    wav = wav.unsqueeze(0)
    if bypass_denoiser:
        return codec(wav)["audio"]
    return full_forward(wav) 


@torch.no_grad()
def prepwav(wav_path: Path):
    """ Reads waveform from path into a tensor of shape (channels, frames) """
    wav, sample_rate = torchaudio.load(wav_path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{wav_path} has sample rate {sample_rate}, expected {SAMPLE_RATE}")
    length = wav.shape[-1]
    right_pad = math.ceil(length / HOP_LENGTH) * HOP_LENGTH - length
    if right_pad > 0:
        wav = nn.functional.pad(wav, (0, right_pad))
    return wav, length


@torch.no_grad()
def main(codec_weights: str,
        denoiser_ckpt: str,
        input_dir: str,
        output_dir: str,
        width: int,
        n_blocks: int,
        device: str = "cpu"
        ):
        
    device = torch.device(device) 

    codec = DAC(encoder_dim=64,
        encoder_rates=[2, 4, 5, 8],
        decoder_dim = 1536,
        decoder_rates = [8, 5, 4, 2],
        n_codebooks = 12,
        codebook_size = 1024,
        codebook_dim = 8,
        sample_rate = SAMPLE_RATE)
    codec.load_state_dict(torch.load(codec_weights, weights_only=True, map_location=device)["state_dict"])
    print("Codec weights loaded")

    denoiser = Denoiser(LATENT_CHANNELS, width, n_blocks)
    checkpoint = torch.load(denoiser_ckpt, weights_only=False, map_location=device)
    denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
    print("Denoiser weights loaded")

    full_forward = FullDenoiser(codec, denoiser).to(device).eval()
    codec = codec.to(device).eval()
    denoiser = denoiser.to(device).eval()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No .wav files found in {input_dir}")

    for wav in wav_files:
        wav_tensor, length = prepwav(wav)
        wav_tensor = wav_tensor.to(device)
        denoised_wav = run_denoise(full_forward, wav_tensor, codec, bypass_denoiser=False)
        denoised_wav = denoised_wav.squeeze(0).cpu()
        denoised_wav = denoised_wav[..., :length]
        torchaudio.save(output_dir / wav.name, denoised_wav, sample_rate=SAMPLE_RATE)

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(**cfg)
