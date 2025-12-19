
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.functional import pad
from scipy.signal import fftconvolve

class DNSAudioDataset(Dataset):
    """
    The denoiser model is trained on the publicly available Interspeech 2021 DNS Challenge wide-
    band (16 kHz) corpus (Reddy et al. 2021) https://github.com/microsoft/DNS-Challenge.

    This is a dataloader created for loading clean/noisy pairs of DNS audio data.
    Returns paired (clean, noisy) 3 s windows with optional convolutional reverb augmentation.

    Required folder layout:
        root_dir/
            clean/*.wav
            noisy/*.wav
            rir/*.wav        (optional; used only when p_reverb > 0)

    Args:
        root_dir: Root directory containing `clean/`, `noisy/`, and optional `rir/`.
        ext: Audio file extension (default: ".wav").
        window_duration: Window length in seconds to draw from each clip.
        min_valid_frac: Minimum fraction of the window that must be real audio
            (clips shorter than this are dropped).
        sample_rate: Expected sample rate; mismatches raise.
        p_reverb: Probability of applying a random RIR convolution.
        seed: Optional base seed for deterministic window-level augmentation.
        max_clips: Truncate the dataset to the first N clips.
    """
    def __init__(self,
                root_dir: str,
                ext: str=".wav",
                window_duration: float=3.0,
                min_valid_frac: float=0.8,
                sample_rate: int=16000,
                p_reverb: float=0.0,
                seed: int=None,
                max_clips: int=None,
):
        if not ext.startswith("."):
            ext = "." + ext
        # collect and match clean/noisy, get rir 
        self.clean_paths = sorted(Path(root_dir+"/clean").glob(f"*{ext}"))
        self.noisy_paths = sorted(Path(root_dir+"/noisy").glob(f"*{ext}"))
        self.rir_paths = list(Path(root_dir+"/rir").rglob(f"*{ext}"))
        if len(self.clean_paths) != len(self.noisy_paths):
            raise ValueError("clean/noisy counts differ "
                             f"({len(self.clean_paths)} vs {len(self.noisy_paths)})")
        if max_clips:
            self.clean_paths = self.clean_paths[:max_clips]
            self.noisy_paths = self.noisy_paths[:max_clips]

        # precompute all constants
        self.target_lufs = -23.0
        self.p_reverb = p_reverb
        self.seed = seed
        self.sr = sample_rate
        self.win_dur = window_duration
        self.win_samps = int(window_duration * sample_rate)
        self.min_valid = int(self.win_samps * min_valid_frac)

        # rng
        self.rng = torch.Generator()

        # build indices from audio lengths
        self.indices = []
        for file_idx, clean_path in enumerate(self.clean_paths):
            info = torchaudio.info(str(clean_path))
            n_frames = info.num_frames
            if n_frames <= 0:
                continue
            n_wins = max(1, n_frames // self.win_samps)
            for win in range(n_wins):
                start = win * self.win_samps
                real = max(0, min(n_frames - start, self.win_samps))
                if real >= self.min_valid:
                    self.indices.append((file_idx, win))
        if not self.indices:
            raise ValueError("No valid windows found; check audio lengths and min_valid_frac.")


    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx, win_idx = self.indices[idx]
        offset_samp = win_idx * self.win_samps

        # load the file
        clean, sr1 = torchaudio.load(str(self.clean_paths[file_idx]),
                                    frame_offset=offset_samp,
                                    num_frames=self.win_samps)
        noisy, sr2 = torchaudio.load(str(self.noisy_paths[file_idx]),
                                    frame_offset=offset_samp,
                                    num_frames=self.win_samps)
        if (sr1 != sr2) or (sr1 != self.sr):
            raise ValueError(f"Sample rate mismatch: clean {sr1}, noisy {sr2}, expected {self.sr}")
        
        # rng
        if self.seed is not None:
            self.rng.manual_seed(self.seed + idx)
        else:
            self.rng.seed()

        # adding reverb 
        if self.rir_paths and torch.rand((), generator=self.rng) <= self.p_reverb:
            rand_idx = torch.randint(low=0, high=len(self.rir_paths), size=(1,), generator=self.rng).item()
            rir_path = self.rir_paths[rand_idx]
            rir, _ = torchaudio.load(str(rir_path))
            rir = rir/torch.sqrt((rir**2).sum())
            noisy = fftconvolve(noisy, rir, mode="full")
            noisy = torch.tensor(noisy[:, :clean.shape[-1]], dtype=clean.dtype)
        
        # loudness normalise
        rms_clean = clean.pow(2).mean(dim=-1, keepdim=True).sqrt()
        rms_db_clean = 20 * torch.log10(rms_clean + 1e-12)

        # 2) compute noisy RMS in dB
        rms_noisy = noisy.pow(2).mean(dim=-1, keepdim=True).sqrt()
        rms_db_noisy = 20 * torch.log10(rms_noisy + 1e-12)

        # 3) compute per‐signal gain needed to hit target_lufs
        gain_db_clean = self.target_lufs - rms_db_clean    # scalar per item
        gain_db_noisy = self.target_lufs - rms_db_noisy

        gain_lin_clean = 10 ** (gain_db_clean / 20.0)
        gain_lin_noisy = 10 ** (gain_db_noisy / 20.0)

        # 4) apply them separately
        clean = clean * gain_lin_clean
        noisy = noisy * gain_lin_noisy
        
        # add random gain
        if torch.rand((), generator=self.rng) < 0.5:
            g_db = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(4.0), generator=self.rng)
            g_lin = torch.pow(torch.tensor(10.0), g_db / 20.0)
            clean *= g_lin 
            noisy *= g_lin

        # pad to window duration
        clean = pad(clean, (0, self.win_samps - clean.size(1)))
        noisy = pad(noisy, (0, self.win_samps - noisy.size(1)))


        return {
            "clean": clean,     # [C, window_samples]
            "noisy": noisy,     # [C, window_samples]
            "sample_rate": self.sr,
            "file_idx": file_idx,
            "window_idx": win_idx,
        }
