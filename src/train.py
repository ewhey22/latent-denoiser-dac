import os
import sys
from dataclasses import dataclass
from typing import List
import argparse
import yaml
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from audiotools.ml.accelerator import Accelerator
from tqdm import tqdm
from denoise_module import Denoiser, Discriminator
from dataset import DNSAudioDataset
from torch.nn import L1Loss
from torch_pesq import PesqLoss
from dac.model.dac import DAC

# Training script for the latent-space denoiser. Relies on Descript Audio Codec
# (DAC) latents as targets; see README for config examples.

def parse_args():
    p = argparse.ArgumentParser(description="Train denoiser")
    p.add_argument("--config", "-c",
        type=str, required=True,
        help="Path to YAML config file")
    return p.parse_args()

# Linear warmup 
def make_linear_warmup(warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@dataclass
class State:
    """Container for model/optimizer/dataset state."""
    codec: torch.nn.Module

    denoiser: torch.nn.Module
    optimizer_g: torch.optim.Optimizer

    disc: torch.nn.Module
    optimizer_d: torch.optim.Optimizer

    pesq_loss: PesqLoss
    l1_loss: L1Loss

    traindata: DNSAudioDataset
    valdata: DNSAudioDataset

    iteration: int
    writer: SummaryWriter


def load(accel: Accelerator,
        codec_weights: str,
        train_data: str,
        val_data: str,
        run_dir: str,
        from_checkpoint: str=None,
        sample_rate: int=16000):
    """
    Build models/optimizers, datasets, and optionally restore from a checkpoint.

    accel: audiotools Accelerator instance (handles DDP/mixed precision).
    codec_weights: path to DAC weights (.pth with key 'state_dict').
    train_data/val_data: dataset root paths (see DNSAudioDataset).
    run_dir: output directory for checkpoints and TensorBoard.
    from_checkpoint: optional path to resume training.
    sample_rate: audio sample rate expected by DAC and dataset.
    """
    codec = DAC(encoder_dim=64,
                encoder_rates=[2, 4, 5, 8],
                decoder_dim = 1536,
                decoder_rates = [8, 5, 4, 2],
                n_codebooks = 12,
                codebook_size = 1024,
                codebook_dim = 8,
                sample_rate = sample_rate)
    if accel.local_rank == 0:
        print(f"Path to codec weights:\n{codec_weights}")
    codec.load_state_dict(torch.load(codec_weights, weights_only=True)["state_dict"])
    denoiser = Denoiser() 
    disc = Discriminator()

    # Loading from checkpoint: checkpoint = torch.load(PATH, weights_only=True)
    # To load the models, first initialize the models and optimizers, 
    # then load the dictionary locally using torch.load(). 
    # From here, you can easily access the saved items by simply querying the dictionary as you would expect.
    if from_checkpoint is not None:
        if accel.local_rank == 0:
            print(f"Loading from checkpoint {from_checkpoint}")
        checkpoint = torch.load(from_checkpoint, weights_only=False)
        denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        iteration = checkpoint["iteration"]
    else:
        iteration = 0

    codec = accel.prepare_model(codec)
    denoiser = accel.prepare_model(denoiser)
    disc = accel.prepare_model(disc)

    if accel.local_rank == 0:
        print(f"Codec:\n{codec}")
        print(f"Denoiser:\n{denoiser}")
        print(f"Discriminator:\n{disc}")
    # Discriminator
    optimizer_d = torch.optim.AdamW(disc.parameters(),
                        lr=1e-4,
                        betas=(0.0, 0.9))
    # Generator
    optimizer_g = torch.optim.AdamW(denoiser.parameters(),
                        lr=4e-4,
                        betas=(0.0, 0.9))

    if from_checkpoint is not None:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])


    # Losses 
    l1_loss = L1Loss(reduction="mean")
    pesq_loss = PesqLoss(1.0, sample_rate=sample_rate) # For tensorboard, compute on full audio samples.

    # Datasets
    traindata = DNSAudioDataset(train_data, p_reverb=0.0)
    valdata = DNSAudioDataset(val_data, seed=42, max_clips=50, p_reverb=0.0)

    # Tensorboard
    writer = SummaryWriter(log_dir=run_dir+"tb")

    return State(
        denoiser=denoiser,
        disc=disc,
        codec=codec,
        optimizer_g=optimizer_g,

        optimizer_d=optimizer_d,

        l1_loss=l1_loss,
        pesq_loss=pesq_loss,
        traindata=traindata,
        valdata=valdata,

        writer=writer,
        iteration=iteration
    )

lambdas = {"l1_loss": 10.0,
            "g_loss": 1.0,
            "feat_loss": 1.0}  

def train_loop(batch, state, accel):
    """
    One GAN iteration: update discriminator (hinge + R1) then generator
    (adv + feature matching + latent L1).
    """
    state.denoiser.train()
    state.disc.train()
    state.codec.eval()
    outputs = {}

    clean = batch["clean"].to(accel.device)
    noisy = batch["noisy"].to(accel.device)
    with torch.no_grad():
        clean_latents = state.codec.module.encoder(clean)
        noisy_latents = state.codec.module.encoder(noisy)

    # Discriminator
    # with accel.autocast():
    fake = state.denoiser(noisy_latents).detach()
    d_fake_D = state.disc(fake, return_feats=False)

    # ---- real (prepare for R1) ----
    real_latents = clean_latents.detach().requires_grad_()
    d_real_D = state.disc(real_latents, return_feats=False)
    
    outputs["loss/d_fake"] = torch.relu(1. + d_fake_D).mean()
    outputs["loss/d_real"] = torch.relu(1. - d_real_D).mean()
    outputs["loss/d_loss"] = outputs["loss/d_fake"] + outputs["loss/d_real"]

    grad_real = torch.autograd.grad(
        outputs=d_real_D.sum(),
        inputs=real_latents,
        create_graph=True,
    )[0]
    # per-sample gradient norm squared, then mean
    r1 = grad_real.pow(2).flatten(1).sum(1).mean()
    r1_term = 0.5 * 20.0 * r1 
    outputs["loss/d_loss"] = outputs["loss/d_loss"] + r1_term
    outputs["loss/r1"] = r1_term

    state.optimizer_d.zero_grad()
    accel.backward(outputs["loss/d_loss"])

    outputs["other/grad_norm_disc"] = clip_grad_norm_(state.disc.parameters(), max_norm=10)
    accel.step(state.optimizer_d)


    # Generator update
    for p in state.disc.parameters():
        p.requires_grad = False
    g_out_G = state.denoiser(noisy_latents)
    d_fake_G, feats_fake = state.disc(g_out_G, return_feats=True)
    with torch.no_grad():
        _, feats_real = state.disc(clean_latents, return_feats=True)
    outputs["loss/l1_loss"] = state.l1_loss(g_out_G, clean_latents)
    outputs["loss/g_loss"] = -d_fake_G.mean()
    
    outputs["loss/feat_loss"] = sum(state.l1_loss(f_fake, f_real)
                                    for f_real, f_fake in zip(feats_real, feats_fake))
    
    outputs["loss/total_loss"] = (outputs["loss/g_loss"] * lambdas["g_loss"]
                                    + outputs["loss/feat_loss"] * lambdas["feat_loss"]
                                    + outputs["loss/l1_loss"] * lambdas["l1_loss"]
                                        )
                        
    state.optimizer_g.zero_grad()
    accel.backward(outputs["loss/total_loss"])
    outputs["other/grad_norm_denoise"] = clip_grad_norm_(state.denoiser.parameters(), max_norm=10)
    accel.step(state.optimizer_g)
    for p in state.disc.parameters():
        p.requires_grad = True

    # Logging
    if state.iteration % 10 == 0:
        for tag, value in outputs.items():
            state.writer.add_scalar(tag, value, global_step=state.iteration)
        state.writer.add_scalar("other/lr_denoise", state.optimizer_g.param_groups[0]['lr'], global_step=state.iteration)
        state.writer.add_scalar("other/lr_disc", state.optimizer_d.param_groups[0]['lr'], global_step=state.iteration)


@torch.no_grad()
def val_loop(batch, state, accel):
    """Compute L1 in latent space for a validation batch."""
    state.codec.eval()
    state.denoiser.eval()

    clean = batch["clean"].to(accel.device)
    noisy = batch["noisy"].to(accel.device)

    clean_latents = state.codec.module.encoder(clean)
    noisy_latents = state.codec.module.encoder(noisy)
    g_out = state.denoiser(noisy_latents)
    return state.l1_loss(g_out, clean_latents)
    

@torch.no_grad()
def save_samples(state, accel, idxs):
    """Log noisy/clean/recon audio and PESQ to TensorBoard for given indices."""
    state.codec.eval()
    state.denoiser.eval()

    batch = [state.valdata[idx*10] for idx in idxs]
    batch = torch.utils.data.default_collate(batch)

    clean = batch["clean"].to(accel.device)
    noisy = batch["noisy"].to(accel.device)


    noisy_latents = state.codec.module.encoder(noisy)
    g_out = state.denoiser(noisy_latents)
    z, _, _, _, _ = state.codec.module.quantizer(g_out)
    recon = state.codec.module.decoder(z)

    pesq = state.pesq_loss.mos(clean.squeeze(1).cpu(), recon.squeeze(1).cpu())

    # Tensorboard
    if state.iteration == 0:
        for i, (n, c) in enumerate(zip(noisy, clean)):
            state.writer.add_audio(f"noisy/audio_{i}", n, global_step=state.iteration, sample_rate=16000)
            state.writer.add_audio(f"clean/audio_{i}", c, global_step=state.iteration, sample_rate=16000)
            
    for i, r in enumerate(recon):
        state.writer.add_audio(f"recon/audio_{i}", r, global_step=state.iteration, sample_rate=16000)
        state.writer.add_scalar(f"pesqmos/recon_{i}", pesq[i], global_step=state.iteration)

def save_checkpoint(state, run_dir, accel, name):
    """Write a training checkpoint (denoiser/disc/opt/iteration)."""
    path = os.path.join(run_dir, name)
    tmp_path = path + ".tmp"
    denoiser = accel.unwrap(state.denoiser)
    disc = accel.unwrap(state.disc)
    torch.save({
        'denoiser_state_dict': denoiser.state_dict(),
        'optimizer_g_state_dict': state.optimizer_g.state_dict(),

        'disc_state_dict': disc.state_dict(),
        'optimizer_d_state_dict': state.optimizer_d.state_dict(),

        'iteration': state.iteration
        }, tmp_path)
    os.replace(tmp_path, path)

def train(accel: Accelerator,
          codec_weights: str,
          train_data: str,
          val_data: str,
          run_dir: str,
          from_checkpoint: str=None,
          total_iters: int=250000,
          train_batch: int=64,
          val_batch: int=64,
          num_workers: int=16,
          val_per: int=1000,
          samples_per: int=1000,
          save_ckpts: List[int]=[1000, 25000, 50000, 100000, 150000, 200000],
          val_idx: List[int]=[1, 3, 6, 8, 12]):
    os.makedirs(run_dir, exist_ok=True)

    state = load(accel, 
                 codec_weights=codec_weights,
                 train_data=train_data,
                 val_data=val_data,
                 run_dir=run_dir,
                 from_checkpoint=from_checkpoint)
    
    train_dataloader = accel.prepare_dataloader(state.traindata,
                                                start_idx=state.iteration*train_batch,
                                                num_workers=num_workers,
                                                batch_size=train_batch,
                                                shuffle=False)
    
    

    val_dataloader = accel.prepare_dataloader(state.valdata,
                                              start_idx=0,
                                              num_workers=num_workers,
                                              batch_size=val_batch,
                                              shuffle=False)

    best_val = float('inf')
    with tqdm(total=total_iters, disable=(accel.local_rank != 0)) as pbar:
        train_iter = get_infinite_loader(train_dataloader)
        while state.iteration < total_iters:
            
            #  Validation
            if state.iteration == 0 or state.iteration % val_per == 0:
                if accel.local_rank == 0:
                    print("Doing validation")
                total_l1 = 0
                with tqdm(total=len(val_dataloader), disable=(accel.local_rank != 0)) as valbar:
                    for batch in val_dataloader:
                        total_l1 += val_loop(batch, state, accel).item()
                        valbar.update(1)
                avg_l1 = total_l1/len(val_dataloader)
                state.writer.add_scalar("val_loss/l1", avg_l1, global_step=state.iteration)
                if avg_l1 < best_val and accel.local_rank == 0:
                    best_val = avg_l1
                    save_checkpoint(state, run_dir=run_dir, accel=accel, name="best.tar")

            # Save samples
            if (state.iteration == 0 or state.iteration % samples_per == 0) and accel.local_rank == 0:
                print("Saving samples")
                save_samples(state, accel, val_idx)
            
            if state.iteration % 100 == 0 and accel.local_rank == 0:
                save_checkpoint(state, run_dir=run_dir, accel=accel, name="latest.tar")
                
            if state.iteration in save_ckpts and accel.local_rank == 0:
                save_checkpoint(state, run_dir=run_dir, accel=accel, name=f"{state.iteration}_ckpt.tar")
            
            batch = next(train_iter)
            train_loop(batch, state, accel)
            state.iteration += 1
            pbar.update(1)


if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    with Accelerator() as accel:
        if accel.local_rank != 0:
            sys.tracebacklimit = 0
        train(accel=accel, **cfg)
