#!/usr/bin/env python3
"""
SDXL LoRA Training Script
Optimized for RTX 3070 (8GB VRAM)

This script trains a LoRA model on SDXL for Pokemon generation.
Based on Hugging Face Diffusers training examples.
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import json

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA training will not work.")


class PokemonDataset(Dataset):
    """Dataset for Pokemon images with optional captions"""
    
    def __init__(self, data_dir, resolution=1024, center_crop=True):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        
        # Find all images
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            self.image_files.extend(list(self.data_dir.glob(ext)))
        
        if not self.image_files:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_files)} images")
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        
        # Load caption if exists
        caption_path = image_path.with_suffix('.txt')
        if caption_path.exists():
            caption = caption_path.read_text().strip()
        else:
            # Default caption
            caption = "a pokemon, official artwork, high quality"
        
        return {
            'pixel_values': image,
            'caption': caption
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDXL LoRA for Pokemon generation")
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA")
    
    # Model
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--resolution", type=int, default=1024)
    
    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    
    # LoRA
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--network_dim", type=int, default=32, help="LoRA rank")
    parser.add_argument("--network_alpha", type=int, default=32, help="LoRA alpha")
    
    # Optimization
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not PEFT_AVAILABLE:
        raise ImportError("Please install peft: pip install peft")
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Loading model: {args.model_id}")
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer",
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float16
    )
    
    # Load UNet
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torch_dtype=torch.float16
    )
    
    # Freeze models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.network_dim,
        lora_alpha=args.network_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("Using 8-bit Adam optimizer")
        except ImportError:
            print("bitsandbytes not available, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    
    # Dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = PokemonDataset(
        data_dir=args.dataset_path,
        resolution=args.resolution
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler"
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    print("Starting training...")
    print(f"Total epochs: {args.max_train_epochs}")
    print(f"Total steps: {args.max_train_steps}")
    print(f"Batch size: {args.train_batch_size}")
    
    global_step = 0
    
    for epoch in range(args.max_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.max_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Encode images
                latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode text
                tokens = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                encoder_hidden_states = text_encoder(tokens)[0]
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
            
            if global_step >= args.max_train_steps:
                break
        
        progress_bar.close()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unet_lora = accelerator.unwrap_model(unet)
                unet_lora.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_lora = accelerator.unwrap_model(unet)
        final_save_path = os.path.join(args.output_dir, "final_lora")
        unet_lora.save_pretrained(final_save_path)
        print(f"Training complete! Final LoRA saved to {final_save_path}")


if __name__ == "__main__":
    main()
