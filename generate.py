import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from pathlib import Path
import random
from datetime import datetime
import os

# Global pipeline cache
pipeline_cache = {}

def load_pipeline(lora_path=None):
    """
    Load SDXL pipeline with optional LoRA
    """
    global pipeline_cache
    
    cache_key = f"base_{lora_path if lora_path else 'none'}"
    
    if cache_key in pipeline_cache:
        print(f"Using cached pipeline: {cache_key}")
        return pipeline_cache[cache_key]
    
    print("Loading SDXL pipeline...")
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Optimize for memory
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    # Use DPM++ scheduler for better quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    
    # Load LoRA if specified
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(str(Path(lora_path).parent), weight_name=Path(lora_path).name)
    
    # Cache the pipeline
    pipeline_cache[cache_key] = pipe
    
    return pipe

def generate_pokemon(
    prompt: str,
    negative_prompt: str = "",
    lora_path: str = None,
    lora_scale: float = 0.8,
    num_images: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = None,
    width: int = 1024,
    height: int = 1024
):
    """
    Generate Pokemon images using SDXL with optional LoRA
    
    Args:
        prompt: Text description of the Pokemon to generate
        negative_prompt: Things to avoid in generation
        lora_path: Path to LoRA weights file (.safetensors)
        lora_scale: Strength of LoRA effect (0-2)
        num_images: Number of images to generate
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility
        width: Image width
        height: Image height
    
    Returns:
        List of generated PIL Images
    """
    
    # Set seed if provided
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Load pipeline
    pipe = load_pipeline(lora_path)
    
    # Enhanced prompt for Pokemon style
    enhanced_prompt = f"{prompt}, high quality, detailed, professional artwork"
    
    # Enhanced negative prompt
    default_negative = "blurry, bad anatomy, bad proportions, deformed, ugly, low quality, watermark, text, signature, worst quality, low res, normal quality, bad hands, extra limbs, mutation, poorly drawn"
    if negative_prompt:
        full_negative = f"{negative_prompt}, {default_negative}"
    else:
        full_negative = default_negative
    
    print(f"Generating {num_images} image(s)...")
    print(f"Prompt: {enhanced_prompt}")
    print(f"Seed: {seed}")
    
    # Generate images
    images = []
    
    for i in range(num_images):
        # Create unique generator for each image if multiple
        if num_images > 1:
            current_seed = seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)
        else:
            current_seed = seed
        
        try:
            result = pipe(
                prompt=enhanced_prompt,
                negative_prompt=full_negative,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                cross_attention_kwargs={"scale": lora_scale} if lora_path else None
            )
            
            image = result.images[0]
            
            # Save image
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pokemon_{timestamp}_seed{current_seed}_{i+1}.png"
            filepath = output_dir / filename
            
            image.save(filepath)
            print(f"Saved: {filepath}")
            
            images.append(image)
            
        except Exception as e:
            print(f"Error generating image {i+1}: {str(e)}")
            continue
    
    return images

def test_generation():
    """Test function to verify generation works"""
    test_prompt = "A cute fire-type pokemon with orange fur and big eyes, official pokemon artwork, ken sugimori style"
    
    print("Running test generation...")
    images = generate_pokemon(
        prompt=test_prompt,
        num_images=1,
        num_inference_steps=25,
        seed=42
    )
    
    print(f"Generated {len(images)} test image(s)")
    return images

if __name__ == "__main__":
    # Test generation
    test_generation()
