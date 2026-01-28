import gradio as gr
import torch
import os
import json
from pathlib import Path
from datetime import datetime
from train_lora import start_training, stop_training, get_training_status
from generate import generate_pokemon
from huggingface_hub import snapshot_download

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("datasets", exist_ok=True)
os.makedirs("loras", exist_ok=True)

# Global variables
training_process = None

def format_training_log(log_text):
    """Format training logs for better readability"""
    if not log_text:
        return "No logs yet..."
    return log_text

def train_wrapper(dataset_path, lora_name, epochs, batch_size, learning_rate, 
                  resolution, use_8bit, gradient_checkpointing, max_train_steps):
    """Wrapper for training function"""
    try:
        config = {
            "dataset_path": dataset_path,
            "lora_name": lora_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "use_8bit_adam": use_8bit,
            "gradient_checkpointing": gradient_checkpointing,
            "max_train_steps": max_train_steps if max_train_steps > 0 else None
        }
        
        result = start_training(config)
        return result["message"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

def stop_training_wrapper():
    """Stop training process"""
    result = stop_training()
    return result["message"]

def get_status_wrapper():
    """Get current training status"""
    status = get_training_status()
    if status["is_training"]:
        return f"🔥 Training in progress...\n" \
               f"Step: {status.get('current_step', 'N/A')}/{status.get('total_steps', 'N/A')}\n" \
               f"Epoch: {status.get('current_epoch', 'N/A')}/{status.get('total_epochs', 'N/A')}\n" \
               f"Loss: {status.get('last_loss', 'N/A')}"
    else:
        return "✅ No training in progress"

def generate_wrapper(prompt, negative_prompt, lora_path, lora_scale, 
                     num_images, steps, guidance_scale, seed, width, height):
    """Wrapper for generation function"""
    try:
        images = generate_pokemon(
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_path=lora_path if lora_path else None,
            lora_scale=lora_scale,
            num_images=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed if seed >= 0 else None,
            width=width,
            height=height
        )
        return images
    except Exception as e:
        return [f"Error generating images: {str(e)}"]

def get_available_loras():
    """Get list of available LoRA models"""
    lora_dir = Path("loras")
    if not lora_dir.exists():
        return []
    loras = [str(f.relative_to(lora_dir)) for f in lora_dir.rglob("*.safetensors")]
    return [""] + loras

def refresh_loras():
    """Refresh LoRA dropdown"""
    return gr.Dropdown(choices=get_available_loras())

# Custom CSS for modern UI
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1400px !important;
}

.title-text {
    background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #FFE66D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.5em;
}

.subtitle-text {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 2em;
}

.tab-nav button {
    font-size: 1.1em;
    font-weight: 600;
}

.generate-btn {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4) !important;
    border: none !important;
    font-weight: bold !important;
}

.train-btn {
    background: linear-gradient(90deg, #4ECDC4, #FFE66D) !important;
    border: none !important;
    font-weight: bold !important;
}
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Maxemon AI - Pokemon Generator") as demo:
    
    gr.HTML("""
        <div class="title-text">⚡ Maxemon AI ⚡</div>
        <div class="subtitle-text">Train & Generate Amazing Pokémon Artwork with AI</div>
    """)
    
    with gr.Tabs():
        # GENERATION TAB
        with gr.Tab("🎨 Generate", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Generation Settings")
                    
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A cute fire-type pokemon with orange fur and big eyes, official pokemon artwork, ken sugimori style",
                        lines=3
                    )
                    
                    gen_negative = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, bad anatomy, bad proportions, deformed, ugly, low quality, watermark",
                        lines=2
                    )
                    
                    with gr.Row():
                        gen_lora = gr.Dropdown(
                            choices=get_available_loras(),
                            label="LoRA Model (Optional)",
                            value="",
                            interactive=True
                        )
                        refresh_btn = gr.Button("🔄", scale=0, min_width=50)
                    
                    gen_lora_scale = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=0.8,
                        step=0.1,
                        label="LoRA Scale"
                    )
                    
                    with gr.Row():
                        gen_width = gr.Slider(512, 1024, value=1024, step=64, label="Width")
                        gen_height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
                    
                    with gr.Row():
                        gen_steps = gr.Slider(20, 100, value=30, step=5, label="Steps")
                        gen_guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                    
                    with gr.Row():
                        gen_num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                        gen_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    
                    generate_btn = gr.Button("🎨 Generate Pokemon!", variant="primary", elem_classes=["generate-btn"])
                
                with gr.Column(scale=1):
                    gr.Markdown("### Generated Images")
                    gen_output = gr.Gallery(
                        label="Results",
                        show_label=False,
                        columns=2,
                        rows=2,
                        height="auto",
                        object_fit="contain"
                    )
            
            # Example prompts
            gr.Markdown("### 💡 Example Prompts")
            gr.Examples(
                examples=[
                    ["A majestic dragon-type legendary pokemon with crystalline wings, flying in the sky, official pokemon artwork, ken sugimori style"],
                    ["A cute grass-type starter pokemon with leaf patterns, sitting pose, official pokemon artwork"],
                    ["A fierce fire and dark type pokemon with flames, mega evolution form, dramatic pose"],
                    ["A water-type mythical pokemon with flowing fins, underwater scene, official pokemon artwork"],
                    ["A ghost and fairy type pokemon with ethereal glow, floating, Gigantamax form"],
                    ["A regional variant of pikachu with ice powers, snowy background, official artwork style"]
                ],
                inputs=[gen_prompt]
            )
        
        # TRAINING TAB
        with gr.Tab("🔥 Train LoRA", id=1):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Training Configuration")
                    gr.Markdown("""
                    **Recommended Dataset Sources:**
                    - [Pokémon Images on Hugging Face](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
                    - [Pokémon Dataset on Kaggle](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)
                    - Your own curated Pokémon images
                    
                    **Dataset Format:** Place images in a folder with optional `.txt` caption files
                    """)
                    
                    train_dataset = gr.Textbox(
                        label="Dataset Path",
                        placeholder="./datasets/pokemon",
                        value="./datasets/pokemon"
                    )
                    
                    train_lora_name = gr.Textbox(
                        label="LoRA Name",
                        placeholder="pokemon_lora_v1",
                        value=f"pokemon_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    with gr.Row():
                        train_epochs = gr.Slider(1, 100, value=10, step=1, label="Epochs")
                        train_batch = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                    
                    train_lr = gr.Number(
                        label="Learning Rate",
                        value=1e-4,
                        step=1e-5
                    )
                    
                    train_resolution = gr.Slider(
                        512, 1024, value=1024, step=64,
                        label="Training Resolution"
                    )
                    
                    train_max_steps = gr.Number(
                        label="Max Steps (0 for epoch-based)",
                        value=0
                    )
                    
                    with gr.Row():
                        train_8bit = gr.Checkbox(label="Use 8-bit Adam (Saves VRAM)", value=True)
                        train_gradient_cp = gr.Checkbox(label="Gradient Checkpointing", value=True)
                    
                    with gr.Row():
                        train_btn = gr.Button("🚀 Start Training", variant="primary", elem_classes=["train-btn"])
                        stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
                        status_btn = gr.Button("📊 Check Status")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Training Progress")
                    train_output = gr.Textbox(
                        label="Training Logs",
                        lines=20,
                        max_lines=30,
                        interactive=False
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        lines=5,
                        interactive=False
                    )
        
        # INFO TAB
        with gr.Tab("ℹ️ Info & Tips", id=2):
            gr.Markdown("""
            # Maxemon AI - Complete Guide
            
            ## 🎯 What is this?
            Maxemon AI is a powerful tool for training custom LoRA models and generating Pokémon artwork using SDXL.
            
            ## 🚀 Quick Start
            
            ### For Generation:
            1. Go to the **Generate** tab
            2. Enter your prompt (e.g., "a fire-type dragon pokemon")
            3. Optionally select a trained LoRA model
            4. Click Generate!
            
            ### For Training:
            1. Prepare your dataset (images + optional captions)
            2. Go to the **Train LoRA** tab
            3. Configure settings (defaults work well for RTX 3070)
            4. Click Start Training
            5. Wait for training to complete
            6. Use your LoRA in the Generate tab!
            
            ## 📊 Recommended Datasets
            
            - **Pokémon Official Artwork**: High-quality Ken Sugimori style artwork
            - **Pokémon Sprites**: Game sprites for pixel-art style
            - **Fan Art Collections**: Diverse artistic styles
            - **Mixed Dataset**: Combine official + fan art for versatility
            
            ### Dataset Tips:
            - **Minimum**: 20-50 images for basic concepts
            - **Recommended**: 100-500 images for best results
            - **Quality over Quantity**: Clean, well-composed images work best
            - **Captions**: Optional but improve results (describe each pokemon)
            
            ## ⚙️ Training Settings Guide
            
            ### For RTX 3070 (8GB VRAM):
            - **Resolution**: 1024px (SDXL native)
            - **Batch Size**: 1-2
            - **Epochs**: 10-20 for small datasets, 5-10 for large
            - **Learning Rate**: 1e-4 (default is good)
            - **8-bit Adam**: ✅ Enabled (saves VRAM)
            - **Gradient Checkpointing**: ✅ Enabled (saves VRAM)
            
            ### LoRA Rank & Alpha:
            - **Rank**: 16-32 (higher = more capacity, more VRAM)
            - **Alpha**: Usually same as rank
            
            ## 🎨 Prompt Engineering Tips
            
            ### Good Prompts:
            - "A majestic water-type legendary pokemon with flowing fins, official pokemon artwork, ken sugimori style"
            - "Fire and flying type pokemon, dragon-like, red scales, breathing flames, dynamic pose"
            - "Cute grass-type starter evolution, final form, battle stance, official artwork"
            
            ### Prompt Elements:
            - **Type**: Fire, water, grass, electric, psychic, etc.
            - **Form**: Starter, evolution, mega evolution, regional variant, legendary
            - **Style**: Ken Sugimori style, official artwork, game sprite
            - **Pose/Action**: Standing, flying, battle pose, sleeping
            - **Details**: Colors, features, size, environment
            
            ### Negative Prompts:
            Include: `blurry, bad anatomy, bad proportions, deformed, ugly, low quality, watermark, text`
            
            ## 🔥 Advanced Tips
            
            1. **Multiple LoRAs**: Train separate LoRAs for different styles (official art, sprites, regional forms)
            2. **LoRA Mixing**: Use LoRA scale 0.5-1.0 for subtle effects, 1.0-1.5 for strong style
            3. **Seed Control**: Use same seed for variations of a design
            4. **Batch Generation**: Generate 2-4 images to pick the best
            
            ## 💾 System Requirements
            
            - **GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)
            - **RAM**: 16GB+ recommended
            - **Storage**: 20GB+ for models and outputs
            - **CUDA**: 11.8 or newer
            
            ## 🐛 Troubleshooting
            
            - **Out of VRAM**: Reduce batch size to 1, enable 8-bit Adam
            - **Slow Training**: Normal for SDXL, expect 1-2 hours for small datasets
            - **Bad Results**: Try more training steps, better dataset, adjust learning rate
            - **LoRA too strong**: Reduce LoRA scale in generation
            
            ## 📚 Model Info
            
            - **Base Model**: SDXL 1.0 (Stable Diffusion XL)
            - **Training Method**: LoRA (Low-Rank Adaptation)
            - **Why SDXL?**: Better quality than SD 1.5, supports higher resolutions, better prompt understanding
            
            ---
            
            **Created with ❤️ for Pokémon fans and AI enthusiasts!**
            """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            gen_prompt, gen_negative, gen_lora, gen_lora_scale,
            gen_num_images, gen_steps, gen_guidance, gen_seed,
            gen_width, gen_height
        ],
        outputs=gen_output
    )
    
    refresh_btn.click(
        fn=refresh_loras,
        inputs=[],
        outputs=gen_lora
    )
    
    train_btn.click(
        fn=train_wrapper,
        inputs=[
            train_dataset, train_lora_name, train_epochs, train_batch,
            train_lr, train_resolution, train_8bit, train_gradient_cp,
            train_max_steps
        ],
        outputs=train_output
    )
    
    stop_btn.click(
        fn=stop_training_wrapper,
        inputs=[],
        outputs=train_output
    )
    
    status_btn.click(
        fn=get_status_wrapper,
        inputs=[],
        outputs=status_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
