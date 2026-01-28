# ⚡ Maxemon AI - Pokémon LoRA Training & Generation

<div align="center">

![Pokemon AI](https://img.shields.io/badge/Pokemon-AI%20Generator-FF6B6B?style=for-the-badge)
![SDXL](https://img.shields.io/badge/SDXL-LoRA-4ECDC4?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-FFE66D?style=for-the-badge)

A powerful Gradio-based application for training custom LoRA models and generating stunning Pokémon artwork using Stable Diffusion XL.

</div>

## 🌟 Features

- 🎨 **Generate Pokémon Artwork**: Create official-style Pokémon artwork, regional variants, mega evolutions, and more
- 🔥 **Train Custom LoRAs**: Train your own LoRA models on custom Pokémon datasets
- 💎 **Modern UI**: Beautiful Gradio interface with intuitive controls
- ⚡ **Optimized for RTX 3070**: Configured for 8GB VRAM GPUs
- 🚀 **SDXL-Based**: Uses Stable Diffusion XL for superior quality
- 📊 **Real-time Monitoring**: Track training progress and generation settings

## 🎯 What Can You Generate?

- Original Pokémon designs
- Existing Pokémon in different poses
- Regional variants (Alolan, Galarian, etc.)
- Legendary and Mythical Pokémon
- Mega Evolutions
- Gigantamax forms
- Fusion designs
- Custom type combinations

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (RTX 3060/3070 or better recommended)
- 16GB+ RAM
- 20GB+ free disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jojogamerCt/maxemon-ai.git
cd maxemon-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

5. (Optional) Install xformers for memory efficiency:
```bash
pip install xformers
```

## 🚀 Quick Start

### Launch the App

```bash
python app.py
```

The app will open at `http://localhost:7860`

### Generate Your First Pokémon

1. Go to the **Generate** tab
2. Enter a prompt:
   ```
   A majestic dragon-type legendary pokemon with crystalline wings,
   flying in the sky, official pokemon artwork, ken sugimori style
   ```
3. Click **Generate Pokemon!**
4. Wait 30-60 seconds for your artwork!

### Train a Custom LoRA

1. Prepare your dataset (see Dataset section below)
2. Go to the **Train LoRA** tab
3. Configure settings:
   - Dataset Path: `./datasets/pokemon`
   - LoRA Name: `my_pokemon_lora`
   - Epochs: 10-20
   - Keep other defaults
4. Click **Start Training**
5. Monitor progress in the logs
6. Use your LoRA in the Generate tab!

## 📊 Dataset Recommendations

### Option 1: Create Your Own Dataset (Recommended)

**Best approach for avoiding copyright issues:**

1. **Use Existing LoRA Models** - Many pre-trained Pokémon LoRAs exist on CivitAI:
   - [CivitAI Pokémon LoRAs](https://civitai.com/tag/pokemon)
   - Search for "pokemon" or "creature design" models
   - Download and place in `./loras/` directory

2. **Scrape Legally** - Use public domain or creative commons images:
   - Fan art with permissive licenses (CC-BY, CC0)
   - Your own artwork or commissions
   - Public domain creature designs

3. **Generate Synthetic Dataset** - Bootstrap using existing models:
   - Use base SDXL to generate 100-200 creature images
   - Use prompts like "fantasy creature, monster design, official artwork"
   - Manually caption each image
   - Train LoRA on this synthetic dataset

### Option 2: Alternative Datasets

**Legal and Available Options:**

- **[Creature Designs Dataset](https://huggingface.co/datasets/search?q=creature)** - Search HF for creature/monster datasets
- **[Kaggle Monster/Creature Datasets](https://www.kaggle.com/search?q=creature+design)** - Various creature design datasets
- **[PokeAPI Images](https://pokeapi.co/)** - Official API (use responsibly, limited images)
- **Custom Web Scraping** - Build your own using:
  - DeviantArt (with artist permission)
  - ArtStation (creative commons only)
  - Your own commissioned artwork

### Option 3: Use Pre-trained LoRAs

**Skip training entirely:**

1. Download pre-trained Pokémon LoRAs from:
   - [CivitAI](https://civitai.com/) - Search "pokemon"
   - [Hugging Face Models](https://huggingface.co/models?search=pokemon)
   
2. Place `.safetensors` file in `./loras/` directory

3. Select in the Generate tab dropdown

### Dataset Preparation Guide

If creating your own dataset:

```
datasets/pokemon/
├── image_001.png
├── image_001.txt  (optional caption: "fire type creature with flames")
├── image_002.png
├── image_002.txt
└── ...
```

**Caption Tips:**
- Keep captions simple and descriptive
- Focus on: type, colors, features, pose
- Example: "blue water-type creature with fins, swimming pose"

### Dataset Guidelines

- **Minimum**: 20-50 images for basic concepts
- **Recommended**: 100-500 images for best results
- **Quality**: Clean, well-composed, consistent style
- **Captions**: Highly recommended for better results
- **Format**: PNG, JPG, or WebP
- **Resolution**: 512px+ (will be resized to 1024)

## ⚙️ Configuration

### For RTX 3070 (8GB VRAM)

**Recommended Training Settings:**
- Resolution: 1024
- Batch Size: 1
- Epochs: 10-20
- Learning Rate: 1e-4
- 8-bit Adam: ✅ Enabled
- Gradient Checkpointing: ✅ Enabled

**Generation Settings:**
- Resolution: 1024x1024
- Steps: 30-50
- Guidance Scale: 7-8
- LoRA Scale: 0.7-1.0

### For Higher-End GPUs (12GB+)

- Batch Size: 2-4
- Can disable 8-bit Adam
- Higher resolutions possible

## 🎨 Prompt Engineering Guide

### Anatomy of a Good Prompt

```
[Type] + [Form/Evolution] + [Appearance] + [Action/Pose] + [Style]
```

### Examples

**Starter Pokémon:**
```
A cute grass-type starter pokemon with leaf patterns on its back,
playful pose, bright green colors, official pokemon artwork
```

**Legendary:**
```
A majestic psychic-type legendary pokemon with flowing ethereal energy,
floating in space, cosmic background, official artwork, ken sugimori style
```

**Mega Evolution:**
```
Mega evolution of a fire and flying type pokemon, dragon-like appearance,
intense flames, battle stance, official pokemon mega evolution artwork
```

**Regional Variant:**
```
Alolan variant of pikachu, ice-type, white and blue fur,
snowy mountain background, official pokemon artwork style
```

### Prompt Tips

- Include "official pokemon artwork" or "ken sugimori style"
- Specify type (fire, water, grass, etc.)
- Mention evolution stage (starter, evolution, final form)
- Describe key features and colors
- Add pose or action
- Use quality modifiers: "high quality", "detailed", "professional"

### Negative Prompts

```
blurry, bad anatomy, bad proportions, deformed, ugly, low quality,
watermark, text, signature, extra limbs, mutation, poorly drawn
```

## 📁 Project Structure

```
maxemon-ai/
├── app.py                 # Main Gradio application
├── train_lora.py         # Training orchestration
├── train_lora_sdxl.py    # SDXL LoRA training script
├── generate.py           # Image generation module
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── datasets/            # Training datasets
├── loras/               # Trained LoRA models
├── outputs/             # Generated images
└── models/              # Downloaded base models
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

- Reduce batch size to 1
- Enable 8-bit Adam
- Enable gradient checkpointing
- Reduce resolution to 768 or 512
- Close other GPU applications

### Slow Training

- Normal for SDXL (1-3 hours for 100 images)
- Install xformers for speedup
- Use smaller datasets for testing

### Poor Quality Results

- Train for more epochs
- Use better quality dataset
- Adjust learning rate
- Try different prompts
- Adjust LoRA scale

### LoRA Too Strong/Weak

- Adjust LoRA scale in generation (0.5-1.5)
- Retrain with different network_dim/alpha
- Train for fewer/more epochs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for educational and creative purposes. Pokémon and related trademarks are property of Nintendo, Game Freak, and The Pokémon Company. 

**Important:**
- Use only datasets you have legal rights to use
- Respect copyright and intellectual property
- Generated images are for personal/educational use
- Do not use for commercial purposes without proper licensing
- Consider using generic "creature" or "monster" concepts instead of copyrighted characters

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion XL
- Hugging Face for Diffusers library
- Gradio team for the amazing UI framework
- Open source AI community
- kohya_ss for LoRA training techniques

## 📞 Support

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Join our community discussions

---

<div align="center">Made with ❤️ for Pokémon fans and AI enthusiasts</div>
