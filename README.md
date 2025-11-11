🧠 README: Mamba-ViT Hybrid Image Dehazing
🌫 Overview

This repository implements a hybrid deep learning model for image dehazing that integrates a Vision Transformer (ViT) encoder and a Mamba recurrent refinement module.
The model enhances feature understanding and spatial consistency by combining:

Local texture extraction (CNN Encoder)

Global context modeling (ViT)

Sequential feature refinement (Mamba GRU)

Image reconstruction (CNN Decoder)

It is trained and evaluated on the SOTS (RESIDE) dataset and outputs clean, dehazed images with high PSNR and SSIM.

🚀 Features

✅ DEM-Free, purely image-based model
✅ End-to-end training with PyTorch
✅ Supports PSNR & SSIM metric evaluation
✅ Visual comparison (Hazy → Dehazed → Ground Truth)
✅ Modular architecture (ViT + Mamba + CNN fusion)
✅ Lightweight, easy to train on mid-range GPUs

🧩 Architecture Diagram
Line Diagram (Model Flow)
Input (Hazy Image)
        │
        ▼
 ┌────────────────────┐
 │  CNN Encoder (3→128)│
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐
 │  Vision Transformer │
 │ (Global Context)    │
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐
 │  Mamba (GRU Block)  │
 │ (Sequential Refinement) │
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐
 │  Projection + Fusion│
 │ (Add to Encoder)    │
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐
 │  CNN Decoder (128→3)│
 └────────────────────┘
        │
        ▼
Output (Dehazed Image)

🧠 Model Components
Component	Description
Encoder	Two-layer CNN for local feature extraction
SimpleViT	Lightweight Vision Transformer-like feature summarizer
SimpleMamba	GRU-based sequential block to refine ViT embeddings
Decoder	CNN block to reconstruct the clean image
Fusion	Adds back the refined global context to spatial encoder features
📂 Dataset

Dataset Used: RESIDE SOTS Outdoor

Folder Structure:

├── hazy_processed/   # input hazy images
├── GT/               # corresponding clear ground truth images

⚙️ Installation
git clone https://github.com/<your-username>/Mamba-ViT-Dehazing.git
cd Mamba-ViT-Dehazing
pip install torch torchvision scikit-learn matplotlib Pillow

🧪 Training
python train_dehazing.py


The model automatically splits data into 80% train / 20% test

Trains for 70 epochs

Saves model weights as mamba_vit_100dddehazing.pth

📈 Evaluation Metrics
Metric	Description
PSNR	Peak Signal-to-Noise Ratio for reconstruction quality
SSIM	Structural Similarity Index for perceptual similarity

Both metrics are computed per image and averaged across the dataset.

🖼 Results Visualization

During testing, the script displays side-by-side comparisons:

Hazy Input | Predicted Output | Ground Truth


Example Output:

Epoch 70, Loss: 0.0084, PSNR: 34.82, SSIM: 0.9231

Test Set - Average PSNR: 35.42, Average SSIM: 0.9287

📊 Example Visualization (Python)

You can generate a diagram showing the model flow using Matplotlib:

import matplotlib.pyplot as plt

stages = [
    "Input (Hazy Image)",
    "CNN Encoder",
    "Vision Transformer (ViT)",
    "Mamba (GRU Refinement)",
    "Feature Fusion",
    "CNN Decoder",
    "Output (Dehazed Image)"
]

plt.figure(figsize=(12, 2))
for i, stage in enumerate(stages):
    plt.text(i * 1.5, 0, stage, fontsize=11, bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle='round,pad=0.3'))
    if i < len(stages) - 1:
        plt.arrow(i * 1.5 + 0.9, 0, 0.5, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.axis('off')
plt.title("Mamba-ViT Hybrid Dehazing Pipeline", fontsize=13, pad=10)
plt.show()

💾 Model Saving

After training:

torch.save(model.state_dict(), "mamba_vit_100dddehazing.pth")

🧍‍♂️ Author

Pranjal Pandey
B.Tech, Mechatronics and Automation
Indian Institute of Information Technology, Bhagalpur
📧 pranjal.230103027@iiitbh.ac.in

🧾 Citation

If you use this work or build upon it, please cite:

@software{pranjal2025mambavitdehazing,
  title={Mamba-ViT Hybrid Image Dehazing},
  author={Pandey, Pranjal},
  year={2025},
  url={https://github.com/<your-username>/Mamba-ViT-Dehazing}
}
