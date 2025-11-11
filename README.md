# 🧠 Mamba-ViT Hybrid Image Dehazing

## 🌫 Overview

This repository implements a **hybrid deep learning model** for **image dehazing** that integrates a **Vision Transformer (ViT)** encoder and a **Mamba recurrent refinement module**.

The model enhances feature understanding and spatial consistency by combining:

- 🧩 **Local texture extraction** — CNN Encoder  
- 🌍 **Global context modeling** — Vision Transformer  
- 🔁 **Sequential feature refinement** — Mamba GRU  
- 🎨 **Image reconstruction** — CNN Decoder  

It is trained and evaluated on the **SOTS (RESIDE)** dataset and outputs clean, dehazed images with **high PSNR and SSIM**.

---

## 🚀 Features

✅ DEM-Free, purely image-based model  
✅ End-to-end training with PyTorch  
✅ Supports PSNR & SSIM metric evaluation  
✅ Visual comparison (Hazy → Dehazed → Ground Truth)  
✅ Modular architecture (ViT + Mamba + CNN fusion)  
✅ Lightweight — runs smoothly on mid-range GPUs  



## 🧠 Model Components

| Component | Description |
|------------|-------------|
| **Encoder** | Two-layer CNN for local feature extraction |
| **SimpleViT** | Lightweight Vision Transformer for global context summarization |
| **SimpleMamba** | GRU-based sequential block to refine ViT embeddings |
| **Decoder** | CNN block for reconstructing the clean image |
| **Fusion** | Adds refined global features back to encoder outputs |

---

## 📂 Dataset

**Dataset Used:** [RESIDE SOTS Outdoor](https://sites.google.com/view/reside-dehaze-datasets)

### Folder Structure
├── hazy_processed/ # input hazy images
├── GT/ # corresponding clear ground truth images



---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/Mamba-ViT-Dehazing.git
cd Mamba-ViT-Dehazing
pip install torch torchvision scikit-learn matplotlib Pillow
🧪 Training
bash

python train_dehazing.py
Automatically splits data into 80% Train / 20% Test

Trains for 70 epochs

Saves model as:

bash
Copy code
mamba_vit_100dddehazing.pth
📈 Evaluation Metrics
Metric	Description
PSNR	Peak Signal-to-Noise Ratio – measures reconstruction fidelity
SSIM	Structural Similarity Index – evaluates perceptual similarity

Both metrics are computed per image and averaged across the dataset.

🖼 Results Visualization
During testing, side-by-side comparisons are displayed as:

Hazy Input → Predicted Output → Ground Truth

Example Output:

yaml
Copy code
Epoch 70, Loss: 0.0084, PSNR: 34.82, SSIM: 0.9231
Test Set — Avg PSNR: 35.42, Avg SSIM: 0.9287
📊 Example Visualization (Optional)
You can visualize the model flow using Matplotlib:

python

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
    plt.text(i * 1.5, 0, stage, fontsize=11,
             bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle='round,pad=0.3'))
    if i < len(stages) - 1:
        plt.arrow(i * 1.5 + 0.9, 0, 0.5, 0,
                  head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.axis('off')
plt.title("Mamba-ViT Hybrid Dehazing Pipeline", fontsize=13, pad=10)
plt.show()
💾 Model Saving
After training, the model weights are saved as:

python

torch.save(model.state_dict(), "mamba_vit_100dddehazing.pth")
🧍‍♂️ Author
Pranjal Pandey
B.Tech — Mechatronics and Automation
Indian Institute of Information Technology, Bhagalpur
📧 pranjal.230103027@iiitbh.ac.in

🧾 Citation
If you use this repository or build upon it, please cite:

bibtex

@software{pranjal2025mambavitdehazing,
  title  = {Mamba-ViT Hybrid Image Dehazing},
  author = {Pandey, Pranjal},
  year   = {2025},
  url    = {https://github.com/<your-username>/Mamba-ViT-Dehazing}
}
