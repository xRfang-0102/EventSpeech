<div align="center">

# EventSpeech


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache](https://img.shields.io/badge/License-Apache-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ / 12.1+
- cuDNN 8.x

### Setup Environment

```bash
# Clone repository
git clone https://github.com/your-username/eventspeech.git
cd eventspeech

# Create conda environment
conda create -n eventspeech python=3.9
conda activate eventspeech

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Preprocessing Pipeline

**Step 1: Audio Preprocessing**

```bash
python data_scripts/preprocess_audio.py \
    --input_dir /path/to/raw_audio \
    --output_dir data/audio
```

Pipeline: 80Hz highpass filter → spectral subtraction denoise → resample to 22050Hz → normalize to -23 LUFS

**Step 2: Event Voxelization**

```bash
python data_scripts/v2e_wrapper.py \
    --input_dir /path/to/raw_video \
    --output_dir data/events
```

V2E parameters: contrast threshold θ=0.15, leak current=0, 20ms time bins

**Step 3: Create Manifest Files**

Organize your data directory as follows:

```
data/
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_manifest.jsonl
├── audio/
│   ├── sample_000001.pt
│   └── ...
├── events/
│   ├── sample_000001.npy
│   └── ...
└── text/
    ├── sample_000001.npy
    └── ...
```

Each manifest line (JSONL format):
```json
{"id": "sample_000001", "emotion_id": 0, "speaker_id": 0}
```

## Training

### Multi-GPU Training (Recommended)

```bash
torchrun --nproc_per_node=6 train.py \
    --config configs/eventspeech_a100.yaml \
    --world_size 6
```

### Single GPU Training

```bash
python train.py \
    --config configs/base_config.yaml \
    --world_size 1
```

### Resume from Checkpoint

```bash
python train.py \
    --config configs/eventspeech_a100.yaml \
    --checkpoint checkpoints/latest_checkpoint.pth
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.8, β₂=0.99) |
| Learning Rate | 2×10⁻⁴ with OneCycleLR |
| Warmup Steps | 10,000 |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |
| Batch Size | 32 (effective) |
| Precision | FP16 Mixed Precision |
| Epochs | 200 |

### Loss Configuration

| Loss | Weight | Schedule |
|------|--------|----------|
| Reconstruction | 1.0 | Constant |
| KL Divergence | 0.8 | Linear annealing (60 epochs) |
| Alignment | 0.3 | Constant |
| Flow Matching | 0.2 | Constant |
| Adversarial | 0.1 | After epoch 90 |

## Inference

### Generate Mel Spectrograms

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/eventspeech_a100.yaml \
    --mode generate \
    --data_dir data \
    --output_dir generated \
    --num_steps 20 \
    --solver euler \
    --batch_size 16
```

### Single Sample Inference

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/eventspeech_a100.yaml \
    --mode single \
    --data_dir data
```

### ODE Solver Options

| Solver | Steps | Speed | Quality |
|--------|-------|-------|---------|
| `euler` | 20 | Faster | Good |
| `rk4` | 20 | Slower | Better |

## Evaluation

### Run Full Evaluation

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/eventspeech_a100.yaml \
    --mode eval \
    --data_dir data \
    --output_dir results \
    --num_steps 20
```

### Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| MCD | ↓ | Mel-Cepstral Distortion (acoustic fidelity) |
| F0-RMSE | ↓ | F0 RMSE in log space (prosody naturalness) |
| LSE-D | ↓ | Lip Sync Error Distance |
| LSE-C | ↑ | Lip Sync Error Confidence |
| WER | ↓ | Word Error Rate via Whisper-large-v3 |

Results are saved to `results/evaluation_results.json`.

## Configuration

### Base Configuration

Edit `configs/base_config.yaml` for general settings:

```yaml
audio:
  sample_rate: 22050
  n_mels: 80
  hop_length: 256

training:
  batch_size: 32
  num_epochs: 200
  learning_rate: 0.0002
```

### Distributed Configuration

Edit `configs/eventspeech_a100.yaml` for multi-GPU settings:

```yaml
ddp:
  backend: nccl
  find_unused_parameters: false

training:
  batch_size: 6  # per GPU
  num_workers: 8
  pin_memory: true
```

## Project Structure

```
EventSpeech/
├── configs/                    # Configuration files
│   ├── base_config.yaml
│   └── eventspeech_a100.yaml
├── data_scripts/               # Data preprocessing
│   ├── v2e_wrapper.py          # V2E simulator
│   ├── voxelizer.py            # Event voxelization
│   └── preprocess_audio.py     # Audio preprocessing
├── datasets/                   # Dataset loaders
│   └── evtspk_dataset.py
├── models/                     # Model architectures
│   ├── event_encoder.py
│   ├── audio_encoder.py
│   ├── alignment.py
│   ├── vits_modules.py
│   └── cfm_decoder.py
├── losses/                     # Loss functions
│   └── multi_task_loss.py
├── utils/                      # Utilities
│   ├── logger.py               # WandB logging
│   ├── ddp_init.py             # DDP setup
│   └── evaluator.py            # Metrics
├── train.py                    # Training script
├── inference.py                # Inference script
└── requirements.txt
```

## License

Apache License. See [LICENSE](LICENSE) for details.
