# DDPM (Denoising Diffusion Probabilistic Models) for MNIST

This repository contains a complete implementation of DDPM (Denoising Diffusion Probabilistic Models) for generating MNIST digits using PyTorch. The implementation includes both unconditional and conditional generation, with comprehensive training, evaluation, and inference scripts.

## Features

- **Unconditional DDPM**: Generate random MNIST digits
- **Conditional DDPM**: Generate specific digit classes
- **DDIM Sampling**: Faster sampling with fewer steps
- **Comprehensive Evaluation**: FID, IS, and other metrics
- **Advanced Visualization**: Training curves, sample quality metrics, noise schedules
- **Flexible Training**: Configurable hyperparameters and experiment tracking
- **Model Checkpointing**: Save and resume training

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ddpm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train an Unconditional DDPM Model

```bash
python train.py --epochs 100 --batch_size 128 --lr 2e-4
```

### 2. Train a Conditional DDPM Model

```bash
python train_conditional.py --epochs 100 --batch_size 128 --lr 2e-4
```

### 3. Generate Samples

```bash
# Generate samples with DDPM (1000 steps)
python inference.py --model_path checkpoints/experiment_name/best_model.pth --method ddpm

# Generate samples with DDIM (50 steps)
python inference.py --model_path checkpoints/experiment_name/best_model.pth --method ddim --ddim_steps 50

# Generate conditional samples
python inference.py --model_path checkpoints/experiment_name/best_model.pth --method ddpm --conditional
```

### 4. Evaluate Model

```bash
python eval.py --model_path checkpoints/experiment_name/best_model.pth --num_samples 1000
```

## File Structure

```
ddpm/
├── ddpm_model.py              # Core DDPM model implementation
├── conditional_ddpm.py        # Conditional DDPM implementation
├── train.py                   # Unconditional training script
├── train_conditional.py       # Conditional training script
├── inference.py               # Sample generation script
├── eval.py                    # Model evaluation script
├── utils.py                   # Utility functions for visualization
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Detailed Usage

### Training

#### Unconditional Training

```bash
python train.py \
    --epochs 100 \
    --batch_size 128 \
    --lr 2e-4 \
    --timesteps 1000 \
    --experiment_name "my_experiment" \
    --save_freq 10 \
    --eval_freq 5
```

#### Conditional Training

```bash
python train_conditional.py \
    --epochs 100 \
    --batch_size 128 \
    --lr 2e-4 \
    --timesteps 1000 \
    --num_classes 10 \
    --experiment_name "conditional_experiment" \
    --save_freq 10 \
    --eval_freq 5
```

#### Training Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 2e-4)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--device`: Device to use ('auto', 'cuda', 'cpu')
- `--experiment_name`: Name for the experiment (auto-generated if not provided)
- `--save_freq`: Frequency to save samples (default: 10)
- `--eval_freq`: Frequency to evaluate model (default: 5)
- `--resume`: Path to checkpoint to resume from

### Inference

#### Basic Sample Generation

```bash
python inference.py \
    --model_path checkpoints/experiment_name/best_model.pth \
    --num_samples 16 \
    --method ddpm
```

#### DDIM Sampling (Faster)

```bash
python inference.py \
    --model_path checkpoints/experiment_name/best_model.pth \
    --num_samples 16 \
    --method ddim \
    --ddim_steps 50 \
    --eta 0.0
```

#### Conditional Generation

```bash
python inference.py \
    --model_path checkpoints/experiment_name/best_model.pth \
    --num_samples 16 \
    --method ddpm \
    --conditional
```

#### Inference Arguments

- `--model_path`: Path to model checkpoint
- `--num_samples`: Number of samples to generate (default: 16)
- `--method`: Sampling method ('ddpm', 'ddim', 'both')
- `--ddim_steps`: Number of DDIM steps (default: 50)
- `--eta`: DDIM eta parameter (default: 0.0)
- `--interpolation`: Generate latent space interpolation
- `--comparison`: Generate DDPM vs DDIM comparison

### Evaluation

#### Basic Evaluation

```bash
python eval.py \
    --model_path checkpoints/experiment_name/best_model.pth \
    --num_samples 1000 \
    --batch_size 50
```

#### Evaluation Arguments

- `--model_path`: Path to model checkpoint
- `--num_samples`: Number of samples for evaluation (default: 1000)
- `--batch_size`: Batch size for evaluation (default: 50)
- `--save_dir`: Directory to save evaluation results

## Model Architecture

### DDPM Model

The implementation uses a U-Net architecture with:

- **Input/Output**: 1 channel (grayscale) 28x28 images
- **Model Channels**: 128 base channels
- **Channel Multipliers**: (1, 2, 2, 2) for different resolution levels
- **Residual Blocks**: 2 per resolution level
- **Time Embedding**: Sinusoidal position embeddings
- **Normalization**: Group normalization
- **Activation**: ReLU

### Conditional DDPM

The conditional version adds:

- **Class Embedding**: Learnable embeddings for each digit class (0-9)
- **Combined Embedding**: Time + class embeddings
- **Conditional Residual Blocks**: Modified to use combined embeddings

## Training Process

1. **Data Loading**: MNIST dataset with normalization to [-1, 1]
2. **Noise Schedule**: Linear beta schedule from 0.0001 to 0.02
3. **Training Loss**: MSE loss between predicted and actual noise
4. **Optimizer**: AdamW with weight decay
5. **Scheduler**: Cosine annealing learning rate schedule
6. **Monitoring**: TensorBoard logging and sample generation

## Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Measures quality and diversity
- **IS (Inception Score)**: Measures quality and diversity
- **Sample Quality Metrics**: Pixel statistics and diversity measures
- **Visual Inspection**: Generated sample grids and comparisons

## Visualization Features

- **Training Curves**: Loss plots over time
- **Sample Grids**: Generated samples in grid format
- **Noise Schedules**: Visualization of diffusion parameters
- **Sample Quality**: Histograms and statistics
- **Class Comparisons**: Side-by-side class generation
- **Interpolation**: Latent space interpolation
- **Denoising Process**: Step-by-step denoising visualization

## Experiment Tracking

Each experiment creates a timestamped directory with:

- **Checkpoints**: Model weights and training state
- **Samples**: Generated images at different epochs
- **Logs**: TensorBoard logs for monitoring
- **Config**: Training configuration and hyperparameters
- **Visualizations**: Training curves and sample quality metrics
- **Summary**: Final training statistics

## Advanced Usage

### Custom Noise Schedules

Modify the beta schedule in the DDPM class:

```python
# Custom beta schedule
self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
```

### Different Architectures

Modify the U-Net architecture:

```python
unet = UNet(
    in_channels=1,
    out_channels=1,
    model_channels=256,  # Increase model capacity
    num_res_blocks=3,    # More residual blocks
    channel_mult=(1, 2, 4, 4),  # Different channel multipliers
)
```

### Custom Datasets

Modify the data loading in `get_data_loaders()`:

```python
# Custom dataset
train_dataset = datasets.CustomDataset(
    root='./data', 
    transform=transform
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model size
2. **Slow Training**: Use fewer timesteps or smaller model
3. **Poor Sample Quality**: Increase training epochs or adjust learning rate
4. **Inconsistent Results**: Set random seeds for reproducibility

### Performance Tips

1. **Use GPU**: Training is much faster on GPU
2. **DDIM Sampling**: Use DDIM for faster inference
3. **Batch Size**: Larger batch sizes generally improve training
4. **Learning Rate**: Start with 2e-4 and adjust based on results

## Results

The model typically achieves:

- **Training Loss**: ~0.02-0.05 after 100 epochs
- **Sample Quality**: High-quality MNIST digits
- **FID**: ~10-20 (lower is better)
- **IS**: ~2.0-3.0 (higher is better)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DDPM paper: "Denoising Diffusion Probabilistic Models" by Ho et al.
- DDIM paper: "Denoising Diffusion Implicit Models" by Song et al.
- PyTorch and torchvision for the deep learning framework
- MNIST dataset for training data

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ddpm_mnist,
  title={DDPM Implementation for MNIST},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ddpm}
}
```
