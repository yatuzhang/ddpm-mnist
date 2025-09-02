#!/usr/bin/env python3
"""
Fast CPU training script for DDPM on MNIST.
Optimized for speed with smaller model and fewer timesteps.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def get_data_loaders(batch_size=256, num_workers=4):
    """Get MNIST data loaders with larger batch size for CPU efficiency."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, epoch, max_batches=50):
    """Train for one epoch with limited batches for speed."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        if batch_idx >= max_batches:  # Limit batches for speed
            break
            
        data = data.to(device)
        
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / min(max_batches, len(train_loader))


def save_samples(model, device, epoch, num_samples=16, save_dir='samples'):
    """Generate and save sample images."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Use fewer sampling timesteps for speed
        samples = model.sample(batch_size=num_samples)
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
    
    # Create a grid of samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/samples_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fast CPU DDPM Training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (larger for CPU efficiency)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (higher for faster convergence)')
    parser.add_argument('--timesteps', type=int, default=200, help='Number of timesteps (fewer for speed)')
    parser.add_argument('--max_batches', type=int, default=50, help='Max batches per epoch')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Create smaller, faster model
    model = Unet(
        dim=32,  # Smaller model
        channels=1,
        dim_mults=(1, 2),  # Fewer layers
        flash_attn=False  # Disable for CPU
    )
    
    # Create diffusion model with fewer timesteps
    diffusion = GaussianDiffusion(
        model,
        image_size=28,
        timesteps=args.timesteps,  # Fewer timesteps
        sampling_timesteps=50,     # Faster sampling
        objective='pred_v'
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in diffusion.parameters())} parameters")
    
    # Create optimizer with higher learning rate
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(diffusion, train_loader, optimizer, device, epoch, args.max_batches)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Time = {epoch_time:.1f}s')
        
        # Save samples every 5 epochs
        if epoch % 5 == 0:
            save_samples(diffusion, device, epoch)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, f'checkpoints/fast_cpu_epoch_{epoch:03d}.pth')
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.1f}s')
    
    # Generate final samples
    save_samples(diffusion, device, args.epochs)


if __name__ == '__main__':
    main()
