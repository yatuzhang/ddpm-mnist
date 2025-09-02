import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from ddpm_model_fixed import DDPM, SimpleUNet


def get_data_loaders(batch_size=128, num_workers=4):
    """Get MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
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


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        optimizer.zero_grad()
        loss = model.training_losses(data)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            loss = model.training_losses(data)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def save_samples(model, device, epoch, num_samples=16, save_dir='samples'):
    """Generate and save sample images."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples)
        samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
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
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST (Simple Version)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory to save samples')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Create model
    unet = SimpleUNet(
        in_channels=1,
        out_channels=1,
        model_channels=64,  # Smaller model for faster training
        num_res_blocks=1,
        dropout=0.1,
        channel_mult=(1, 2, 2),  # Simpler architecture
    )
    
    model = DDPM(
        model=unet,
        timesteps=args.timesteps,
        device=device
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        test_loss = evaluate(model, test_loader, device)
        
        print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}')
        
        # Save samples
        if epoch % 2 == 0:
            save_samples(model, device, epoch, save_dir=args.sample_dir)
        
        # Save checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, f'{args.save_dir}/best_model.pth')
    
    # Generate final samples
    save_samples(model, device, args.epochs, save_dir=args.sample_dir)
    
    print('Training completed!')
    print(f'Best test loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
