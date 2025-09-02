import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_data_loaders(batch_size=64, num_workers=4):
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


def create_model(device):
    """Create DDPM model."""
    # Create U-Net model
    model = Unet(
        dim=64,  # Model dimension
        channels=1,  # Input channels (grayscale)
        dim_mults=(1, 2, 4),  # Dimension multipliers
        flash_attn=True  # Use flash attention if available
    )
    
    # Create diffusion model
    diffusion = GaussianDiffusion(
        model,
        image_size=28,  # MNIST image size
        timesteps=1000,
        sampling_timesteps=250,  # Number of sampling timesteps
        objective='pred_v'  # Use v-parameterization
    ).to(device)
    
    return diffusion


def save_samples_grid(samples, save_path, title="Generated Samples", nrow=4, ncol=4):
    """Save samples in a grid format."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
    
    # Handle case where we have only one subplot
    if nrow == 1 and ncol == 1:
        axes = [axes]
    elif nrow == 1 or ncol == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(min(nrow * ncol, len(samples))):
        axes[i].imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(nrow * ncol, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_untrained_model():
    """Demonstrate DDPM with an untrained model."""
    print("=== DDPM Demonstration with Untrained Model ===")
    
    # Use MPS (Apple Silicon GPU) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create model
    diffusion = create_model(device)
    print(f"Model created with {sum(p.numel() for p in diffusion.parameters())} parameters")
    
    # Create output directory
    os.makedirs('demo_outputs', exist_ok=True)
    
    # Generate samples with untrained model
    print("Generating samples with untrained model...")
    with torch.no_grad():
        samples = diffusion.sample(batch_size=16)
    
    # Save samples
    save_samples_grid(samples, 'demo_outputs/untrained_samples.png', 
                     title="Untrained DDPM Samples", nrow=4, ncol=4)
    print("Untrained samples saved to demo_outputs/untrained_samples.png")
    
    return diffusion


def train_quick_demo(diffusion, device, epochs=3):
    """Quick training demonstration."""
    print(f"\n=== Quick Training Demo ({epochs} epochs) ===")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Create optimizer
    optimizer = optim.AdamW(diffusion.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(epochs):
        diffusion.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            optimizer.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Only train on first 100 batches for demo
            if batch_idx >= 100:
                break
        
        avg_loss = total_loss / min(100, len(train_loader))
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')
        
        # Generate samples after each epoch
        with torch.no_grad():
            samples = diffusion.sample(batch_size=16)
        
        save_samples_grid(samples, f'demo_outputs/epoch_{epoch}_samples.png', 
                         title=f"DDPM Samples - Epoch {epoch}", nrow=4, ncol=4)
        print(f"Epoch {epoch} samples saved to demo_outputs/epoch_{epoch}_samples.png")
    
    return diffusion


def demonstrate_noise_process(diffusion, device):
    """Demonstrate the noise addition process."""
    print("\n=== Noise Process Demonstration ===")
    
    # Get a sample from MNIST
    train_loader, _ = get_data_loaders(batch_size=1)
    original_image = next(iter(train_loader))[0][0:1].to(device)
    
    # Show original
    save_samples_grid(original_image, 'demo_outputs/original_image.png', 
                     title="Original MNIST Image", nrow=1, ncol=1)
    
    # Add noise at different timesteps
    timesteps = [0, 100, 200, 500, 800, 999]
    noisy_images = []
    
    with torch.no_grad():
        for t in timesteps:
            # Add noise manually
            noise = torch.randn_like(original_image)
            alpha_t = diffusion.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            noisy_image = sqrt_alpha_t * original_image + sqrt_one_minus_alpha_t * noise
            noisy_images.append(noisy_image)
    
    # Save noise progression
    all_images = [original_image] + noisy_images
    all_images = torch.cat(all_images, dim=0)
    
    save_samples_grid(all_images, 'demo_outputs/noise_progression.png', 
                     title="Noise Addition Process", nrow=1, ncol=7)
    print("Noise progression saved to demo_outputs/noise_progression.png")


def main():
    """Main demonstration function."""
    print("DDPM MNIST Demonstration")
    print("=" * 50)
    
    # Demonstrate with untrained model
    diffusion = demonstrate_untrained_model()
    
    # Quick training demo
    # Use MPS (Apple Silicon GPU) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    trained_diffusion = train_quick_demo(diffusion, device, epochs=3)
    
    # Demonstrate noise process
    demonstrate_noise_process(trained_diffusion, device)
    
    print("\n" + "=" * 50)
    print("Demonstration completed!")
    print("Check the 'demo_outputs' directory for generated images.")
    print("\nGenerated files:")
    print("- untrained_samples.png: Random samples from untrained model")
    print("- epoch_X_samples.png: Samples after each training epoch")
    print("- original_image.png: Original MNIST image")
    print("- noise_progression.png: Noise addition process")


if __name__ == '__main__':
    main()
