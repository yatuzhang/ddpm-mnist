import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os
import json
from typing import List, Tuple, Optional, Dict, Any
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image


def denormalize_images(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1]."""
    return torch.clamp((images * std + mean), 0, 1)


def normalize_images(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Normalize images from [0, 1] to [-1, 1]."""
    return (images - mean) / std


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 8, 
                   title: str = "", figsize: Tuple[int, int] = (12, 12)):
    """Save a grid of images."""
    # Denormalize if needed
    if images.min() < 0:
        images = denormalize_images(images)
    
    # Create grid
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Convert to numpy and transpose for matplotlib
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        plt.title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_individual_images(images: torch.Tensor, save_dir: str, prefix: str = "image"):
    """Save individual images."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize if needed
    if images.min() < 0:
        images = denormalize_images(images)
    
    for i, img in enumerate(images):
        # Convert to PIL Image
        if img.shape[0] == 1:  # Grayscale
            img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='L')
        else:  # RGB
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        
        img_pil.save(os.path.join(save_dir, f"{prefix}_{i:04d}.png"))


def create_gif_from_images(image_paths: List[str], save_path: str, duration: int = 500):
    """Create a GIF from a list of image paths."""
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )


def plot_training_curves(log_file: str, save_path: str = None):
    """Plot training curves from log file."""
    # This would need to be adapted based on your logging format
    # For now, we'll create a placeholder function
    pass


def visualize_noise_schedule(model, save_path: str = "noise_schedule.png"):
    """Visualize the noise schedule used in the model."""
    betas = model.betas.cpu().numpy()
    alphas = model.alphas.cpu().numpy()
    alphas_cumprod = model.alphas_cumprod.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Beta schedule
    axes[0, 0].plot(betas)
    axes[0, 0].set_title('Beta Schedule')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Beta')
    
    # Alpha schedule
    axes[0, 1].plot(alphas)
    axes[0, 1].set_title('Alpha Schedule')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Alpha')
    
    # Alpha cumprod schedule
    axes[1, 0].plot(alphas_cumprod)
    axes[1, 0].set_title('Alpha Cumprod Schedule')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Alpha Cumprod')
    
    # Sqrt alpha cumprod schedule
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    axes[1, 1].plot(sqrt_alphas_cumprod)
    axes[1, 1].set_title('Sqrt Alpha Cumprod Schedule')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Sqrt Alpha Cumprod')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_denoising_process(model, x_start: torch.Tensor, timesteps: List[int], 
                              save_path: str = "denoising_process.png"):
    """Visualize the denoising process."""
    model.eval()
    device = next(model.parameters()).device
    x_start = x_start.to(device)
    
    # Add noise at different timesteps
    noisy_images = []
    with torch.no_grad():
        for t in timesteps:
            t_tensor = torch.full((x_start.shape[0],), t, device=device, dtype=torch.long)
            x_t, _ = model.q_sample(x_start, t_tensor)
            noisy_images.append(x_t)
    
    # Create visualization
    num_timesteps = len(timesteps)
    fig, axes = plt.subplots(2, num_timesteps, figsize=(num_timesteps * 3, 6))
    
    if num_timesteps == 1:
        axes = axes.reshape(2, 1)
    
    # Original images
    for i in range(min(2, x_start.shape[0])):
        axes[i, 0].imshow(denormalize_images(x_start[i]).squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
    
    # Noisy images at different timesteps
    for j, (t, noisy_img) in enumerate(zip(timesteps, noisy_images)):
        for i in range(min(2, noisy_img.shape[0])):
            axes[i, j].imshow(denormalize_images(noisy_img[i]).squeeze().cpu().numpy(), cmap='gray')
            axes[i, j].set_title(f't={t}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_denoising_animation(model, x_start: torch.Tensor, save_path: str = "denoising_animation.gif"):
    """Create an animation of the denoising process."""
    model.eval()
    device = next(model.parameters()).device
    x_start = x_start.to(device)
    
    # Generate denoising sequence
    images = []
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn_like(x_start)
        images.append(denormalize_images(x).squeeze().cpu().numpy())
        
        # Denoise step by step
        for i in reversed(range(model.timesteps)):
            t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            x = model.p_sample(x, t)
            if i % 50 == 0:  # Save every 50 steps
                images.append(denormalize_images(x).squeeze().cpu().numpy())
    
    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    def animate(frame):
        ax.clear()
        ax.imshow(images[frame], cmap='gray')
        ax.set_title(f'Denoising Step {frame * 50}')
        ax.axis('off')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=200, repeat=True)
    anim.save(save_path, writer='pillow', fps=5)


def calculate_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate the number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def save_model_info(model: nn.Module, save_path: str):
    """Save model information to a JSON file."""
    info = calculate_model_parameters(model)
    
    # Add model architecture info
    info['model_class'] = model.__class__.__name__
    info['model_str'] = str(model)
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)


def load_model_info(save_path: str) -> Dict[str, Any]:
    """Load model information from a JSON file."""
    with open(save_path, 'r') as f:
        return json.load(f)


def create_latent_space_visualization(model, device, save_path: str = "latent_space.png"):
    """Create a 2D visualization of the latent space."""
    # Generate samples from different parts of the latent space
    z1 = torch.randn(1, 1, 28, 28, device=device)
    z2 = torch.randn(1, 1, 28, 28, device=device)
    
    # Create interpolation
    num_steps = 10
    interpolations = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Generate sample from interpolated noise
        with torch.no_grad():
            sample = model.p_sample_loop(z_interp.shape)
            interpolations.append(sample)
    
    # Save interpolation
    interpolations = torch.cat(interpolations, dim=0)
    save_image_grid(interpolations, save_path, nrow=num_steps, 
                   title="Latent Space Interpolation")


def compare_sampling_methods(model, ddim_model, device, save_path: str = "sampling_comparison.png"):
    """Compare different sampling methods."""
    batch_size = 16
    
    # Generate samples with different methods
    with torch.no_grad():
        # DDPM (1000 steps)
        ddpm_samples = model.sample(batch_size)
        
        # DDIM (50 steps)
        ddim_samples_50 = ddim_model.ddim_sample_fast(batch_size, num_steps=50)
        
        # DDIM (10 steps)
        ddim_samples_10 = ddim_model.ddim_sample_fast(batch_size, num_steps=10)
    
    # Create comparison grid
    all_samples = torch.cat([ddpm_samples, ddim_samples_50, ddim_samples_10], dim=0)
    
    # Create labels
    labels = ['DDPM (1000)'] * batch_size + ['DDIM (50)'] * batch_size + ['DDIM (10)'] * batch_size
    
    # Save comparison
    save_image_grid(all_samples, save_path, nrow=batch_size, 
                   title="Sampling Method Comparison")


def create_training_visualization(train_losses: List[float], test_losses: List[float], 
                                save_path: str = "training_curves.png"):
    """Create training curves visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss', color='blue')
    ax.plot(epochs, test_losses, label='Test Loss', color='red')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Test Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_sample_quality_metrics(samples: torch.Tensor, save_path: str = "sample_quality.png"):
    """Create visualization of sample quality metrics."""
    # Denormalize samples
    samples = denormalize_images(samples)
    
    # Calculate basic statistics
    mean_intensity = samples.mean().item()
    std_intensity = samples.std().item()
    min_intensity = samples.min().item()
    max_intensity = samples.max().item()
    
    # Create histogram
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sample grid
    grid = make_grid(samples[:16], nrow=4, padding=2, normalize=False)
    axes[0, 0].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Sample Grid')
    axes[0, 0].axis('off')
    
    # Intensity histogram
    axes[0, 1].hist(samples.flatten().cpu().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title('Intensity Distribution')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    
    # Statistics
    stats_text = f"""
    Mean: {mean_intensity:.4f}
    Std: {std_intensity:.4f}
    Min: {min_intensity:.4f}
    Max: {max_intensity:.4f}
    """
    axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes, 
                    fontsize=12, verticalalignment='center')
    axes[1, 0].set_title('Sample Statistics')
    axes[1, 0].axis('off')
    
    # Sample diversity (variance across samples)
    sample_vars = samples.var(dim=0).mean().item()
    axes[1, 1].bar(['Sample Variance'], [sample_vars], color='green', alpha=0.7)
    axes[1, 1].set_title('Sample Diversity')
    axes[1, 1].set_ylabel('Average Variance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_noise_visualization(model, x_start: torch.Tensor, save_path: str = "noise_visualization.png"):
    """Visualize the noise addition process."""
    device = next(model.parameters()).device
    x_start = x_start.to(device)
    
    # Add noise at different timesteps
    timesteps = [0, 100, 200, 500, 800, 999]
    noisy_images = []
    
    with torch.no_grad():
        for t in timesteps:
            t_tensor = torch.full((x_start.shape[0],), t, device=device, dtype=torch.long)
            x_t, noise = model.q_sample(x_start, t_tensor)
            noisy_images.append((x_t, noise))
    
    # Create visualization
    fig, axes = plt.subplots(3, len(timesteps), figsize=(len(timesteps) * 3, 9))
    
    for j, (t, (x_t, noise)) in enumerate(zip(timesteps, noisy_images)):
        # Original
        axes[0, j].imshow(denormalize_images(x_start[0]).squeeze().cpu().numpy(), cmap='gray')
        axes[0, j].set_title(f'Original (t=0)')
        axes[0, j].axis('off')
        
        # Noisy
        axes[1, j].imshow(denormalize_images(x_t[0]).squeeze().cpu().numpy(), cmap='gray')
        axes[1, j].set_title(f'Noisy (t={t})')
        axes[1, j].axis('off')
        
        # Noise
        axes[2, j].imshow(denormalize_images(noise[0]).squeeze().cpu().numpy(), cmap='gray')
        axes[2, j].set_title(f'Noise (t={t})')
        axes[2, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_model_architecture_diagram(model: nn.Module, save_path: str = "model_architecture.png"):
    """Create a diagram of the model architecture."""
    # This is a simplified visualization
    # For a more detailed diagram, you might want to use tools like torchviz
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a simple text-based diagram
    model_str = str(model)
    lines = model_str.split('\n')
    
    y_pos = 0.9
    for line in lines[:20]:  # Show first 20 lines
        ax.text(0.05, y_pos, line, transform=ax.transAxes, fontsize=10, 
                fontfamily='monospace', verticalalignment='top')
        y_pos -= 0.04
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Model Architecture', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comprehensive_visualization(model, device, save_dir: str = "comprehensive_viz"):
    """Create a comprehensive visualization of the model."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate some samples
    with torch.no_grad():
        samples = model.sample(16)
    
    # Create various visualizations
    save_image_grid(samples, os.path.join(save_dir, "sample_grid.png"), 
                   title="Generated Samples", nrow=4)
    
    visualize_noise_schedule(model, os.path.join(save_dir, "noise_schedule.png"))
    
    create_sample_quality_metrics(samples, os.path.join(save_dir, "sample_quality.png"))
    
    create_model_architecture_diagram(model, os.path.join(save_dir, "model_architecture.png"))
    
    # Save model info
    save_model_info(model, os.path.join(save_dir, "model_info.json"))
    
    print(f"Comprehensive visualization saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions for DDPM visualization and analysis")
    print("Import this module to use the visualization functions")
