import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from ddpm_model import DDPM, UNet


class DDIM(DDPM):
    """DDIM (Denoising Diffusion Implicit Models) for faster sampling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def ddim_sample(self, x, t, t_prev, eta=0.0):
        """DDIM sampling step."""
        # Predict noise
        predicted_noise = self.model(x, t)
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Reshape for broadcasting
        alpha_t = alpha_t.reshape(-1, 1, 1, 1)
        alpha_t_prev = alpha_t_prev.reshape(-1, 1, 1, 1)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t)) * predicted_noise
        
        # Noise
        noise = torch.randn_like(x) if eta > 0 else 0
        
        # DDIM step
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + eta * torch.sqrt(1 - alpha_t) * noise
        
        return x_prev
    
    def ddim_sample_loop(self, shape, num_steps=50, eta=0.0):
        """Generate samples using DDIM sampling."""
        device = next(self.parameters()).device
        b = shape[0]
        
        # Create timestep schedule
        timesteps = np.linspace(self.timesteps - 1, 0, num_steps, dtype=int)
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_tensor = torch.full((b,), t_prev, device=device, dtype=torch.long)
            
            img = self.ddim_sample(img, t_tensor, t_prev_tensor, eta)
        
        return img
    
    def ddim_sample_fast(self, batch_size=16, num_steps=50, eta=0.0):
        """Fast DDIM sampling for MNIST."""
        return self.ddim_sample_loop((batch_size, 1, 28, 28), num_steps, eta)


def load_model(model_path, device):
    """Load DDPM model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model architecture
    unet = UNet(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
    )
    
    model = DDPM(
        model=unet,
        timesteps=1000,
        device=device
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def create_ddim_model(model_path, device):
    """Create DDIM model from DDPM checkpoint."""
    print(f"Loading DDIM model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model architecture
    unet = UNet(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
    )
    
    model = DDIM(
        model=unet,
        timesteps=1000,
        device=device
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def save_samples_grid(samples, save_path, title="Generated Samples", nrow=4, ncol=4):
    """Save samples in a grid format."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(min(nrow * ncol, len(samples))):
        row, col = i // ncol, i % ncol
        axes[row, col].imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(nrow * ncol, len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_individual_samples(samples, save_dir, prefix="sample"):
    """Save individual samples as separate images."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    for i, sample in enumerate(samples):
        # Convert to PIL Image
        img_array = (sample.squeeze().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(os.path.join(save_dir, f"{prefix}_{i:04d}.png"))


def generate_interpolation(model, device, num_steps=10, save_dir="interpolation"):
    """Generate interpolation between two random samples."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate two random samples
    z1 = torch.randn(1, 1, 28, 28, device=device)
    z2 = torch.randn(1, 1, 28, 28, device=device)
    
    # Create interpolation
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
    save_samples_grid(interpolations, os.path.join(save_dir, "interpolation.png"), 
                     title="Latent Space Interpolation", nrow=2, ncol=5)


def generate_ddim_comparison(ddpm_model, ddim_model, device, save_dir="ddim_comparison"):
    """Compare DDPM and DDIM sampling."""
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = 16
    
    # Generate samples with DDPM (1000 steps)
    print("Generating samples with DDPM...")
    with torch.no_grad():
        ddpm_samples = ddpm_model.sample(batch_size)
    
    # Generate samples with DDIM (50 steps)
    print("Generating samples with DDIM (50 steps)...")
    with torch.no_grad():
        ddim_samples_50 = ddim_model.ddim_sample_fast(batch_size, num_steps=50)
    
    # Generate samples with DDIM (10 steps)
    print("Generating samples with DDIM (10 steps)...")
    with torch.no_grad():
        ddim_samples_10 = ddim_model.ddim_sample_fast(batch_size, num_steps=10)
    
    # Save comparisons
    save_samples_grid(ddpm_samples, os.path.join(save_dir, "ddpm_samples.png"), 
                     title="DDPM Samples (1000 steps)", nrow=4, ncol=4)
    save_samples_grid(ddim_samples_50, os.path.join(save_dir, "ddim_samples_50.png"), 
                     title="DDIM Samples (50 steps)", nrow=4, ncol=4)
    save_samples_grid(ddim_samples_10, os.path.join(save_dir, "ddim_samples_10.png"), 
                     title="DDIM Samples (10 steps)", nrow=4, ncol=4)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(4, 12, figsize=(24, 8))
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # DDPM
        axes[row, col*3].imshow((ddpm_samples[i].squeeze().cpu() + 1) / 2, cmap='gray')
        axes[row, col*3].axis('off')
        if row == 0:
            axes[row, col*3].set_title('DDPM (1000)', fontsize=10)
        
        # DDIM 50
        axes[row, col*3+1].imshow((ddim_samples_50[i].squeeze().cpu() + 1) / 2, cmap='gray')
        axes[row, col*3+1].axis('off')
        if row == 0:
            axes[row, col*3+1].set_title('DDIM (50)', fontsize=10)
        
        # DDIM 10
        axes[row, col*3+2].imshow((ddim_samples_10[i].squeeze().cpu() + 1) / 2, cmap='gray')
        axes[row, col*3+2].axis('off')
        if row == 0:
            axes[row, col*3+2].set_title('DDIM (10)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate samples using DDPM/DDIM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--save_dir', type=str, default='generated_samples', help='Directory to save samples')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim', 'both'], 
                       help='Sampling method to use')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter')
    parser.add_argument('--interpolation', action='store_true', help='Generate interpolation')
    parser.add_argument('--comparison', action='store_true', help='Generate DDPM vs DDIM comparison')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.method in ['ddpm', 'both']:
        print("Loading DDPM model...")
        ddpm_model = load_model(args.model_path, device)
        
        print(f"Generating {args.num_samples} samples with DDPM...")
        with torch.no_grad():
            ddpm_samples = ddpm_model.sample(args.num_samples)
        
        save_samples_grid(ddpm_samples, os.path.join(args.save_dir, "ddpm_samples.png"), 
                         title=f"DDPM Samples ({args.num_samples} samples)", nrow=4, ncol=4)
        save_individual_samples(ddpm_samples, os.path.join(args.save_dir, "ddpm_individual"))
    
    if args.method in ['ddim', 'both']:
        print("Loading DDIM model...")
        ddim_model = create_ddim_model(args.model_path, device)
        
        print(f"Generating {args.num_samples} samples with DDIM ({args.ddim_steps} steps)...")
        with torch.no_grad():
            ddim_samples = ddim_model.ddim_sample_fast(args.num_samples, args.ddim_steps, args.eta)
        
        save_samples_grid(ddim_samples, os.path.join(args.save_dir, f"ddim_samples_{args.ddim_steps}.png"), 
                         title=f"DDIM Samples ({args.ddim_steps} steps)", nrow=4, ncol=4)
        save_individual_samples(ddim_samples, os.path.join(args.save_dir, "ddim_individual"))
    
    if args.interpolation:
        print("Generating interpolation...")
        if args.method in ['ddpm', 'both']:
            generate_interpolation(ddpm_model, device, save_dir=os.path.join(args.save_dir, "ddpm_interpolation"))
        if args.method in ['ddim', 'both']:
            generate_interpolation(ddim_model, device, save_dir=os.path.join(args.save_dir, "ddim_interpolation"))
    
    if args.comparison and args.method == 'both':
        print("Generating DDPM vs DDIM comparison...")
        generate_ddim_comparison(ddpm_model, ddim_model, device, 
                               save_dir=os.path.join(args.save_dir, "ddim_comparison"))
    
    print(f"Generated samples saved to {args.save_dir}")


if __name__ == '__main__':
    main()
