import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import numpy as np

from ddpm_model import SinusoidalPositionEmbeddings, ResidualBlock


class ConditionalUNet(nn.Module):
    """Conditional U-Net architecture for DDPM with class conditioning."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 2, 2),
        conv_resample: bool = True,
        num_heads: int = 4,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        num_classes: int = 10,  # Number of classes for conditioning
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_classes = num_classes
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ConditionalResidualBlock(ch, model_channels * mult, time_embed_dim, dropout)
                )
                ch = model_channels * mult
                
            if level != len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ds *= 2
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle block
        self.middle_block1 = ConditionalResidualBlock(ch, ch, time_embed_dim, dropout)
        self.middle_block2 = ConditionalResidualBlock(ch, ch, time_embed_dim, dropout)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ConditionalResidualBlock(ch + model_channels * mult, model_channels * mult, time_embed_dim, dropout)
                )
                ch = model_channels * mult
                
            if level != 0:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.up_samples.append(nn.Identity())
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.ReLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Class embedding
        class_emb = self.class_embed(y)
        
        # Combine time and class embeddings
        emb = time_emb + class_emb
        
        # Input
        h = self.input_conv(x)
        hs = [h]
        
        # Downsampling
        for i, (down_block, down_sample) in enumerate(zip(self.down_blocks, self.down_samples)):
            h = down_block(h, emb)
            hs.append(h)
            h = down_sample(h)
        
        # Middle
        h = self.middle_block1(h, emb)
        h = self.middle_block2(h, emb)
        
        # Upsampling
        for up_block, up_sample in zip(self.up_blocks, self.up_samples):
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, emb)
            h = up_sample(h)
        
        # Output
        return self.output_conv(h)


class ConditionalResidualBlock(nn.Module):
    """Residual block with time and class embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.relu(h)
        
        return h + self.residual_conv(x)


class ConditionalDDPM(nn.Module):
    """Conditional Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.num_classes = num_classes
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> tuple:
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t, y)."""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t, y) / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape: tuple, y: torch.Tensor) -> torch.Tensor:
        """Generate samples by iteratively denoising."""
        device = next(self.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, y)
            
        return img

    def sample(self, batch_size: int = 16, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate samples with optional class conditioning."""
        if y is None:
            # Generate random classes
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        return self.p_sample_loop((batch_size, 1, 28, 28), y)

    def sample_class(self, class_label: int, batch_size: int = 16) -> torch.Tensor:
        """Generate samples for a specific class."""
        y = torch.full((batch_size,), class_label, device=self.device, dtype=torch.long)
        return self.sample(batch_size, y)

    def training_losses(self, x_start: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute training losses."""
        b = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to x_start
        x_t, _ = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t, y)
        
        # Compute loss
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss


class ConditionalDDIM(ConditionalDDPM):
    """Conditional DDIM for faster sampling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def ddim_sample(self, x, t, t_prev, y, eta=0.0):
        """DDIM sampling step."""
        # Predict noise
        predicted_noise = self.model(x, t, y)
        
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
    
    def ddim_sample_loop(self, shape, y, num_steps=50, eta=0.0):
        """Generate samples using DDIM sampling."""
        device = next(self.parameters()).device
        b = shape[0]
        
        # Create timestep schedule
        timesteps = np.linspace(self.timesteps - 1, 0, num_steps, dtype=int)
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_tensor = torch.full((b,), t_prev, device=device, dtype=torch.long)
            
            img = self.ddim_sample(img, t_tensor, t_prev_tensor, y, eta)
        
        return img
    
    def ddim_sample_fast(self, batch_size=16, y=None, num_steps=50, eta=0.0):
        """Fast DDIM sampling for MNIST."""
        if y is None:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        return self.ddim_sample_loop((batch_size, 1, 28, 28), y, num_steps, eta)


def create_conditional_model(num_classes=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Create a conditional DDPM model."""
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_classes=num_classes,
    )
    
    model = ConditionalDDPM(
        model=unet,
        timesteps=1000,
        device=device,
        num_classes=num_classes
    )
    
    return model


def create_conditional_ddim_model(num_classes=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Create a conditional DDIM model."""
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_classes=num_classes,
    )
    
    model = ConditionalDDIM(
        model=unet,
        timesteps=1000,
        device=device,
        num_classes=num_classes
    )
    
    return model
