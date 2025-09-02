import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
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


class SimpleUNet(nn.Module):
    """Simplified U-Net architecture for DDPM."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 2, 2),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            # Add residual blocks for this level
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(ch, model_channels * mult, time_embed_dim, dropout)
                )
                ch = model_channels * mult
            
            # Add downsampling (except for the last level)
            if level != len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle blocks
        self.middle_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.middle_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # Add residual blocks for this level
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(ch + model_channels * mult, model_channels * mult, time_embed_dim, dropout)
                )
                ch = model_channels * mult
            
            # Add upsampling (except for the first level)
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

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input
        h = self.input_conv(x)
        hs = [h]
        
        # Downsampling
        for i, (down_block, down_sample) in enumerate(zip(self.down_blocks, self.down_samples)):
            h = down_block(h, time_emb)
            hs.append(h)
            h = down_sample(h)
        
        # Middle
        h = self.middle_block1(h, time_emb)
        h = self.middle_block2(h, time_emb)
        
        # Upsampling
        for up_block, up_sample in zip(self.up_blocks, self.up_samples):
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, time_emb)
            h = up_sample(h)
        
        # Output
        return self.output_conv(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
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

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape: tuple) -> torch.Tensor:
        """Generate samples by iteratively denoising."""
        device = next(self.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            
        return img

    def sample(self, batch_size: int = 16) -> torch.Tensor:
        """Generate samples."""
        return self.p_sample_loop((batch_size, 1, 28, 28))

    def training_losses(self, x_start: torch.Tensor) -> torch.Tensor:
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
        predicted_noise = self.model(x_t, t)
        
        # Compute loss
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss
