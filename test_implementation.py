#!/usr/bin/env python3
"""
Test script for DDPM implementation.

This script runs basic tests to verify the implementation works correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from ddpm_model import DDPM, UNet
from conditional_ddpm import ConditionalDDPM, ConditionalUNet
from utils import save_image_grid, denormalize_images


def test_unconditional_model():
    """Test unconditional DDPM model."""
    print("Testing unconditional DDPM model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    unet = UNet(
        in_channels=1,
        out_channels=1,
        model_channels=64,  # Smaller for testing
        num_res_blocks=1,
        dropout=0.1,
        channel_mult=(1, 2),
    )
    
    model = DDPM(
        model=unet,
        timesteps=100,  # Fewer timesteps for testing
        device=device
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    
    # Test noise prediction
    with torch.no_grad():
        noise_pred = model.model(x, t)
        assert noise_pred.shape == x.shape, f"Expected {x.shape}, got {noise_pred.shape}"
    
    # Test training loss
    loss = model.training_losses(x)
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(batch_size)
        assert samples.shape == (batch_size, 1, 28, 28), f"Expected {(batch_size, 1, 28, 28)}, got {samples.shape}"
    
    print("✓ Unconditional model tests passed")


def test_conditional_model():
    """Test conditional DDPM model."""
    print("Testing conditional DDPM model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        model_channels=64,  # Smaller for testing
        num_res_blocks=1,
        dropout=0.1,
        channel_mult=(1, 2),
        num_classes=10,
    )
    
    model = ConditionalDDPM(
        model=unet,
        timesteps=100,  # Fewer timesteps for testing
        device=device,
        num_classes=10
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    # Test noise prediction
    with torch.no_grad():
        noise_pred = model.model(x, t, y)
        assert noise_pred.shape == x.shape, f"Expected {x.shape}, got {noise_pred.shape}"
    
    # Test training loss
    loss = model.training_losses(x, y)
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(batch_size)
        assert samples.shape == (batch_size, 1, 28, 28), f"Expected {(batch_size, 1, 28, 28)}, got {samples.shape}"
        
        # Test class-specific sampling
        class_samples = model.sample_class(5, batch_size)
        assert class_samples.shape == (batch_size, 1, 28, 28), f"Expected {(batch_size, 1, 28, 28)}, got {class_samples.shape}"
    
    print("✓ Conditional model tests passed")


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test image normalization/denormalization
    images = torch.randn(4, 1, 28, 28)
    
    # Test denormalization
    denorm_images = denormalize_images(images)
    assert denorm_images.min() >= 0 and denorm_images.max() <= 1, "Denormalized images should be in [0, 1]"
    
    # Test saving (create a temporary directory)
    os.makedirs('test_outputs', exist_ok=True)
    save_image_grid(images, 'test_outputs/test_grid.png', nrow=2, title="Test Grid")
    assert os.path.exists('test_outputs/test_grid.png'), "Grid image should be saved"
    
    print("✓ Utility functions tests passed")


def test_noise_schedule():
    """Test noise schedule properties."""
    print("Testing noise schedule...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DDPM(
        model=nn.Module(),  # Dummy model
        timesteps=1000,
        device=device
    )
    
    # Test beta schedule
    assert model.betas.shape == (1000,), f"Expected (1000,), got {model.betas.shape}"
    assert model.betas.min() > 0, "Betas should be positive"
    assert model.betas.max() < 1, "Betas should be less than 1"
    
    # Test alpha schedule
    assert model.alphas.shape == (1000,), f"Expected (1000,), got {model.alphas.shape}"
    assert torch.allclose(model.alphas, 1 - model.betas), "Alphas should be 1 - betas"
    
    # Test alpha cumprod
    assert model.alphas_cumprod.shape == (1000,), f"Expected (1000,), got {model.alphas_cumprod.shape}"
    assert model.alphas_cumprod[0] == model.alphas[0], "First alpha cumprod should equal first alpha"
    assert model.alphas_cumprod[-1] < model.alphas_cumprod[0], "Alpha cumprod should decrease"
    
    print("✓ Noise schedule tests passed")


def test_q_sample():
    """Test q_sample function."""
    print("Testing q_sample function...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DDPM(
        model=nn.Module(),  # Dummy model
        timesteps=1000,
        device=device
    )
    
    # Test q_sample
    x_start = torch.randn(4, 1, 28, 28, device=device)
    t = torch.randint(0, 1000, (4,), device=device)
    
    x_t, noise = model.q_sample(x_start, t)
    
    assert x_t.shape == x_start.shape, f"Expected {x_start.shape}, got {x_t.shape}"
    assert noise.shape == x_start.shape, f"Expected {x_start.shape}, got {noise.shape}"
    
    # Test that noise is actually added
    assert not torch.allclose(x_t, x_start), "Noise should be added to x_start"
    
    print("✓ q_sample tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running DDPM implementation tests...")
    print("=" * 50)
    
    try:
        test_noise_schedule()
        test_q_sample()
        test_unconditional_model()
        test_conditional_model()
        test_utils()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("The DDPM implementation is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def cleanup():
    """Clean up test files."""
    import shutil
    if os.path.exists('test_outputs'):
        shutil.rmtree('test_outputs')
        print("Cleaned up test files")


if __name__ == '__main__':
    success = run_all_tests()
    cleanup()
    
    if success:
        print("\nYou can now proceed with training:")
        print("  python train.py --epochs 10")
        print("  python train_conditional.py --epochs 10")
    else:
        print("\nPlease fix the issues before proceeding.")
        sys.exit(1)
