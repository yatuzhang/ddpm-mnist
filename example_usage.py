#!/usr/bin/env python3
"""
Example usage script for DDPM on MNIST.

This script demonstrates how to use the DDPM implementation for training,
inference, and evaluation.
"""

import torch
import os
import argparse
from ddpm_model import DDPM, UNet
from conditional_ddpm import ConditionalDDPM, ConditionalUNet
from utils import save_image_grid, create_comprehensive_visualization


def example_unconditional_training():
    """Example of training an unconditional DDPM model."""
    print("=== Unconditional DDPM Training Example ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
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
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate some samples
    print("Generating samples...")
    with torch.no_grad():
        samples = model.sample(16)
    
    # Save samples
    os.makedirs('example_outputs', exist_ok=True)
    save_image_grid(samples, 'example_outputs/unconditional_samples.png', 
                   title="Unconditional DDPM Samples", nrow=4)
    
    print("Samples saved to example_outputs/unconditional_samples.png")


def example_conditional_training():
    """Example of training a conditional DDPM model."""
    print("\n=== Conditional DDPM Training Example ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create conditional model
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_classes=10,
    )
    
    model = ConditionalDDPM(
        model=unet,
        timesteps=1000,
        device=device,
        num_classes=10
    ).to(device)
    
    print(f"Conditional model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate samples for each class
    print("Generating conditional samples...")
    os.makedirs('example_outputs', exist_ok=True)
    
    with torch.no_grad():
        # Generate one sample from each class
        all_samples = []
        for class_label in range(10):
            sample = model.sample_class(class_label, 1)
            all_samples.append(sample)
        
        all_samples = torch.cat(all_samples, dim=0)
    
    # Save samples
    save_image_grid(all_samples, 'example_outputs/conditional_samples.png', 
                   title="Conditional DDPM Samples (0-9)", nrow=10)
    
    print("Conditional samples saved to example_outputs/conditional_samples.png")


def example_ddim_sampling():
    """Example of DDIM sampling for faster inference."""
    print("\n=== DDIM Sampling Example ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
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
    
    # Generate samples with different methods
    print("Generating samples with DDPM (1000 steps)...")
    with torch.no_grad():
        ddpm_samples = model.sample(16)
    
    # For DDIM, we would need to implement the DDIM class
    # This is just a placeholder for the example
    print("DDIM sampling would be implemented here...")
    
    # Save samples
    os.makedirs('example_outputs', exist_ok=True)
    save_image_grid(ddpm_samples, 'example_outputs/ddpm_samples.png', 
                   title="DDPM Samples (1000 steps)", nrow=4)
    
    print("DDPM samples saved to example_outputs/ddpm_samples.png")


def example_model_visualization():
    """Example of model visualization and analysis."""
    print("\n=== Model Visualization Example ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
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
    
    # Create comprehensive visualization
    print("Creating model visualizations...")
    create_comprehensive_visualization(model, device, 'example_outputs/model_viz')
    
    print("Model visualizations saved to example_outputs/model_viz/")


def main():
    parser = argparse.ArgumentParser(description='DDPM Example Usage')
    parser.add_argument('--example', type=str, default='all', 
                       choices=['unconditional', 'conditional', 'ddim', 'visualization', 'all'],
                       help='Which example to run')
    
    args = parser.parse_args()
    
    print("DDPM MNIST Example Usage")
    print("=" * 50)
    
    if args.example in ['unconditional', 'all']:
        example_unconditional_training()
    
    if args.example in ['conditional', 'all']:
        example_conditional_training()
    
    if args.example in ['ddim', 'all']:
        example_ddim_sampling()
    
    if args.example in ['visualization', 'all']:
        example_model_visualization()
    
    print("\n" + "=" * 50)
    print("Example completed! Check the 'example_outputs' directory for results.")
    print("\nTo train a full model, use:")
    print("  python train.py --epochs 100")
    print("  python train_conditional.py --epochs 100")
    print("\nTo generate samples, use:")
    print("  python inference.py --model_path checkpoints/experiment_name/best_model.pth")
    print("\nTo evaluate a model, use:")
    print("  python eval.py --model_path checkpoints/experiment_name/best_model.pth")


if __name__ == '__main__':
    main()
