import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from datetime import datetime

from conditional_ddpm import ConditionalDDPM, ConditionalUNet, create_conditional_model
from utils import (
    save_image_grid, save_individual_images, create_training_visualization,
    visualize_noise_schedule, create_sample_quality_metrics, save_model_info
)


def get_data_loaders(batch_size=128, num_workers=4):
    """Get MNIST data loaders with labels."""
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


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        loss = model.training_losses(data, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, test_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            loss = model.training_losses(data, labels)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def save_conditional_samples(model, device, epoch, num_samples=16, save_dir='samples', writer=None):
    """Generate and save conditional sample images."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Generate samples for each class
        all_samples = []
        class_labels = []
        
        for class_label in range(10):  # MNIST has 10 classes
            samples = model.sample_class(class_label, num_samples // 10)
            all_samples.append(samples)
            class_labels.extend([class_label] * (num_samples // 10))
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
    
    # Save grid using utility function
    save_image_grid(all_samples, f'{save_dir}/conditional_samples_epoch_{epoch:03d}.png', 
                   nrow=10, title=f'Conditional Generated Samples - Epoch {epoch}')
    
    # Save individual samples
    save_individual_images(all_samples, f'{save_dir}/epoch_{epoch:03d}_conditional_individual')
    
    # Log to tensorboard
    if writer is not None:
        # Convert samples to grid for tensorboard
        grid = torch.cat([all_samples[i:i+10] for i in range(0, len(all_samples), 10)], dim=0)
        writer.add_images('Conditional_Generated_Samples', grid, epoch)
    
    return all_samples


def save_class_comparison(model, device, epoch, save_dir='samples', writer=None):
    """Generate and save samples for each class separately."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Generate samples for each class
        for class_label in range(10):
            samples = model.sample_class(class_label, 16)
            
            # Save grid for this class
            save_image_grid(samples, f'{save_dir}/class_{class_label}_epoch_{epoch:03d}.png', 
                           nrow=4, title=f'Class {class_label} - Epoch {epoch}')
    
    # Create a comparison grid with one sample from each class
    comparison_samples = []
    with torch.no_grad():
        for class_label in range(10):
            sample = model.sample_class(class_label, 1)
            comparison_samples.append(sample)
    
    comparison_samples = torch.cat(comparison_samples, dim=0)
    save_image_grid(comparison_samples, f'{save_dir}/class_comparison_epoch_{epoch:03d}.png', 
                   nrow=10, title=f'Class Comparison - Epoch {epoch}')


def main():
    parser = argparse.ArgumentParser(description='Train Conditional DDPM on MNIST')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory to save samples')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for logging')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency to save samples')
    parser.add_argument('--eval_freq', type=int, default=5, help='Frequency to evaluate model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"conditional_ddpm_mnist_{timestamp}"
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create experiment-specific directories
    exp_save_dir = os.path.join(args.save_dir, args.experiment_name)
    exp_sample_dir = os.path.join(args.sample_dir, args.experiment_name)
    exp_log_dir = os.path.join(args.log_dir, args.experiment_name)
    
    os.makedirs(exp_save_dir, exist_ok=True)
    os.makedirs(exp_sample_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'timesteps': args.timesteps,
        'device': str(device),
        'experiment_name': args.experiment_name,
        'save_freq': args.save_freq,
        'eval_freq': args.eval_freq,
        'num_classes': args.num_classes
    }
    
    with open(os.path.join(exp_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Create model
    model = create_conditional_model(num_classes=args.num_classes, device=device).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create tensorboard writer
    writer = SummaryWriter(exp_log_dir)
    
    # Save model info
    save_model_info(model, os.path.join(exp_save_dir, 'model_info.json'))
    
    # Visualize noise schedule
    visualize_noise_schedule(model, os.path.join(exp_save_dir, 'noise_schedule.png'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    test_losses = []
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        train_losses.append(train_loss)
        
        # Evaluate
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            test_loss = evaluate(model, test_loader, device)
            test_losses.append(test_loss)
        else:
            test_loss = test_losses[-1] if test_losses else float('inf')
        
        # Update scheduler
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Time = {epoch_time:.2f}s')
        
        # Save samples
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            samples = save_conditional_samples(model, device, epoch, save_dir=exp_sample_dir, writer=writer)
            save_class_comparison(model, device, epoch, save_dir=exp_sample_dir, writer=writer)
            # Create sample quality metrics
            create_sample_quality_metrics(samples, os.path.join(exp_save_dir, f'sample_quality_epoch_{epoch:03d}.png'))
        
        # Save checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'config': config,
            }, os.path.join(exp_save_dir, 'best_model.pth'))
        
        # Save regular checkpoint
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'config': config,
            }, os.path.join(exp_save_dir, f'checkpoint_epoch_{epoch:03d}.pth'))
    
    # Generate final samples
    final_samples = save_conditional_samples(model, device, args.epochs, save_dir=exp_sample_dir, writer=writer)
    save_class_comparison(model, device, args.epochs, save_dir=exp_sample_dir, writer=writer)
    
    # Create final training visualization
    create_training_visualization(train_losses, test_losses, 
                                os.path.join(exp_save_dir, 'training_curves.png'))
    
    # Create final sample quality metrics
    create_sample_quality_metrics(final_samples, os.path.join(exp_save_dir, 'final_sample_quality.png'))
    
    # Save final training summary
    total_time = time.time() - start_time
    training_summary = {
        'total_training_time': total_time,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': best_loss,
        'total_epochs': args.epochs,
        'config': config
    }
    
    with open(os.path.join(exp_save_dir, 'training_summary.json'), 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    writer.close()
    print(f'Training completed! Total time: {total_time:.2f}s')
    print(f'Best test loss: {best_loss:.4f}')
    print(f'Results saved to: {exp_save_dir}')


if __name__ == '__main__':
    main()
