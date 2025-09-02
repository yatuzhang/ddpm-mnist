#!/usr/bin/env python3
"""
Distributed DDPM Training Script
Supports both DataParallel and DistributedDataParallel
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import argparse
from tqdm import tqdm
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def get_data_loaders(batch_size=128, num_workers=4, distributed=False, rank=0, world_size=1):
    """Get MNIST data loaders with distributed support."""
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
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, sampler=train_sampler, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, sampler=test_sampler, pin_memory=True
    )
    
    return train_loader, test_loader

def create_model(device):
    """Create DDPM model."""
    model = Unet(
        dim=64,
        channels=1,
        dim_mults=(1, 2, 4),
        flash_attn=True
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size=28,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_v'
    ).to(device)
    
    return diffusion

def train_epoch_distributed(diffusion, train_loader, optimizer, device, epoch, rank=0):
    """Train for one epoch with distributed support."""
    diffusion.train()
    total_loss = 0
    
    # Set epoch for distributed sampler
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=rank != 0)
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        loss = diffusion(data)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def test_epoch_distributed(diffusion, test_loader, device, rank=0):
    """Test for one epoch with distributed support."""
    diffusion.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device, non_blocking=True)
            loss = diffusion(data)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)

def save_checkpoint_distributed(diffusion, optimizer, epoch, train_loss, test_loss, save_dir, rank=0):
    """Save checkpoint (only on rank 0)."""
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': diffusion.module.state_dict() if hasattr(diffusion, 'module') else diffusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth'))

def train_distributed(rank, world_size, args):
    """Main distributed training function."""
    # Setup distributed
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"🚀 Starting distributed training on {world_size} GPUs")
        print(f"📊 Batch size per GPU: {args.batch_size}")
        print(f"📊 Total effective batch size: {args.batch_size * world_size}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        distributed=True,
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    diffusion = create_model(device)
    
    # Wrap with DDP
    diffusion = DDP(diffusion, device_ids=[rank])
    
    # Create optimizer
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch_distributed(diffusion, train_loader, optimizer, device, epoch, rank)
        
        # Test
        test_loss = test_epoch_distributed(diffusion, test_loader, device, rank)
        
        # Save checkpoint
        save_checkpoint_distributed(diffusion, optimizer, epoch, train_loss, test_loss, args.save_dir, rank)
        
        if rank == 0:
            print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}')
    
    # Cleanup
    cleanup_distributed()

def train_dataparallel(args):
    """Train with DataParallel (single node, multiple GPUs)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    
    print(f"🚀 Starting DataParallel training on {world_size} GPUs")
    print(f"📊 Batch size per GPU: {args.batch_size}")
    print(f"📊 Total effective batch size: {args.batch_size * world_size}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        distributed=False
    )
    
    # Create model
    diffusion = create_model(device)
    
    # Wrap with DataParallel
    diffusion = torch.nn.DataParallel(diffusion)
    
    # Create optimizer
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        diffusion.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss = total_loss / len(train_loader)
        
        # Test
        diffusion.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device, non_blocking=True)
                loss = diffusion(data)
                test_loss += loss.item()
        
        test_loss = test_loss / len(test_loader)
        
        # Save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': diffusion.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch:03d}.pth'))
        
        print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Distributed DDPM Training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints_distributed', help='Directory to save checkpoints')
    parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--dataparallel', action='store_true', help='Use DataParallel')
    
    args = parser.parse_args()
    
    if args.distributed:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("❌ Distributed training requires at least 2 GPUs")
            return
        
        print(f"🚀 Starting distributed training on {world_size} GPUs")
        mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size, join=True)
    
    elif args.dataparallel:
        if torch.cuda.device_count() < 2:
            print("❌ DataParallel training requires at least 2 GPUs")
            return
        
        train_dataparallel(args)
    
    else:
        print("❌ Please specify --distributed or --dataparallel")
        print("💡 Use --help for more options")

if __name__ == '__main__':
    main()
