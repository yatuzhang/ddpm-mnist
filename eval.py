import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy import linalg
from sklearn.metrics import accuracy_score
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from ddpm_model import DDPM, UNet


class InceptionV3(nn.Module):
    """Inception V3 model for computing FID and IS metrics."""
    
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 147 x 147
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 73 x 73
        x = self.Mixed_5b(x)
        # N x 256 x 73 x 73
        x = self.Mixed_5c(x)
        # N x 288 x 73 x 73
        x = self.Mixed_5d(x)
        # N x 288 x 73 x 73
        x = self.Mixed_6a(x)
        # N x 768 x 35 x 35
        x = self.Mixed_6b(x)
        # N x 768 x 35 x 35
        x = self.Mixed_6c(x)
        # N x 768 x 35 x 35
        x = self.Mixed_6d(x)
        # N x 768 x 35 x 35
        x = self.Mixed_6e(x)
        # N x 768 x 35 x 35
        x = self.Mixed_7a(x)
        # N x 1280 x 17 x 17
        x = self.Mixed_7b(x)
        # N x 2048 x 17 x 17
        x = self.Mixed_7c(x)
        # N x 2048 x 17 x 17
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def get_inception_features(images, model, device, batch_size=50):
    """Extract features using Inception V3 model."""
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(device)
            # Convert grayscale to RGB by repeating channels
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            # Resize to 299x299 for Inception V3
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            feat = model(batch)
            features.append(feat.cpu())
    
    return torch.cat(features, dim=0)


def calculate_fid(real_features, fake_features):
    """Calculate Fréchet Inception Distance (FID)."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate score
    fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid


def calculate_is(preds, splits=10):
    """Calculate Inception Score (IS)."""
    # Convert to probabilities
    preds = F.softmax(preds, dim=1)
    
    # Calculate IS for each split
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        py = part.mean(axis=0)
        scores.append(np.exp((part * (np.log(part) - np.log(py[None, :]))).sum(axis=1).mean()))
    
    return np.mean(scores), np.std(scores)


def get_data_loaders(batch_size=128, num_workers=4):
    """Get MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return test_loader


def evaluate_model(model_path, device, num_samples=1000, batch_size=50):
    """Evaluate the DDPM model."""
    print(f"Loading model from {model_path}")
    
    # Load model
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
    
    print("Generating samples...")
    # Generate samples
    generated_samples = []
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            batch_size_actual = min(batch_size, num_samples - i)
            samples = model.sample(batch_size_actual)
            generated_samples.append(samples.cpu())
    
    generated_samples = torch.cat(generated_samples, dim=0)[:num_samples]
    
    # Get real samples
    print("Loading real samples...")
    test_loader = get_data_loaders(batch_size=batch_size)
    real_samples = []
    for data, _ in test_loader:
        real_samples.append(data)
        if len(torch.cat(real_samples, dim=0)) >= num_samples:
            break
    
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    
    # Load Inception V3 model
    print("Loading Inception V3 model...")
    inception_model = InceptionV3(transform_input=False).to(device)
    
    # Extract features
    print("Extracting features from real samples...")
    real_features = get_inception_features(real_samples, inception_model, device, batch_size)
    
    print("Extracting features from generated samples...")
    fake_features = get_inception_features(generated_samples, inception_model, device, batch_size)
    
    # Calculate metrics
    print("Calculating FID...")
    fid = calculate_fid(real_features.numpy(), fake_features.numpy())
    
    print("Calculating IS...")
    is_mean, is_std = calculate_is(fake_features)
    
    print(f"\nEvaluation Results:")
    print(f"FID: {fid:.4f}")
    print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
    
    return {
        'fid': fid,
        'is_mean': is_mean,
        'is_std': is_std,
        'generated_samples': generated_samples,
        'real_samples': real_samples
    }


def save_comparison_images(results, save_dir='evaluation_results'):
    """Save comparison images between real and generated samples."""
    os.makedirs(save_dir, exist_ok=True)
    
    real_samples = results['real_samples']
    generated_samples = results['generated_samples']
    
    # Denormalize from [-1, 1] to [0, 1]
    real_samples = (real_samples + 1) / 2
    generated_samples = (generated_samples + 1) / 2
    real_samples = torch.clamp(real_samples, 0, 1)
    generated_samples = torch.clamp(generated_samples, 0, 1)
    
    # Save real samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(real_samples[i].squeeze().numpy(), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title('Real')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/real_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save generated samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(generated_samples[i].squeeze().numpy(), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title('Generated')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/generated_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save side-by-side comparison
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        # Real sample
        axes[row, col*2].imshow(real_samples[i].squeeze().numpy(), cmap='gray')
        axes[row, col*2].axis('off')
        axes[row, col*2].set_title('Real')
        # Generated sample
        axes[row, col*2+1].imshow(generated_samples[i].squeeze().numpy(), cmap='gray')
        axes[row, col*2+1].axis('off')
        axes[row, col*2+1].set_title('Generated')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate DDPM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Evaluate model
    results = evaluate_model(args.model_path, device, args.num_samples, args.batch_size)
    
    # Save comparison images
    save_comparison_images(results, args.save_dir)
    
    print(f"Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
