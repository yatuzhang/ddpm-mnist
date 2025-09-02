#!/usr/bin/env python3
"""
Setup script for DDPM MNIST project.

This script helps set up the environment and run initial tests.
"""

import os
import sys
import subprocess
import torch
import torchvision


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python version: {sys.version}")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'tqdm', 'tensorboard', 'PIL', 'scipy', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability."""
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA is not available, will use CPU")


def create_directories():
    """Create necessary directories."""
    directories = ['data', 'checkpoints', 'samples', 'logs', 'example_outputs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def test_imports():
    """Test if all modules can be imported."""
    try:
        from ddpm_model import DDPM, UNet
        print("✓ ddpm_model imported successfully")
        
        from conditional_ddpm import ConditionalDDPM, ConditionalUNet
        print("✓ conditional_ddpm imported successfully")
        
        from utils import save_image_grid
        print("✓ utils imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def run_example():
    """Run a simple example to test the setup."""
    try:
        print("\nRunning example...")
        from example_usage import example_unconditional_training
        example_unconditional_training()
        print("✓ Example ran successfully")
        return True
    except Exception as e:
        print(f"✗ Example failed: {e}")
        return False


def main():
    """Main setup function."""
    print("DDPM MNIST Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and run setup again.")
        sys.exit(1)
    
    # Check CUDA
    print("\nChecking CUDA...")
    check_cuda()
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Test imports
    print("\nTesting imports...")
    if not test_imports():
        print("Import test failed. Please check your installation.")
        sys.exit(1)
    
    # Run example
    print("\nTesting example...")
    if not run_example():
        print("Example test failed. Please check your installation.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("\nYou can now:")
    print("1. Train a model: python train.py --epochs 10")
    print("2. Train conditional model: python train_conditional.py --epochs 10")
    print("3. Generate samples: python inference.py --model_path checkpoints/.../best_model.pth")
    print("4. Evaluate model: python eval.py --model_path checkpoints/.../best_model.pth")
    print("5. Run examples: python example_usage.py")


if __name__ == '__main__':
    main()
