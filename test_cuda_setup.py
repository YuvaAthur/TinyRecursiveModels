"""
Test script to verify CUDA and PyTorch setup for TinyRecursiveModels

Run this in your virtual environment:
    python test_cuda_setup.py
"""

import sys
print(f"Python Version: {sys.version}")
print("-" * 50)

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test a simple CUDA operation
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print(f"✓ CUDA Operations: Working")
    else:
        print("⚠ CUDA not available - will run on CPU")
        print("  Check:")
        print("  1. NVIDIA GPU is installed")
        print("  2. CUDA Toolkit is installed")
        print("  3. PyTorch CUDA version matches CUDA Toolkit")
        
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

print("-" * 50)

# Test other dependencies
dependencies = ['numpy', 'transformers', 'flask']
for dep in dependencies:
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {dep}: {version}")
    except ImportError:
        print(f"✗ {dep}: Not installed")

print("-" * 50)
print("\nSetup Status:")
if torch.cuda.is_available():
    print("✓ Ready for GPU-accelerated TinyRecursiveModels!")
else:
    print("⚠ Will run on CPU (slower)")

print("\nNext steps:")
print("1. Navigate to TinyRecursiveModels directory")
print("2. Run your model server")
print("3. Start Urmi browser")
