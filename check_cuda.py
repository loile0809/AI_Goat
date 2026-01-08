print("Checking CUDA...")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("Torch not installed")
except Exception as e:
    print(f"Error: {e}")
