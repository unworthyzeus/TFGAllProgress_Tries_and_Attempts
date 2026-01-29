import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("Success! Your 3050 Ti is ready for training.")
else:
    print("Error: PyTorch is still using the CPU.")