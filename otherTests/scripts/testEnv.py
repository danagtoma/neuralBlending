import sys
import torch
import numpy as np
import meshio
import igl
import skimage
import matplotlib
import pyevtk

print("--- Environment Check ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

print("\n--- Dependencies Loaded Successfully ---")
print(f"meshio version: {meshio.__version__}")
print(f"skimage version: {skimage.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print("libigl and pyevtk imported without errors!")