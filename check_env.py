import torch
import timm
import transformers
import pandas as pd
import numpy as np

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Timm version:", timm.__version__)
print("Transformers version:", transformers.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)