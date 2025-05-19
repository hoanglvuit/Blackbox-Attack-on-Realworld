import numpy as np
import matplotlib.pyplot as plt
from src.utils import to_pytorch
from src.model import SignNN
import torch

# Load dữ liệu từ file
data = np.load('result\ex.npy', allow_pickle=True).item()
adv = data['adversary']
orig = data['orig']

# display
plt.imshow(adv)            
plt.axis('off')            
plt.title("Ảnh RGB (0–1)")
plt.show()