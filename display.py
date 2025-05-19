import numpy as np
import matplotlib.pyplot as plt
from src.utils import to_pytorch


# Load dữ liệu từ file
data = np.load('result\ex1.npy', allow_pickle=True).item()
adv = data['adversary']
orig = data['orig']
success = data['success']
print(success)

# display
plt.imshow(adv)            
plt.axis('off')            
plt.title("Ảnh RGB (0–1)")
plt.show()