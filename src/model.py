import torch
from torchvision import models as torch_models
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class SignNN(nn.Module):
    def __init__(self): 
        super(SignNN, self).__init__() 
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3), 
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 

            nn.Conv2d(4, 8, kernel_size=3), 
            nn.BatchNorm2d(8), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, kernel_size=3), 
            nn.BatchNorm2d(16),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3), 
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(4*64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        x = x.to(device)  
        return self.model(x)

    def predict_mulprob(self, x):
        return self.forward(x) 
    
    def predict_maxprob(self, x): 
        x = x.to(device) 
        pred = self.model(x) 
        max_val, max_ind = torch.max(pred, dim=1)
        return max_val.item(), max_ind.item()