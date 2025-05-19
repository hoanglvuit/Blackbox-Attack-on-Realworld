import os
import torch
import argparse
import numpy as np
from torch import nn 
from src.model import SignNN 
from src.utils import transform
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, f1_score



parser = argparse.ArgumentParser(description="Train Sign classifier model")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01) 
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--schedule',  action='store_true') 
parser.add_argument('--saved_path', type=str, default='saved_model')
args = parser.parse_args()
epochs = args.epochs 
lr = args.lr 
saved_path = args.saved_path
use_schedule = args.schedule
weight_decay = args.weight_decay

# train loader
train_data = datasets.ImageFolder(root='data/TRAIN', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# valid loader
valid_data = datasets.ImageFolder(root='data/TEST', transform=transform)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
print(train_data.class_to_idx)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize
model = SignNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
schedule = OneCycleLR(optimizer, max_lr=lr, total_steps = len(train_loader)*epochs)

best_f1 = 0.0  
train_acc_log = []
train_f1_log = [] 
eval_acc_log = [] 
eval_f1_log = [] 


for epoch in range(epochs): 
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []

    for X, y in train_loader: 
        X, y = X.to(device), y.to(device)
        pred = model(X) 
        loss = loss_function(pred, y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_schedule: 
            schedule.step()

        total_loss += loss.item()

        preds = torch.argmax(pred, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(y.cpu().numpy())

    train_acc = accuracy_score(all_train_labels, all_train_preds)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

    # Validation
    model.eval()
    all_valid_preds = []
    all_valid_labels = []
    with torch.no_grad():
        for X, y in valid_loader: 
            X, y = X.to(device), y.to(device)
            pred = model(X) 
            preds = torch.argmax(pred, dim=1)
            all_valid_preds.extend(preds.cpu().numpy())
            all_valid_labels.extend(y.cpu().numpy())

    val_acc = accuracy_score(all_valid_labels, all_valid_preds)
    val_f1 = f1_score(all_valid_labels, all_valid_preds, average='macro')

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Valid Acc: {val_acc:.4f} | Valid F1: {val_f1:.4f}")
    train_acc_log.append(train_acc)
    train_f1_log.append(train_f1) 
    eval_acc_log.append(val_acc)
    eval_f1_log.append(val_f1)

    # Lưu mô hình tốt nhất dựa trên F1 score
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(saved_path,'best_f1.pt'))
        print("Saved Best Model (New Best F1 Score)")

ran = np.arange(epochs)
plt.plot(ran, train_acc_log, label='Train Accuracy', marker='o')
plt.plot(ran, eval_acc_log, label='Validation Accuracy', marker='o')
plt.plot(ran, train_f1_log, label='Train F1 Score', marker='s')
plt.plot(ran, eval_f1_log, label='Validation F1 Score', marker='s')

plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Training and Validation Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(saved_path, 'plot.png'))
