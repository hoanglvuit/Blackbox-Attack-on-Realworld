from src.model import SignNN 
import torch
from torch import nn 
from src.utils import transform
from torchvision import datasets
from torch.utils.data import DataLoader 
from sklearn.metrics import accuracy_score, f1_score

# train loader
train_data = datasets.ImageFolder(root='data/TRAIN', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# valid loader
valid_data = datasets.ImageFolder(root='data/TEST', transform=transform)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize
model = SignNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10

best_f1 = 0.0  

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

    # Lưu mô hình tốt nhất dựa trên F1 score
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'saved_model/best_model.pth')
        print("Saved Best Model (New Best F1 Score)")
