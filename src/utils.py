from torchvision import datasets, transforms

transform = transforms.Compose([
  transforms.Resize((128,128)), 
  transforms.ToTensor(),
])
