#%%
import torch
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
from models.implements.cnn.cifar10.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils import get_data_dataset, train, test_top1
from types import SimpleNamespace
#%%
torch.manual_seed(42)

# train_loader, val_loader, crierion = get_data_dataset('cifar10', '/work/dataset', 512, 512)


# # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# #%%

# args = SimpleNamespace(log_interval=1000)
# trains = {
#     'resnet18': ResNet18,
#     # 'resnet34': ResNet34,
#     # 'resnet50': ResNet50,
#     # 'resnet101': ResNet101,
#     # 'resnet152': ResNet152,
#           }

# for name in trains.keys():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = trains[name]()
#     model.to(device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
#     for e in range(1000):
#         s = time.time()
#         train(args, model, device, train_loader, crierion, optimizer, e)
#         print(test_top1(model, device, crierion, val_loader))
#         print(time.time() - s)
#         if e % 100 == 0:
#             torch.save(model.state_dict(), f'./model_zoo/cifar10_{name}_{e}.pth')

# # %%
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 512
num_epochs = 300
learning_rate = 0.1

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# CIFAR-10 dataset
train_dataset = datasets.CIFAR10(os.path.join('/work/dataset', 'cifar10'), train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(os.path.join('/work/dataset', 'cifar10'), train=False, download=True, transform=transform_test)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define a ResNet18 model
model = ResNet18()
model = model.to(device)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 학습 스케줄러
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 학습 루프
def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.3f} | Acc: {100.*correct/total:.3f}%")

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f"Test Acc: {100.*correct/total:.3f}%")

# 전체 학습
start_time = time.time()
for epoch in range(200):
    train(epoch)
    test()
    scheduler.step()

# Training loop
end_time = time.time()
print(f"Training completed in: {end_time - start_time:.2f} seconds")
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Model saved as resnet18_cifar10.pth")

# %%
