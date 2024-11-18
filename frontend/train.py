#%%
import torch
import time
from models.implements.cnn.cifar10.resnet import ResNet18
from utils import get_data_dataset, train, test_top1
from types import SimpleNamespace
#%%
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18()
# model.load_state_dict(torch.load('cifar10_model3.pth'))
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

train_loader, val_loader, crierion = get_data_dataset('cifar10', '/work/dataset', 512, 512)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%

args = SimpleNamespace(log_interval=1000)
for e in range(1000):
    s = time.time()
    train(args, model, device, train_loader, crierion, optimizer, e)
    print(test_top1(model, device, crierion, val_loader))
    print(time.time() - s)
    if e % 100 == 0:
        torch.save(model.state_dict(), f'cifar10_model_{e}.pth')
        




# %%
