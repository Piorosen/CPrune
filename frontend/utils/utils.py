#%%
from torchvision import datasets, transforms
import torch
import os
import ssl

#%%
def get_data_dataset(dataset, data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }
    
    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
        
    elif dataset == 'imagenet':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        trainset = datasets.ImageNet(os.path.join(data_dir, 'imagenet'), split='train', download=None, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        
        val_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        valset = datasets.ImageNet(os.path.join(data_dir, 'imagenet'), split='val', download=None, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    
    elif dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(data_dir, 'mnist'), train=True, transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.RandomHorizontalFlip(),
                transforms.Resize(28),
                transforms.ToTensor(),
            ]), download=False),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(data_dir, 'mnist'), train=False, transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize(28),
                transforms.ToTensor(),
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    return train_loader, val_loader, criterion

def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file_object = open('./train_epoch.txt', 'a')
            file_object.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file_object.close()

# Top-1 and Top-5 accuracy test (ImageNet)
def test(model, device, criterion, val_loader):
    model.eval()
    total_len = len(val_loader.dataset)
    test_loss = 0
    correct = 0
    correct_5 = 0
    count = 0
    with torch.no_grad():
        for data, target in val_loader:
            count += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            _, pred = output.topk(5, 1, True, True)
            temp_1 = pred.eq(target.view(1, -1).expand_as(pred))
            temp_5 = temp_1[:5].view(-1)
            correct_5 += temp_5.sum().item()
            if count % 5000 == 0 and count != total_len:
                print('Top-1: {}/{} ({:.4f}%), Top-5: {}/{} ({:.4f}%)'.format(correct, count, 100.*(correct/count), correct_5, count, 100.*(correct_5/count)))

    test_loss /= total_len
    accuracy = correct / total_len
    accuracy_5 = correct_5 / total_len

    print('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, total_len, 100. * accuracy, correct_5, total_len, 100. * accuracy_5))
    file_object = open('./train_epoch.txt', 'a')
    file_object.write('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, total_len, 100. * accuracy, correct_5, total_len, 100. * accuracy_5))
    file_object.close()

    return accuracy, accuracy_5

# Only top-1 accuracy test (CIFAR-10)
def test_top1(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))
    file_object = open('./train_epoch.txt', 'a')
    file_object.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))
    file_object.close()

    return accuracy, accuracy


def get_dummy_input(size, batch_size):
    p = size.copy()
    p[0] = batch_size
    dummy_input = torch.randn(p)
    return dummy_input

def get_input_size(dataset):
    if dataset == 'cifar10':
        input_size = [1, 3, 32, 32]
    elif dataset == 'imagenet':
        input_size = [1, 3, 224, 224]
    elif dataset == 'mnist':
        input_size = [1, 1, 28, 28]
    return input_size


