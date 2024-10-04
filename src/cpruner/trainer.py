import torch

class Trainer():
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

        return accuracy
