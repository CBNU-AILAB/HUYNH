import time
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from new_simulator.softLIF.model import AlexNet

def current_conversion(data, gain=1., bias=0.):
    return data * gain + bias


def train(train_loader, model, criterion, optimizer):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Convert to current before training
        images = current_conversion(images)
        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def testing(test_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Convert to current before training
        images = current_conversion(images)
        preds = model(images)

        loss = criterion(preds, labels)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(24),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers)
    model = AlexNet(num_classes=opt.num_classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        print(start)
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end - start, epoch, train_acc,
                                                                                 train_loss))

        val_acc, val_loss = testing(test_loader, model, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()}
            torch.save(state, opt.pretrained)
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, val_loss))

        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='CIFAR10')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--pretrained', default='pretrained/alexnet.pt')

    app(parser.parse_args())
