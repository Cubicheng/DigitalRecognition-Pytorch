from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms

from model import NeuralNetwork

def train(dataloader,device,model,loss_function,optimizer):
    model.train()
    running_loss = 0.0
    for batch, (inputs,labels) in enumerate(dataloader):
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'loss: {running_loss/len(dataloader):>0.3f}')

def test(dataloader,device,model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'accuracy: {100.0 * correct / total:>0.2f} %')

def main():

    data_transform= transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),scale=(0.5, 1.2)),
        transforms.ToTensor(),
    ])

    print("Loading training data...")
    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=data_transform,
    )
    print("Loading test data...")
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=data_transform,
    )

    train_dataloader = DataLoader(train_data,batch_size=64)
    test_dataloader = DataLoader(test_data,batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")

    model = NeuralNetwork().to(device)
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    epochs = 10

    for t in range(epochs):
        start_time = time()
        print(f"\nepoch{t+1}/{epochs}\n----------------------")
        train(train_dataloader,device,model,loss_function,optimizer)
        test(test_dataloader,device,model)
        end_time = time()
        print(f'time: {end_time - start_time:>0.2f} seconds')

    print("done!")
    path = "mnist.pth"
    torch.save(model.state_dict(),path)
    print(f'model saved: {path}')

if __name__ == '__main__':
    main()