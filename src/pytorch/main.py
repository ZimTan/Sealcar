import torch
import time
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader


import nvidia_speed
import loader
from loader import MyDataSet
from nvidia_speed import NvidiaSpeed

#Use GPU cuda if possible
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device use is : ", device)


dataset_path = "robotracingleague_dataset/rrl1"

"""

   use for testing a model juste after creating it

   forward advancement with random input

"""
def test_model_random(net, input_shape):
    logits = net(torch.randn(input_shape).to(device))
    print(f"Logits shape: {logits.shape}")

"""

    use for evaluate a model

    input -> the model, the inputs loader and the loss function
    give accuraty and loss

"""
def eval_model(net, loader, loss_fn):
    net.eval()
    acc, loss = 0., 0.
    c = 0
    for x, y in loader:
        #x, y = x.to(device), y.to(device)

        with torch.no_grad():
            # No need to compute gradient here thus we avoid storing intermediary activations
            logits = net(x.to(device)).cpu()

        print(logits)
        loss += loss_fn(logits, y).item()
        preds = logits.argmax(dim=1)
        #acc += (preds.numpy() == y.numpy()).sum()
        c += len(x)

    acc /= c
    loss /= len(loader)
    net.train()
    return acc, loss


def train_model(net, train_loader, val_loader, nb_epochs, optimizer):
    
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []

    #cross_entropy = nn.CrossEntropyLoss() #Does softmax + CE
    mse = nn.MSELoss()

    for epoch in range(nb_epochs):
        start = time.time()
        running_acc, running_loss = 0., 0.
        c = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # Clear previous gradients
            logits = net(x)
            loss = mse(logits, y)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights with gradients

            #calculate on cpu acc and loss
            #running_acc += ((logits.argmax(dim=1).cpu().numpy() - y.cpu().numpy())**2).sum()
            running_loss += loss.item()
            c += len(x)

        train_acc, train_loss = running_acc / c, running_loss / len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        val_acc, val_loss = eval_model(net, val_loader, mse)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{nb_epochs}, "
            f"train acc/loss: {round(100 * train_acc, 2)}/{round(train_loss, 4)}, "
            f"val acc/loss: {round(100 * val_acc, 2)}/{round(val_loss, 4)}, "
            f"time {int(time.time() - start)}s"
         )


def plot_acc_loss(train_accs, val_accs, train_losses, val_losses, nb_epochs):
    plt.subplot(1, 2, 1)
    plt.plot(list(range(nb_epochs)), train_accs, label="Train")
    plt.plot(list(range(nb_epochs)), val_accs, label="Val")
    plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(list(range(nb_epochs)), train_losses, label="Train")
    plt.plot(list(range(nb_epochs)), val_losses, label="Val")
    plt.title("Loss")

if __name__ == '__main__':

    mean = 0
    std = 0

    #create transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean, std),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean, std),
    ])

    train_dataset = MyDataSet([dataset_path], transform=train_transforms) 

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset) + 1)]
    )

    val_dataset.transform = test_transforms
    test_dataset = MyDataSet([dataset_path], transform=test_transforms, train=False) 

    print(f"Nb images in train: {len(train_dataset)}")
    print(f"Nb images in val: {len(val_dataset)}")
    print(f"Nb images in test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8)

    print(f"\nNb batches in train: {len(train_loader)}")
    print(f"Nb batches in val: {len(val_loader)}")
    print(f"Nb batches in test: {len(test_loader)}")

    net = NvidiaSpeed()

    net.to(device)

    #test_model_random(net, (2, 3, 120, 160))

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)#, momentum=0.9, nesterov=True)

    train_model(net, train_loader, val_loader, 30, optimizer)

    mse = nn.MSELoss()
    print(eval_model(net, val_loader, mse))

    torch.save(net.state_dict(), "models/nvidia_speed")
