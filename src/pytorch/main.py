import torch
import time
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader


import nvidia_speed
import cnn_segmentation
import loader
import config

from loader import SegDataSet
from loader import MyDataSet

from nvidia_speed import NvidiaSpeed
from cnn_segmentation import CNNSeg
from seg_nvidia import SegNvidia

#Use GPU cuda if possible
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device use is : ", device)


dataset_path = config.DATASETS_PATH

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
def eval_model(net, loader, loss_fn, show=False):
    net.eval()
    acc, loss = 0., 0.
    c = 0

    if show:
        for x, y in loader:
            print(type(x))

            with torch.no_grad():
                logits = net(x.to(device)).cpu()

            f, ax = plt.subplots(2)
            ax[0].imshow(x[0].squeeze(dim=0), cmap='gray')
            ax[1].imshow(logits[0].squeeze(dim=0), cmap='gray')
            plt.show()
        return 0, 0;

    for x, y in loader:
        #x, y = x.to(device), y.to(device)

        with torch.no_grad():
            # No need to compute gradient here thus we avoid storing intermediary activations
            logits = net(x.to(device)).cpu()


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


#TODO faire une classe qui claque sa mère

def head_train_model(net, train_loader, val_loader, test_loader, save_path, epoch):

    net.to(device)

    #test_model_random(net, (2, 3, 120, 160))

    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)#, momentum=0.9, nesterov=True)
   # optimizer = torch.optim.SGD(net.parameters(), lr=config.LEARNING_RATE)#, momentum=0.9, nesterov=True)

    train_model(net, train_loader, val_loader, epoch, optimizer)

    mse = nn.MSELoss()

    #tkt
   # if epoch < 22:
    #    eval_model(net, test_loader, mse, show=True)

    torch.save(net.state_dict(), save_path)
    return net


if __name__ == '__main__':

    #create transforms
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std),
    ])

    train_seg_dataset = SegDataSet(config.DATASETS_SEG_PATH, transform=train_transforms) 
    train_seg_dataset, val_seg_dataset = torch.utils.data.random_split(
        train_seg_dataset,
        [int(config.TRAIN_SIZE * len(train_seg_dataset)), len(train_seg_dataset) - int(config.TRAIN_SIZE * len(train_seg_dataset))]
    )

    val_seg_dataset.transform = train_transforms
    test_seg_dataset = SegDataSet(config.DATASETS_SEG_PATH, transform=train_transforms, train=False) 

    print(f"Nb images in train: {len(train_seg_dataset)}")
    print(f"Nb images in val: {len(val_seg_dataset)}")
    print(f"Nb images in test: {len(test_seg_dataset)}")

    train_seg_loader = DataLoader(train_seg_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=config.BATCH_SIZE, num_workers=8)
    test_seg_loader = DataLoader(test_seg_dataset, batch_size=128, num_workers=8)

    conv_seg = head_train_model(CNNSeg(), train_seg_loader, val_seg_loader, test_seg_loader, config.MODEL_SAVE_PATH_SEG, 20)

    print(f"\nNb batches in train: {len(train_seg_loader)}")
    print(f"Nb batches in val: {len(val_seg_loader)}")
    print(f"Nb batches in test: {len(test_seg_loader)}")

    train_dataset = MyDataSet(config.DATASETS_PATH, transform=train_transforms, train=True)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [int(config.TRAIN_SIZE * len(train_dataset)), len(train_dataset) - int(config.TRAIN_SIZE * len(train_dataset))]
    )

    val_dataset.transform = train_transforms
    test_dataset = MyDataSet(config.DATASETS_PATH, transform=train_transforms, train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8)

    net = SegNvidia()
    net.conv_seg = conv_seg
    head_train_model(net, train_loader, val_loader, test_loader, config.MODEL_SAVE_PATH, 20)
