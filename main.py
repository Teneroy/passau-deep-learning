from time import perf_counter
from typing import Tuple

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
import sklearn.model_selection
import matplotlib.pyplot as plt


# Define a model
class FeedForwardNet(nn.Module):
    def __init__(
            self,
            hidden_dim,
            out_dim=10,
            img_shape=(28, 28),
            n_layers: int = 3,
            p: float = 0.5,
    ) -> None:
        super().__init__()
        in_dim = img_shape[0] * img_shape[1]
        self.img_shape = img_shape
        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()))
        for _ in range(n_layers - 2):
            lin = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(lin.weight)
            self.layers.append(
                nn.Sequential(
                    lin, nn.ReLU(), nn.Dropout(p=p)
                )
            )
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        """x has shape (batch_size, *img_size)"""
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNet(nn.Module):
    def __init__(
            self,
            hidden_dim,
            out_dim=10,
            img_shape=(28, 28),
            p: float = 0.5,
            n_channels=1
    ) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.layers = nn.ModuleList()
        filters_size = [n_channels, 20, 50]
        for i in range(1, len(filters_size)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(kernel_size=(5, 5), in_channels=filters_size[i - 1], out_channels=filters_size[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )
            )
        self.layers.append(nn.Sequential(
            nn.Linear(filters_size[len(filters_size) - 1] * 4 * 4, hidden_dim),
            nn.ReLU()
        ))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        i = 1
        for layer in self.layers:
            if i == len(self.layers) - 1:
                x = torch.flatten(x, start_dim=1)
            x = layer(x)
            i += 1
        return x


def train_epoch(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        l1_coeff: float = 0
) -> float:
    """Train a model for one epoch

    Args:
        model (torch.nn.Module): model to be trained
        loader (torch.utils.data.DataLoader): Dataloader for training data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        l1_coeff (float): coefficient of L1 loss

    Returns:
        float: total loss over one epoch
    """
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)  # I am using device as a global variable, but you could pass it as well
        out = model(x)
        if l1_coeff != 0:
            params = torch.concat([params.view(-1) for params in model.parameters()])
            l1_loss = F.l1_loss(params, torch.zeros_like(params))
            loss = criterion(out, y) + l1_coeff * l1_loss
        else:
            loss = criterion(out, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss


@torch.no_grad()  # we dont want these operations to be recorded for automatic differentation, saves memory
def validate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module = None,
) -> Tuple[float, float]:
    """Compute total loss and accuracy

    Args:
        model (torch.nn.Module): model to be evaluated
        loader (torch.utils.data.DataLoader): Dataloader for evaluation data
        criterion (torch.nn.Module, optional): loss function. Defaults to None.

    Returns:
        Tuple[float, float]: total loss, accuracy
    """
    total_loss = 0
    total_correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            loss = criterion(out, y)
            total_loss += loss.item()
        total_correct += (out.argmax(dim=1) == y).sum().item()
    return total_loss, total_correct / len(loader.dataset)


torch.manual_seed(0)

# load data

train_transforms = torchvision.transforms.ToTensor()  # you can try out torchvision.transforms for augmentation as well

train_set_full = torchvision.datasets.FashionMNIST(
    "./data", train=True, download=True, transform=train_transforms
)
test_set = torchvision.datasets.FashionMNIST(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)

img, target = train_set_full[231]
plt.imshow(img.view(28, 28), cmap="binary")

val_size = 0.2
train_indices, val_indices = sklearn.model_selection.train_test_split(
    range(len(train_set_full)),
    stratify=train_set_full.targets,
    test_size=val_size,
    random_state=0,
)
train_set = torch.utils.data.Subset(train_set_full, train_indices)
val_set = torch.utils.data.Subset(train_set_full, val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=500, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-3

model = ConvNet(hidden_dim=500, out_dim=10, n_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

n_epochs = 10  # change this as needed
start = perf_counter()
print("TRAINING:")
for epoch in range(n_epochs):
    train_epoch(model, train_loader, criterion, optimizer, l1_coeff=0)
    train_loss, train_acc = validate(model, train_loader, criterion=criterion)
    val_loss, val_acc = validate(model, val_loader, criterion=criterion)
    print(
        f"{perf_counter() - start:.1f}s {epoch=}: {train_loss=:.3f}, {train_acc=:.3f}, {val_loss=:.3f}, {val_acc=:.3f}"
    )
