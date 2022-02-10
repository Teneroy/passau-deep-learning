from time import perf_counter
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import sklearn.model_selection
import matplotlib.pyplot as plt


# Define a model
# You can reuse the model from exercise 7 and add a dropout parameter
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
        # add model definition here
        # self.l0 = nn.Linear(in_dim, hidden_dim)
        # self.act0 = nn.ReLU()
        # self.l1 = nn.Linear(hidden_dim, out_dim)
        self.linear_relu_stack = nn.Sequential()
        self.linear_relu_stack.add_module('l0', nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers - 1):
            self.linear_relu_stack.add_module('dropout' + str(i), nn.Dropout(p=p))
            self.linear_relu_stack.add_module('act' + str(i), nn.ReLU())
            inner_d = hidden_dim
            outer_d = hidden_dim
            if i == n_layers - 2:
                outer_d = out_dim
            self.linear_relu_stack.add_module('l' + str(i + 1), nn.Linear(inner_d, outer_d))

    def forward(self, x):
        """x has shape (batch_size, *img_size)"""
        x = torch.flatten(x, start_dim=1)
        logits = self.linear_relu_stack(x)
        return logits


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

        # What should happen if the l1_coeff is not zero?
        # Add it here
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
train_set_full = torchvision.datasets.FashionMNIST(
    "./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
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


train_loader = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=500, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_dim = 200
learning_rate = 1e-1

model = FeedForwardNet(hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # add L2 regularization
#criterion = nn.L1Loss()
criterion = nn.CrossEntropyLoss()

n_epochs = 10  # change this as needed
for epoch in range(n_epochs):
    train_epoch(model, train_loader, criterion, optimizer)
    train_loss, train_acc = validate(model, train_loader, criterion=criterion)
    val_loss, val_acc = validate(model, val_loader, criterion=criterion)
    print(
        f"{epoch=}: {train_loss=:.3f}, {train_acc=:.3f}, {val_loss=:.3f}, {val_acc=:.3f}"
    )
