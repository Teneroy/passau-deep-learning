from typing import Tuple
import matplotlib.pyplot as plt
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torchvision

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


# Define a model
class FeedForwardNet(nn.Module):
    def __init__(self, target_number, img_size, dim_number) -> None:
        super().__init__()
        self.l0 = nn.Linear(img_size * img_size, dim_number)
        self.act0 = nn.ReLU()
        self.l1 = nn.Linear(dim_number, target_number)

    def forward(self, x):
        """x has shape (batch_size, *img_size)"""
        x = x.view(-1, 28*28)
        x = self.l0(x)
        x = self.act0(x)
        x = self.l1(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> float:
    """Train a model for one epoch

    Args:
        model (torch.nn.Module): model to be trained
        loader (torch.utils.data.DataLoader): Dataloader for training data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        float: total loss over one epoch
    """
    total_loss = 0
    model.train()
    for x, y in loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


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
    # testing
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            pred = pred.argmax(dim=1)
            correct += (pred == y).sum().item()
            test_loss += loss.item()
    return (test_loss / len(loader.dataset)), (correct / len(loader.dataset))


learning_rate = 1e-1
sz = img.shape[1]
model = FeedForwardNet(10, sz, 100).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

n_epochs = 10  # change this as needed
for epoch in range(n_epochs):
    train_epoch(model, train_loader, criterion, optimizer)  # add your code here
    train_loss, train_acc = validate(model, train_loader, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)  # add your code here
    print(
        #f"{epoch=}: {train_loss=:.3f}, {val_loss=:.3f}, {val_acc=:.3f}"
        f"{epoch=}: {train_loss=:.3f}, {train_acc=:.3f}, {val_loss=:.3f}, {val_acc=:.3f}"
    )
