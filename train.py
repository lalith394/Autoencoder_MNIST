from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from model import AutoEncoder_MNIST
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import torch
import os, sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(torch.cuda.is_available()):
    print("Using Cuda device!")
else:
    print("Using CPU")

""" Hyperparameters """
batch_size = 64
epochs = 240
lr = 1e-4
model_dir = "trained_models"
model_name = "model3.pth"

""" Handling model save dir """
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# --- Initializing model, loss functions and optimizer ---
model = AutoEncoder_MNIST().to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

# --- Loading train and test data ---
transform = transforms.Compose([transforms.ToTensor()])

train_data = MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_data = MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)


train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test = DataLoader(test_data, batch_size=batch_size, shuffle=True)


sample_images, sample_labels = next(iter(train))
print("Sample Images shape: ", sample_images.shape)
print("Sample labels shape: ", sample_labels.shape)

# --- Training ---
best_loss = float('inf')
for ep in range(epochs):
    model.train()
    running_loss = 0.0

    for img, _ in tqdm(train):
        img = img.to(device)

        """ Forward """
        outputs = model(img)
        loss = criterion(outputs, img)

        """ Backward """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss/len(train)
    print(f"Epoch [{ep+1}/{epochs}] - Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(model_dir, model_name))
        print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")