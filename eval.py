import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from model import AutoEncoder_MNIST
from tqdm import tqdm

""" Visualization of predictions """
def visualize_reconstructions(model, test_loader, device):
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        outputs = model(imgs)

    imgs = imgs.cpu()
    outputs = outputs.cpu()

    n = 8  # number of images to show
    plt.figure(figsize=(16, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
model_path = 'tained_models/model2.pth'

# --- Load test data ---
transform = transforms.Compose([transforms.ToTensor()])
test_data = MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# --- Load model ---
model = AutoEncoder_MNIST().to(device)
model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
model.eval()

# --- Define loss function ---
criterion = nn.MSELoss()

# --- Evaluate ---
total_loss = 0.0
with torch.no_grad():
    for imgs, _ in tqdm(test_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)
print(f"\n🔍 Test Reconstruction Loss (MSE): {avg_loss:.6f}")

# --- visualize_reconstructions (8 recontstructions) ---
visualize_reconstructions(model, test_loader, device)