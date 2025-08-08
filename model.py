import torch
import torch.nn as nn

class AutoEncoder_MNIST(nn.Module):
	def __init__(self):
		super().__init__()
		self.Encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True)
		)
		
		self.Dense = nn.Sequential(
			nn.Flatten(start_dim=1, end_dim=-1),
			nn.Linear(in_features=64*7*7, out_features=10),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=10, out_features=64*7*7),
			nn.ReLU(inplace=True),
			nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
		)

		self.Decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		#print("Input shape: ", x.shape)
		x = self.Encoder(x)
		#print("After Encoder shape: ", x.shape)
		x = self.Dense(x)
		#print("After Dense shape: ", x.shape)
		x = self.Decoder(x)
		#print("Output shape: ", x.shape)
		return x

if __name__ == '__main__':
	model = AutoEncoder_MNIST()
	tensor = torch.rand(64, 1, 28, 28)
	output = model(tensor)
