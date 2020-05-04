import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import argparse

import os

if not os.path.exists('saved_models/'):
	os.makedirs('saved_models/')

use_cuda = torch.cuda.is_available()

class classifier(nn.Module):
	def __init__(self):
		super(classifier, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		out = F.log_softmax(x, dim=1)

		return out

def train(args, model, device, train_loader, optimizer, epoch):
	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		print(data.shape)
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		out = model(data)
		loss = F.nll_loss(out, target)
		loss.backward()
		optimizer.step()
		if not batch_idx % args.log_interval:
			print(f'\nTraining epoch {epoch} [{100. * batch_idx / len(train_loader):.3f} complete]\tLoss:{loss.item():.4f}')

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			out = model(data)
			test_loss += F.nll_loss(out, target, reduction='sum').item()
			prediction = out.argmax(dim=1, keepdim=True)
			correct += prediction.eq(target.view_as(prediction)).sum().item()

	test_loss /= len(test_loader.dataset)

	print(f'\nTesting\t Loss: {test_loss} Accuracy: {100. * correct / len(test_loader.dataset)}')

def main():
	parser = argparse.ArgumentParser(description='basic conv mnist classifier')

	parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train for (default: 14)')
	parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 1.0)')
	parser.add_argument('--gamma', type=float, default=0.7, help='learning rate step gamma (default: 0.7)')
	parser.add_argument('--log_interval', type=int, default=10, help='training logs every x steps (default: 10)')

	args = parser.parse_args()

	device = torch.device('cuda' if use_cuda else 'cpu')

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data/', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
				])
			), batch_size=args.batch_size, shuffle=True, **kwargs
		)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data/', train=False, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
				])
			), batch_size=args.batch_size, shuffle=True, **kwargs
		)

	model = classifier().to(device)
	optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

	scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
	for epoch in range(1, args.epochs+1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)
		scheduler.step()

	torch.save(model.state_dict(), 'saved_models/mnist_cnn.pt')

if __name__ == '__main__':
	main()