# Import libraries
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
from torch import optim, nn

# Create a Transformation
def create_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform

# Download and Load Datasets
def load_datasets(transform):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# Visualize Images
def visualize_images(image):
    image = image.numpy()  
    plt.imshow(image[0], cmap="gray")  
    plt.show()

# Build a Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Calculate Cross-Entropy Loss
def get_loss():
    return nn.CrossEntropyLoss()

# Obtain the Stochastic Gradient Descent Optimizer
def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the Model
def train_model(model, train_loader, loss_fn, optimizer, epochs=5):
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}")

# Get the Predicted Label
def get_predicted_label(image, model):
    with torch.no_grad():
        logits = model(image)
        return logits.argmax()

# Test the Model
def test_model(model, test_loader):
    model.eval()
    total, correct = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            pred_label = get_predicted_label(images[i].unsqueeze(0), model)
            correct += (pred_label == labels[i]).type(torch.float).sum().item()
        total += len(labels)
    print(f"Tested {total} images. Accuracy: {100 * correct / total:.2f}%")

# Main execution function
if __name__ == "__main__":
    transform = create_transform()
    train_data, test_data = load_datasets(transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    
    model = NeuralNetwork()
    loss_fn = get_loss()
    optimizer = get_optimizer(model)

    train_model(model, train_loader, loss_fn, optimizer)
    test_model(model, test_loader)
