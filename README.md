# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## THEORY
A convolutional autoencoder for image denoising learns to compress images into a lower-dimensional representation and then reconstructs clean images from noisy inputs. It consists of encoder layers that extract features, a bottleneck that captures essential information, and decoder layers that reconstruct the denoised image. The model is trained to minimize the difference between the output and the clean target images. This approach effectively removes noise while preserving important details in the images.

## DESIGN STEPS
### STEP 1: 
Load MNIST data and add noise to images.
### STEP 2: 
Build a convolutional autoencoder.
### STEP 3: 
Train the model with noisy images, minimizing MSE loss.
### STEP 4: 
Update weights using backpropagation.
### STEP 5: 
Test the model and visualize original, noisy, and denoised images.
### STEP 6: 
Repeat through multiple epochs for better denoising performance.


## PROGRAM

### Name: Manisha selvakumari.S.S.

### Register Number: 212223220055

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1), # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Name : Manisha selvakumari.S.S.")
print("Reg No : 212223220055")
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name : Manisha selvakumari.S.S.")
    print("Reg No : 212223220055") 
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name : Manisha selvakumari.S.S.")
    print("Reg No : 212223220055")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")
        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)


torch.save(model.state_dict(),'manisha ex07.pth')

```

### OUTPUT

### Model Summary

![Screenshot 2025-05-22 090111](https://github.com/user-attachments/assets/a1a498e0-cced-4f0a-a66f-b6045ee961eb)


### Training loss

![Screenshot 2025-05-22 090135](https://github.com/user-attachments/assets/c1699ab7-7527-4edc-bbbb-0b78c45d05ef)

## Original vs Noisy Vs Reconstructed Image

![Screenshot 2025-05-22 090150](https://github.com/user-attachments/assets/881aab49-5d6e-4c80-8ca5-a1124dacc498)


## RESULT
Thus, a convolutional autoencoder for image denoising application has been developed.
