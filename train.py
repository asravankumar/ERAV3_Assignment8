import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model import CIFAR10Net
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchsummary import summary

torch.manual_seed(1)

class CIFAR10Albumentation(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 0.001

# Calculate CIFAR10 mean and std
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

# Albumentations transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(
        max_holes=1, max_height=16, max_width=16,
        min_holes=1, min_height=16, min_width=16,
        fill_value=[x * 255 for x in cifar10_mean],
        mask_fill_value=None,
        p=0.5
    ),
    A.Normalize(mean=cifar10_mean, std=cifar10_std),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=cifar10_mean, std=cifar10_std),
    ToTensorV2()
])

# Load CIFAR10 dataset
train_dataset_raw = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
test_dataset_raw = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True)

train_dataset = CIFAR10Albumentation(train_dataset_raw, train_transform)
test_dataset = CIFAR10Albumentation(test_dataset_raw, test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = CIFAR10Net().to(device)
summary(model, input_size=(3, 32, 32))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Calculate accuracy for progress bar
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar with batch information
        pbar.set_postfix({
            'batch': f'{batch_idx}/{len(train_loader)}',
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        val_pbar = tqdm(test_loader, desc='Validation')
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update validation progress bar
            val_pbar.set_postfix({
                'val_loss': f'{val_loss/(val_pbar.n+1):.4f}',
                'val_acc': f'{100 * correct / total:.2f}%'
            })
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    
    # Save model if accuracy > 85%
    if accuracy > 85.0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'val_loss': val_loss/len(test_loader)
        }
        torch.save(checkpoint, f'cifar10_model_acc_{accuracy:.2f}.pth')
        print(f'Model saved with accuracy: {accuracy:.2f}%')
        break
    
    print(f'Epoch {epoch+1} Summary:')
    print(f'Training Loss: {running_loss/len(train_loader):.4f}')
    print(f'Validation Loss: {val_loss/len(test_loader):.4f}')
    print(f'Validation Accuracy: {accuracy:.2f}%')
    print('-' * 50) 