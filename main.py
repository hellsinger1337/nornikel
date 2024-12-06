import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
class ContaminationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  
        
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

def extract_aberration_features(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
    
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.hypot(sobelx, sobely)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())
    
    
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method='default')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    
    
    features = np.stack([laplacian, sobel, lbp], axis=-1)
    return features

class ContaminationDatasetWithFeatures(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        features = extract_aberration_features(image)
        
        
        combined = np.concatenate([image, features], axis=-1)
        
        
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  
        
        
        if self.transform:
            augmented = self.transform(image=combined, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits

train_image_dir = 'path/to/train/images'
train_mask_dir = 'path/to/train/masks'
val_image_dir = 'path/to/val/images'
val_mask_dir = 'path/to/val/masks'


train_dataset = ContaminationDatasetWithFeatures(train_image_dir, train_mask_dir, transform=train_transform)
val_dataset = ContaminationDatasetWithFeatures(val_image_dir, val_mask_dir, transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch.optim as optim


model = UNet(n_channels=6, n_classes=1)  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        
        optimizer.zero_grad()
        
        
        outputs = model(images)
        outputs = outputs.squeeze(1)
        
        
        loss = criterion(outputs, masks)
        
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            outputs = model(images)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    
    val_epoch_loss = val_loss / len(val_loader.dataset)
    print(f'Validation Loss: {val_epoch_loss:.4f}')

def calculate_mIoU(loader, model, threshold=0.5):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.int)
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(1)
            preds = (outputs > threshold).int()
            
            
            intersection = (preds & masks).float().sum((1, 2))
            union = (preds | masks).float().sum((1, 2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.append(iou.mean().item())
    
    mIoU = np.mean(iou_scores)
    return mIoU


mIoU = calculate_mIoU(val_loader, model)
print(f'mIoU on validation set: {mIoU:.4f}')

def visualize_predictions(loader, model, num_images=5):
    model.eval()
    images_shown = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            masks = masks.cpu().numpy()
            images = images.cpu().numpy()
            
            for i in range(images.shape[0]):
                if images_shown >= num_images:
                    return
                img = images[i][:3]  
                img = np.transpose(img, (1, 2, 0))
                mask = masks[i]
                pred = outputs[i]
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title('Исходное изображение')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title('Истинная маска')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='gray')
                plt.title('Предсказанная маска')
                plt.axis('off')
                
                plt.show()
                images_shown += 1


visualize_predictions(val_loader, model)