import os
import cv2
import shutil  # Убедитесь, что shutil импортирован
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Определение архитектуры UNet

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
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)        # [B, 32, H, W]
        x2 = self.down1(x1)     # [B, 64, H/2, W/2]
        x3 = self.down2(x2)     # [B, 128, H/4, W/4]
        x4 = self.down3(x3)     # [B, 256, H/8, W/8]
        x5 = self.down4(x4)     # [B, 512, H/16, W/16]
        
        x = self.up1(x5)        # [B, 256, H/8, W/8]
        x = torch.cat([x, x4], dim=1)  # [B, 512, H/8, W/8]
        x = self.conv1(x)       # [B, 256, H/8, W/8]
        
        x = self.up2(x)         # [B, 128, H/4, W/4]
        x = torch.cat([x, x3], dim=1)  # [B, 256, H/4, W/4]
        x = self.conv2(x)       # [B, 128, H/4, W/4]
        
        x = self.up3(x)         # [B, 64, H/2, W/2]
        x = torch.cat([x, x2], dim=1)  # [B, 128, H/2, W/2]
        x = self.conv3(x)       # [B, 64, H/2, W/2]
        
        x = self.up4(x)         # [B, 32, H, W]
        x = torch.cat([x, x1], dim=1)  # [B, 64, H, W]
        x = self.conv4(x)       # [B, 32, H, W]
        
        logits = self.outc(x)   # [B, n_classes, H, W]
        return logits

# Функция для извлечения дополнительных признаков
def extract_aberration_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Лапласиан
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
    
    # Собель
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.hypot(sobelx, sobely)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())
    
    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='default')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    
    features = np.stack([laplacian, sobel, lbp], axis=-1)  # [H, W, 3]
    return features

# Определение класса датасета
class ContaminationDatasetWithFeatures(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        # Проверка соответствия количества изображений и масок
        if len(self.images) != len(self.masks):
            print(f"Количество изображений: {len(self.images)}")
            print(f"Количество масок: {len(self.masks)}")
            print("Количество изображений и масок не совпадает.")
            # Перечислим отсутствующие маски
            image_basenames = set(os.path.splitext(f)[0] for f in self.images)
            mask_basenames = set(os.path.splitext(f)[0] for f in self.masks)
            missing_masks = image_basenames - mask_basenames
            if missing_masks:
                print("Отсутствующие маски для следующих изображений:")
                for name in missing_masks:
                    print(f"{name}.png")
            raise ValueError("Количество изображений и масок не совпадает.")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Извлечение дополнительных признаков
        features = extract_aberration_features(image)  # [H, W, 3]
        
        # Комбинирование исходного изображения с признаками
        combined = np.concatenate([image, features], axis=-1)  # [H, W, 6]
        
        # Проверка формы combined
        # print(f"Loading image: {img_name}, Combined shape: {combined.shape}")
        
        # Загрузка маски
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Не удалось загрузить маску для изображения: {img_name}")
        mask = mask / 255.0
        
        if self.transform:
            augmented = self.transform(image=combined, mask=mask)
            combined = augmented['image']
            mask = augmented['mask']
        
        return combined, mask

# Определение трансформаций
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(
        mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 6 элементов
        std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)     # 6 элементов
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 6 элементов
        std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)     # 6 элементов
    ),
    ToTensorV2()
])

def main():
    # Пути к данным
    IMAGES_DIR = './train_dataset/cv_open_dataset/open_img'  # Путь к вашему датасету с изображениями
    MASKS_DIR = './train_dataset/cv_open_dataset/open_msk'  # Путь к вашему датасету с масками
    OUTPUT_DIR = './datasets/train_data'  # Путь к выходной директории

    # Создание выходных директорий
    os.makedirs(os.path.join(OUTPUT_DIR, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels/val'), exist_ok=True)

    # Получение списка файлов
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]

    # Проверка соответствия
    if len(image_files) != len(mask_files):
        print(f"Количество изображений: {len(image_files)}")
        print(f"Количество масок: {len(mask_files)}")
        print("Количество изображений и масок не совпадает.")
        # Перечислим отсутствующие маски
        image_basenames = set(os.path.splitext(f)[0] for f in image_files)
        mask_basenames = set(os.path.splitext(f)[0] for f in mask_files)
        missing_masks = image_basenames - mask_basenames
        if missing_masks:
            print("Отсутствующие маски для следующих изображений:")
            for name in missing_masks:
                print(f"{name}.png")
        raise ValueError("Количество изображений и масок не совпадает.")

    # Разделение на обучающую и валидационную выборки
    TRAIN_SIZE = 0.8
    train_images, val_images = train_test_split(image_files, train_size=TRAIN_SIZE, random_state=42)

    # Копирование изображений и масок в соответствующие папки
    for img in train_images:
        src_img_path = os.path.join(IMAGES_DIR, img)
        dst_img_path = os.path.join(OUTPUT_DIR, 'images/train', img)
        shutil.copy(src_img_path, dst_img_path)
        
        mask_name = os.path.splitext(img)[0] + '.png'
        src_mask_path = os.path.join(MASKS_DIR, mask_name)
        dst_mask_path = os.path.join(OUTPUT_DIR, 'labels/train', mask_name)
        if os.path.exists(src_mask_path):
            shutil.copy(src_mask_path, dst_mask_path)
        else:
            print(f"Маска не найдена для изображения: {img}")

    for img in val_images:
        src_img_path = os.path.join(IMAGES_DIR, img)
        dst_img_path = os.path.join(OUTPUT_DIR, 'images/val', img)
        shutil.copy(src_img_path, dst_img_path)
        
        mask_name = os.path.splitext(img)[0] + '.png'
        src_mask_path = os.path.join(MASKS_DIR, mask_name)
        dst_mask_path = os.path.join(OUTPUT_DIR, 'labels/val', mask_name)
        if os.path.exists(src_mask_path):
            shutil.copy(src_mask_path, dst_mask_path)
        else:
            print(f"Маска не найдена для изображения: {img}")

    # Создание файла data.yaml
    data_yaml_content = f"""
train: datasets/train_data/images/train
val: datasets/train_data/images/val

nc: 1  # Количество классов (1 для загрязнения)
names: ['contaminated']  # Название класса
"""

    with open('data.yaml', 'w') as f:
        f.write(data_yaml_content)

    print("Датасет успешно разбит и сохранен в структуре проекта.")

    # Создание датасетов и загрузчиков
    try:
        train_dataset = ContaminationDatasetWithFeatures(
            image_dir=os.path.join(OUTPUT_DIR, 'images/train'),
            mask_dir=os.path.join(OUTPUT_DIR, 'labels/train'),
            transform=train_transform
        )
    except ValueError as e:
        print(e)
        return

    try:
        val_dataset = ContaminationDatasetWithFeatures(
            image_dir=os.path.join(OUTPUT_DIR, 'images/val'),
            mask_dir=os.path.join(OUTPUT_DIR, 'labels/val'),
            transform=val_transform
        )
    except ValueError as e:
        print(e)
        return

    # На Windows рекомендуется устанавливать num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # Инициализация модели, критерия и оптимизатора
    model = UNet(n_channels=6, n_classes=1)  # 3 канала RGB + 3 дополнительных признака
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Функция расчета mIoU
    def calculate_mIoU(loader, model, threshold=0.5):
        model.eval()
        iou_scores = []
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.int)
                
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                preds = (outputs > threshold).int()
                
                intersection = (preds & masks).float().sum((1, 2))
                union = (preds | masks).float().sum((1, 2))
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                # Отладочные выводы
                print(f"Batch IoU: {iou.cpu().numpy()}")
                
                iou_scores.extend(iou.cpu().numpy())
        
        mIoU = np.mean(iou_scores)
        return mIoU

    # Функция для визуализации предсказаний (опционально)
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
                    img = images[i][:3]  # Первые 3 канала RGB
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

    # Цикл обучения
    num_epochs = 25
    best_mIoU = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            outputs = outputs.squeeze(1)  # [B, H, W]
            
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Валидация
        val_mIoU = calculate_mIoU(val_loader, model)
        print(f'Validation mIoU: {val_mIoU:.4f}')
        
        # Сохранение лучшей модели
        if(val_mIoU<0.98):
            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                torch.save(model.state_dict(), 'model.pth')
                print(f'Лучшая модель сохранена с mIoU: {best_mIoU:.4f}')
        
        # (Опционально) Визуализация предсказаний
        visualize_predictions(val_loader, model, num_images=10)
    
    print(f'Обучение завершено. Лучшая модель имеет mIoU: {best_mIoU:.4f}')
    
    # Окончательное сохранение модели (если не была сохранена в цикле)
    if not os.path.exists('model.pth'):
        torch.save(model.state_dict(), 'model.pth')
        print("Модель успешно сохранена как model.pth")
    
    # Функция для инференса на отдельных изображениях (опционально)
    def infer_image(image_path, model, device, output_path='./mask_image.png', threshold=0.5):
        model.eval()
        with torch.no_grad():
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Извлечение дополнительных признаков
            features = extract_aberration_features(image_rgb)
            
            # Комбинирование исходного изображения с признаками
            combined = np.concatenate([image_rgb, features], axis=-1)  # [H, W, 6]
            
            # Применение трансформаций
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                ToTensorV2()
            ])
            
            augmented = transform(image=combined)
            image_tensor = augmented['image'].unsqueeze(0).to(device, dtype=torch.float32)  # [1, 6, 256, 256]
            
            # Инференс
            output = model(image_tensor)
            output = torch.sigmoid(output).squeeze(1).cpu().numpy()  # [256, 256]
            
            # Пороговая обработка
            mask = (output > threshold).astype(np.uint8) * 255  # [256, 256]
            
            # Масштабирование маски до оригинального размера
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Сохранение маски
            cv2.imwrite(output_path, mask_resized)
            print(f"Маска сохранена по пути: {output_path}")
    
    # Пример использования функции инференса (опционально)
    # infer_image('./cv_test_dataset/test_img/_08.jpg', model, device)
    
    # Создание файла data.yaml для соответствия бейзлайну
    data_yaml_content = f"""
train: datasets/train_data/images/train
val: datasets/train_data/images/val

nc: 1  # Количество классов (1 для загрязнения)
names: ['contaminated']  # Название класса
"""
    
    with open('data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print("Файл data.yaml создан.")

if __name__ == '__main__':
    main()