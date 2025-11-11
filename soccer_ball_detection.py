import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class BallDetectionCNN(nn.Module):
    def __init__(self):
        super(BallDetectionCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BallDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        df = pd.read_csv(csv_path)
        self.data = df[df['class'].str.contains('ball', case=False, na=False)]
        self.data = self.data.drop_duplicates('filename').reset_index(drop=True)
        
        self.img_width = self.data['width'].iloc[0]
        self.img_height = self.data['height'].iloc[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_path = os.path.join(self.img_dir, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        xmin = row['xmin'] / self.img_width
        ymin = row['ymin'] / self.img_height
        xmax = row['xmax'] / self.img_width
        ymax = row['ymax'] / self.img_height
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        
        return img, bbox


def get_transform(training=True):
    if training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def iou_loss(pred, target):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
    union = pred_area + tgt_area - inter
    
    iou = inter / (union + 1e-6)
    return (1 - iou).mean()


def calculate_iou(pred, target):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred[0], pred[1], pred[2], pred[3]
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target[0], target[1], target[2], target[3]
    
    inter_x1 = max(pred_x1, tgt_x1)
    inter_y1 = max(pred_y1, tgt_y1)
    inter_x2 = min(pred_x2, tgt_x2)
    inter_y2 = min(pred_y2, tgt_y2)
    
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
    union = pred_area + tgt_area - inter
    
    return inter / (union + 1e-6)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    mse = nn.MSELoss()
    
    for imgs, targets in tqdm(loader, desc='Training'):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        preds = model(imgs)
        loss = 0.5 * mse(preds, targets) + 0.5 * iou_loss(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    ious = []
    mse = nn.MSELoss()
    
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc='Validating'):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            preds = model(imgs)
            loss = 0.5 * mse(preds, targets) + 0.5 * iou_loss(preds, targets)
            total_loss += loss.item()
            
            for pred, target in zip(preds, targets):
                iou = calculate_iou(pred.cpu(), target.cpu())
                ious.append(iou)
    
    return total_loss / len(loader), np.mean(ious)


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_dataset = BallDataset('Soccerball/train', 'Soccerball/train/_annotations.csv', get_transform(True))
    val_size = int(0.15 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)
    
    model = BallDetectionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    best_iou = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(100):
        print(f'\nEpoch {epoch+1}/100')
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
        
        scheduler.step(val_loss)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({'model': model.state_dict(), 'iou': val_iou}, 'models/best_model.pth')
            print(f'Saved best model (IoU: {best_iou:.4f})')
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'])
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.savefig('results/training.png')
    print(f'\nTraining complete. Best IoU: {best_iou:.4f}')


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = BallDataset('Soccerball/test', 'Soccerball/test/_annotations.csv', get_transform(False))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = BallDetectionCNN().to(device)
    checkpoint = torch.load('models/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    ious = []
    results = []
    
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc='Testing'):
            imgs = imgs.to(device)
            preds = model(imgs).cpu()
            
            for i, (pred, target) in enumerate(zip(preds, targets)):
                iou = calculate_iou(pred, target)
                ious.append(iou)
                
                pred_px = [
                    int(pred[0] * test_dataset.img_width),
                    int(pred[1] * test_dataset.img_height),
                    int(pred[2] * test_dataset.img_width),
                    int(pred[3] * test_dataset.img_height)
                ]
                
                results.append({
                    'pred_xmin': pred_px[0], 'pred_ymin': pred_px[1],
                    'pred_xmax': pred_px[2], 'pred_ymax': pred_px[3],
                    'iou': iou
                })
    
    mean_iou = np.mean(ious)
    acc_50 = np.mean([iou >= 0.5 for iou in ious]) * 100
    acc_75 = np.mean([iou >= 0.75 for iou in ious]) * 100
    
    print(f'\nTest Results:')
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Accuracy @ IoU=0.50: {acc_50:.2f}%')
    print(f'Accuracy @ IoU=0.75: {acc_75:.2f}%')
    
    pd.DataFrame(results).to_csv('results/predictions.csv', index=False)
    print('Results saved to results/predictions.csv')


def predict_image(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BallDetectionCNN().to(device)
    checkpoint = torch.load('models/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    img_resized = cv2.resize(img, (224, 224))
    transform = get_transform(False)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img_tensor)[0].cpu().numpy()
    
    xmin = int(pred[0] * w)
    ymin = int(pred[1] * h)
    xmax = int(pred[2] * w)
    ymax = int(pred[3] * h)
    
    print(f'Predicted bounding box:')
    print(f'xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}')
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (220, 20, 60), 3)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('results/prediction.png', bbox_inches='tight')
    plt.show()
    
    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python train.py train')
        print('  python train.py test')
        print('  python train.py predict <image_path>')
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'train':
        train_model()
    elif mode == 'test':
        test_model()
    elif mode == 'predict' and len(sys.argv) == 3:
        predict_image(sys.argv[2])
    else:
        print('Invalid command')