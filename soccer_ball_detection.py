import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the CNN model
class BallDetectorNet(nn.Module):
    def __init__(self):
        super(BallDetectorNet, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        # Fourth conv block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 4)  # output 4 coordinates
    
    def forward(self, x):
        # pass through conv layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layers
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class BallDataset(Dataset):
    def __init__(self, image_folder, annotations_file, training=False):
        self.image_folder = image_folder
        self.training = training
        
        # read the csv file
        df = pd.read_csv(annotations_file)
        
        # filter only ball rows
        ball_df = df[df['class'].str.lower().str.contains('ball')]
        
        # remove duplicate filenames
        self.annotations = ball_df.drop_duplicates(subset=['filename']).reset_index(drop=True)
        
        print(f'Found {len(self.annotations)} images with ball')
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        
        # load image
        img_file = os.path.join(self.image_folder, row['filename'])
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        # get bbox coordinates and normalize them
        x1 = row['xmin'] / width
        y1 = row['ymin'] / height
        x2 = row['xmax'] / width
        y2 = row['ymax'] / height
        
        # data augmentation for training
        if self.training and np.random.random() > 0.5:
            # horizontal flip
            image = np.fliplr(image)
            x1_new = 1.0 - x2
            x2_new = 1.0 - x1
            x1, x2 = x1_new, x2_new
        
        # resize image
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0
        
        # convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        
        return image, bbox


def calculate_iou(pred_box, true_box):
    # convert to numpy if needed
    if torch.is_tensor(pred_box):
        pred_box = pred_box.detach().cpu().numpy()
    if torch.is_tensor(true_box):
        true_box = true_box.detach().cpu().numpy()
    
    # get coordinates
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
    true_x1, true_y1, true_x2, true_y2 = true_box
    
    # intersection
    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    union_area = pred_area + true_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def train():
    # check if gpu available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # load datasets
    train_data = BallDataset('Soccerball/train', 'Soccerball/train/_annotations.csv', training=True)
    val_data = BallDataset('Soccerball/valid', 'Soccerball/valid/_annotations.csv', training=False)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    
    # create model
    model = BallDetectorNet()
    model = model.to(device)
    
    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training
    epochs = 150
    best_val_iou = 0
    losses_history = []
    iou_history = []
    
    print('\nStarting training...\n')
    
    for epoch in range(epochs):
        # train
        model.train()
        total_loss = 0
        
        for images, bboxes in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            bboxes = bboxes.to(device)
            
            # forward pass
            predictions = model(images)
            loss = loss_fn(predictions, bboxes)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses_history.append(avg_loss)
        
        # validation
        model.eval()
        val_ious = []
        
        with torch.no_grad():
            for images, bboxes in val_loader:
                images = images.to(device)
                predictions = model(images)
                predictions = predictions.cpu()
                
                for pred, true in zip(predictions, bboxes):
                    iou = calculate_iou(pred, true)
                    val_ious.append(iou)
        
        avg_iou = np.mean(val_ious)
        iou_history.append(avg_iou)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val IoU = {avg_iou:.4f}')
        
        # save best model
        if avg_iou > best_val_iou:
            best_val_iou = avg_iou
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'  Model saved! Best IoU: {best_val_iou:.4f}')
    
    # save training plots
    if not os.path.exists('results'):
        os.makedirs('results')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(iou_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    
    print(f'\nTraining finished!')
    print(f'Best validation IoU: {best_val_iou:.4f}')


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load test data
    test_data = BallDataset('Soccerball/test', 'Soccerball/test/_annotations.csv', training=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    # load model
    model = BallDetectorNet()
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f'Testing on {len(test_data)} images...')
    
    # test
    all_ious = []
    results = []
    
    with torch.no_grad():
        for images, bboxes in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            predictions = model(images)
            predictions = predictions.cpu()
            
            for pred, true in zip(predictions, bboxes):
                iou = calculate_iou(pred, true)
                all_ious.append(iou)
                
                results.append({
                    'pred_xmin': pred[0].item(),
                    'pred_ymin': pred[1].item(),
                    'pred_xmax': pred[2].item(),
                    'pred_ymax': pred[3].item(),
                    'iou': iou
                })
    
    # compute metrics
    mean_iou = np.mean(all_ious)
    median_iou = np.median(all_ious)
    iou_50 = sum([1 for iou in all_ious if iou >= 0.5]) / len(all_ious) * 100
    iou_75 = sum([1 for iou in all_ious if iou >= 0.75]) / len(all_ious) * 100
    
    print('\n' + '='*50)
    print('TEST RESULTS')
    print('='*50)
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Median IoU: {median_iou:.4f}')
    print(f'Accuracy @ IoU >= 0.5: {iou_50:.1f}%')
    print(f'Accuracy @ IoU >= 0.75: {iou_75:.1f}%')
    print('='*50)
    
    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/test_predictions.csv', index=False)
    print(f'\nResults saved to results/test_predictions.csv')


def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load model
    model = BallDetectorNet()
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # load and preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # predict
    with torch.no_grad():
        prediction = model(img_tensor)
    
    prediction = prediction.cpu().numpy()[0]
    
    # convert to pixel coordinates
    x1 = int(prediction[0] * orig_w)
    y1 = int(prediction[1] * orig_h)
    x2 = int(prediction[2] * orig_w)
    y2 = int(prediction[3] * orig_h)
    
    print(f'\nPredicted bounding box:')
    print(f'xmin: {x1}, ymin: {y1}, xmax: {x2}, ymax: {y2}')
    
    # draw bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # save
    if not os.path.exists('results'):
        os.makedirs('results')
    cv2.imwrite('results/prediction.jpg', img)
    print('Saved to results/prediction.jpg')


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python ball_detection.py train')
        print('  python ball_detection.py test')
        print('  python ball_detection.py predict <image_path>')
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    elif mode == 'predict':
        if len(sys.argv) < 3:
            print('Please provide image path')
            sys.exit(1)
        predict(sys.argv[2])
    else:
        print(f'Unknown command: {mode}')