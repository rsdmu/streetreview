import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(filename="model_output.log", level=logging.INFO, format='%(asctime)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CustomDataset(Dataset):
    def __init__(self, evaluations_path, logits_path, transform=None, encoding='utf-8', max_length=15000):
        self.evaluations_path = evaluations_path
        self.logits_path = logits_path
        self.transform = transform
        self.encoding = encoding
        self.max_length = max_length

        # Read evaluations
        self.evaluations = {}
        for file in os.listdir(evaluations_path):
            if file.endswith('.csv'):
                point_id = file.split('_')[0]
                eval_path = os.path.join(evaluations_path, file)
                try:
                    self.evaluations[point_id] = pd.read_csv(eval_path, encoding=self.encoding)
                except UnicodeDecodeError:
                    self.evaluations[point_id] = pd.read_csv(eval_path, encoding='ISO-8859-1')

        # Collect all logits files
        self.logits_files = []
        for subdir in os.listdir(logits_path):
            subdir_path = os.path.join(logits_path, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.csv'):
                        self.logits_files.append(os.path.join(subdir_path, file))

    def __len__(self):
        return len(self.logits_files)

    def __getitem__(self, idx):
        logits_file = self.logits_files[idx]
        try:
            logits_data = pd.read_csv(logits_file, encoding=self.encoding)
        except UnicodeDecodeError:
            logging.info(f"Error reading {logits_file}. Trying different encoding.")
            logits_data = pd.read_csv(logits_file, encoding='ISO-8859-1')

        if 'class_label' in logits_data.columns:
            logits_data.drop(columns=['class_label'], inplace=True)

        # Convert to tensor
        logits_data = logits_data.astype(float).fillna(0)
        logits_tensor = torch.tensor(logits_data.values, dtype=torch.float32)

        # Normalize
        logits_tensor = self.normalize_data(logits_tensor)

        # Pad or truncate
        if logits_tensor.size(0) < self.max_length:
            pad_size = self.max_length - logits_tensor.size(0)
            logits_tensor = F.pad(logits_tensor, (0, 0, 0, pad_size), mode='constant', value=0)
        else:
            logits_tensor = logits_tensor[:self.max_length, :]

        if self.transform:
            logits_tensor = self.transform(logits_tensor)

        # Identify point ID and get evaluation
        point_id = os.path.basename(os.path.dirname(logits_file))
        evaluation_data = self.evaluations.get(point_id)

        if evaluation_data is None:
            # If no evaluation data, create a zero-vector
            evaluation_tensor = torch.zeros(28, dtype=torch.float32)
        else:
            if 'pointid' in evaluation_data.columns:
                evaluation_data.drop(columns=['pointid'], inplace=True, errors='ignore')
            evaluation_data = evaluation_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            evaluation_tensor = torch.tensor(evaluation_data.values, dtype=torch.float32)
            # Flatten
            if evaluation_tensor.ndim > 1:
                evaluation_tensor = evaluation_tensor.view(-1)
            # Pad or truncate to size 28
            if evaluation_tensor.size(0) < 28:
                pad_size = 28 - evaluation_tensor.size(0)
                evaluation_tensor = F.pad(evaluation_tensor, (0, pad_size), mode='constant', value=0)
            else:
                evaluation_tensor = evaluation_tensor[:28]

        return logits_tensor, evaluation_tensor

    def normalize_data(self, data_tensor):
        data_mean = data_tensor.mean(dim=0, keepdim=True)
        data_std = data_tensor.std(dim=0, keepdim=True)
        return (data_tensor - data_mean) / (data_std + 1e-6)


def to_tensor(data):
    if torch.is_tensor(data):
        return data.clone().detach()
    else:
        return torch.tensor(data, dtype=torch.float32)


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // self.num_heads
        assert self.head_dim * self.num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_out = torch.matmul(attention_weights, V)
        out = self.fc_out(attention_out)
        return out


class MultiOutputModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 12
        num_heads = 6
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.attention = MultiHeadAttention(feature_dim=feature_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(feature_dim, 160)
        self.fc2 = nn.Linear(160, 320)
        self.fc3 = nn.Linear(320, 640)
        self.fc4 = nn.Linear(640, 640)
        self.fc5 = nn.Linear(640, 320)
        self.fc6 = nn.Linear(320, 160)
        self.output = nn.Linear(160, 28)

    def forward(self, x):
        # Attention
        x = self.attention(x)
        # Mean pooling
        x = x.mean(dim=1)
        # Feedforward layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # Output layer
        return self.output(x)


if __name__ == "__main__":
    # Example usage
    evaluations_path = 'path_to_evaluations'
    logits_path = 'path_to_logits'
    dataset = CustomDataset(evaluations_path, logits_path, transform=to_tensor)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    def custom_collate_fn(batch):
        logits, evaluations = zip(*batch)
        logits = torch.stack(logits, dim=0)
        evaluations = torch.stack(evaluations, dim=0)
        return logits, evaluations

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=custom_collate_fn)

    # Model
    model = MultiOutputModelWithAttention().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    train_losses, valid_losses = [], []
    train_r2_scores, valid_r2_scores = [], []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_predictions, all_targets = [], []

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_r2 = r2_score(all_targets, all_predictions)
        train_r2_scores.append(train_r2)

        # Validation
        valid_loss = 0.0
        all_predictions, all_targets = [], []
        model.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        valid_loss /= len(test_loader)
        valid_losses.append(valid_loss)
        valid_r2 = r2_score(all_targets, all_predictions)
        valid_r2_scores.append(valid_r2)

        logging.info(
            f'Epoch {epoch+1}/{epochs} '
            f'Train Loss: {train_loss:.4f} '
            f'Valid Loss: {valid_loss:.4f} '
            f'Train R2: {train_r2:.4f} '
            f'Valid R2: {valid_r2:.4f}'
        )

    # Save the model
    torch.save(model.state_dict(), 'model_extra_large.pth')

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.close()
    logging.info("Training & validation loss plot saved.")
