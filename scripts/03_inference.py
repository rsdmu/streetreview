import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

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
        x = self.attention(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.output(x)

# Load model
model = MultiOutputModelWithAttention().to(device)
model_path = 'model_extra_large.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def normalize_data(data_tensor):
    data_mean = data_tensor.mean(dim=0, keepdim=True)
    data_std = data_tensor.std(dim=0, keepdim=True)
    return (data_tensor - data_mean) / (data_std + 1e-6)

def predict(model, new_logit_file_path, label_names, max_length=10000):
    try:
        new_logit_data = pd.read_csv(new_logit_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        new_logit_data = pd.read_csv(new_logit_file_path, encoding='ISO-8859-1')

    if 'class_label' in new_logit_data.columns:
        new_logit_data.drop('class_label', axis=1, inplace=True)

    new_logit_tensor = torch.tensor(new_logit_data.values, dtype=torch.float32)
    new_logit_tensor = normalize_data(new_logit_tensor)

    if new_logit_tensor.size(0) < max_length:
        pad_size = max_length - new_logit_tensor.size(0)
        new_logit_tensor = F.pad(new_logit_tensor, (0, 0, 0, pad_size), mode='constant', value=0)
    else:
        new_logit_tensor = new_logit_tensor[:max_length, :]

    new_logit_tensor = new_logit_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(new_logit_tensor)

    prediction_np = prediction.cpu().numpy().flatten()
    prediction_df = pd.DataFrame({'Label': label_names, 'Score': prediction_np})
    return prediction_df

def batch_predict(model, directory, label_names, max_length=20000):
    all_predictions = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            prediction_df = predict(model, file_path, label_names, max_length)
            all_predictions[filename] = prediction_df['Score'].tolist()

    results_df = pd.DataFrame.from_dict(all_predictions, orient='index', columns=label_names)
    results_df.to_csv('predictions.csv', index_label='Filename')
    print("Batch predictions saved to predictions.csv")

if __name__ == "__main__":
    label_names = [
        'Lgbtqia2+ Practicality', 'Lgbtqia2+ Inclusivity', 'Lgbtqia2+ Aesthetics', 'Lgbtqia2+ Accessibility',
        'Handicapped Practicality', 'Handicapped Inclusivity', 'Handicapped Aesthetics', 'Handicapped Accessibility',
        'Elderly Female Practicality', 'Elderly Female Inclusivity', 'Elderly Female Aesthetics', 'Elderly Female Accessibility',
        'Elderly Male Practicality', 'Elderly Male Inclusivity', 'Elderly Male Aesthetics', 'Elderly Male Accessibility',
        'Young Male Practicality', 'Young Male Inclusivity', 'Young Male Aesthetics', 'Young Male Accessibility',
        'Young Female Practicality', 'Young Female Inclusivity', 'Young Female Aesthetics', 'Young Female Accessibility',
        'Group Practicality', 'Group Inclusivity', 'Group Aesthetics', 'Group Accessibility'
    ]

    csv_directory = 'path_to_directory_containing_csvs'
    batch_predict(model, csv_directory, label_names, max_length=20000)
