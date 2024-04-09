import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


data = pd.read_csv("study_data.csv",low_memory=False)

from scipy import stats

cols = data.columns.values

# Groups the companies by 'Years Since Founded' and standardizes non-binary features in each group
for col in cols[:-2]:
    if col.startswith('Details.Description') or col.startswith('Website.') or col.startswith('Overview') or col.startswith('Education') or col.startswith('Major'):
        if col not in ["Overview.Gender.Agender", "Overview.Gender.Non-Binary"]:
            data[col] = data.groupby('Details.Years Since Founded')[col].transform(lambda x : stats.zscore(x,ddof=1,nan_policy='omit'))

            
# Splits the data into features and target
Y = data[data.columns[-2:]].copy()
X = data.drop(columns=['Target', 'Details.Years Since Founded'])


# Assuming 'data' has been loaded and preprocessed
class TabularDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = dataframe.iloc[:, :-2].values.astype(np.float32)  # Exclude the last two columns
        self.targets = dataframe.iloc[:, -2:].values.astype(np.float32)   # Last two columns as targets

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        targets = torch.tensor(self.targets[idx], dtype=torch.float)
        return features, targets

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 1)  # Output layer for classification
        self.regressor = nn.Linear(32, 1)   # Output layer for regression

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        classification_output = torch.sigmoid(self.classifier(x))  # Sigmoid activation for binary classification
        regression_output = self.regressor(x)  # No activation for regression
        return classification_output, regression_output

# Load your data into the dataset
dataset = TabularDataset(data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model, loss functions, and optimizer
model = SimpleNN(input_size=X.shape[1])  # Assuming 'X' is your features matrix
classification_loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
regression_loss_function = nn.MSELoss()      # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, targets in data_loader:
        optimizer.zero_grad()
        classification_preds, regression_preds = model(features)
        
        # Compute the losses for both outputs
        classification_loss = classification_loss_function(classification_preds, targets[:, 0].unsqueeze(1))
        regression_loss = regression_loss_function(regression_preds, targets[:, 1].unsqueeze(1))
        
        # Combine the losses
        loss = classification_loss + regression_loss
        total_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')
