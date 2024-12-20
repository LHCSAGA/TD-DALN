import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\CE_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CE_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
CE_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
CE_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
CE_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)
CE_combined_tensor = torch.stack([CE_current_tensor, CE_position_tensor, CE_ref_tensor, CE_velocity_tensor], dim=1)
mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\CI_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)
CI_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
CI_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
CI_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
CI_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)
CI_combined_tensor = torch.stack([CI_current_tensor, CI_position_tensor, CI_ref_tensor, CI_velocity_tensor], dim=1)
mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\CP_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)
CP_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
CP_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
CP_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
CP_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)
CP_combined_tensor = torch.stack([CP_current_tensor, CP_position_tensor, CP_ref_tensor, CP_velocity_tensor], dim=1)
mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\DZ_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)
DZ_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
DZ_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
DZ_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
DZ_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)
DZ_combined_tensor = torch.stack([DZ_current_tensor, DZ_position_tensor, DZ_ref_tensor, DZ_velocity_tensor], dim=1)
all_data = torch.cat((CE_combined_tensor, CI_combined_tensor, CP_combined_tensor, DZ_combined_tensor),dim=0)
all_labels = torch.cat((torch.zeros(54), torch.ones(54), 2 * torch.ones(54), 3 * torch.ones(54)), dim=0).long()


X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class AdvancedLSTMClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_classes=4, dropout_rate=0.5):
        super(AdvancedLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_size = 4
hidden_size = 128
num_layers = 2
num_classes = 4
num_heads = 8
dropout_rate = 0.1

model = AdvancedLSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3000
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Accuracy on test set: {100 * accuracy:.2f}%')

class_names = ['CE_A', 'CI_A', 'CP_A', 'DZ_A']
conf_matrix = confusion_matrix(all_labels, all_preds)


