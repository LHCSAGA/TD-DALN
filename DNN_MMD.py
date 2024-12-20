import torch
import torch.nn as nn
import data_load
import torch.optim as optim
from utils import set_model_mode

source_test_loader = data_load.source_test_loader
target_test_loader = data_load.target_test_loader

source_train_loader = data_load.source_train_loader
target_train_loader = data_load.target_train_loader

all_true_labels = []
all_pred_labels = []



class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class FCNetwork(nn.Module):
    def __init__(self):
        super(FCNetwork, self).__init__()

        self.fc1 = nn.Linear(4 * 501, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 20)
        self.fc5 = nn.Linear(20, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

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

input_size = 4
hidden_size = 128
num_layers = 2
num_classes = 4
num_heads = 8
dropout_rate = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 1000

def calculate_accuracy(correct, total):
    return 100. * correct / total

def print_accuracy( accuracies):
    for key, value in accuracies.items():
        print(f"{key} Accuracy: {value['correct']}/{value['total']} ({value['accuracy']:.2f}%)")

def compute_output(classifier, images):
    outputs = classifier(images)

    preds = outputs.data.max(1, keepdim=True)[1]
    return preds

def tester(classifier, source_test_loader, target_test_loader):
    classifier.cuda()
    set_model_mode('eval', [classifier])

    source_correct = 0
    target_correct = 0

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        source_image, source_label = source_data
        target_image, target_label = target_data

        source_image, source_label = source_image.cuda(), source_label.cuda()
        target_image, target_label = target_image.cuda(), target_label.cuda()

        source_pred = compute_output(classifier, source_image)
        target_pred = compute_output(classifier, target_image)

        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).sum().item()
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).sum().item()


    source_dataset_len = len(source_test_loader.dataset)
    target_dataset_len = len(target_test_loader.dataset)

    accuracies = {
        "Source": {
            "correct": source_correct,
            "total": source_dataset_len,
            "accuracy": calculate_accuracy(source_correct, source_dataset_len)
        },
        "Target": {
            "correct": target_correct,
            "total": target_dataset_len,
            "accuracy": calculate_accuracy(target_correct, target_dataset_len)
        }
    }

    print_accuracy(accuracies)

def DANN_MMD(classifier, source_train_loader, target_train_loader, source_test_loader, target_test_loader):

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [classifier])

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            target_image, target_label = target_data
            target_image, target_label = target_image.cuda(), target_label.cuda()  # 32

            optimizer.zero_grad()
            source_pred = classifier(source_image)

            class_loss = classifier_criterion(source_pred, source_label)

            target_pred = classifier(target_image)
            mmd_loss = MMD(source_pred, target_pred)

            loss_all = class_loss + mmd_loss
            loss_all.backward()
            optimizer.step()
            if (batch_idx + 1) % 40 == 0:
                total_processed = batch_idx * len(source_image)
                total_dataset = len(source_train_loader.dataset)
                percentage_completed = 100. * batch_idx / len(source_train_loader)
                print(f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tclass_loss: {class_loss.item():.4f}')
                print(
                    f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tmmd_loss: {mmd_loss.item():.4f}')
                print(
                    f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tloss_all: {loss_all.item():.4f}')


        tester(classifier, source_test_loader, target_test_loader)


MMD = MMDLoss()
model = FCNetwork().to(device)
DANN_MMD(model, source_train_loader, target_train_loader, source_test_loader, target_test_loader)



