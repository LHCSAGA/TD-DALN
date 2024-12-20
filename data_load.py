import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import params

mat_file_path = 'D:\code\DANN_LSTM\DATA\CE_A_data.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CE_current_tensor = torch.tensor(mat_data['Current'], dtype=torch.float32)
CE_position_tensor = torch.tensor(mat_data['Position'], dtype=torch.float32)
CE_ref_tensor = torch.tensor(mat_data['Ref'], dtype=torch.float32)
CE_velocity_tensor = torch.tensor(mat_data['Velocity'], dtype=torch.float32)

source_CE_combined_tensor = torch.stack([CE_current_tensor, CE_position_tensor, CE_ref_tensor, CE_velocity_tensor], dim=1)
with open('CE_selected_indices.txt', 'r') as file:
    selected_indices = [int(line.strip()) for line in file.readlines()]

source_CE_combined_tensor = source_CE_combined_tensor[selected_indices]

mat_file_path = 'D:\code\DANN_LSTM\DATA\CI_A_data.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CI_current_tensor = torch.tensor(mat_data['Current'], dtype=torch.float32)
CI_position_tensor = torch.tensor(mat_data['Position'], dtype=torch.float32)
CI_ref_tensor = torch.tensor(mat_data['Ref'], dtype=torch.float32)
CI_velocity_tensor = torch.tensor(mat_data['Velocity'], dtype=torch.float32)

source_CI_combined_tensor = torch.stack([CI_current_tensor, CI_position_tensor, CI_ref_tensor, CI_velocity_tensor], dim=1)
with open('CI_selected_indices.txt', 'r') as file:
    selected_indices = [int(line.strip()) for line in file.readlines()]

source_CI_combined_tensor = source_CI_combined_tensor[selected_indices]

mat_file_path = 'D:\code\DANN_LSTM\DATA\CP_A_data.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CP_current_tensor = torch.tensor(mat_data['Current'], dtype=torch.float32)
CP_position_tensor = torch.tensor(mat_data['Position'], dtype=torch.float32)
CP_ref_tensor = torch.tensor(mat_data['Ref'], dtype=torch.float32)
CP_velocity_tensor = torch.tensor(mat_data['Velocity'], dtype=torch.float32)

source_CP_combined_tensor = torch.stack([CP_current_tensor, CP_position_tensor, CP_ref_tensor, CP_velocity_tensor], dim=1)
with open('CP_selected_indices.txt', 'r') as file:
    selected_indices = [int(line.strip()) for line in file.readlines()]

source_CP_combined_tensor = source_CP_combined_tensor[selected_indices]

mat_file_path = 'D:\code\DANN_LSTM\DATA\DZ_A_data.mat'
mat_data = scipy.io.loadmat(mat_file_path)

DZ_current_tensor = torch.tensor(mat_data['Current'], dtype=torch.float32)
DZ_position_tensor = torch.tensor(mat_data['Position'], dtype=torch.float32)
DZ_ref_tensor = torch.tensor(mat_data['Ref'], dtype=torch.float32)
DZ_velocity_tensor = torch.tensor(mat_data['Velocity'], dtype=torch.float32)


source_DZ_combined_tensor = torch.stack([DZ_current_tensor, DZ_position_tensor, DZ_ref_tensor, DZ_velocity_tensor], dim=1)
with open('DZ_selected_indices.txt', 'r') as file:
    selected_indices = [int(line.strip()) for line in file.readlines()]

source_DZ_combined_tensor = source_DZ_combined_tensor[selected_indices]

mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\CE_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CE_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
CE_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
CE_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
CE_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)

target_CE_combined_tensor = torch.stack([CE_current_tensor, CE_position_tensor, CE_ref_tensor, CE_velocity_tensor], dim=1)


mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\CI_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CI_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
CI_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
CI_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
CI_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)

target_CI_combined_tensor = torch.stack([CI_current_tensor, CI_position_tensor, CI_ref_tensor, CI_velocity_tensor], dim=1)

mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\CP_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)

CP_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
CP_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
CP_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
CP_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)

target_CP_combined_tensor = torch.stack([CP_current_tensor, CP_position_tensor, CP_ref_tensor, CP_velocity_tensor], dim=1)

mat_file_path = 'D:\code\DANN_LSTM\AD_data_Physical\DZ_A_data_Physical.mat'
mat_data = scipy.io.loadmat(mat_file_path)

DZ_current_tensor = torch.tensor(mat_data['Current'].astype(np.float32), dtype=torch.float32)
DZ_position_tensor = torch.tensor(mat_data['Position'].astype(np.float32), dtype=torch.float32)
DZ_ref_tensor = torch.tensor(mat_data['Ref'].astype(np.float32), dtype=torch.float32)
DZ_velocity_tensor = torch.tensor(mat_data['Velocity'].astype(np.float32), dtype=torch.float32)

target_DZ_combined_tensor = torch.stack([DZ_current_tensor, DZ_position_tensor, DZ_ref_tensor, DZ_velocity_tensor], dim=1)

source_all_data = torch.cat((source_CE_combined_tensor, source_CI_combined_tensor, source_CP_combined_tensor, source_DZ_combined_tensor),dim=0)
source_all_labels = torch.cat((torch.zeros(158), torch.ones(139), 2 * torch.ones(261), 3 * torch.ones(201)), dim=0).long()

source_X1_train, source_X1_test, source_y1_train, source_y1_test = train_test_split(source_all_data, source_all_labels, test_size=0.2, random_state=42)

target_all_data = torch.cat((target_CE_combined_tensor, target_CI_combined_tensor, target_CP_combined_tensor, target_DZ_combined_tensor),dim=0)
target_all_labels = torch.cat((torch.zeros(54), torch.ones(54), 2 * torch.ones(54), 3 * torch.ones(54)), dim=0).long()
target_all_data = target_all_data.repeat(11,1,1)
target_all_labels = target_all_labels.repeat(11)

source_X2, target_X2, source_y2, target_y2 = train_test_split(target_all_data, target_all_labels, test_size=0.66, random_state=42)
source_X2_train, source_X2_test, source_y2_train, source_y2_test = train_test_split(source_X2, source_y2, test_size=0.2, random_state=42)
target_X2_train, target_X2_test, target_y2_train, target_y2_test = train_test_split(target_X2, target_y2, test_size=0.2, random_state=42)

source_all_X_train = torch.cat((source_X1_train, source_X2_train), dim=0)
source_all_y_train = torch.cat((source_y1_train, source_y2_train), dim=0)
source_all_X_test = torch.cat((source_X1_test, source_X2_test), dim=0)
source_all_y_test = torch.cat((source_y1_test, source_y2_test), dim=0)

target_all_X_train = target_X2_train
target_all_y_train = target_y2_train
target_all_X_test = target_X2_test
target_all_y_test = target_y2_test

source_train_dataset = TensorDataset(source_all_X_train, source_all_y_train)
source_test_dataset = TensorDataset(source_all_X_test, source_all_y_test)

target_train_dataset = TensorDataset(target_all_X_train, target_all_y_train)
target_test_dataset = TensorDataset(target_all_X_test, target_all_y_test)

source_train_loader = DataLoader(source_train_dataset, batch_size=params.batch_size)
source_test_loader = DataLoader(source_test_dataset, batch_size=params.batch_size)

target_train_loader = DataLoader(target_train_dataset, batch_size=params.batch_size)
target_test_loader = DataLoader(target_test_dataset, batch_size=params.batch_size)