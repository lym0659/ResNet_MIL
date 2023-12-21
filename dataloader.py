"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import pandas as pd
import torch
import os
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split


class CTScanDataset(Dataset):
    def __init__(self, root_dir, label_file):
        self.root_dir = root_dir
        self.labels = pd.read_csv(label_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patient_id = self.labels.iloc[idx]['PatientID']
        file_name = self.labels.iloc[idx]['FileName']
        label = torch.tensor([self.labels.iloc[idx]['Label']], dtype=torch.float32)

        # print(label)
        

        img_dir = os.path.join(self.root_dir, f'{file_name}')
        images = []
        i = 0
        for img_name in os.listdir(img_dir):
            if i% 10 == 0:
                img_path = os.path.join(img_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                images.append(img)
            i += 1
        bag = torch.stack(images)
        return bag, label

class PeXScanDataset(Dataset):

    def __init__(self, root_dir, label_file, patient_file):
        self.root_dir = root_dir
        self.labels = pd.read_csv(label_file)
        self.patient_file = pd.read_csv(patient_file)

        merged_df = pd.merge(self.patient_file, self.labels, on='FileName', how='inner')
        patient_file_only_df = self.patient_file[~self.patient_file['FileName'].isin(merged_df['FileName'])]
        
        # 'Label' 컬럼을 가져올 때 self.labels["Label"]로 가져오기
        self.new_labels_df = pd.DataFrame({'FileName': patient_file_only_df['FileName'], 'Label': self.labels["Label"].iloc[0]})

        # 두 데이터프레임을 합치기
        self.final_merged_df = pd.concat([merged_df, self.new_labels_df], ignore_index=True)
        self.final_merged_df = self.final_merged_df.dropna()
        
        # print(self.final_merged_df)
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.final_merged_df)

    def __getitem__(self, idx):
        # self.final_merged_df.to_csv("pex_label.csv")
        file_name = self.final_merged_df.iloc[idx]['FileName']
        label = torch.tensor([self.final_merged_df.iloc[idx]['Label']], dtype=torch.float32)

        img_dir = os.path.join(self.root_dir, f'{file_name}')
        images = []

        # print("file_name : %s, label : %s"%(file_name, label))
        for side in ['left', 'middle', 'right']:
            side_dir = os.path.join(img_dir, side)
            # i = 0
            for img_name in os.listdir(side_dir):
                # if i % 2 == 0:
                img_path = os.path.join(side_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                images.append(img)
                # i += 1
     
        bag = torch.stack(images)
        return bag, label

if __name__ == "__main__":

    root_dir = "/root/datasets/PeX-ray"
    label_file = "/root/datasets/Implant_Marking_pex.csv"
    patient_file = "/root/datasets/PeX_info.csv"
    dataset = PeXScanDataset(root_dir=root_dir, label_file=label_file, patient_file=patient_file)

    train_indices, temp_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_data_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        # mnist_bags_train += label[0].numpy()[0]
        mnist_bags_train += label.numpy()[0]

    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_data_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(valid_dataset):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label.numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(valid_dataset),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
