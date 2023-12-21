from __future__ import print_function

import numpy as np
import argparse
import os
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# from dataloader import MnistBags, CTScanDataset
from dataloader import *
# from model import Attention, GatedAttention, BinaryResNet50Model
from model import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='resnet50', help='Choose b/w resnet50 and unet')
parser.add_argument('--resume', type=str, default=None, help='Your trained model')
parser.add_argument('--logdir', type=str, help='Path to save your model')
parser.add_argument('--dataset', type=str, default='CT')
parser.add_argument('--inference', type=bool, default=False)

# python main.py --model unet --logdir ./logs/UNet/CT/first --dataset CT 
# python main.py --model resnet50 --logdir ./logs/resnet50/CT/first --dataset CT 
# python main.py --inference True --logdir ./logs --dataset CT --model resnet50 --resume model_epoch_60

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.dataset == "CT":
    root_dir = "/root/datasets/CBCT"
    label_file = "/root/datasets/Implant_Marking.csv"
    dataset = CTScanDataset(root_dir=root_dir, label_file=label_file)

    train_indices, temp_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

elif args.dataset == "PeX":
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

save_path = args.logdir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()

elif args.model == 'resnet50':
    model = BinaryResNet50Model()

# elif args.model == 'unet':
#     model = UNet()

elif args.model == 'resnext':
    model = ResNext()

elif args.model == 'resnet':
    model = ResNet()

print("Your model : %s"%(args.model))
# model = BinaryResNet50Model()
print("loading...")
if args.cuda:
    model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def save_model(model, epoch, save_path):
    model_name = f"model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), save_path + "/" + model_name)
    # torch.save(model.state_dict(), './logs/best_model.pth')
    print(f"Model saved at: {save_path}")

def train_resume(epoch, model_name):
    loaded_model = model_name + ".pth"
    model.load_state_dict(torch.load(os.path.join(save_path, loaded_model)))

    trained_epoch = int(model_name[-2:]) + epoch

    model.train()
    train_loss = 0.0
    train_error = 0.0

    writer = SummaryWriter(save_path + "/tensorboard")

    for (data, label) in tqdm(train_data_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
        bag_label = label
        if args.cuda:
            data, bag_label = data.to(device), label.to(device)

        optimizer.zero_grad()
        data.requires_grad_(True)
        bag_label.requires_grad_(True)

        y_pred = model.forward(data)

        try:
            loss, _ = model.calculate_objective(data, bag_label)
        except:
            loss = model.calculate_objective(data, bag_label)
            # loss = model.calculate_objective(data, bag_label)

        # loss = loss = F.binary_cross_entropy_with_logits(y_pred, bag_label)
        loss.requires_grad_(True)
        train_loss += loss.item()
        
        try:
            error, _ = model.calculate_classification_error(data, bag_label)
        except:
            error = model.calculate_classification_error(data, bag_label)
            # error = model.calculate_classification_error(data, bag_label)
        train_error += error

        loss.backward()
        optimizer.step()

    train_loss /= len(train_data_loader)
    train_error /= len(train_data_loader)

    print('Epoch: {}, Train Loss: {:.4f}, Train error: {:.4f}'.format(trained_epoch, train_loss, train_error))

    writer.add_scalar('Loss/Train', train_loss, trained_epoch)

    model.eval()
    valid_loss = 0.0
    valid_error = 0.0

    with torch.no_grad():
        for (data, label) in tqdm(valid_data_loader, desc='Validation', leave=False):
            if args.cuda:
                data, label = data.to(device), label.to(device)
            data, label = Variable(data), Variable(label)

            loss, _ = model.calculate_objective(data, label)
            valid_loss += loss.data.item()
            error, _ = model.calculate_classification_error(data, label)
            valid_error += error

        valid_loss /= len(valid_data_loader)
        valid_error /= len(valid_data_loader)

        print('Epoch: {}, Valid Loss: {:.4f}, Validation error: {:.4f}'.format(trained_epoch, valid_loss, valid_error))

        writer.add_scalar('Loss/Validation', valid_loss, trained_epoch)
    
    if epoch % 10 == 0:
        save_model(model, trained_epoch, save_path)

def train(epoch):
    model.train()
    train_loss = 0.0
    train_error = 0.0
    # early_stop = False
    # best_loss = float('inf')
    # patience = 5
    # num_bad_epochs = 0
    
    writer = SummaryWriter(save_path + "/tensorboard")

    for (data, label) in tqdm(train_data_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
        bag_label = label
        if args.cuda:
            data, bag_label = data.to(device), label.to(device)

        optimizer.zero_grad()
        data.requires_grad_(True)
        bag_label.requires_grad_(True)

        y_pred = model.forward(data)

        try:
            loss, _ = model.calculate_objective(data, bag_label)
        except:
            loss = model.calculate_objective(data, bag_label)
            print("^^")

        loss.requires_grad_(True)
        train_loss += loss.item()
        
        try:
            error, _ = model.calculate_classification_error(data, bag_label)
        except:
            error = model.calculate_classification_error(data, bag_label)
        train_error += error

        loss.backward()
        optimizer.step()

    train_loss /= len(train_data_loader)
    train_error /= len(train_data_loader)

    print('Epoch: {}, Train Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))

    writer.add_scalar('Loss/Train', train_loss, epoch)
    # writer.add_scalar('Error/Train', train_error, epoch)

    model.eval()
    valid_loss = 0.0
    valid_error = 0.0

    with torch.no_grad():
        for (data, label) in tqdm(valid_data_loader, desc='Validation', leave=False):
            if args.cuda:
                data, label = data.to(device), label.to(device)
            data, label = Variable(data), Variable(label)

            loss, _ = model.calculate_objective(data, label)
            valid_loss += loss.data.item()
            error, _ = model.calculate_classification_error(data, label)
            valid_error += error

        valid_loss /= len(valid_data_loader)
        valid_error /= len(valid_data_loader)

        print('Epoch: {}, Valid Loss: {:.4f}, Validation error: {:.4f}'.format(epoch, valid_loss, valid_error))

        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        # writer.add_scalar('Error/Validation', valid_error, epoch)
    
    if epoch % 10 == 0:
        save_model(model, epoch, save_path)

            # if valid_loss < best_loss:
            #     best_loss = valid_loss
            #     num_bad_epochs = 0
            #     torch.save(model.state_dict(), './logs/best_model.pth')
            # else:
            #     num_bad_epochs += 1
            #     if num_bad_epochs >= patience:
            #         print(f'Early stopping at epoch {epoch}')
            #         early_stop = True
            #         break


def test(model_name):
    model_path = os.path.join(save_path, model_name+".pth")
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    model.eval()

    y_gt = []
    y_pred = []

    # for (data, label) in tqdm(test_data_loader, desc='Validation', leave=False):
    for data, label in test_data_loader:
        bag_label = label
        if args.cuda:
            data, bag_label = data.to(device), bag_label.to(device)
       
        bag_label = bag_label.item()
        pred = model.forward(data).item()

        if pred > 0.85:
            isImplant = 1
        
        else:
            isImplant = 0

        y_gt.append(bag_label)
        y_pred.append(isImplant)

        if(bag_label): 
            print("Implant! Model Predict %s"%pred)
        else : 
            print("No Implant! Model Predict %s"%pred)

    cm = confusion_matrix(y_gt, y_pred)
    recall = recall_score(y_gt, y_pred)
    precision = precision_score(y_gt, y_pred)
    accuracy = accuracy_score(y_gt, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("accuracy: {:.4f}".format(accuracy))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Implant", "Implant"], yticklabels=["No Implant", "Implant"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    print('Start Training')
    print(model)
    if args.inference:
        test(args.resume)

    else:
        print("train")
        if args.resume == None:
            for epoch in range(1, args.epochs + 1):
                train(epoch)
    
        else:
            for epoch in range(1, args.epochs + 1):
                train_resume(epoch, args.resume)

# python main.py --inference True --logdir ./logs/resnet50/PeX/seventh/ --dataset PeX --model resnet50 --resume model_epoch_20
# python main.py --inference True --logdir ./logs/resnet50/CT/fourth --dataset CT --model resnet50 --resume model_epoch_50