import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from torchvision import transforms



class BinaryResNet50Model(nn.Module):
    def __init__(self):
        super(BinaryResNet50Model, self).__init__()
        # ResNet-50 모델 불러오기 (pre-trained 가중치 사용)
        self.resnet50 = models.resnet50(pretrained=True)

        in_features = self.resnet50.fc.in_features
        self.dropout = nn.Dropout(p=0.5)
        self.resnet50.fc = nn.Sequential(
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 2)  # 이진 분류
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x.requires_grad_(True)
        x = x.squeeze(0)
        x = self.resnet50(x)
        # print(x.shape)
        x = self.sigmoid(x)
        x = x.view(2, -1)
        x = x[0]
        x = torch.max(x)
        x = x.view(1, 1)
        return x

    def calculate_classification_error(self, X, Y,):
        Y = Y.float()
        Y_hat = self.forward(X).to(device=Y.device)
        
        error = 1. - Y_hat.eq(Y).float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob = self.forward(X)

        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5).to(device=Y.device)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        neg_log_likelihood.requires_grad_(True)

        return neg_log_likelihood, Y

class ResNext(nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        # ResNet-50 모델 불러오기 (pre-trained 가중치 사용)
        self.resnext = models.resnext50_32x4d()
        # self.resnet50 = models.resnet50(pretrained=False)
        # self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # 마지막 레이어의 출력을 2로 조정 (이진 분류의 경우)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, 2)  # 2는 이진 분류의 클래스 수
        # self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        # self.bag_classification = nn.Sequential(

        # )

    def forward(self, x):
        x.requires_grad_(True)
        x = x.squeeze(0) 
        x = self.resnext(x)
        # print(x.shape)
        x = self.sigmoid(x)
        x = x.view(2, -1)
        x = x[0]
        x = torch.max(x)
        x = x.view(1, 1)
        return x

    def calculate_classification_error(self, X, Y,):
        Y = Y.float()
        # print(len(self.foward(X)))
        Y_hat = self.forward(X).to(device=Y.device)
        
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        error = 1. - Y_hat.eq(Y).float().mean().item()
        # error = torch.Tensor(error, requires_grad=True)

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob = self.forward(X)

        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5).to(device=Y.device)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        neg_log_likelihood.requires_grad_(True)
        # neg_log_likelihood_(True)
        # print(neg_log_likelihood.requires_grad)


        return neg_log_likelihood, Y

