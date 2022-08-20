import numpy as np
from typing import List
from preprocess import CompDataset
from preprocess import get_user_data
from collections import Counter
import pickle
import torch
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import random


# 每个worker的定义，训练函数是user_round_train
class NNWorker(object):
    def __init__(self, user_idx):
        self.user_idx = user_idx
        self.data = get_user_data(self.user_idx)  # The worker can only access its own data
        self.ps_info = {}

    def preprocess_data(self):
        '''
        #x, y = self.data
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        '''
        x, y = self.data
        x[x == np.inf] = np.nan
        # x[np.isnan(x)] = 0.
        df = pd.DataFrame(x)
        # 线性补全缺失值
        df = df.fillna(df.interpolate(method='linear', limit_direction='forward', axis=0))
        x = df.to_numpy().astype(np.float32)
        x[np.isnan(x)] = 0.
        self.data = (x, y)

    # 计算本worker的均值和标准差，server调用此函数进行梯度聚合
    def send_mean_std(self):
        x, y = self.data
        feature_mean = np.zeros(79)
        feature_std = np.zeros(79)
        for i in range(0, 79):
            feature_value = x[:, i]
            feature_mean[i] = np.mean(feature_value)
            feature_std[i] = np.std(feature_value)
        return feature_mean, feature_std

    def get_mean_std(self, global_feature_mean, global_feature_std):

        x, y = self.data
        for row_idx in range(0, len(x)):
            for col_idx in range(0, len(x[0])):
                if global_feature_std[col_idx] != 0:
                    x[row_idx][col_idx] = (x[row_idx][col_idx] - global_feature_mean[col_idx]) / global_feature_std[
                        col_idx]
                else:
                    x[row_idx][col_idx] = (x[row_idx][col_idx] - global_feature_mean[col_idx])
        # print(x[:,0])
        self.data = (x, y)

    def round_data(self, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            n_round: int, round number
            n_round_samples: int, the number of samples this round
        """

        if n_round_samples == -1:
            return self.data

        n_samples = len(self.data[1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self.data[0][choices], self.data[1][choices]

    def receive_server_info(self, info):  # receive info from PS
        self.ps_info = info

    def process_mean_round_train_acc(self):  # process the "mean_round_train_acc" info from server
        mean_round_train_acc = self.ps_info["mean_round_train_acc"]
        # You can go on to do more processing if needed

    def user_round_train(self, model, device, n_round, batch_size, n_round_samples=-1, debug=False):

        X, Y = self.round_data(n_round, n_round_samples)
        data = CompDataset(X=X, Y=Y)
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
        )

        model.train()

        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model = model.to(device)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # import ipdb
            # ipdb.set_trace()
            # print(data.shape, target.shape)
            data = torch.unsqueeze(data, dim=1)
            data = data.float()
            output = model(data)
            target = target.long()
            loss = torch.nn.functional.cross_entropy(output, target)
            total_loss += loss
            loss.backward()
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            prediction.extend(pred.reshape(-1).tolist())
            real.extend(target.reshape(-1).tolist())

        grads = {'n_samples': data.shape[0], 'named_grads': {}}
        for name, param in model.named_parameters():
            # print('User {}'.format(self.user_idx))
            # print(type(param))
            # print(type(param.grad))
            if isinstance(param.grad, Tensor):
                # print(name)
                # print(param.grad.detach().cpu().numpy())
                grads['named_grads'][name] = param.grad.detach().cpu().numpy()
                # print(np.array(param.grad.detach().cpu().numpy()).shape)
                # print(grads['named_grads'][name])
        worker_info = {}
        worker_info["train_acc"] = correct / len(train_loader.dataset)

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(train_loader.dataset)))

        return grads, worker_info
