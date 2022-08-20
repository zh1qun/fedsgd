from datetime import datetime
import os
import shutil
import unittest
import torch
import numpy as np
from preprocess import get_test_data
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

from context import FederatedAveragingGrads, PytorchModel
from learning_model import FLModel
import random


class NNParameterServer(object):
    def __init__(self, init_model_path, testworkdir, resultdir):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.worker_info = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedAveragingGrads(
            model=PytorchModel(torch=torch,
                               model_class=FLModel,
                               init_model_path=self.init_model_path,
                               optim_name='Adam'),
            framework='pytorch',
        )
        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.test_data_x, self.test_data_y = get_test_data()

        self.round_train_acc = []
        self.global_feature_mean = []
        self.global_feature_std = []

        # self.error_array = np.zeros(shape=(256,512),dtype=float)

    def aggr_mean_std(self, feature_mean, feature_std, worker_num):
        global_feature_mean = np.zeros(79)
        global_feature_std = np.zeros(79)

        for i in range(0, 79):
            cur_mean = 0
            cur_std = 0
            for j in range(0, worker_num):
                cur_mean += feature_mean[j][i]
                cur_std += feature_std[j][i]
            global_feature_mean[i] = cur_mean / worker_num
            global_feature_std[i] = cur_std / worker_num

        self.global_feature_mean = global_feature_mean
        self.global_feature_std = global_feature_std

    def test_data_norm(self):
        for i in range(0, len(self.test_data_x)):
            for j in range(0, len(self.test_data_x[0])):
                if self.global_feature_std[j] != 0:
                    self.test_data_x[i][j] = (self.test_data_x[i][j] - self.global_feature_mean[j]) / \
                                             self.global_feature_std[j]
                else:
                    self.test_data_x[i][j] = self.test_data_x[i][j] - self.global_feature_mean[j]

    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads):  # receive grads info from worker
        self.current_round_grads.append(grads)

    def receive_worker_info(self, info):  # receive worker info from worker
        self.worker_info = info

    def process_round_train_acc(self):  # process the "round_train_acc" info from worker
        self.round_train_acc.append(self.worker_info["train_acc"])

    def print_round_train_acc(self):
        mean_round_train_acc = np.mean(self.round_train_acc) * 100
        print("\nMean_round_train_acc in train data is : ", "%.2f%%" % (mean_round_train_acc))
        self.round_train_acc = []
        return {"mean_round_train_acc": mean_round_train_acc}

    def aggregate(self):
        self.aggr(self.current_round_grads)

        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1
        self.current_round_grads = []

        return info

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray,)):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self, model, device, test_batch_size):
        loader = torch.utils.data.DataLoader(
            self.test_data_x,
            batch_size=test_batch_size,
            shuffle=False,
        )
        prediction = []
        with torch.no_grad():
            for data in loader:
                data = torch.unsqueeze(data, dim=1)
                pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                prediction.extend(pred.reshape(-1).tolist())
        assert len(prediction) == len(self.test_data_y)
        right_count = 0
        for i in range(0, len(prediction)):
            if (prediction[i] == self.test_data_y[i]):
                right_count += 1
        test_data_accuracy = right_count / len(prediction)
        print("The accuracy in test data is {}.".format(test_data_accuracy))
        self.save_prediction(prediction)
