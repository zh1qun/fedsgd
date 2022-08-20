from datetime import datetime
import os
import shutil
import unittest
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from worker import NNWorker
from server import NNParameterServer

import preprocess
from learning_model import FLModel


class NNTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        self.seed = 0
        self.use_cuda = False
        self.batch_size = 1024
        self.test_batch_size = 1024
        self.lr = 0.001
        self.n_max_rounds = 200
        self.log_interval = 20
        self.n_round_samples = 2048
        self.testbase = self.TEST_BASE_DIR
        self.n_users = 4
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        torch.manual_seed(self.seed)

        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.ps = NNParameterServer(init_model_path=self.init_model_path,
                                    testworkdir=self.testworkdir, resultdir=self.RESULT_DIR)

        self.workers = []
        for u in range(0, self.n_users):
            self.workers.append(NNWorker(user_idx=u))

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def nn_test(self):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu")

        # let workers preprocess data
        # for u in range(0, self.n_users):
        # self.workers[u].preprocess_data()
        global_feature_mean = []
        global_feature_std = []
        for u in range(0, self.n_users):
            self.workers[u].preprocess_data()
            cur_usr_feature_mean, cur_usr_feature_std = self.workers[u].send_mean_std()
            global_feature_mean.append(cur_usr_feature_mean)
            global_feature_std.append(cur_usr_feature_std)
        global_feature_mean = np.array(global_feature_mean)
        global_feature_std = np.array(global_feature_std)

        self.ps.aggr_mean_std(global_feature_mean, global_feature_std, self.n_users)
        assert len(self.ps.global_feature_mean) == 79
        assert len(self.ps.global_feature_std) == 79
        # print(self.ps.global_feature_mean)
        # print(self.ps.global_feature_std)
        for u in range(0, self.n_users):
            self.workers[u].get_mean_std(self.ps.global_feature_mean, self.ps.global_feature_std)
        self.ps.test_data_norm()

        training_start = datetime.now()
        model = None
        for r in range(1, self.n_max_rounds + 1):
            path = self.ps.get_latest_model()
            start = datetime.now()
            for u in range(0, self.n_users):
                model = FLModel()
                model.load_state_dict(torch.load(path))
                # print(type(model.named_parameters()))
                model = model.to(device)
                grads, worker_info = self.workers[u].user_round_train(model=model, device=device, n_round=r,
                                                                      batch_size=self.batch_size,
                                                                      n_round_samples=self.n_round_samples)

                self.ps.receive_grads_info(grads=grads)
                self.ps.receive_worker_info(
                    worker_info)  # The transfer of information from the worker to the server requires a call to the "ps.receive_worker_info"
                self.ps.process_round_train_acc()

            self.ps.aggregate()
            print('\nRound {} cost: {}, total training cost: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
            ))

            if model is not None and r % self.log_interval == 0:
                server_info = self.ps.print_round_train_acc()  # print average train acc and return
                for u in range(0, self.n_users):  # transport average train acc to each worker
                    self.workers[u].receive_server_info(
                        server_info)  # The transfer of information from the server to the worker requires a call to the "worker.receive_server_info"
                    self.workers[u].process_mean_round_train_acc()  # workers do processing

                self.ps.save_testdata_prediction(model=model, device=device, test_batch_size=self.test_batch_size)

        if model is not None:
            self.ps.save_testdata_prediction(model=model, device=device, test_batch_size=self.test_batch_size)


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
    suite.addTest(NNTestSuit('nn_test'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    print('cpu count: ', os.cpu_count())
    main()
