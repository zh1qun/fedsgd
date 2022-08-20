import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from datetime import datetime
from preprocess import CompDataset

TRAINDATA_DIR = './train - 副本'
ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


class Net(nn.Module):
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x
    '''

    def __init__(self):
        super().__init__()

        self.conv_layer1 = nn.Conv1d(1, 10, 3)
        self.pool_layer1 = nn.MaxPool1d(5)
        self.fc1 = nn.Linear(150, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 14)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.pool_layer1(x)
        x = x.view(-1, 150)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output


def extract_features(data, has_label=True):
    data['SimillarHTTP'] = 0.
    if has_label:
        return data.iloc[:, -80:-1]

    return data.iloc[:, -79:]


def get_all_data():
    fpath = ""
    data_x = []
    data_y = []

    for root, dirs, fnames in os.walk(TRAINDATA_DIR):
        # fname = fnames[user_idx]
        # fpath = os.path.join(root, fname)
        for fname in fnames:
            fpath = os.path.join(root, fname)
            print('Load {} Data: '.format(fname))
            data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
            x = extract_features(data)
            y = np.array([
                ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
                for t in data.iloc[:, -1]
            ])
            x = x.to_numpy().astype(np.float32)
            data_x.extend(x)
            data_y.extend(y)
        break
    if not fpath.endswith('csv'):
        return

    return data_x, data_y


def get_test_data():
    # with open(TESTDATA_PATH, 'rb') as fin:
    # data = pickle.load(fin)

    # return data['X']
    data = pd.read_csv("D:/2020Federated/test/type-total-8-150000-samples.csv", skipinitialspace=True, low_memory=False)
    x = extract_features(data)
    y = np.array([
        ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
        for t in data.iloc[:, -1]
    ])
    x = x.to_numpy().astype(np.float32)
    return x, y


def save_testdata_prediction(model, f_mean, f_std):
    x, y = get_test_data()
    for i in range(0, 79):
        for j in range(0, len(x)):
            if (f_std[i] != 0):
                x[j][i] = (x[j][i] - f_mean[i]) / f_std[i]
            else:
                x[j][i] = (x[j][i] - f_mean[i])

    loader = torch.utils.data.DataLoader(
        x,
        batch_size=512,
        shuffle=False,
    )
    prediction = []
    with torch.no_grad():
        for data in loader:
            data = torch.unsqueeze(data, dim=1)
            pred = model(data.to("cuda")).argmax(dim=1, keepdim=True)
            prediction.extend(pred.reshape(-1).tolist())
    assert len(prediction) == len(y)
    right_count = 0
    for i in range(0, len(prediction)):
        if (prediction[i] == y[i]):
            right_count += 1
    test_data_accuracy = right_count / len(prediction)
    print("The accuracy in test data is {}.".format(test_data_accuracy))


x, y = get_all_data()
print(len(x[0]))
x_arr = np.array(x, dtype=np.float)
y_arr = np.array(y, dtype=np.float)
x_arr[x_arr == np.inf] = 1.
x_arr[np.isnan(x_arr)] = 0.

f_mean = np.zeros(shape=(79,), dtype=np.float)
f_std = np.zeros(shape=(79,), dtype=np.float)
print(f_mean)
print("Start to norm.")

for i in range(0, 79):
    feature_mean = np.mean(x_arr[:, i])
    feature_std = np.std(x_arr[:, i])
    f_mean[i] = feature_mean
    f_std[i] = feature_std
    print("Feature {} :".format(i))
    for j in range(0, len(x_arr)):
        if (feature_std != 0):
            x_arr[j][i] = (x_arr[j][i] - feature_mean) / feature_std
        else:
            x_arr[j][i] = (x_arr[j][i] - feature_mean)

input_x = torch.tensor(data=x_arr, dtype=torch.float32)
label = torch.tensor(data=y_arr, dtype=torch.float32)

data = CompDataset(X=input_x, Y=label)
train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=512,
    shuffle=True,
)

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# SGD:随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss
print("Start to train.")
net = net.to("cuda")
total_time = datetime.now()
for i in range(400):
    total_loss = 0
    optimizer.zero_grad()
    # label = label.long()
    start = datetime.now()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to("cuda")
        target = target.to("cuda")
        data = torch.unsqueeze(data, dim=1)
        data = data.float()
        out = net(data)
        target = target.long()
        loss = torch.nn.functional.cross_entropy(out, target)
        total_loss += loss
        loss.backward()
    optimizer.step()
    print("Round {} trainied ,time cost : {}.".format(i, datetime.now() - start))
    if (i % 20 == 0 or i >= 399):
        save_testdata_prediction(net.to("cuda"), f_mean, f_std)
print("Total time cost{}".format(datetime.now() - total_time))
# input_x = input_x.float()
# label = label.long()
# out = net(input_x)
# loss = torch.nn.functional.cross_entropy(out, label)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
'''
out = net(input_x)
pred = out.argmax(
    dim=1, keepdim=True)
print(pred.shape)
print(label.shape)
count =0
for i in range (0,len(label)):
    if(pred[i][0] == label):
        count+=1
print(count/len(label))
# out是一个计算矩阵
#prediction = torch.max(out, 1)[1]
#pred_y = prediction.numpy()
# 预测y输出数列
#target_y = label.data.numpy()
'''
