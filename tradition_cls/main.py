# -*- coding: utf-8 -*-

import os
import time
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn  as nn
import torch.optim as optim
from ccf.tradition_cls.data_pro import load_data_labels,Data
#from data_pro import load_data_labels, Data
from ccf.tradition_cls.models.config import Config
from ccf.tradition_cls.models.RNN import RNN

os.environ['CUDA_VISIBLE_DEVICES']='1'
#device_ids = [1,2]

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

opt = Config()


def train():
    opt = Config()
    x_text, y = load_data_labels("./data/dev_5.csv")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y,test_size=opt.test_size,random_state=0)

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print("{} train data: {}, test data: {}".format(now(), len(train_data), len(test_data)))
    model = RNN(opt)
    if opt.use_gpu:
        model = model.cuda()
        #model = nn.DataParallel(model)
    #model = LSTMClassifier( output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    print("{} init model finished".format(now()))
    for epoch in range(opt.epochs):
        total_loss = 0.0
        model.train()
        for step, batch_data in enumerate(train_loader):
            x, labels = batch_data
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                labels = labels.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            print('loss:',loss.item())
            total_loss = loss.item()
        acc = test(model,test_loader)
        print("{} {} epoch: loss: {}, acc: {}".format(now(), epoch, total_loss, acc))
def test(model, test_loader):
    correct = 0
    num = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, labels = data
            num += len(labels)
            output = model(x)
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                output = output.cpu()
            predict = torch.max(output.data, 1)[1]
            correct += (predict == labels).sum().item()
        return correct * 1.0 / num
if __name__ == "__main__":
    train()
