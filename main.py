import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt'
TEST_DIR = 'test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'

## parameter setting
epochs = 50  ## 計算 50 次
batch_size = 5  ## 每次訓練 5 個樣本
use_gpu = torch.cuda.is_available()
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:  ##管理參數組
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    ### parameter setting
    embedding_dim = 100
    hidden_dim = 50
    sentence_len = 32
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)  ##train_file: 'data\\train_txt.txt'
    test_file = os.path.join(DATA_DIR, TEST_FILE)  ##test_file: 'data\\test_txt.txt'
    fp_train = open(train_file, 'r')  
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]  ##讀取data\\train_txt裡面的每一個 txt 文檔, list:       ['train_txt\\1.txt', 'train_txt\\2.txt', 'train_txt\\3.txt', ...]
    filenames = copy.deepcopy(train_filenames)
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]  ##讀取data\\test_txt裡面的每一個 txt  文檔, list:['test_txt\\1.txt', 'test_txt\\2.txt', 'test_txt\\3.txt', ...]
    fp_test.close()
    filenames.extend(test_filenames)  ##將讀取出的 list:['train_txt\\1.txt', 'train_txt\\2.txt', 'train_txt\\3.txt', ...]和 讀取出的 list:['test_txt\\1.txt', 'test_txt\\2.txt', 'test_txt\\3.txt', ...]接在一起

    corpus = DP.Corpus(DATA_DIR, filenames)
    nlabel = 8

    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                           vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    if use_gpu:  ##忽略
        model = model.cuda()  ##忽略
    ### data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)  ## 轉成 tensor 型式
    
    ### 將 dtrain_set 放入 dataloader
    train_loader = DataLoader(dtrain_set, ## dtrain_set 需為 tensor 型式
                          batch_size=batch_size,
                          shuffle=True,  ## 要不要打乱数据 (打乱比较好)
                          num_workers=4  ## 多線程讀取數據
                         )
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)  ## 轉成 tensor ??

    test_loader = DataLoader(dtest_set, ## dtest_set
                          batch_size=batch_size,
                          shuffle=False,  ## 要不要打乱数据 (打乱比较好)
                          num_workers=4  ## 多線程讀取數據
                         )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  ## 優化器 ??
    loss_function = nn.CrossEntropyLoss()  ## 针对单目标分类问题, 结合了 nn.LogSoftmax() 和 nn.NLLLoss() 来计算 loss.
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    for epoch in range(epochs):  ##開始訓練，第一次  第二次....
        optimizer = adjust_learning_rate(optimizer, epoch) ##訓練一次，調整一次learning rate

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader): ## 將數據從 train_loader 裡面讀出來，一次讀取的樣本數是 5 個
            train_inputs, train_labels = traindata  ##為什麼會有 tensor 的 list 出現
            train_labels = torch.squeeze(train_labels)  ## 壓縮矩陣  ??  降維 ??

            if use_gpu:  ##忽略
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()  ## 忽略
            else: train_inputs = Variable(train_inputs)

            model.zero_grad()   ## 梯度歸 0 ??
            model.batch_size = len(train_labels) 
            model.hidden = model.init_hidden()
            output = model(train_inputs.t())

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.data[0]  ## 將 total_loss 加到 loss.data矩陣裡面
            #total_loss += loss.item()


        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, testdata in enumerate(test_loader):
            test_inputs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else: test_inputs = Variable(test_inputs)

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t())

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.item()
            #total_loss += loss.data[0]
        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = sentence_len

    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param

    if use_plot:
        import PlotFigure as PF
        PF.PlotFigure(result, use_save)
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)
