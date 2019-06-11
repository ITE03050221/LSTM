import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

use_plot = True  ## create a tag that plot the figure or not
use_save = True  ## create a tag that save the data or not

if use_save:  ## if True, import the following packet
    import pickle    ## pickle 會記錄已經序列化的物件，如果後續有物件參考到相同物件，才不會再度被序列化
    from datetime import datetime    ## datetime是Python处理日期和时间的标准库

DATA_DIR = 'data'    ## DATA_DIR = data 資料夾
TRAIN_DIR = 'train_txt'    ##TRAIN_DIR = data/train_txt 資料夾
TEST_DIR = 'test_txt'    ## TEST_DIR = data/text_txt 資料夾
TRAIN_FILE = 'train_txt.txt'   ## TRAIN_FILE = data/train_txt.txt 文檔
TEST_FILE = 'test_txt.txt'    ## TEST_FILE = data/test_txt.txt 文檔
TRAIN_LABEL = 'train_label.txt'    ## TRAIN_LABEL = data/train_label.txt 文檔
TEST_LABEL = 'test_label.txt'    ## TEST_LABEL = data/test_label.txt 文檔

## parameter setting
epochs = 50  ## 計算 50 次
batch_size = 5  ## 每次訓練 5 個樣本
use_gpu = torch.cuda.is_available()  ## not using GPU, ignore
learning_rate = 0.01  ## 先設定 learning 為 0.01

def adjust_learning_rate(optimizer, epoch):    ## 調整learning rate ## epoch = 訓練幾次
    lr = learning_rate * (0.1 ** (epoch // 10))   
        ## Sets the learning rate to the initial 'lr' decayed by 10 every 10 epochs
        ## optimizer 通过 param_group 来管理参数组，param_group中保存了参数组及其对应的学习率,动量等等 ##
    for param_group in optimizer.param_groups:    
        param_group['lr'] = lr  ## 通過更改param_group[‘lr’]的值來更改對應參數組的學習率
    return optimizer  

if __name__=='__main__':
    ### parameter setting
    embedding_dim = 100
    hidden_dim = 50
    sentence_len = 32
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)  ## train_file: 'data\\train_txt.txt'
    test_file = os.path.join(DATA_DIR, TEST_FILE)  ## test_file: 'data\\test_txt.txt'
    fp_train = open(train_file, 'r')  ## just read the document 'data\\train_txt.txt'
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]  
            ## 讀取data\\train_txt裡面的每一個 txt 文檔
            ## list:['train_txt\\1.txt', 'train_txt\\2.txt', 'train_txt\\3.txt', ...]
        
    filenames = copy.deepcopy(train_filenames)  
            ## Hard copy 'filenames', a independent copy version of 'train_filenames'
            ##  改变原有被复制对象不会对已经复制出来的新对象产生影响
    fp_train.close()
    
    
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]  
            ##讀取data\\test_txt裡面的每一個 txt  文檔
            ## list:['test_txt\\1.txt', 'test_txt\\2.txt', 'test_txt\\3.txt', ...]
    fp_test.close()
    filenames.extend(test_filenames)  
    ##將讀取出的 list:['train_txt\\1.txt', 'train_txt\\2.txt', 'train_txt\\3.txt', ...]和 
    ##  讀取出的 list:['test_txt\\1.txt', 'test_txt\\2.txt', 'test_txt\\3.txt', ...]接在一起
    ## "train_txt\\1.txt" plus "test_txt\\1.txt"

    corpus = DP.Corpus(DATA_DIR, filenames)  ## ??
    ## return ids  
    nlabel = 8  ## Labels 0 ~ 7

    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                       vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    
    
    if use_gpu:  ## not useful, ignore
        model = model.cuda()  ## not useful, ignore
    ### data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)  
            ## 轉成 tensor 型式
    
    ### 將 dtrain_set 放入 dataloader
    train_loader = DataLoader(dtrain_set, ## Input dataset "dtrain_set"
                          batch_size=batch_size,  ## 每次導入 5 個樣本
                          shuffle=True,  ## 要不要打乱数据 (打乱比较好)
                          num_workers=4  ## 多線程讀取數據
                         )
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)  
            ## 轉成 tensor ??

    test_loader = DataLoader(dtest_set, ## Input the dataset "dtest_set"
                          batch_size=batch_size,  ## ## 每次導入 5 個樣本
                          shuffle=False,  ## 要不要打乱数据 (打乱比较好)
                          num_workers=4  ## 多線程讀取數據
                         )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  ##  Optimaztion: SGD 算法
    ## class torch.optim.SGD(params, lr=, momentum = 0,dampening = 0,weight_decay = 0,nesterov = False)
    ## params(iterable) – 待优化参数的iterable或者是定义了参数组的dict
    ## lr(float) – 学习率
    
    
    loss_function = nn.CrossEntropyLoss()  
            ## 针对单目标分类问题, 结合了 nn.LogSoftmax() 和 nn.NLLLoss() 来计算 loss.
    
    
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    
    ### training procedure
    for epoch in range(epochs):  ##開始訓練，第一次  第二次.... 總共訓練 50 次
        optimizer = adjust_learning_rate(optimizer, epoch) ##訓練一次，調整一次learning rate

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader): 
                ## 創建 iter index for 'train_loader'
                ## 將數據從 train_loader 裡面讀出來，一次讀取的樣本數是 5 個，每次都打亂
                ## enumerate 將數據進行編號，並且將編號和數據一起列出    
                ## iter  traindata
                ## 0     这
                ## 1     是
                ## 2     一个
                ## 3     测试
                                                                                                            
            train_inputs, train_labels = traindata 
            # train_input = 5 * 32 的 tensor
            # train_labels = tensor([2, 0, 0, 7, 0])
                
            train_labels = torch.squeeze(train_labels)  
            # 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，
            # 比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。
            # squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。
            # a.squeeze(N) 就是去掉a中指定的维数为一的维度。


            if use_gpu:  ## not using GPU, ignore
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()   
                # not using GPU, ignore
            else: train_inputs = Variable(train_inputs) # Variable(train_inputs) = 5 * 32 的 tensor

            model.zero_grad()   # Pytorch 会累加梯度
                                # 每次训练前需要清空梯度值
            
            model.batch_size = len(train_labels)  ## batch_size = 5
            
            model.hidden = model.init_hidden()  # 此外还需要清空 LSTM 的隐状态
                                                # 将其从上个实例的历史中分离出来
                                                # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错
            
            output = model(train_inputs.t())    # 将train_inputs进行转置
                                                # output 為 5 * 8 的 tensor

            # 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度    
            loss = loss_function(output, Variable(train_labels)) 
            # train_labels = tensor([2, 0, 0, 7, 0])
            # Calculate error cross_entropy(predicted value, class )

            loss.backward()  
            # 执行.backward()来自动计算出所有需要的梯度

            # 前向传播，然后计算loss，
            # 反向传播，接着优化参数，特别注意的是在每次反向传播的时候需要将参数的梯度归零
            
            optimizer.step()
            # apply所有的梯度以更新parameter的值.因为step（）更新所有参数，所以不用指明梯度

            # calc training acc
            _, predicted = torch.max(output.data, 1)   
            # 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
            # output.data = 5 * 8 的 tensor
            # predicted = tensor ([1, 4, 1, 3, 2])
                                            
           
            
            total_acc += (predicted == train_labels).sum()   ## 有多少個訓練正確
            total += len(train_labels)  ## len(train_labels) = 5, 每次训练一次加5
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
