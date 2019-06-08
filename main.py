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
    lr = learning_rate * (0.1 ** (epoch // 10))   ## Sets the learning rate to the initial 'lr' decayed by 10 every 10 epochs
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
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]  ## 讀取data\\train_txt裡面的每一個 txt 文檔
                                                         ## list:['train_txt\\1.txt', 'train_txt\\2.txt', 'train_txt\\3.txt', ...]
        
    filenames = copy.deepcopy(train_filenames)  ## Hard copy 'filenames', a independent copy version of 'train_filenames'
      ##  改变原有被复制对象不会对已经复制出来的新对象产生影响
    fp_train.close()
    
    
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]  ##讀取data\\test_txt裡面的每一個 txt  文檔 
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
    ## len(corpus.dictionary) 23590
    ## len(corpus.dictionary) --- return len(self.idx2word)
    ## LSTMClassifier(
    ##   (word_embeddings): Embedding(23590, 100)
    ##   (lstm): LSTM(100, 50)
    ##  (hidden2label): Linear(in_features=50, out_features=8, bias=True)
    
    
    if use_gpu:  ## not useful, ignore
        model = model.cuda()  ## not useful, ignore
    ### data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)  ## 轉成 tensor 型式
    
    ### 將 dtrain_set 放入 dataloader
    train_loader = DataLoader(dtrain_set, ## Input dataset "dtrain_set"
                          batch_size=batch_size,  ## 每次導入 5 個樣本
                          shuffle=True,  ## 要不要打乱数据 (打乱比较好)
                          num_workers=4  ## 多線程讀取數據
                         )
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)  ## 轉成 tensor ??

    test_loader = DataLoader(dtest_set, ## Input the dataset "dtest_set"
                          batch_size=batch_size,  ## ## 每次導入 5 個樣本
                          shuffle=False,  ## 要不要打乱数据 (打乱比较好)
                          num_workers=4  ## 多線程讀取數據
                         )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  ##  Optimaztion: SGD 算法
    ## class torch.optim.SGD(params, lr=, momentum = 0,dampening = 0,weight_decay = 0,nesterov = False)
    ## params(iterable) – 待优化参数的iterable或者是定义了参数组的dict
    ## lr(float) – 学习率
    ## momentum(float, 可选) – 动量因子（默认：0）
    ## weight_decay(float, 可选) – 权重衰减（L2惩罚）（默认：0）
    ## dampening(float, 可选) – 动量的抑制因子（默认：0）
    ## nesterov(bool, 可选) – 使用Nesterov动量（默认：False）
    
    
    loss_function = nn.CrossEntropyLoss()  ## 针对单目标分类问题, 结合了 nn.LogSoftmax() 和 nn.NLLLoss() 来计算 loss.
    
    
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
        for iter, traindata in enumerate(train_loader): ## 創建 iter index for 'train_loader'
                                                        ## 將數據從 train_loader 裡面讀出來，一次讀取的樣本數是 5 個，每次都打亂
                                                        ## enumerate 將數據進行編號，並且將編號和數據一起列出    ##iter  traindata
                                                                                                             ## 0     这
                                                                                                             ## 1     是
                                                                                                             ## 2     一个
                                                                                                             ## 3     测试
                                                                                                            
            train_inputs, train_labels = traindata  ## train_inputs torch.Size([5, 32] 5来自batch_size, 32 来自sen_len
                                                    ## train_labels torch.Size([5, 1])  5来自batch_size,真实label
                
            train_labels = torch.squeeze(train_labels)  ## torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
                                                        ## squeeze(a)就是将a中所有为1的维度删掉
                
                 #print('Epoch: ', epoch, '| Step: ', iter, '| train_inputs: ',train_inputs.numpy(), '| train_labels: ', train_labels.size(), '| train_labels:.size ', train_inputs.size())
            #   Epoch:  0 | Step:  1084        train_inputs:.size  torch.Size([5, 32]          train_labels: [5 0 0 0 1]    torch.Size([5])


            if use_gpu:  ## not using GPU, ignore
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()  ## not using GPU, ignore
            else: train_inputs = Variable(train_inputs)

            model.zero_grad()   #清空梯度缓存
           # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
           # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
           # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
           # 现在还不是很理解
            
            model.batch_size = len(train_labels)  ## batch_size = 5
            model.hidden = model.init_hidden()  ## model.hidden: tuple type return (h0, c0)  size 5 * 50 tensors,
            # print(train_inputs.shape)  torch.Size([5, 32])
            # print(train_inputs.t().shape)    torch.Size([32, 5])
            
            output = model(train_inputs.t())  ## Transpose train_inputs tensor and use it as input
                                              ## Output torch.Size([5, 8])

             ## print(Variable(train_labels).size())
            loss = loss_function(output, Variable(train_labels)) ## Calculate error cross_entropy(predicted value, class )
                                                                  # train_labels torch.Size([5, 1])
                                                                  # Variable(train_labels)  torch.Size([5]
                                                                  # 公式在官方文档里，这里不注释了
            loss.backward()  ## torch.autograd.backward(variables, grad_variables=None, retain_graph=None, create_graph=False)
                                 # 这里是默认情况，相当于out.backward(torch.Tensor([1.0]))
             
            # 给定图的叶子节点variables, 计算图中变量的梯度和。 计算图可以通过链式法则求导。如果variables中的任何一个variable是 非标量(non-scalar)的，且requires_grad=True。
# 那么此函数需要指定grad_variables，它的长度应该和variables的长度匹配，里面保存了相关variable的梯度(对于不需要gradient tensor的variable，None是可取的)。
# 此函数累积leaf variables计算的梯度。你可能需要在调用此函数之前将leaf variable的梯度置零。
# 参数：
#
# variables（变量的序列） - 被求微分的叶子节点，即 ys 。
# grad_variables（（张量，变量）的序列或无） - 对应variable的梯度。仅当variable不是标量且需要求梯度的时候使用。
# retain_graph（bool，可选） - 如果为False，则用于释放计算grad的图。请注意，在几乎所有情况下，没有必要将此选项设置为True，通常可以以更有效的方式解决。默认值为create_graph的值。
# create_graph（bool，可选） - 如果为True，则将构造派生图，允许计算更高阶的派生产品。默认为False

            # 更新的三步：
            #1.
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            # 这一句代码中optimizer获取到了所有parameters的引用，每个parameter都包含梯度（gradient），optimizer可以把梯度应用上去更新parameter。
            #2.
            # loss = loss_function(output,Variable(train_labels))
            # prediction和true class之间进行比对（熵或者其他lossfunction），产生最初的梯度
            # loss.backward()
            # 反向传播到整个网络的所有链路和节点。节点与节点之间有联系，因此可以反向链式传播梯度
            # 3.
            optimizer.step()
            # apply所有的梯度以更新parameter的值.因为step（）更新所有参数，所以不用指明梯度

            # calc training acc
            _, predicted = torch.max(output.data, 1)   ##  返回每一行中最大值的那个元素，且返回其索引
                                                     #  Predicted 前面那个逗号是为了返回索引，而不是具体的值，但是具体怎么看代码不知道 ？？？？？？？
                                                     #  输入：The size of output.data torch.Size([5, 8]),
                                                     #  输出：predicted  torch.Size([5])
            # train_loss_ = []  # 初始化训练和测试集正确率和损失
            # test_loss_ = []
            # train_acc_ = []
            # test_acc_ = []
            # print(train_labels.size())
            
            total_acc += (predicted == train_labels).sum()   ## 多少个训练对了，是个size 0的tensor，要用。item（）来看
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
