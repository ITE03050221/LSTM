import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    # class torch.nn.Module

    #   model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim ,hidden_dim=hidden_dim,
    #                   vocab_size=len(corpus.dictionary), label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu): 
        super(LSTMClassifier, self).__init__()   ### _init__()确保父类被正确的初始化了：
        self.hidden_dim = hidden_dim    # hidden_dim = 50
        self.batch_size = batch_size    # 一次訓練 5 個樣本
        self.use_gpu = use_gpu          # 不使用GPU，省略
                                        # embedding_dim = 100
        # 定义词嵌入
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  
                # vocab_size個單詞，单词的词嵌入维度为embedding_dim
                # LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
                # nn.Embedding(vocab_size, embed_dim)类，它是Module类的子类，
                # 这里它接受最重要的两个初始化参数：词汇量大小，每个词汇向量表示的向量维度。
                # Embedding类返回的是一个以索引表示的大表，
                # 表内每一个索引对应的元素都是表示该索引指向的单词的向量表示，大表具体是以矩阵的形式存储的。
                # Pytorch官网的解释是：
                # 一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。
                # 模块的输入是一个下标的列表，输出是对应的词嵌入。
                # 個人理解：
                # 这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小，
                # 宽是用来表示字典中每个元素的属性向量，向量的维度根据你想要表示的元素的复杂度而定。
                # 类实例化之后可以根据字典中元素的下标来查找元素对应的向量。
                # num_embeddings(int) - 嵌入字典的大小
                # embedding_dim(int) - 每个嵌入向量的大小

      
        # 变量：
        # weight(Tensor) - 形状为(num_embeddings, embedding_dim)的模块中可学习的权值
        # 形状：
        # 输入： LongTensor(N, W), N = mini - batch, W = 每个mini - batch中提取的下标数
        # 输出： (N, W, embedding_dim)

        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  
                # LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
                # 输入维度是embedding_dim, 输出维度是hidden_dim
                # 输入维度是embedding_dim，隐层hidden_dim维
        
        # 官方文档 https://pytorch.org/docs/stable/nn.html
        # input_size – 输入的特征维度
        # hidden_size – 隐状态的特征维度
        # num_layers – 层数（和时序展开要区分开）

        
        self.hidden2label = nn.Linear(hidden_dim, label_size)
                # torch.nn.Linear(in_features, out_features, bias=True)
                # in_features：每个输入样本的大小
                # out_features：每个输出样本的大小
                # 线性层将隐状态空间映射到标注空间
                # 对输入数据做线性变换：y=Ax+b
                
                # 形状：
                # 输入：(N,in_features)
                # 输出：(N,out_features)
 
        self.hidden = self.init_hidden()  
                # 返回保存着batch中每个元素的初始化隐状态的Tensor

    def init_hidden(self):
        if self.use_gpu:  # 忽略
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())  # 忽略
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())  # 忽略
        else:                            

            # 开始时刻, 没有隐状态
            # 关于维度设置的详情,请参考 Pytorch 文档
            # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
                    # 在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 
                    # 裡面的值為torch的Tensor
                    # torch.zeros：用0填充的矩陣
                    # 創建 5 * 50 的 tensor h0 and c0，用 0 填充      
                   
        return (h0, c0)
    
        # 在定义好forward后，该module调用forward，将按forward进行前向传播，并构建网络
    def forward(self, sentence):
        # forward()方法，用于定义该Module进行forward时的运算，forward()方法接受一个输入，
        # 然后通过其他modules或者其他Function运算，来进行forward，返回一个输出结果

        
        embeds = self.word_embeddings(sentence)
                # 预处理文本转成稠密向量

        # x = embeds
        x = embeds.view(len(sentence), self.batch_size, -1)
        # x 成為了 5 * 32 的 tensor      -1 意思??

        # a.view(3, 3) 
        #   1   2   3   
        #   4   5   6   
        #   7   8   9
        # [torch.FloatTensor of size 4x4]
        
        lstm_out, self.hidden = self.lstm(x, self.hidden)
                # 根据文本的稠密向量训练网络
                # x 為上面的 embeds.view(len(sentence), self.batch_size, -1)


        y  = self.hidden2label(lstm_out[-1])
        # lstm_out[-1]是lstm_out的最後一個值
        
        
        return y   # 最后返回的是一个4.8的tensor  torch.Size([5, 8])    torch.Size([4, 8])
    
