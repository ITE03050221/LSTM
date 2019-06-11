import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}  ## dict: ['champion','products','ch',......]
        self.idx2word = []  ## list: {'champion':0, 'products':1, 'ch':2,.....}

    def add_word(self, word):  ##加入 word
        if word not in self.word2idx:  ## 如果 word 不在 dict 裡面 
            self.idx2word.append(word)  ## 將 此word 加到 list 裡面
            self.word2idx[word] = len(self.idx2word) - 1  ## Add index for next word
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)   ## 有多少不同的 words 在 list 裡面

class Corpus(object):
    # corpus = DP.Corpus(DATA_DIR, filenames)         DATA_DIR = 'data'
    # Now "filenames" have both test and train file name
    # 0001 = {str} 'train_txt\\2.txt'          plus  0971 = {str} 'test_txt\\1.txt'
    # 0000 = {str} 'train_txt\\1.txt'                0972 = {str} 'test_txt\\2.txt'
    # 0002 = {str}'train_txt\\3.txt'                 0973 = {str} 'test_txt\\3.txt'
    
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
                # 字典是另一种可变容器模型，且可存储任意类型对象。
                # 字典的每个键值 key=>value 对用冒号 : 分割，每个键值对之间用逗号 , 分割
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:                    
                    ## each train and test file. E.g  train_txt\\2.txt and test_txt\\1.txt
            path = os.path.join(DATA_DIR, filename)   
                    ##合併目錄: 'data\\train_txt\\1.txt', 'data\\train_txt\\2.txt' # Path: data+ train_txt\\1.txt
            with open(path, 'r') as f:                
                    ## e.g <_io.TextIOWrapper name='data\\train_txt\\1571.txt' mode='r' encoding='cp936'>
                tokens = 0                            
                        ## 初始化一个token，对每一个文件
                for line in f:  
                            ## 一行一行讀取        
                            
                    words = line.split() + ['<eos>']  ## 切割單詞     
                                                      
                    tokens += len(words)              
                            ## 有幾個單詞，tokens 就是多少    # Sum up all different words length, save as token
                                                      # Token是一个文档里面所有的词的长度
                    for word in words:                
                        self.dictionary.add_word(word) # 把字加到字典裏面

            # Tokenize file content  ## ??
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)                      
                        ## # Create new tensor with the dimension “tokens”
                token = 0
                        ## 初始化一个token，对每一个文件
                for line in f:
                             ## 一行一行讀取        
                            # For each line of read document
                    words = line.split() + ['<eos>']    ## 切割單詞    
                                                      
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        # 比如说一个文档里面有100个词，但是不重复的只有5个，
                        # ids = torch.LongTensor(tokens)会创建一个100的Long_tensor
                        # 然后对于word2idx，对于每个不重复的WORD都会有一个index，然后ids[token]会创建一个长度为100
                        # 但是每个元素会是这个词在word2idx里对应的index
                        # E.g  “I book a book” ----   tensor[1,2,3,2]--- size 4 tensor
                        
                        token += 1

        return ids      ## Tokenize Tensor

class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):  
        # dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)
        ## data_path='data', txt_path='train_txt', txt_filename = 'train_txt.txt', label_filename = 'train_label.txt'
        
        self.txt_path = os.path.join(data_path, txt_path)   ## 合併目錄: data/train_txt
                                                            ## data_path = "data"  txt_path = "train_txt"
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)  
                ##合併目錄： data/train_txt.txt  # Txt_filename = "train_txt.txt"
        fp = open(txt_filepath, 'r')  ##讀取 data/train_txt.txt       
                                    ## Open train_txt files, read the list of all train files
        self.txt_filename = [x.strip() for x in fp]  ## 去空格        
                                                     ## txt_filename will be likes 1.txt|2.txt|3.txt
        fp.close()
        
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)  ##合併目錄： data/train_label.txt
        fp_label = open(label_filepath, 'r')         ## 讀取  data/train_label.txt   # Open train_label.txt file
        labels = [int(x.strip()) for x in fp_label]  ## 去空格            
                                                     # Read all label
        fp_label.close()
        self.label = labels  ## 標籤
        self.corpus = corpus  ## 字集
        self.sen_len = sen_len # sen_len = 32


    def __getitem__(self, index):
        
        # txt_path = data + train_txt
        # txt_filename will be likes 1.txt|2.txt|3.txt
        filename = os.path.join(self.txt_path, self.txt_filename[index])  ## data/train_txt/train_txt[index]
        fp = open(filename, 'r')    ## fp: 打開 data/train_txt/train_txt[index] 
                                    # 打开第 INDEX 个， train文件   1.txt
        
        # sen_len = 32
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))   # 创建一个32 size全为0的tensor
                                                                        # dtype:数据类型
                                                                       
        count = 0
        clip = False  ## ??
        for words in fp:                                      # 对于打开的第 index 文件的所有词
            for word in words.split():                        ## 分割 data/train_txt/train_txt[index] 裡面的 words
                                                              # 按照空格拆分word
                if word.strip() in self.corpus.dictionary.word2idx:    
                    ## 如果上述 word 有在 dict 裡面
                     # strip()  默认删除空白符
                     # 如果当前的词已经在dictionary中
                     # ids[token] = self.dictionary.word2idx[word] [1,2,3,1]
                    if count > self.sen_len - 1: ## 对txt tensor 32个词都赋值完之后，跳出
                        clip = True  ##  ??
                        break
                         # ids[token] = self.dictionary.word2idx[word]
                    txt[count] = self.corpus.dictionary.word2idx[word.strip()]    
                            # 32 個空間的txt tensor， 每個詞有對應的index
                            # 比如说 a b c d 对应 1，2，3，4，然后刚好文本长度为32，为 a,b,c,d,a,b
                            # 对应的tensor为[1, 2, 3, 4, 1, 2, ...] 
                            
                    count += 1
            if clip: break
           
        label = torch.LongTensor([self.label[index]]) 
        # 創建 longtensor，裡面包含了第幾個index對應的label

        return txt, label              
                                      
    def __len__(self):
        return len(self.txt_filename)
