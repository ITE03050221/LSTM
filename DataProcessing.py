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
            self.word2idx[word] = len(self.idx2word) - 1  ## 
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename) ##合併目錄: 'data/_txt/1.txt', 'data/train_txt/2.txt'
            with open(path, 'r') as f:
                tokens = 0
                for line in f:  ## 一行一行讀取
                    words = line.split() + ['<eos>']  ## 切割單詞
                    tokens += len(words)  ## 有幾個單詞，tokens 就加幾次
                    for word in words:
                        self.dictionary.add_word(word) ##把字加到字典裏面

            # Tokenize file content  ## ??
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens) ## ??
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids

class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):  ## data_path='data', txt_path='train_txt', txt_filename = 'train_txt.txt', label_filename = 'train_label.txt'
        self.txt_path = os.path.join(data_path, txt_path)  ##合併目錄: data/train_txt
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)  ##合併目錄： data/train_txt.txt
        fp = open(txt_filepath, 'r')  ##讀取 data/train_txt.txt
        self.txt_filename = [x.strip() for x in fp]  ##去空格
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)  ##合併目錄： data/train_label.txt
        fp_label = open(label_filepath, 'r')  ## 讀取  data/train_label.txt
        labels = [int(x.strip()) for x in fp_label]  ## 去空格
        fp_label.close()
        self.label = labels  ## 標籤
        self.corpus = corpus  ## 字集
        self.sen_len = sen_len


    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])  ## data/train_txt/train_txt[index]
        fp = open(filename, 'r') ## fp: 打開 data/train_txt/train_txt[index] 
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))  ## np.zeros:用 0 填充 32 位元矩陣  ##longTensor ??
        count = 0
        clip = False  ## ??
        for words in fp:
            for word in words.split():  ## 分割 data/train_txt/train_txt[index] 裡面的 words
                if word.strip() in self.corpus.dictionary.word2idx:  ## 如果上述 word 有在 dict 裡面
                    if count > self.sen_len - 1: ## 32 - 1
                        clip = True  ##  ??
                        break
                    txt[count] = self.corpus.dictionary.word2idx[word.strip()] 
                    count += 1
            if clip: break
        label = torch.LongTensor([self.label[index]])  ## ??
        return txt, label
    def __len__(self):
        return len(self.txt_filename)
