import sys
from sys import argv
import os
import re
import numpy
import string
import json
import torch
# from matplotlib import pyplot as plt

import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Tester

from config import Config

def data_analysis(data_path):
    data_set = DataSet()
    sample_num = 0
    sample_len = []
    scores = []
    if os.path.exists(data_path):
        with open(data_path,'r',encoding='utf-8') as fin:
            for lid, line in enumerate(fin):
                joke = json.loads(line)
                if len(joke['content']) > 0:
                    scores.append(joke['support'])
                    sample_num += 1
                    sample_len.append(len(joke['content'] ))
    else:
        print("the data path doesn't  exit.")
    print("Got {} samples from file.".format(sample_num))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.hist(scores, bins=50, range=(0,1500))
    plt.savefig("./sample_scores.jpg")
    count = 0
    for i in scores:
        if i >= 3:
            count += 1
    print(count,'/',len(sample_len))
    return

def get_joke_data(data_path):
    data_set = DataSet()
    sample_num = 0
    sample_len = []
    if os.path.exists(data_path):
        with open(data_path,'r',encoding='utf-8') as fin:
            for lid, line in enumerate(fin):
                joke = json.loads(line)
                if joke['support'] > 0:
                    if len(joke['content']) == 0:
                        continue
                    else:
                        instance = Instance(raw_joke=joke['content'])
                        data_set.append(instance)
                        sample_num += 1
                        sample_len.append(len(joke['content']))
    else:
        print("the data path doesn't  exit.")
    print("Got {} samples from file.".format(sample_num))
    for i in range(5):
        import random
        id = random.randint(0,sample_num)
        print("sample {}: {}".format(id, data_set[id]['raw_joke']))
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.hist(sample_len, bins=50, range=(0,1000))
    plt.savefig("./examples.jpg")
    count = 0
    for i in sample_len:
        if i < 255:
            count += 1
    print(count,'/',len(sample_len))
    return data_set

class JokeData(object):
    data_set = None
    train_data = None
    test_data = None
    vocab = None
    data_num = 0
    vocab_size = 0
    max_seq_len = 0

    def __init__(self, conf):
        print(conf.data_path)
        self.data_set = get_joke_data(conf.data_path)
        self.data_num = len(self.data_set)
        self.data_set.apply(self.split_chinese_sent,new_field_name='words')
        self.max_seq_len = min(self.max_seq_len,conf.max_seq_len)
        self.data_set.apply(lambda x : len(x['words']),new_field_name='seq_len')
        self.train_data,self.test_data = self.data_set.split(0.2)

    def split_chinese_sent(self,ins,remove_punc=False):
        line = ins['raw_joke'].strip()
        words = ['<START>']
        for c in line:
            if c in ['，','。','？','！']:
                if remove_punc:
                    continue
                else:
                    words.append(c)
            else:
                words.append(c)
        words.append('<EOS>')
        self.max_seq_len = max(self.max_seq_len,len(words))
        return words
    
    def split_sent(self,ins,remove_punc=False):
        words = ['<START>'] + ins['raw_joke'].split() + ['<EOS>']
        self.max_seq_len = max(self.max_seq_len,len(words))
        return words

    def pad_seq(self,ins):
        words = ins['words']
        if(len(words) < self.max_seq_len):
            words = [0]*(self.max_seq_len-len(words)) + words
        else:
            words = words[:self.max_seq_len]
        return words
        
    def get_vocab(self):
        self.vocab = Vocabulary(min_freq=1)
        self.train_data.apply(lambda x : [self.vocab.add(word) for word in x['words']])
        self.vocab.build_vocab()
        self.vocab.build_reverse_vocab()
        self.vocab_size = self.vocab.__len__()

        self.train_data.apply(lambda x : [self.vocab.to_index(word) for word in x['words']],new_field_name='words')
        self.train_data.apply(self.pad_seq,new_field_name='pad_words')
        
        self.test_data.apply(lambda x : [self.vocab.to_index(word) for word in x['words']],new_field_name='words')
        self.test_data.apply(self.pad_seq,new_field_name='pad_words')


if __name__ == "__main__":
    # get_joke_data('/home/ubuntu/nlp-project/JokeGenerator/data/chinese_joke/duanzi.json')
    # conf = Config()
    # data = JokeData(conf)
    # print(len(data.data_set))
    # print(len(data.train_data))
    # data.get_vocab()
    # import pickle
    # with open('/home/ubuntu/nlp-project/JokeGenerator/data/vocab/vocab1_cn.pickle', 'wb') as fout:
    #     pickle.dump(data, fout)


    data_analysis('/home/ubuntu/nlp-project/JokeGenerator/data/chinese_joke/duanzi.json')