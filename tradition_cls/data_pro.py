# -*- coding: utf-8 -*-

import re
import os
import sys
from pyltp import Segmentor
from pyltp import SentenceSplitter
import numpy as np
import pickle
import os

from torch.utils.data import Dataset
segmentor = Segmentor()
cws_model_path = os.path.join('model/ltp_data_v3.4.0', 'cws.model')
segmentor.load(cws_model_path)


class Data(Dataset):
    def __init__(self, x, y):
        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]


def split_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    seg_list = []
    sents = SentenceSplitter.split(string)
    for sent in sents:
        seg_list.extend(segmentor.segment(sent))
    return seg_list


def load_data_labels(data_file):
    x_text = []
    y = []
    with open(data_file,'r',encoding = 'utf-8') as f:
        f.readline()
        for line in f:
            s = line.split(',')
            if len(s) == 4:
                id, label, title, content = s
            else:
                continue
            x_text.append(split_str(title.strip()) + split_str(content.strip()))
            y.append(int(label))
    y = np.array(y)
    return x_text,y
#   将word2id glove 的数据转化一下
def read_word2id(w2v_path,dim):
    fw = open(w2v_path,'r')
    fw.readline()
    vecs = []
    w2id = {}
    id2w = {}
    vecs.append([0] * dim)
    w2id['PAD'] = 0
    id2w[0] = 'PAD'
    vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))
    w2id['UNK'] = 1
    id2w[1] ='UNK'
    id = 2
    for line in fw:
        line = line.strip().split(' ')
        vec = list(map(float, line[1:]))
        if len(vec) != dim:
            continue
        word = line[0]
        vecs.append(vec)
        w2id[word] = id
        id2w[id] = word
        id += 1
    print('id',len(id2w),'vec',len(vecs),'dim',len(vecs[6]))
    print("vocab size: {}, dim: {}".format(len(w2id), dim))
    np.save("model/w2v/w2v_{}d.npy".format(dim), np.array(vecs, dtype=np.float32))
    np.save("model/w2v/word2id.npy", w2id)
    np.save("model/w2v/id2word.npy", id2w)

def read_glove(glove_path,dim):
    fw = open(glove_path,'r')
    fw.readline()
    vecs = []
    w2id = {}
    id2w = {}
    vecs.append([0] * dim)
    w2id['PAD'] = 0
    id2w[0] = 'PAD'
    vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))
    w2id['UNK'] = 1
    id2w[1] ='UNK'
    id = 2
    for line in fw:
        line = line.strip().split(' ')
        word = line[0]
        vec = list(map(np.float32, line[1:]))
        vecs.append(vec)
        w2id[word] = id
        id2w[id] = word
        id += 1
    print("vocab size: {}, dim: {}".format(len(w2id), dim))
    np.save("model/glove/glove_{}d.npy".format(dim), np.array(vecs, dtype=np.float32))
    np.save("model/glove/word2id.npy", w2id)
    np.save("model/glove/id2word.npy", id2w)


def extract_vocab(data_file):
    '''
    extract vocab from txt
    '''
    examples = list(open(data_file, "r", encoding='utf-8').readlines())
    examples = [s.strip() for s in examples]
    x_text = [split_str(sent) for sent in examples]
    vocab = []
    for line in x_text:
        vocab.extend(line)
    vocab = list(set(vocab))
    print("vocab size: {}.".format(len(vocab)))
    open("./data/vocab.txt", "w").write("\n".join(vocab))

if __name__ == "__main__":
    read_word2id('./model/Tencent_AILab_ChineseEmbedding.txt',200)
    #read_glove

