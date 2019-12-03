# -*- coding: utf-8 -*-

from elmoformanylangs import Embedder
import torch
import torch.nn as nn
import numpy as np


class DataModel():
    def __init__(self, opt):

        super(DataModel, self).__init__()
        self.opt = opt
        self.use_gpu = self.opt.use_gpu

        if self.opt.emb_method == 'elmo':
            self.init_elmo()
        elif self.opt.emb_method == 'glove':
            self.init_glove()
        elif self.opt.emb_method == 'word2vec':
            self.init_word2vec()
        elif self.opt.emb_method == 'elmo_word2vec':
            self.init_word2vec()
            self.init_elmo()
            self.word_dim = self.opt.elmo_dim + self.opt.word2vec_dim
        elif self.opt.emb_method == 'elmo_glove':
            self.init_elmo()
            self.init_glove()
            self.word_dim = self.opt.elmo_dim + self.opt.glove_dim
        elif self.opt.emb_method == 'all':
            self.init_elmo()
            self.init_glove()
            self.init_word2vec()
            self.word_dim = self.opt.elmo_dim + self.opt.glove_dim + self.opt.word2vec_dim

    def init_elmo(self):
        '''
        initilize the ELMo model
        '''
        self.elmo = Embedder(self.opt.elmo_model,batch_size=self.opt.batch_size)
    def init_word2vec(self):
        self.word2vec_word2id = np.load(self.opt.word2vec_w2id_file).tolist()
        self.word2vec_vocab_size = len(self.word2vec_word2id)
        self.word2vec = nn.Embedding(self.word2vec_vocab_size, self.opt.word2vec_dim)
        emb = torch.from_numpy(np.load(self.opt.word2vec_file))
        self.word2vec.weight.data.copy_(emb)

    def init_glove(self):
        self.glove_word2id = np.load(self.opt.glove_w2id_file).tolist()
        self.glove_vocab_size = len(self.glove_word2id)
        self.glove = nn.Embedding(self.glove_vocab_size, self.opt.glove_dim)
        emb = torch.from_numpy(np.load(self.opt.glove_file))
        self.glove.weight.data.copy_(emb)

    def get_elmo(self, sentence_lists):
        '''
        get the ELMo word embedding vectors for a sentences
        '''
        max_len = max(map(lambda x: len(x), sentence_lists))
        sentence_lists =  self.elmo.sents2elmo(sentence_lists)
        sentence_end = []
        # 统一长度
        for sentence in sentence_lists:
            sentence = sentence.tolist()
            for i in range(max_len - len(sentence)):
                sentence.append([0] * self.opt.elmo_dim)
            sentence_end.append(sentence)
        return torch.FloatTensor(sentence_end)

    def get_word2vec(self,sentence_lists):
        # 分词之后的 向量
        max_len = max(map(lambda x: len(x), sentence_lists))
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2vec_word2id.get(w, 1), x)), sentence_lists))
        # 补充 全部是 0  的向量
        sentence_lists = list(map(lambda x: x + [0] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists)
        embeddings = self.word2vec(sentence_lists)
        return embeddings
    #分字之后的向量
    def get_glove(self, sentence_lists):
        '''
        get the glove word embedding vectors for a sentences
        '''
        max_len = max(map(lambda x: len(x), sentence_lists))
        # UNK   --> 1
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 1), x)), sentence_lists))
        # 补充全部是 0  的向量
        sentence_lists = list(map(lambda x: x + [0] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists)
        embeddings = self.glove(sentence_lists)
        return embeddings

    def get_data(self, x):
        if self.opt.emb_method == 'elmo':
            word_embs = self.get_elmo(x)
        elif self.opt.emb_method == 'glove':
            word_embs = self.get_glove(x)
        elif self.opt.emb_method == 'elmo_word2vec':
            word2vec = self.get_word2vec(x)
            elmo = self.get_elmo(x)
            word_embs = torch.cat([elmo,word2vec],-1)
        elif self.opt.emb_method == 'elmo_glove':
            glove = self.get_glove(x)
            elmo = self.get_elmo(x)
            word_embs = torch.cat([elmo, glove], -1)
        elif self.opt.emb_method == 'all':
            glove = self.get_glove(x)
            word2vec = self.get_word2vec(x)
            elmo = self.get_elmo(x)
            word_embs = torch.cat([elmo, glove, word2vec], -1)
        return word_embs
