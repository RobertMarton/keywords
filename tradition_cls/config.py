# -*- coding: utf-8 -*-
class Config():
    word2vec_file = "model/w2v/w2v_200d.npy"
    word2vec_w2id_file = "model/w2v/word2id.npy"

    glove_file = "model/glove/glove_300d.npy"
    glove_w2id_file = "model/glove/"
    elmo_model = "model/ELMoForManyLangs/zhs.model/"
    emb_method = 'elmo_word2vec'  # elmo/glove/elmo_word2vec/elmo_glove/word2vec/all
    num_filters = 100
    k = [3, 4, 5]
    glove_dim = 300
    word2vec_dim = 200
    elmo_dim = 1024
    num_labels = 3
    use_gpu = False
    dropout = 0.5
    epochs = 1
    test_size = 0.3
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 4

