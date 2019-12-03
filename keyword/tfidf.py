# -*- coding: utf-8 -*-
# @Author  : renxiaokai
# @Email   : renhongkai27@163.com
# @File    : tfidf.py
# @Time    : 2019-09-10 11:13

import argparse
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


def get_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_zr_deal/train_1_mini.csv',
                        help='train_path')
    parser.add_argument('--train_tfidf_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_zr_deal/train_1_mini_tfidf.csv',
                        help='train_path')
    parser.add_argument('--dev_path', default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_zr_deal/dev_1_mini.csv',
                        help='eval_path')
    parser.add_argument('--dev_tfidf_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_zr_deal/dev_1_mini_tfidf.csv',
                        help='eval_path')
    parser.add_argument('--test_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_zr_deal/Test_DataSet_mini.csv',
                        help='test_path')
    parser.add_argument('--test_tfidf_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_zr_deal/Test_DataSet_mini_tfidf.csv',
                        help='test_path')
    parser.add_argument('--stop_words', default='/home/rhk/projects/gcp_vm/keywords/data/stop_words.txt', help='')
    args = parser.parse_args()
    return args


def get_corpus(file, stop_file):
    stop = [line.strip() for line in open(stop_file, 'r', encoding='utf-8').readlines()]
    dataset = pd.read_csv(file, sep=',')
    corpus = []
    for row in dataset.iterrows():
        content_remove_stop = []
        content_seg = jieba.cut(str(row[1]['content']), cut_all=False)
        for word in content_seg:
            if word not in stop:
                content_remove_stop.append(word)
        corpus.append(" ".join(content_remove_stop))

    return corpus


def calc_tfidf(vectorizer, data):
    return vectorizer.transform(data).toarray()
c

def write_train_dev(file, new_file, tfidf, vocabulary):
    dataset = pd.read_csv(file, sep=',')

    content_tfidf = []
    for i in range(len(tfidf)):
        content_dict = {}
        for j in range(len(vocabulary)):
            if tfidf[i][j] != 0:
                content_dict[vocabulary[j]] = tfidf[i][j]
                print(str(vocabulary[j]) + '/' + str(tfidf[i][j]))
        print('')
        content_dict_sort = sorted(content_dict.items(), key=lambda t: t[1])
        content_tfidf.append("_".join([item[0] for item in content_dict_sort]))
    dataframe = pd.DataFrame(
        {'id': dataset['id'], 'label': dataset['label'], 'title': dataset['title'], 'content': dataset['content'],
         'content_keywords': content_tfidf})
    dataframe.to_csv(new_file, index=False, sep=',', columns=['id', 'label', 'title', 'content', 'content_keywords'])


def write_test(file, new_file, tfidf, vocabulary):
    dataset = pd.read_csv(file, sep=',')

    content_tfidf = []
    for i in range(len(tfidf)):
        content_dict = {}
        for j in range(len(vocabulary)):
            if tfidf[i][j] != 0:
                content_dict[vocabulary[j]] = tfidf[i][j]
        content_dict_sort = sorted(content_dict.items(), key=lambda t: t[1])
        content_dict_sort = content_dict_sort[:len(content_dict_sort)//2]
        content_tfidf.append("_".join([item[0] for item in content_dict_sort]))
    dataframe = pd.DataFrame(
        {'id': dataset['id'], 'title': dataset['title'], 'content': dataset['content'],
         'content_keywords': content_tfidf})
    dataframe.to_csv(new_file, index=False, sep=',', columns=['id', 'title', 'content', 'content_keywords'])


def main():
    args = get_parse()
    print("Reading corpus...")
    corpus = get_corpus(args.train_path, args.stop_words)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names()

    print("Reading train...")
    train = get_corpus(args.train_path, args.stop_words)
    print("Reading dev...")
    dev = get_corpus(args.dev_path, args.stop_words)
    print("Reading test...")
    test = get_corpus(args.test_path, args.stop_words)

    print("Calc train...")
    train_tfidf = calc_tfidf(vectorizer, train)
    print("Calc dev...")
    dev_tfidf = calc_tfidf(vectorizer, dev)
    print("Calc test...")
    test_tfidf = calc_tfidf(vectorizer, test)

    write_train_dev(args.train_path, args.train_tfidf_path, train_tfidf, vocabulary)
    write_train_dev(args.dev_path, args.dev_tfidf_path, dev_tfidf, vocabulary)
    write_test(args.test_path, args.test_tfidf_path, test_tfidf, vocabulary)


if __name__ == '__main__':
    main()
