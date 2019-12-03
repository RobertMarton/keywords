# -*- coding: utf-8 -*-
# @Author  : renxiaokai
# @Email   : renhongkai27@163.com
# @File    : jieba_tfidf.py
# @Time    : 2019-09-12 11:41


import argparse
import pandas as pd
import jieba
import jieba.analyse


def get_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/train.csv',
                        help='train_path')
    parser.add_argument('--train_tfidf_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/train_jiebatfidf.csv',
                        help='train_path')
    parser.add_argument('--dev_path', default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/dev.csv',
                        help='eval_path')
    parser.add_argument('--dev_tfidf_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/dev_jiebatfidf.csv',
                        help='eval_path')
    parser.add_argument('--test_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/Test_DataSet.csv',
                        help='test_path')
    parser.add_argument('--test_tfidf_path',
                        default='/home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/Test_DataSet_jiebatfidf.csv',
                        help='test_path')
    parser.add_argument('--stop_words', default='/home/rhk/projects/gcp_vm/keywords/data/stop_words.txt', help='')
    args = parser.parse_args()
    return args


def write_train_dev(file, new_file):
    dataset = pd.read_csv(file, sep=',')

    dataset_keywords = []
    for row in dataset.iterrows():
        try:
            keywords = jieba.analyse.extract_tags(row[1]['content'], topK=20)
            dataset_keywords.append("_".join(keywords))
        except:
            dataset_keywords.append(row[1]['content'])

    dataframe = pd.DataFrame(
        {'id': dataset['id'], 'label': dataset['label'], 'title': dataset['title'], 'content': dataset['content'],
         'content_keywords': dataset_keywords})
    dataframe.to_csv(new_file, index=False, sep=',', columns=['id', 'label', 'title', 'content', 'content_keywords'])


def write_test(file, new_file):
    dataset = pd.read_csv(file, sep=',')

    dataset_keywords = []
    for row in dataset.iterrows():
        try:
            keywords = HanLP.extractKeyword(row[1]['content'], 20)
            dataset_keywords.append("_".join(keywords))
        except:
            dataset_keywords.append(row[1]['content'])

    dataframe = pd.DataFrame(
        {'id': dataset['id'], 'title': dataset['title'], 'content': dataset['content'],
         'content_keywords': dataset_keywords})
    dataframe.to_csv(new_file, index=False, sep=',', columns=['id', 'title', 'content', 'content_keywords'])


def main():
    args = get_parse()

    write_train_dev(args.train_path, args.train_tfidf_path)
    write_train_dev(args.dev_path, args.dev_tfidf_path)
    write_test(args.test_path, args.test_tfidf_path)


if __name__ == '__main__':
    main()
