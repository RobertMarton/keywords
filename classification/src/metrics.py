# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-09-03
Description :
auther : wcy
"""
# import modules
import os, sys
import numpy as np

curr_path = os.getcwd()
sys.path.append(curr_path)
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

__all__ = []


# define function
def get_metrics_ops(labels, predictions, num_labels, weights):
    cm, op = _streaming_confusion_matrix(labels, predictions, num_labels, weights)
    tf.logging.info(type(cm))
    tf.logging.info(type(op))

    return (tf.convert_to_tensor(cm), op)


def get_metrics(conf_mat, num_labels):
    precisions = []
    recalls = []
    f1s = []
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0
        f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)


    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1_mean = sum(f1s) / len(f1s)
    # f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1_mean


# main
if __name__ == '__main__':
    conf_mat = np.array([[97,  54,   9],
                         [77, 609,  41],
                         [7, 115, 462]])
    ret = get_metrics(conf_mat=conf_mat, num_labels=3)
    print(ret)

    # TURE 0.7404276583412609


