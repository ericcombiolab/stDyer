import numpy as np
import scipy.special
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import normalized_mutual_info_score, accuracy_score


def recover_labels(y_true, y_pred):
    print("Will change ari results, use it with caution")
    unique_cluster, y_pred = np.unique(y_pred, return_inverse=True)
    basic_counting_mat = contingency_matrix(y_true, y_pred)
    if len(unique_cluster) == len(np.unique(y_true)):
        counting_mat = basic_counting_mat
    else:
        lcm = np.lcm.reduce(basic_counting_mat.shape)
        counting_mat = basic_counting_mat
        for i in range(lcm // basic_counting_mat.shape[0] - 1):
            counting_mat = np.vstack([counting_mat, basic_counting_mat])
        zero_mat = np.zeros((lcm, lcm - basic_counting_mat.shape[1]))
        counting_mat = np.hstack([counting_mat, zero_mat])
    row_idx, col_idx = linear_sum_assignment(counting_mat.max() - counting_mat)
    y_pred_anno_dict = {c: r % basic_counting_mat.shape[0] for r, c in zip(row_idx, col_idx) if c < basic_counting_mat.shape[1]}
    y_pred_anno = np.array([y_pred_anno_dict[k] for k in y_pred])
    print("label mapping dict:", y_pred_anno_dict)
    print("y_true:", np.unique(y_true, return_counts=True))
    print("y_pred:", np.unique(y_pred, return_counts=True))
    print("y_pred_anno:", np.unique(y_pred_anno, return_counts=True))
    return y_pred_anno

# Code taken from the work
# VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
def cluster_acc(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max()-w)
    return sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0/Y_pred.size

def acc(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return accuracy_score(Y, recover_labels(Y, Y_pred))

def nmi(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y, Y_pred, average_method='arithmetic')
