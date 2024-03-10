import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from data import clusters, get_cluster_weights

def weighted_roc_auc(y_true, y_pred, labels, weights_dict):
    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)

cluster_weights_ = get_cluster_weights()
weights_dict_ = cluster_weights_["unnorm_weight"].to_dict()

def score(y_true, y_pred):
    return weighted_roc_auc(y_true, y_pred, labels=clusters, weights_dict=weights_dict_)