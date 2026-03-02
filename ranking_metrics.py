import numpy as np
import pandas as pd


def precision_at_k(y_true, y_scores, k=8):
    """
    Precision@K = Relevant items in top K / K
    """
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[sorted_indices]) / k


def recall_at_k(y_true, y_scores, k=8):
    """
    Recall@K = Relevant items in top K / Total relevant items
    """
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[sorted_indices]) / np.sum(y_true)


def ndcg_at_k(y_true, y_scores, k=8):
    """
    NDCG@K (Normalized Discounted Cumulative Gain)
    """
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    gains = y_true[sorted_indices]
    
    discounts = 1 / np.log2(np.arange(2, k + 2))
    
    dcg = np.sum(gains * discounts)
    
    ideal_sorted = np.sort(y_true)[::-1][:k]
    ideal_dcg = np.sum(ideal_sorted * discounts)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0